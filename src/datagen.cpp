#include "datagen.h"
#include "search.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <memory>

namespace {

std::mutex g_outMutex;
std::FILE* g_log = nullptr;
std::atomic<uint64_t> g_gamesDone{0}, g_positions{0}, g_whiteWins{0}, g_blackWins{0}, g_draws{0}, g_totalPlies{0};
std::atomic<uint64_t> g_randomPlies{0};      // plies that drew a random move from the window
std::atomic<uint64_t> g_alternatives{0};     // ... and played another move than the best
std::atomic<uint64_t> g_noAlternative{0};    // plies that were to get a random move but had no other move in the window
std::atomic<uint64_t> g_costly{0};           // alternatives that cost more than the window by the next search
std::atomic<uint64_t> g_noResult{0};         // positions recorded without a result label (flags bit 1)
std::atomic<uint64_t> g_duplicates{0};       // positions not recorded because they were recorded earlier in this run
std::atomic<uint64_t> g_shallow{0};          // positions not recorded because the search did not complete depth 1
std::atomic<uint64_t> g_openingsTried{0};    // random-play positions searched for the score window ...
std::atomic<uint64_t> g_openingsKept{0};     // ... and found within it

constexpr uint32_t RANDOM_OPENING = UINT32_MAX;   // a game's opening when it is not a book position (openings_of)

// Keys of the positions recorded so far, one slot per key hash, always replaced (Stockfish's
// was_seen_before).  Many games open alike, so without this their first plies would be recorded over and
// over.  Races between threads only cost an occasional duplicate.
std::unique_ptr<std::atomic<uint64_t>[]> g_seen;
size_t g_seenMask = 0;

// Console formatting: "6,646,476", "49m32s".
std::string with_commas(uint64_t v) {
    std::string s = std::to_string(v), out;
    int n = 0;
    for (int i = int(s.size()) - 1; i >= 0; --i) {
        out.insert(out.begin(), s[i]);
        if (++n % 3 == 0 && i > 0) out.insert(out.begin(), ',');
    }
    return out;
}

std::string duration(double secs) {
    char buf[32];
    long t = long(secs + 0.5);
    if (t < 60) std::snprintf(buf, sizeof buf, "%lds", t);
    else if (t < 3600) std::snprintf(buf, sizeof buf, "%ldm%02lds", t / 60, t % 60);
    else std::snprintf(buf, sizeof buf, "%ldh%02ldm", t / 3600, (t % 3600) / 60);
    return buf;
}

// Progress is reported at most 50 times per run, at a round number of games (every 1000 of 50,000).
uint64_t progress_step(uint64_t games) {
    uint64_t step = 1;
    for (bool five = true; games / step > 50; five = !five) step *= five ? 5 : 2;   // 1, 5, 10, 50, 100, 500, 1000, ...
    return step;
}

std::chrono::steady_clock::time_point g_runStart;
uint64_t g_gamesAtStart = 0, g_positionsAtStart = 0, g_progressStep = 50;

bool seen_before(Key key) {
    if (!g_seen) return false;
    std::atomic<uint64_t>& slot = g_seen[size_t(key) & g_seenMask];
    if (slot.load(std::memory_order_relaxed) == key) return true;
    slot.store(key, std::memory_order_relaxed);
    return false;
}

// The dedup key of a position is a hash of its record (board and side to move), not the position's
// Zobrist key, so that a resumed run can rebuild the table from the records already on disk.
uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

Key record_key(const TrainingRecord& r) {
    uint64_t h = mix64(r.occ);
    uint64_t w[2];
    std::memcpy(w, r.pcs, 16);
    h = mix64(h ^ w[0]);
    h = mix64(h ^ w[1]);
    return Key(mix64(h ^ r.stm));
}

// What an interrupted run left behind: the games in the log (one line each, "fen|moves|result") with
// their results and lengths, and the positions in the output file, which go back into the dedup table.
struct Progress {
    uint64_t games = 0, whiteWins = 0, blackWins = 0, draws = 0, plies = 0, positions = 0;
};

Progress reload_progress(const DatagenOptions& opt) {
    Progress p;
    std::ifstream log(opt.log);
    std::string line;
    while (std::getline(log, line)) {
        p.games++;
        size_t bar1 = line.find('|'), bar2 = bar1 == std::string::npos ? bar1 : line.find('|', bar1 + 1);
        if (bar2 == std::string::npos) continue;   // a line cut short by the interruption
        for (size_t i = bar1 + 1; i < bar2; ++i) p.plies += line[i] == ' ';   // every move is followed by a space
        std::string result = line.substr(bar2 + 1);
        if (result == "1-0") p.whiteWins++; else if (result == "0-1") p.blackWins++; else p.draws++;
    }
    if (std::FILE* f = std::fopen(opt.out.c_str(), "rb")) {
        TrainingRecord recs[4096];
        size_t n;
        while ((n = std::fread(recs, sizeof(TrainingRecord), 4096, f)) > 0) {
            for (size_t i = 0; i < n; ++i) seen_before(record_key(recs[i]));
            p.positions += n;
        }
        std::fclose(f);
    }
    return p;
}

// The soft node budget of a search (NODES_HARD_FACTOR in datagen.h).
Limits search_limits(uint64_t nodes) {
    Limits lim;
    lim.softNodes = std::max<uint64_t>(1, nodes);
    lim.nodes = lim.softNodes <= UINT64_MAX / NODES_HARD_FACTOR ? lim.softNodes * NODES_HARD_FACTOR : UINT64_MAX;
    return lim;
}

// A random move, as Stockfish's random_multi_pv: a second search with several principal variations, then one of
// the top lines within randomMultiPVDiff of the best (the best move included) is played, so that the game goes on
// at engine strength and its result still says something about the earlier positions.  Returns MOVE_NONE when no
// other move lies within the window: a forcing position gets the best move rather than a blunder.  Each line gets
// half the move's node budget: the lines only choose the move (the position's label comes from the search before),
// and a shallower look at them hardly changes the choice.
Move choose_multipv_move(Engine& engine, const Limits& lim, int legalMoves, const DatagenOptions& opt, PRNG& rng) {
    const int pvs = std::min(opt.randomMultiPV, legalMoves);
    engine.shared.params.multiPV = pvs;
    SearchResult r = engine.search(search_limits(lim.softNodes * pvs / 2));
    engine.shared.params.multiPV = 1;
    const std::vector<RootLine>& lines = r.lines;
    if (lines.size() < 2 || lines[0].move == MOVE_NONE || is_mate_score(lines[0].score)) return MOVE_NONE;
    size_t n = 1;
    while (n < lines.size() && lines[n].move != MOVE_NONE && lines[0].score - lines[n].score <= opt.randomMultiPVDiff) n++;
    if (n < 2) return MOVE_NONE;
    return lines[rng.rand_below(n)].move;
}

void encode(const Position& pos, TrainingRecord& r) {
    std::memset(&r, 0, sizeof(r));
    r.occ = pos.pieces();
    Bitboard b = r.occ;
    int i = 0;
    while (b) {
        Square s = pop_lsb(b);
        uint8_t code = uint8_t(pos.piece_on(s));
        if (i & 1) r.pcs[i >> 1] |= uint8_t(code << 4);
        else r.pcs[i >> 1] |= code;
        i++;
    }
    r.stm = uint8_t(pos.side_to_move());
    r.ply = uint8_t(std::min(pos.game_ply(), 255));
}

// The positions of --book: one per line that holds a legal position, as a FEN or an EPD line (operations after the
// four position fields are ignored, as in the Stockfish books).  Other lines are counted and left out.
struct Book {
    std::vector<std::string> fens;
    uint64_t skipped = 0;
};

bool load_book(const std::string& file, Book& book) {
    std::ifstream in(file);
    if (!in) { std::printf("datagen: cannot open the book %s\n", file.c_str()); return false; }
    Position pos;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // Windows line ends
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        if (pos.set(line)) book.fens.push_back(pos.fen());
        else book.skipped++;
    }
    if (book.fens.empty()) {
        std::printf("datagen: %s holds no legal position (a FEN or an EPD line per position; tools/gen_book.py reads PGN books)\n", file.c_str());
        return false;
    }
    return true;
}

// The opening of each game: RANDOM_OPENING for the games that start from a random-play position, randomOpeningPct
// of every hundred games in a shuffled order, and for the others the index of their book position, the book taken
// in a random order and each position once before any comes round again (fishtest's order=random).  Fixed by the
// run's own seed, not the one --resume moves on, so that a resumed run keeps the assignment.
std::vector<uint32_t> openings_of(const DatagenOptions& opt, size_t bookSize, uint64_t seed) {
    PRNG rng(seed ^ 0x2545F4914F6CDD1DULL);
    auto shuffled = [&rng](size_t n) {
        std::vector<uint32_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = uint32_t(i);
        for (size_t i = n; i > 1; --i) std::swap(v[i - 1], v[rng.rand_below(i)]);
        return v;
    };
    const std::vector<uint32_t> order = shuffled(bookSize);
    std::vector<uint32_t> openings(opt.games), slots;
    size_t fromBook = 0;
    for (uint64_t g = 0; g < opt.games; ++g) {
        if (g % 100 == 0) slots = shuffled(100);
        const bool random = slots[g % 100] < uint32_t(opt.randomOpeningPct);
        openings[g] = random ? RANDOM_OPENING : bookSize ? order[fromBook++ % bookSize] : 0;
    }
    return openings;
}

// A random-play opening: randomPliesMin to randomPliesMax uniformly random legal plies from the start position, tried
// again when that ends the game and, with a score window, when a search at the game's node budget scores the
// position outside it (for either side).  Within the window one side has the upper hand, which makes the game more
// often decisive than a balanced opening does, and the position is not yet decided.
Position random_opening(Engine& engine, const DatagenOptions& opt, PRNG& rng) {
    for (int tries = 1;; ++tries) {
        Position pos;
        pos.set(START_FEN);
        MoveList ml;
        const int plies = opt.randomPliesMin + int(rng.rand_below(uint64_t(opt.randomPliesMax - opt.randomPliesMin + 1)));
        for (int i = 0; i < plies; ++i) {
            pos.generate_legal(ml);
            if (ml.size == 0) break;
            pos.do_move(ml.moves[rng.rand_below(ml.size)]);
        }
        pos.generate_legal(ml);
        if (ml.size == 0 || pos.is_dead_position()) continue;
        if (opt.openingMaxScore > 0) {
            g_openingsTried++;
            engine.new_game();
            engine.set_position(pos, {});
            const SearchResult r = engine.search(search_limits(opt.nodes));
            const bool within = r.depth >= 1 && !is_mate_score(r.score) && std::abs(r.score) >= opt.openingMinScore
                             && std::abs(r.score) <= opt.openingMaxScore;
            if (!within && tries < 1000) continue;
            if (!within) {   // a window that random play cannot reach: better a game outside it than none
                std::lock_guard<std::mutex> lock(g_outMutex);
                std::printf("  warning: no random-play position within %d-%d millipawns in 1000 tries; one outside it is played\n",
                            opt.openingMinScore, opt.openingMaxScore);
                std::fflush(stdout);
            } else g_openingsKept++;
        }
        return pos;
    }
}

void worker(const DatagenOptions& opt, int threadId, const Book& book, const std::vector<uint32_t>& openings, std::FILE* out) {
    PRNG rng(opt.seed * 0x9E3779B97F4A7C15ULL + 0xABCDEF * (threadId + 1));
    Engine engine;
    engine.set_hash(opt.hashMB);
    engine.shared.params.silent = true;

    std::vector<TrainingRecord> game;
    std::vector<Key> keys;

    while (true) {
        uint64_t g = g_gamesDone.fetch_add(1);
        if (g >= opt.games) break;

        Position pos;
        if (openings[g] == RANDOM_OPENING) pos = random_opening(engine, opt, rng);
        else pos.set(book.fens.empty() ? std::string(START_FEN) : book.fens[openings[g]]);
        keys.clear();
        keys.push_back(pos.key());
        game.clear();
        engine.new_game();

        int result = 0;       // from White's point of view
        bool decided = false;
        std::string startFen = pos.fen(), moveLog;
        int drawStreak = 0;
        int plies = 0;
        // The records before this index get no result label (flags bit 1).  A random move is chosen within a window
        // of the best on the strength of the MultiPV search, which is shallow and can be wrong.  When the next search
        // shows that it cost more than the window allowed, the game was steered by a mistake the engine would not
        // have made, so its result no longer says what the earlier positions were worth; their scores stand.
        size_t noResultUpTo = 0;
        bool lastPlayedAlternative = false;   // the previous ply played a random move other than the best ...
        int lastBestScore = 0;                // ... whose score was this ...
        size_t recordsAtAlternative = 0;      // ... with this many records written up to it, its own position's included

        while (!decided) {
            MoveList ml;
            pos.generate_legal(ml);
            if (ml.size == 0) {
                result = pos.in_check() ? (pos.side_to_move() == WHITE ? 1 : -1) : 0;
                break;
            }
            if (pos.rule50_count() >= 100 || pos.is_dead_position() || plies >= opt.maxPlies) { result = 0; break; }
            int reps = 0;
            for (Key k : keys) reps += k == pos.key();
            if (reps >= 3) { result = 0; break; }

            std::vector<Key> history(keys.begin(), keys.end() - 1);
            engine.set_position(pos, history);
            const Limits lim = search_limits(opt.nodes * (80 + rng.rand_below(41)) / 100);
            SearchResult r = engine.search(lim);
            if (r.bestMove == MOVE_NONE) break;  // cannot happen: the position has legal moves
            const int score = r.score;

            // What the random move of the previous ply cost, as this search sees it: the best move's score then
            // against this position's, both from the mover's side (r.depth 0: a search cut off by the runaway-search
            // limit inside its first iteration, which has no score to give).
            if (lastPlayedAlternative && r.depth >= 1 && lastBestScore + score > opt.randomMultiPVDiff) {
                noResultUpTo = std::max(noResultUpTo, recordsAtAlternative);
                g_costly++;
            }
            lastPlayedAlternative = false;
            // Not recorded: positions in check or whose best move is a capture (their score is a tactical one), the
            // first plies of the game, a search that did not complete depth 1, and positions recorded earlier in this
            // run (checked last, so that the table holds only recorded positions).
            const bool eligible = !pos.in_check() && !pos.is_capture(r.bestMove) && plies >= opt.writeMinPly;
            if (eligible && r.depth < 1) g_shallow++;
            if (eligible && r.depth >= 1) {
                TrainingRecord rec;
                encode(pos, rec);
                if (seen_before(record_key(rec))) g_duplicates++;
                else {
                    rec.score = int16_t(std::max(-32000, std::min(32000, score)));
                    game.push_back(rec);
                }
            }

            // Draw adjudication.  With a bare king on the board the game is drawn unless the search finds a forced
            // mate of it (Thread::evaluate), so the rule applies from any ply there.
            const bool bareKing = pos.bare_king(WHITE) || pos.bare_king(BLACK);
            if ((bareKing || pos.game_ply() >= opt.adjDrawMinPly) && std::abs(score) <= opt.adjDrawScore) {
                if (++drawStreak >= opt.adjDrawPlies) { result = 0; decided = true; }
            } else drawStreak = 0;

            Move played = r.bestMove;
            if (opt.randomMovePct > 0 && !is_mate_score(score) && plies >= opt.randomMoveMinPly && plies < opt.randomMoveMaxPly
                && int(rng.rand_below(100)) < opt.randomMovePct) {
                const Move alt = choose_multipv_move(engine, lim, int(ml.size), opt, rng);
                if (alt == MOVE_NONE) g_noAlternative++;
                else {
                    g_randomPlies++;
                    if (alt != played) {
                        g_alternatives++;
                        lastPlayedAlternative = true;
                        lastBestScore = score;
                        recordsAtAlternative = game.size();
                    }
                    played = alt;
                }
            }
            pos.do_move(played);
            keys.push_back(pos.key());
            plies++;
            if (g_log) moveLog += move_to_string(played) + " ";
        }

        for (size_t i = 0; i < game.size(); ++i) {
            game[i].result = int8_t(result);
            if (i < noResultUpTo) game[i].flags |= 2;
        }
        g_noResult += noResultUpTo;
        {
            std::lock_guard<std::mutex> lock(g_outMutex);
            if (!game.empty()) std::fwrite(game.data(), sizeof(TrainingRecord), game.size(), out);
            std::fflush(out);
            if (g_log) {
                std::fprintf(g_log, "%s|%s|%s\n", startFen.c_str(), moveLog.c_str(), result > 0 ? "1-0" : result < 0 ? "0-1" : "1/2-1/2");
                std::fflush(g_log);
            }
        }
        g_positions += game.size();
        g_totalPlies += plies;
        if (result > 0) g_whiteWins++; else if (result < 0) g_blackWins++; else g_draws++;

        uint64_t done = g_whiteWins + g_blackWins + g_draws;
        if (done % g_progressStep == 0 || done == opt.games) {
            std::lock_guard<std::mutex> lock(g_outMutex);
            const double secs = elapsed_ms(g_runStart) / 1000.0;
            const uint64_t gamesThisRun = done - g_gamesAtStart, posThisRun = g_positions.load() - g_positionsAtStart;
            const double eta = gamesThisRun > 0 ? secs * double(opt.games - done) / double(gamesThisRun) : 0.0;
            std::printf("  %8s  %7s  %7s  %7s  %12s  %10.1f  %7.0f  %8s  %8s\n",
                        with_commas(done).c_str(), with_commas(g_whiteWins.load()).c_str(), with_commas(g_blackWins.load()).c_str(),
                        with_commas(g_draws.load()).c_str(), with_commas(g_positions.load()).c_str(),
                        double(g_totalPlies.load()) / std::max<uint64_t>(1, done), posThisRun / std::max(secs, 1e-9),
                        duration(secs).c_str(), done == opt.games ? "-" : duration(eta).c_str());
            std::fflush(stdout);
        }
    }
}

}  // namespace

bool run_datagen(const DatagenOptions& options) {
    DatagenOptions opt = options;   // --resume moves the seed on (below)
    if (opt.resume && opt.log.empty()) {
        std::printf("datagen: --resume needs --log (the log is where the games are counted)\n");
        return false;
    }
    std::printf("datagen\n");
    Book book;
    if (!opt.book.empty() && !load_book(opt.book, book)) return false;
    const std::vector<uint32_t> openings = openings_of(opt, book.fens.size(), options.seed);
    if (opt.dedupMB > 0) {
        size_t entries = 1;
        while (entries * 2 * sizeof(uint64_t) <= size_t(opt.dedupMB) << 20) entries *= 2;
        g_seen.reset(new std::atomic<uint64_t>[entries]());
        g_seenMask = entries - 1;
    }
    // The settings in effect, one per line.
    std::string randomMoves = "none";
    if (opt.randomMovePct > 0)
        randomMoves = std::to_string(opt.randomMovePct) + "% of plies " + std::to_string(opt.randomMoveMinPly) + "-" +
                      std::to_string(opt.randomMoveMaxPly - 1) + ": one of the top " + std::to_string(opt.randomMultiPV) +
                      " moves within " + std::to_string(opt.randomMultiPVDiff) + " millipawns of the best (one that costs more by the next "
                      "search leaves the positions before it without a result label)";
    std::string start;
    if (opt.randomOpeningPct > 0) {
        start = std::to_string(opt.randomOpeningPct) + "% random-play positions (" + std::to_string(opt.randomPliesMin) + "-" +
                std::to_string(opt.randomPliesMax) + " random plies";
        if (opt.openingMaxScore > 0)
            start += ", kept when a search scores them within " + std::to_string(opt.openingMinScore) + "-" +
                     std::to_string(opt.openingMaxScore) + " millipawns";
        start += ")";
    }
    if (opt.randomOpeningPct < 100) {
        if (!start.empty()) start += ", " + std::to_string(100 - opt.randomOpeningPct) + "% ";
        if (book.fens.empty()) start += "the start position";
        else {
            start += opt.book + " (" + with_commas(book.fens.size()) + " positions, taken in a random order";
            if (book.skipped > 0) start += "; " + with_commas(book.skipped) + " lines without a legal position left out";
            start += ")";
        }
    }
    const std::string limit = with_commas(opt.nodes) + " nodes per move";
    std::printf("  %-16s%s\n", "evaluation", NNUE::current()->path.c_str());
    std::printf("  %-16s%s\n", "openings", start.c_str());
    std::printf("  %-16s%s at %s, %d thread%s, seed %llu\n", "games", with_commas(opt.games).c_str(), limit.c_str(), opt.threads,
                opt.threads == 1 ? "" : "s", (unsigned long long)opt.seed);
    std::printf("  %-16s%s\n", "random moves", randomMoves.c_str());
    std::printf("  %-16sfrom ply %d; positions already recorded are skipped (table %d MB)\n", "recording", opt.writeMinPly, opt.dedupMB);
    std::printf("  %-16s%s%s%s\n", "output", opt.out.c_str(), opt.log.empty() ? "" : ", game log ", opt.log.c_str());

    uint64_t positionsBefore = 0;
    if (opt.resume) {
        const Progress p = reload_progress(opt);
        if (p.games >= opt.games) {
            std::printf("  %-16s%s games already in %s (%s positions): nothing to do\n", "resume",
                        with_commas(p.games).c_str(), opt.log.c_str(), with_commas(p.positions).c_str());
            g_seen.reset();
            return true;
        }
        if (p.games > 0) {
            std::printf("  %-16s%s games already played (%s positions), %s to go\n", "resume",
                        with_commas(p.games).c_str(), with_commas(p.positions).c_str(), with_commas(opt.games - p.games).c_str());
            // The counters carry on from the games done, so the progress lines continue where they stopped.
            g_gamesDone = p.games;
            g_whiteWins = p.whiteWins; g_blackWins = p.blackWins; g_draws = p.draws;
            g_totalPlies = p.plies;
            g_positions = positionsBefore = p.positions;
            opt.seed = mix64(opt.seed ^ (p.games * 0x5851F42D4C957F2DULL));   // new games, not the old ones again
        }
    }
    std::FILE* out = std::fopen(opt.out.c_str(), "ab");
    if (!out) { std::printf("datagen: cannot open %s\n", opt.out.c_str()); g_seen.reset(); return false; }
    if (!opt.log.empty()) {
        g_log = std::fopen(opt.log.c_str(), "a+");
        // A line cut short by an interruption is ended, so that the next game starts a line of its own.
        if (g_log && std::fseek(g_log, -1, SEEK_END) == 0) {
            const int last = std::fgetc(g_log);
            std::fseek(g_log, 0, SEEK_END);   // (a stream must be positioned between a read and a write)
            if (last != '\n') std::fputc('\n', g_log);
        }
    }
    g_runStart = now();
    g_gamesAtStart = g_gamesDone.load();
    g_positionsAtStart = positionsBefore;
    g_progressStep = progress_step(opt.games);
    std::printf("\n  %8s  %7s  %7s  %7s  %12s  %10s  %7s  %8s  %8s\n", "games", "white", "black", "draws", "positions", "plies/game", "pos/s", "elapsed", "ETA");
    std::fflush(stdout);
    std::vector<std::thread> th;
    for (int t = 0; t < opt.threads; ++t) th.emplace_back(worker, std::cref(opt), t, std::cref(book), std::cref(openings), out);
    for (auto& t : th) t.join();
    std::fclose(out);
    if (g_log) { std::fclose(g_log); g_log = nullptr; }
    const double secs = elapsed_ms(g_runStart) / 1000.0;
    const uint64_t games = g_whiteWins + g_blackWins + g_draws;
    const uint64_t newPositions = g_positions.load() - positionsBefore;
    const double pct = 100.0 / std::max<uint64_t>(1, games);
    std::printf("\n  %-16s%s games: white wins %s (%.1f%%), black wins %s (%.1f%%), draws %s (%.1f%%), %.1f plies per game\n", "done",
                with_commas(games).c_str(), with_commas(g_whiteWins.load()).c_str(), g_whiteWins.load() * pct,
                with_commas(g_blackWins.load()).c_str(), g_blackWins.load() * pct, with_commas(g_draws.load()).c_str(), g_draws.load() * pct,
                double(g_totalPlies.load()) / std::max<uint64_t>(1, games));
    if (positionsBefore > 0)
        std::printf("  %-16s%s in %s (%s of them in this run, in %s: %.0f pos/s)\n", "positions", with_commas(g_positions.load()).c_str(),
                    opt.out.c_str(), with_commas(newPositions).c_str(), duration(secs).c_str(), newPositions / std::max(secs, 1e-9));
    else
        std::printf("  %-16s%s in %s (%s, %.0f pos/s)\n", "positions", with_commas(g_positions.load()).c_str(), opt.out.c_str(),
                    duration(secs).c_str(), newPositions / std::max(secs, 1e-9));
    if (g_openingsTried > 0)
        std::printf("  %-16s%s random-play positions searched: %.1f%% lay within %d-%d millipawns and opened a game, the rest were "
                    "drawn again\n", "openings", with_commas(g_openingsTried.load()).c_str(),
                    100.0 * double(g_openingsKept.load()) / double(g_openingsTried.load()), opt.openingMinScore, opt.openingMaxScore);
    if (g_duplicates > 0)
        std::printf("  %-16s%s positions recorded earlier in this run (%.1f%% of those eligible)\n", "skipped",
                    with_commas(g_duplicates.load()).c_str(),
                    100.0 * double(g_duplicates.load()) / std::max<uint64_t>(1, g_duplicates.load() + newPositions));
    if (g_shallow > 0)
        std::printf("  %-16s%s positions whose search was cut off inside its first iteration (runaway-search limit)\n", "not searched",
                    with_commas(g_shallow.load()).c_str());
    g_seen.reset();
    if (g_randomPlies + g_noAlternative > 0) {
        std::printf("  %-16s%s plies drew one of the top %d moves within %d millipawns (%s of them an alternative to the best move); "
                    "%s plies had no alternative in the window and kept the best move\n", "random moves",
                    with_commas(g_randomPlies.load()).c_str(), opt.randomMultiPV, opt.randomMultiPVDiff,
                    with_commas(g_alternatives.load()).c_str(), with_commas(g_noAlternative.load()).c_str());
        std::printf("  %-16s%s of the alternatives (%.1f%%) cost more than %d millipawns by the next search: the positions before them have "
                    "no result label; %.1f%% of all the positions keep theirs\n", "", with_commas(g_costly.load()).c_str(),
                    100.0 * double(g_costly.load()) / std::max<uint64_t>(1, g_alternatives.load()), opt.randomMultiPVDiff,
                    100.0 * (1.0 - double(g_noResult.load()) / std::max<uint64_t>(1, newPositions)));
    }
    return true;
}

// ---- rescore --------------------------------------------------------------------------------------------

namespace {

// A record's position (src/datagen.h): the board alone - no castling rights, no en-passant square, as the
// trainer sees it.  False for a malformed record.
bool decode_record(const TrainingRecord& r, Position& pos) {
    Piece board[SQUARE_NB];
    for (int s = 0; s < SQUARE_NB; ++s) board[s] = NO_PIECE;
    Bitboard b = r.occ;
    int i = 0, kings = 0;
    while (b) {
        Square s = pop_lsb(b);
        uint8_t nib = (i & 1) ? (r.pcs[i >> 1] >> 4) : (r.pcs[i >> 1] & 15);
        if (nib >= 12) return false;
        board[s] = Piece(nib);
        kings += type_of(Piece(nib)) == KING;
        i++;
    }
    if (kings != 2) return false;
    pos.set_board(board, r.stm ? BLACK : WHITE);
    return true;
}

std::atomic<uint64_t> g_rescoreNext{0}, g_rescoreDone{0}, g_rescoreBad{0}, g_rescoreShallow{0};

// The hash table carries over from one position to the next within a worker, as it does within a game in
// datagen (consecutive records are consecutive positions of a game, so it helps), which is why two runs are
// not bit-identical: which positions share a worker differs.
void rescore_worker(const RescoreOptions& opt, std::vector<TrainingRecord>& recs) {
    Engine engine;
    engine.set_hash(opt.hashMB);
    engine.shared.params.silent = true;
    engine.new_game();
    constexpr uint64_t CHUNK = 256;
    while (true) {
        const uint64_t a = g_rescoreNext.fetch_add(CHUNK);
        if (a >= recs.size()) break;
        const uint64_t b = std::min<uint64_t>(a + CHUNK, recs.size());
        for (uint64_t k = a; k < b; ++k) {
            TrainingRecord& r = recs[k];
            Position pos;
            MoveList ml;
            if (!decode_record(r, pos) || (pos.generate_legal(ml), ml.size == 0)) { g_rescoreBad++; continue; }
            engine.set_position(pos, {});
            SearchResult res = engine.search(search_limits(opt.nodes));
            if (res.bestMove == MOVE_NONE) { g_rescoreBad++; continue; }
            if (res.depth < 1) { g_rescoreShallow++; continue; }   // cut off by the runaway-search limit: keep the old score
            r.score = int16_t(std::max(-32000, std::min(32000, res.score)));
            r.next = 0;                // the unused field (and its flag bit 2), which earlier versions of datagen set
            r.flags &= ~uint8_t(4);
        }
        g_rescoreDone += b - a;
    }
}

}  // namespace

bool run_rescore(const RescoreOptions& opt) {
    std::printf("rescore\n");
    // The files are rescored one at a time and appended to the output as they finish: the archive need not fit
    // in memory (a quarter of a billion records are 8 GB), only its largest file.
    uint64_t total = 0;
    for (const std::string& f : opt.in) {
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        if (!in) { std::printf("  cannot open %s\n", f.c_str()); return false; }
        const std::streamoff bytes = in.tellg();
        if (bytes % sizeof(TrainingRecord) != 0) {
            std::printf("  %s is not a record file (size not a multiple of %zu)\n", f.c_str(), sizeof(TrainingRecord));
            return false;
        }
        total += uint64_t(bytes) / sizeof(TrainingRecord);
    }
    std::printf("  %-16s%zu file%s, %s positions\n", "data", opt.in.size(), opt.in.size() == 1 ? "" : "s", with_commas(total).c_str());
    std::printf("  %-16s%s\n", "evaluation", NNUE::current()->path.c_str());
    std::printf("  %-16s%s node%s per position, %d thread%s\n", "search", with_commas(opt.nodes).c_str(), opt.nodes == 1 ? "" : "s",
                opt.threads, opt.threads == 1 ? "" : "s");
    std::printf("  %-16s%s\n", "output", opt.out.c_str());
    std::FILE* out = std::fopen(opt.out.c_str(), "wb");
    if (!out) { std::printf("  cannot write %s\n", opt.out.c_str()); return false; }
    std::printf("\n  %12s  %8s  %8s  %8s\n", "positions", "pos/s", "elapsed", "ETA");
    const auto start = std::chrono::steady_clock::now();
    const uint64_t step = progress_step(total);
    uint64_t doneBefore = 0, reported = 0;
    for (const std::string& f : opt.in) {
        std::vector<TrainingRecord> recs;
        {
            std::ifstream in(f, std::ios::binary | std::ios::ate);
            const std::streamoff bytes = in.tellg();
            in.seekg(0);
            recs.resize(size_t(bytes) / sizeof(TrainingRecord));
            in.read(reinterpret_cast<char*>(recs.data()), bytes);
        }
        g_rescoreNext = 0;
        g_rescoreDone = 0;
        std::vector<std::thread> pool;
        for (int t = 0; t < std::max(1, opt.threads); ++t) pool.emplace_back(rescore_worker, std::cref(opt), std::ref(recs));
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const uint64_t done = doneBefore + g_rescoreDone.load();
            const bool fileDone = g_rescoreDone.load() >= recs.size();
            if (done / step > reported / step || (fileDone && done >= total)) {
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double rate = done / std::max(secs, 1e-9);
                std::printf("  %12s  %8.0f  %8s  %8s\n", with_commas(done).c_str(), rate, duration(secs).c_str(),
                            done ? duration((total - done) / std::max(rate, 1e-9)).c_str() : "?");
                std::fflush(stdout);
                reported = done;
            }
            if (fileDone) break;
        }
        for (auto& t : pool) t.join();
        std::fwrite(recs.data(), sizeof(TrainingRecord), recs.size(), out);
        std::fflush(out);
        doneBefore += recs.size();
    }
    std::fclose(out);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("\n  %-16s%s positions rescored in %s (%.0f pos/s)%s\n", "done", with_commas(total).c_str(), duration(secs).c_str(),
                total / std::max(secs, 1e-9),
                g_rescoreBad.load() ? (", " + with_commas(g_rescoreBad.load()) + " left as they were (no legal move or malformed)").c_str() : "");
    if (g_rescoreShallow > 0)
        std::printf("  %-16s%s positions kept their old score: the search was cut off before depth 1 (runaway-search safety limit)\n",
                    "too shallow", with_commas(g_rescoreShallow.load()).c_str());
    return true;
}
