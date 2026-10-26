#include "uci.h"
#include "cpu.h"
#include "search.h"
#include "nnue.h"
#include "perft.h"
#include "datagen.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#if defined(_WIN32)
#include "memory.h"   // large_pages_used(), reported on Windows only
#include <windows.h>  // IWYU pragma: keep (GetModuleFileNameA is documented as included through it)
#else
#include <unistd.h>
#endif

namespace {

constexpr char VERSION[] = "1.0";   // a release is tagged v<VERSION>

struct UciState {
    Engine engine;
    Position pos;
    std::vector<Key> history;
    std::string evalFile = NNUE::DEFAULT_NET;
    int hashMB = 64;
    int threads = 1;
    std::string exeDir;   // directory of the executable (networks are also looked for there)
};

std::string next_token(std::istringstream& is) { std::string t; is >> t; return t; }

void try_load_net(UciState& st, bool verbose) {
    std::string err;
    // Relative paths are tried as given and then next to the executable (GUIs start engines from anywhere).
    bool ok = NNUE::load(st.evalFile, &err);
    bool absolute = !st.evalFile.empty() && (st.evalFile[0] == '/' || st.evalFile[0] == '\\' || st.evalFile.find(':') != std::string::npos);
    if (!ok && !st.exeDir.empty() && !st.evalFile.empty() && !absolute) {
        std::string alt = st.exeDir + "/" + st.evalFile;
        std::string err2;
        if (NNUE::load(alt, &err2)) { ok = true; st.evalFile = alt; }
    }
    if (ok) {
        if (verbose) print_line("info string loaded network " + st.evalFile + " (" + NNUE::current()->describe() + ")");
    } else {
        NNUE::current().reset();   // not the previous net either: a search runs with the net asked for or not at all
        if (verbose) print_line("info string " + err + " - set EvalFile to a network before searching");
    }
}

// The network is Tsumi's only evaluation: a command that searches without one says why and exits, as for a
// hash table that cannot be allocated, rather than play some other way (a GUI shows the engine stopping and
// its log has the reason).
void require_network(const UciState& st) {
    if (NNUE::available()) return;
    print_line("info string no network loaded (EvalFile " + st.evalFile + "): Tsumi cannot search without one");
    std::exit(EXIT_FAILURE);
}

void cmd_position(UciState& st, std::istringstream& is) {
    std::string token = next_token(is), fen;
    if (token == "startpos") {
        fen = START_FEN;
        token = next_token(is);
    } else if (token == "fen") {
        while (is >> token && token != "moves") fen += token + " ";
    } else return;
    Position p;
    if (!p.set(fen)) { print_line("info string invalid fen"); return; }
    st.history.clear();
    if (token == "moves" || token.empty()) {
        while (is >> token) {
            Move m = parse_uci_move(p, token);
            if (m == MOVE_NONE) { print_line("info string illegal move " + token); break; }
            st.history.push_back(p.key());
            p.do_move(m);
        }
    }
    st.pos = p;
    st.engine.set_position(st.pos, st.history);
}

void cmd_go(UciState& st, std::istringstream& is) {
    Limits lim;
    std::string token;
    while (is >> token) {
        if (token == "wtime") { is >> lim.time[WHITE]; lim.clock = true; }
        else if (token == "btime") { is >> lim.time[BLACK]; lim.clock = true; }
        else if (token == "winc") is >> lim.inc[WHITE];
        else if (token == "binc") is >> lim.inc[BLACK];
        else if (token == "movestogo") is >> lim.movestogo;
        else if (token == "depth") is >> lim.depth;
        else if (token == "nodes") is >> lim.nodes;
        else if (token == "movetime") is >> lim.movetime;
        else if (token == "mate") is >> lim.mate;
        else if (token == "infinite") lim.infinite = true;
        else if (token == "searchmoves") {
            // Consume tokens as long as they are legal moves in the current position.
            std::streampos pos;
            while ((pos = is.tellg()), (is >> token)) {
                Move m = parse_uci_move(st.pos, token);
                if (m == MOVE_NONE) { is.clear(); is.seekg(pos); break; }
                lim.searchMoves.push_back(m);
            }
        }
        else if (token == "perft") { int d = 1; is >> d; perft_divide(st.pos, d); return; }
    }
    require_network(st);
    st.engine.set_position(st.pos, st.history);
    st.engine.start_search_async(lim);
}

void cmd_setoption(UciState& st, std::istringstream& is) {
    std::string token, name, value;
    is >> token;  // "name"
    while (is >> token && token != "value") name += (name.empty() ? "" : " ") + token;
    while (is >> token) value += (value.empty() ? "" : " ") + token;
    if (name == "Hash") {
        st.hashMB = std::max(1, std::min(1 << 20, std::atoi(value.c_str())));
        st.engine.set_hash(st.hashMB);   // exits, saying why, if the system cannot provide the memory (tt.cpp)
#if defined(_WIN32)
        // Large pages spare every probe a page-table walk (memory.h); ordinary pages work too, a little slower.
        print_line("info string hash table: " + std::to_string(st.hashMB) + " MB in "
                   + (large_pages_used() ? "Windows large pages" : "ordinary pages (large pages need the 'Lock pages in memory' right)"));
#endif
    }
    else if (name == "Threads") { st.threads = std::max(1, std::min(512, std::atoi(value.c_str()))); st.engine.set_threads(st.threads); }
    else if (name == "EvalFile") { st.evalFile = value; try_load_net(st, true); }
    else if (name == "MoveOverhead") st.engine.shared.params.moveOverhead = std::max(0, std::atoi(value.c_str()));
    else if (name == "MultiPV") st.engine.shared.params.multiPV = std::max(1, std::min(256, std::atoi(value.c_str())));
    else print_line("info string unknown option " + name);
}

void cmd_eval(UciState& st) {
    std::cout << st.pos.pretty();
    if (!NNUE::available()) std::cout << "no network loaded (EvalFile " << st.evalFile << ")" << std::endl;
    else {
        const NNUE::Network& net = *NNUE::current();
        std::vector<int16_t> a(net.acc_width()), b(net.acc_width());
        NNUE::refresh(net, st.pos, st.pos.side_to_move(), a.data());
        NNUE::refresh(net, st.pos, ~st.pos.side_to_move(), b.data());
        MoveList moves;
        st.pos.generate_legal(moves);
        const int v = NNUE::evaluate(net, st.pos, a.data(), b.data(), NNUE::mobility(moves));
        std::cout << "NNUE (side to move): " << v << std::endl;   // internal units; the line train/verify_net.py reads
        std::ostringstream o;
        o << std::showpos << std::fixed << std::setprecision(2) << double(v) / (100 * UNITS_PER_CP);
        std::cout << "NNUE in pawns: " << o.str() << " (+1.00 is a 50% chance of winning)" << std::endl;
    }
    if (st.pos.bare_king(WHITE) || st.pos.bare_king(BLACK))
        std::cout << "bare king: the search evaluates this position as a draw unless it finds a forced mate (a shade below the draw "
                     "for the side with material, so that it ends the game when it can; see Thread::evaluate)" << std::endl;
    MoveList ml;
    st.pos.generate_legal(ml);
    std::cout << "Legal moves: " << ml.size << ", in check: " << st.pos.in_check() << ", dead: " << st.pos.is_dead_position() << std::endl;
}

void cmd_bench(UciState& st, int depth) {
    require_network(st);
    static const char* fens[] = {
        START_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "8/8/8/3k4/8/8/4K3/6Q1 w - - 0 1",
        "r1b1kb1r/2pp1ppp/1np1q3/p3P3/2P5/1P6/PB1NQPPP/R3KB1R b KQkq - 0 1",
        "3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1",
    };
    st.engine.set_threads(1);
    st.engine.shared.params.silent = true;
    uint64_t total = 0;
    auto t0 = now();
    for (const char* f : fens) {
        Position p;
        p.set(f);
        st.engine.new_game();
        st.engine.set_position(p, {});
        Limits lim;
        lim.depth = depth;
        SearchResult r = st.engine.search(lim);
        total += r.nodes;
        std::cout << "bench " << f << " : " << move_to_string(r.bestMove) << " " << score_to_uci(r.score) << " nodes " << r.nodes << std::endl;
    }
    int64_t ms = std::max<int64_t>(1, elapsed_ms(t0));
    std::cout << "===========================\nTotal time (ms) : " << ms << "\nNodes searched  : " << total
              << "\nNodes/second    : " << total * 1000 / ms << std::endl;
    st.engine.shared.params.silent = false;
    st.engine.set_threads(st.threads);
}

void print_uci_options(const UciState& st) {
    print_line(std::string("id name Tsumi ") + VERSION + "\n"
               "id author Sean Adamson\n"
               "option name Hash type spin default " + std::to_string(st.hashMB) + " min 1 max 1048576\n"
               "option name Threads type spin default " + std::to_string(st.threads) + " min 1 max 512\n"
               "option name EvalFile type string default " + st.evalFile + "\n"
               "option name MultiPV type spin default 1 min 1 max 256\n"
               "option name MoveOverhead type spin default 30 min 0 max 5000\n"
               "uciok");
}

bool handle_command(UciState& st, const std::string& line) {
    std::istringstream is(line);
    std::string cmd = next_token(is);
    if (cmd.empty()) return true;
    if (cmd == "uci") print_uci_options(st);
    else if (cmd == "isready") print_line("readyok");
    else if (cmd == "setoption") cmd_setoption(st, is);
    else if (cmd == "ucinewgame") st.engine.new_game();
    else if (cmd == "position") cmd_position(st, is);
    else if (cmd == "go") cmd_go(st, is);
    else if (cmd == "stop") { st.engine.stop(); st.engine.wait(); }
    else if (cmd == "ponderhit") {}
    else if (cmd == "wait") st.engine.wait();   // non-standard: block until the search finishes (useful in scripts)
    else if (cmd == "quit") { st.engine.stop(); st.engine.wait(); return false; }
    else if (cmd == "d") std::cout << st.pos.pretty() << std::endl;
    else if (cmd == "eval") cmd_eval(st);
    else if (cmd == "crosscheck") {
        // Test helper: read FENs from stdin until EOF; for each print the sorted legal moves ('+' = gives
        // check, verified against the slow check test and a consistency check), the FEN and the dead flag.
        std::string fenLine;
        while (std::getline(std::cin, fenLine)) {
            Position p;
            if (!p.set(fenLine)) { std::cout << "badfen" << std::endl; continue; }
            MoveList ml;
            p.generate_legal(ml);
            const CheckInfo ci = p.check_info();
            std::vector<std::string> v;
            for (Move m : ml) {
                Position next = p;
                next.do_move(m);
                bool gc = p.gives_check(m, ci);
                if (gc != next.in_check() || gc != p.gives_check(m)) std::cout << "GIVESCHECK_MISMATCH " << move_to_string(m) << " ";
                if (!next.consistent()) std::cout << "INCONSISTENT " << move_to_string(m) << " ";
                v.push_back(move_to_string(m) + (gc ? "+" : ""));
            }
            std::sort(v.begin(), v.end());
            for (auto& x : v) std::cout << x << " ";
            std::cout << "| " << p.fen() << " | " << p.is_dead_position() << std::endl;
        }
        return false;
    }
    else if (cmd == "legal") { MoveList ml; st.pos.generate_legal(ml); for (Move m : ml) std::cout << move_to_string(m) << (st.pos.gives_check(m) ? "+ " : " "); std::cout << "(" << ml.size << ")" << std::endl; }
    else if (cmd == "perft") { int d = 1; is >> d; perft_divide(st.pos, d); }
    else if (cmd == "test") std::cout << (run_perft_tests() ? "all perft tests passed" : "PERFT TESTS FAILED") << std::endl;
    else if (cmd == "bench") { int d = 10; is >> d; cmd_bench(st, d); }
    else if (cmd == "nnuetest") {
        // Walk random legal moves, comparing incremental accumulator updates against full refreshes.
        if (!NNUE::available()) { std::cout << "no network loaded" << std::endl; return true; }
        int plies = 2000; is >> plies;
        const NNUE::Network& net = *NNUE::current();
        const int W = net.acc_width();
        std::vector<int16_t> inc[2] = {std::vector<int16_t>(W), std::vector<int16_t>(W)};
        std::vector<int16_t> fresh[2] = {std::vector<int16_t>(W), std::vector<int16_t>(W)};
        PRNG rng(12345);
        Position pos = st.pos;
        for (Color c : {WHITE, BLACK}) NNUE::refresh(net, pos, c, inc[c].data());
        NNUE::AccumulatorCache cache;
        cache.reset(net.H);
        std::vector<int16_t> viaCache(W);
        int errors = 0, games = 0;
        for (int i = 0; i < plies; ++i) {
            MoveList ml;
            pos.generate_legal(ml);
            if (ml.size == 0 || pos.is_dead_position() || pos.rule50_count() > 100) {
                pos.set(START_FEN); games++;
                for (Color c : {WHITE, BLACK}) NNUE::refresh(net, pos, c, inc[c].data());
                continue;
            }
            Move m = ml.moves[rng.rand_below(ml.size)];
            DirtyPiece dp;
            Position next = pos;
            next.do_move(m, &dp);
            NNUE::ThreatDiff diff;   // shared by both perspectives, as in the search
            for (Color c : {WHITE, BLACK}) {
                bool kingMoved = false;
                for (int k = 0; k < dp.count; ++k) kingMoved |= dp.piece[k] == make_piece(c, KING);
                std::vector<int16_t> child(W);
                if (kingMoved) NNUE::refresh_cached(net, next, c, child.data(), cache, &pos, &dp, inc[c].data(), &diff);
                else NNUE::update(net, pos, next, c, dp, inc[c].data(), child.data(), diff);
                inc[c] = child;
                NNUE::refresh(net, next, c, fresh[c].data());
                if (fresh[c] != inc[c]) { errors++; if (errors < 5) std::cout << "accumulator mismatch after " << move_to_string(m) << " in " << pos.fen() << std::endl; inc[c] = fresh[c]; }
                if (rng.rand_below(4) == 0) {
                    NNUE::refresh_cached(net, next, c, viaCache.data(), cache);
                    if (fresh[c] != viaCache) { errors++; if (errors < 5) std::cout << "cache refresh mismatch in " << next.fen() << std::endl; }
                }
            }
            pos = next;
        }
        std::cout << "nnuetest: " << plies << " plies, " << games << " restarts, " << errors << " errors" << std::endl;
    }
    else print_line("info string unknown command: " + cmd);
    return true;
}

}  // namespace

void init_engine() {
    Bitboards::init();
    Zobrist::init();
    init_search_tables();
    NNUE::select_kernels(CPU::level());
}

int uci_loop(int argc, char** argv) {
    print_line(std::string("Tsumi ") + VERSION + " selfmate chess engine (" + CPU::description() + ")");
    UciState st;
    {
        // Directory of the executable, so a network next to the binary is found from any working directory.
        char buf[4096];
        std::string exe;
#if defined(_WIN32)
        DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf) - 1);
        if (n > 0) exe.assign(buf, n);
#else
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) exe.assign(buf, n);
#endif
        if (exe.empty() && argc > 0) exe = argv[0];
        size_t slash = exe.find_last_of("/\\");
        if (slash != std::string::npos) st.exeDir = exe.substr(0, slash);
    }
    st.pos.set(START_FEN);
    st.engine.set_hash(st.hashMB);
    try_load_net(st, true);

    // Commands given on the command line are executed in order, and then the engine exits: tsumi bench 13,
    // tsumi "position startpos" "go depth 12".  (bench and perft also take their arguments as separate words.)
    bool keepGoing = true;
    if (argc > 2 && (std::string(argv[1]) == "bench" || std::string(argv[1]) == "perft")) {
        std::string cmd = argv[1];
        for (int i = 2; i < argc; ++i) cmd += std::string(" ") + argv[i];
        handle_command(st, cmd);
        return 0;
    }
    for (int i = 1; i < argc && keepGoing; ++i) {
        keepGoing = handle_command(st, argv[i]);
        if (std::string(argv[i]).rfind("go", 0) == 0) st.engine.wait();
    }
    if (argc > 1) { st.engine.stop(); st.engine.wait(); return 0; }

    std::string line;
    while (keepGoing && std::getline(std::cin, line)) keepGoing = handle_command(st, line);
    st.engine.stop();
    st.engine.wait();
    return 0;
}

// datagen and rescore: the network given with --net, or the engine's own; without one they cannot run.
static bool load_net_or_report(const std::string& evalFile) {
    std::string err;
    if (NNUE::load(evalFile.empty() ? NNUE::DEFAULT_NET : evalFile, &err)) return true;
    std::printf("error: %s (a network is needed: --net FILE)\n", err.c_str());
    return false;
}

// Parses `--name value` options: `value(name)` returns the value when argv[i] is the option, advancing i past it.
struct OptionParser {
    int argc;
    char** argv;
    int i = 2;
    const char* value(const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc ? argv[++i] : nullptr; }
};

int datagen_main(int argc, char** argv) {
    DatagenOptions opt;
    std::string evalFile;
    const char* usage =
        "usage: tsumi datagen [--net FILE] [--threads N] [--games N] [--nodes N] [--out FILE] [--log FILE] [--resume]\n"
        "                     [--book FILE] [--seed N] [--hash MB] [--max-plies N] [--write-min-ply N] [--dedup-mb MB]\n"
        "                     [--random-move-pct P] [--random-move-minply N] [--random-move-maxply N]\n"
        "                     [--random-multi-pv N] [--random-multi-pv-diff D]\n"
        "                     [--adj-draw-minply N] [--adj-draw-score D] [--adj-draw-plies N]\n"
        "Plays self-play games and writes training records (README: Data generation).  --net defaults to the engine's own\n"
        "network (src/nnue.h); the other defaults are in src/datagen.h.\n";
    for (OptionParser p{argc, argv}; p.i < argc; ++p.i) {
        const std::string a = argv[p.i];
        const char* v;
        if (a == "--help" || a == "-h") { std::printf("%s", usage); return 0; }
        if ((v = p.value("--net"))) evalFile = v;
        else if ((v = p.value("--threads"))) opt.threads = std::max(1, std::atoi(v));
        else if ((v = p.value("--games"))) opt.games = std::strtoull(v, nullptr, 10);
        else if ((v = p.value("--nodes"))) opt.nodes = std::strtoull(v, nullptr, 10);
        else if ((v = p.value("--out"))) opt.out = v;
        else if ((v = p.value("--log"))) opt.log = v;
        else if (a == "--resume") opt.resume = true;
        else if ((v = p.value("--book"))) opt.book = v;
        else if ((v = p.value("--seed"))) opt.seed = std::strtoull(v, nullptr, 10);
        else if ((v = p.value("--hash"))) opt.hashMB = std::max(1, std::atoi(v));
        else if ((v = p.value("--max-plies"))) opt.maxPlies = std::atoi(v);
        else if ((v = p.value("--write-min-ply"))) opt.writeMinPly = std::atoi(v);
        else if ((v = p.value("--dedup-mb"))) opt.dedupMB = std::max(0, std::atoi(v));
        else if ((v = p.value("--random-move-pct"))) opt.randomMovePct = std::atoi(v);
        else if ((v = p.value("--random-move-minply"))) opt.randomMoveMinPly = std::atoi(v);
        else if ((v = p.value("--random-move-maxply"))) opt.randomMoveMaxPly = std::atoi(v);
        else if ((v = p.value("--random-multi-pv"))) opt.randomMultiPV = std::atoi(v);
        else if ((v = p.value("--random-multi-pv-diff"))) opt.randomMultiPVDiff = std::atoi(v);
        else if ((v = p.value("--adj-draw-minply"))) opt.adjDrawMinPly = std::atoi(v);
        else if ((v = p.value("--adj-draw-score"))) opt.adjDrawScore = std::atoi(v);
        else if ((v = p.value("--adj-draw-plies"))) opt.adjDrawPlies = std::atoi(v);
        else { std::printf("unknown or incomplete datagen option %s\n%s", a.c_str(), usage); return 1; }
    }
    if (opt.nodes == 0) { std::printf("datagen: --nodes must be at least 1\n"); return 1; }
    if (opt.randomMovePct > 0 && opt.randomMultiPV < 2) { std::printf("datagen: --random-multi-pv must be at least 2 (--random-move-pct 0 for no random moves)\n"); return 1; }
    if (!load_net_or_report(evalFile)) return 1;
    return run_datagen(opt) ? 0 : 1;
}

int rescore_main(int argc, char** argv) {
    RescoreOptions opt;
    std::string evalFile;
    const char* usage =
        "usage: tsumi rescore --in FILE [FILE ...] --out FILE [--net FILE] [--nodes N] [--threads N] [--hash MB]\n"
        "Searches every position of the record files again with the network (--net, default the engine's own) and writes\n"
        "the records with the new scores.  --nodes 1 (the default) stops each search after its first iteration.\n";
    for (OptionParser p{argc, argv}; p.i < argc; ++p.i) {
        const std::string a = argv[p.i];
        const char* v;
        if (a == "--help" || a == "-h") { std::printf("%s", usage); return 0; }
        if (a == "--in") { while (p.i + 1 < argc && argv[p.i + 1][0] != '-') opt.in.push_back(argv[++p.i]); }
        else if ((v = p.value("--out"))) opt.out = v;
        else if ((v = p.value("--net"))) evalFile = v;
        else if ((v = p.value("--nodes"))) opt.nodes = std::strtoull(v, nullptr, 10);
        else if ((v = p.value("--threads"))) opt.threads = std::max(1, std::atoi(v));
        else if ((v = p.value("--hash"))) opt.hashMB = std::max(1, std::atoi(v));
        else { std::printf("unknown or incomplete rescore option %s\n%s", a.c_str(), usage); return 1; }
    }
    if (opt.in.empty()) { std::printf("rescore: --in needs at least one record file\n%s", usage); return 1; }
    if (opt.nodes == 0) { std::printf("rescore: --nodes must be at least 1\n"); return 1; }
    if (!load_net_or_report(evalFile)) return 1;
    return run_rescore(opt) ? 0 : 1;
}
