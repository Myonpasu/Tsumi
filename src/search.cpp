#include "search.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <algorithm>

namespace {

int LMRTable[64][64];
int LMPTable[16];

// Piece values for ordering captures (most valuable victim, then least valuable attacker); NO_PIECE_TYPE = 0.
constexpr int PieceValue[PIECE_TYPE_NB + 1] = {100, 300, 300, 500, 900, 0, 0};

constexpr int HISTORY_MAX = 16384;
constexpr int LMR_CHECK_MOVES = 4;   // checking moves are reduced from this move count on (quiet moves from the 2nd)

inline void history_update(int16_t& h, int bonus) {
    bonus = std::max(-HISTORY_MAX, std::min(HISTORY_MAX, bonus));
    h += int16_t(bonus - int(h) * std::abs(bonus) / HISTORY_MAX);
}

inline int stat_bonus(int depth) { return std::min(16 * depth * depth + 32 * depth, 1600); }

}  // namespace

void init_search_tables() {
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            LMRTable[d][m] = int(0.6 + std::log(double(d)) * std::log(double(m)) / 2.3);
    for (int d = 0; d < 16; ++d) LMPTable[d] = 3 + d * d;
}

// A score in UCI terms: centipawns, the internal millipawns / UNITS_PER_CP (see types.h), or a mate distance.
std::string score_to_uci(int score) {
    std::ostringstream ss;
    if (score >= VALUE_MATE_IN_MAX_PLY) ss << "mate " << (VALUE_MATE - score + 1) / 2;       // we get mated: we win
    else if (score <= -VALUE_MATE_IN_MAX_PLY) ss << "mate -" << (VALUE_MATE + score + 1) / 2;  // we must mate: we lose
    else ss << "cp " << std::lround(double(score) / UNITS_PER_CP);
    return ss.str();
}

uint64_t SharedState::total_nodes() const {
    uint64_t n = 0;
    for (Thread* t : threads) n += t->nodes;
    return n;
}

// ---------------------------------------------------------------------------
// Thread
// ---------------------------------------------------------------------------
Thread::Thread(SharedState& s, int id_) : id(id_), shared(s) {
    clear_history();
}

void Thread::clear_history() {
    std::memset(mainHistory, 0, sizeof(mainHistory));
    std::memset(captureHistory, 0, sizeof(captureHistory));
    std::memset(continuationHistory, 0, sizeof(continuationHistory));
    for (auto& row : counterMoves) for (auto& m : row) m = MOVE_NONE;
}

void Thread::set_position(const Position& root, const std::vector<Key>& history) {
    stack[0] = root;
    historySize = history.size();
    keys.assign(historySize + MAX_PLY + 8, 0);
    std::copy(history.begin(), history.end(), keys.begin());
    keys[historySize] = root.key();
    // The network is the only evaluation.  The commands that search - go, bench, datagen, rescore - refuse to
    // start without one (uci.cpp), so this only stops a caller that forgets to check.
    net = NNUE::current();
    if (!net) {
        std::printf("info string no network loaded: cannot search\n");
        std::fflush(stdout);
        std::_Exit(EXIT_FAILURE);
    }
    if (accs.width() != net->acc_width()) accs.resize(*net);
    NNUE::refresh(*net, stack[0], WHITE, accs.acc(0, WHITE));
    NNUE::refresh(*net, stack[0], BLACK, accs.acc(0, BLACK));
    accs.computed[0][0] = accs.computed[0][1] = true;
    nodes = 0;
    selDepth = 0;
    completedDepth = 0;
    result = SearchResult();
}

bool Thread::is_repetition(int ply) const {
    int idx = int(historySize) + ply;
    Key k = keys[idx];
    int end = std::max(0, idx - stack[ply].rule50_count());
    for (int i = idx - 4; i >= end; i -= 2)
        if (keys[i] == k) return true;
    return false;
}

int Thread::repetitions(int ply) const {
    int idx = int(historySize) + ply, n = 0;
    Key k = keys[idx];
    int end = std::max(0, idx - stack[ply].rule50_count());
    for (int i = idx - 4; i >= end; i -= 2)
        if (keys[i] == k) ++n;
    return n;
}

// A repeated position is a draw: both sides can keep repeating, and whichever would rather not had its other
// moves searched at the first occurrence.  Not so against a bare king.  The bare side wants the game to go on
// (its only chance is a mistake) and never has to repeat - a king usually has somewhere else to go - so the side
// with the material cannot force a draw this way, however often it repeats; scoring the repetition as a draw
// would let it believe it could, and it would shuffle its pieces rather than go for the stalemate.  Against a
// bare king a repeated position is therefore just the game going on, with the usual bare-king evaluation (which
// also rewards confining the king), unless this is its third occurrence, when the draw can be claimed.
int Thread::repetition_value(int ply) {
    const Position& pos = stack[ply];
    const bool bareUs = pos.bare_king(pos.side_to_move()), bareThem = pos.bare_king(~pos.side_to_move());
    if (bareUs == bareThem || repetitions(ply) >= 2) return VALUE_DRAW;
    return evaluate(ply);
}

// Called by the main thread every 1024 nodes: the node and time limits apply once the first iteration
// has completed, so that a best move always exists.
void Thread::check_time() {
    if (id != 0) return;
    const Limits& lim = shared.limits;
    const uint64_t total = shared.total_nodes();
    // Safety net: stop at 64x the node budget (+1M) however the search got there, even inside the first
    // iteration.  The position is reported so that a runaway search can be reproduced and fixed.
    if (lim.nodes && total >= (lim.nodes >= (UINT64_MAX - 1000000) / 64 ? UINT64_MAX : lim.nodes * 64 + 1000000)) {
        if (!shared.stop) {
            std::fprintf(stderr, "warning: search cut off at %llu nodes (budget %llu, depth %d completed): position fen %s\n",
                         (unsigned long long)total, (unsigned long long)lim.nodes, completedDepth, shared.rootPos.fen().c_str());
            std::fflush(stderr);
        }
        shared.stop = true;
        return;
    }
    if (lim.infinite || completedDepth < 1) return;
    if (lim.nodes && total >= lim.nodes) shared.stop = true;
    if (shared.tm.maximum() > 0 && elapsed_ms(shared.startTime) >= shared.tm.maximum()) shared.stop = true;
}

// The squares a bare king can legally step to (not attacked once it has left its square; a capture of an
// undefended piece counts).  Zero with the king to move is stalemate, the draw the side with the material
// is playing for.
static int bare_king_room(const Position& pos, Color bare) {
    const Square k = pos.king_square(bare);
    const Bitboard occ = pos.pieces() ^ square_bb(k);
    const Bitboard them = pos.pieces(~bare);
    int room = 0;
    for (Bitboard b = Bitboards::KingAttacks[k]; b; b &= b - 1)
        if (!(pos.attackers_to(lsb(b), occ) & them)) ++room;
    return room;
}

void Thread::update_pv(int ply, Move m) {
    pvTable[ply][0] = m;
    int len = pvLength[ply + 1];
    for (int i = 0; i < len; ++i) pvTable[ply][i + 1] = pvTable[ply + 1][i];
    pvLength[ply] = len + 1;
}

int Thread::evaluate(int ply, const MoveList* ml) {
    const Position& pos = stack[ply];
    // A bare king gives no check, so the side facing it can never be checkmated and can no longer win; the
    // bare side wins only if its opponent is left with nothing but mating moves, which the search has to
    // prove move by move.  Under correct play these positions are drawn, so they are evaluated as such
    // rather than by a network that has only seen the winning side's material (this also teaches the
    // search not to capture the opponent's last piece: it is the only mating material left).
    // A shade is taken off the draw for the side with the material: it has nothing left to gain by playing
    // on and the bare side nothing left to lose, so a position still being played scores -BARE_KING_CONTEMPT
    // for it (and +BARE_KING_CONTEMPT for the bare king), while the draws the search can actually force -
    // stalemate, the fifty-move rule, a dead position, a claimable repetition (repetition_value) - score
    // exactly 0.  The search then ends the game when it can, and the bare side keeps the game going, its
    // only chance.  A further BARE_KING_ROOM per square the bare king can still step to points the search
    // towards the stalemate when none is yet in sight: the king is driven to the edge and hemmed in, and the
    // stalemate comes into the horizon.
    {
        const bool bareUs = pos.bare_king(pos.side_to_move()), bareThem = pos.bare_king(~pos.side_to_move());
        if (bareUs && bareThem) return VALUE_DRAW;
        if (bareThem) return -BARE_KING_CONTEMPT - BARE_KING_ROOM * bare_king_room(pos, ~pos.side_to_move());
        if (bareUs) return BARE_KING_CONTEMPT + BARE_KING_ROOM * bare_king_room(pos, pos.side_to_move());
    }
    // Bring both accumulators up to date lazily, replaying the moves since the last computed ancestor.  A move of
    // the perspective's own king changes its king bucket, so the piece-square half is then rebuilt from the cache
    // (refresh_cached) rather than updated, while the threat half is still derived from the parent's.  When more
    // than REPLAY_BEFORE_KING moves would have to be replayed before the last king move, the accumulator at that
    // move is computed without its parent instead (piece-square half from the cache, threat half from scratch),
    // and the replay starts there.
    constexpr int REPLAY_BEFORE_KING = 4;
    for (Color c : {WHITE, BLACK}) {
        if (accs.computed[ply][c]) continue;
        const Piece king = make_piece(c, KING);
        auto kingMoved = [&](int p) {
            for (int i = 0; i < dirty[p].count; ++i) if (dirty[p].piece[i] == king) return true;
            return false;
        };
        int p = ply, lastKing = -1;
        while (p > 0 && !accs.computed[p][c]) {
            if (lastKing < 0 && kingMoved(p)) lastKing = p;
            --p;
        }
        if (lastKing >= 0 && lastKing - p > REPLAY_BEFORE_KING) {
            NNUE::refresh_cached(*net, stack[lastKing], c, accs.acc(lastKing, c), accs.cache);
            accs.computed[lastKing][c] = true;
            p = lastKing;
        }
        for (int q = p + 1; q <= ply; ++q) {
            if (kingMoved(q))
                NNUE::refresh_cached(*net, stack[q], c, accs.acc(q, c), accs.cache, &stack[q - 1], &dirty[q], accs.acc(q - 1, c), &accs.diff[q]);
            else
                NNUE::update(*net, stack[q - 1], stack[q], c, dirty[q], accs.acc(q - 1, c), accs.acc(q, c), accs.diff[q]);
            accs.computed[q][c] = true;
        }
    }
    MoveList own;
    if (!ml) { pos.generate_legal(own); ml = &own; }
    const int v = NNUE::evaluate(*net, pos, accs.acc(ply, pos.side_to_move()), accs.acc(ply, ~pos.side_to_move()), NNUE::mobility(*ml));
    return std::max(-VALUE_MAX_EVAL + 1, std::min(VALUE_MAX_EVAL - 1, v));
}

void Thread::update_quiet_stats(Stack* s, int ply, Move best, Move* quiets, int quietCount, int depth) {
    const Position& pos = stack[ply];
    Color us = pos.side_to_move();
    int bonus = stat_bonus(depth);
    if (s->killers[0] != best) {
        s->killers[1] = s->killers[0];
        s->killers[0] = best;
    }
    if (ply > 0) {
        Move prev = ss[ply - 1].currentMove;
        if (prev != MOVE_NONE) counterMoves[ss[ply - 1].movedPiece][to_sq(prev)] = best;
    }
    auto update_cont = [&](Move m, int b) {
        Piece pc = pos.moved_piece(m);
        Square to = to_sq(m);
        if (ply >= 1 && ss[ply - 1].contHist) history_update(ss[ply - 1].contHist[pc * 64 + to], b);
        if (ply >= 2 && ss[ply - 2].contHist) history_update(ss[ply - 2].contHist[pc * 64 + to], b);
    };
    history_update(main_hist(us, best), bonus);
    update_cont(best, bonus);
    for (int i = 0; i < quietCount; ++i) {
        if (quiets[i] == best) continue;
        history_update(main_hist(us, quiets[i]), -bonus);
        update_cont(quiets[i], -bonus);
    }
}

// ---------------------------------------------------------------------------
// Quiescence search: captures, promotions and (at the first ply) quiet checks.  When in check all
// evasions are searched, which also makes checkmate detection exact there.
// ---------------------------------------------------------------------------
constexpr int QS_MAX_PLY = 10;     // quiescence depth bound (see below)
constexpr int QS_RECAPTURES = 5;   // after this many quiescence plies, a node not in check searches only recaptures

int Thread::qsearch(int alpha, int beta, int ply, int qdepth) {
    Position& pos = stack[ply];
    nodes++;
    if ((nodes & 1023) == 0) check_time();
    if (shared.stop.load(std::memory_order_relaxed)) return VALUE_DRAW;
    if (ply >= MAX_PLY) return pos.in_check() ? VALUE_DRAW : evaluate(ply);
    pvLength[ply] = 0;

    const bool inCheck = pos.in_check();
    const bool pvNode = beta - alpha > 1;

    // Draws by repetition or dead position cannot be checkmates, so they are settled before the moves are
    // generated; the fifty-move rule waits for the mate check below, since a mate on the hundredth half-move
    // is a mate.  The transposition table is probed before move generation too: a position with no legal
    // moves is never stored (the search returns before storing), so a hit that ends the node saves
    // generating them.
    const bool rule50Draw = pos.rule50_count() >= 100;
    if (!rule50Draw) {
        if (pos.is_dead_position()) return VALUE_DRAW;
        if (is_repetition(ply)) return repetition_value(ply);
    }

    const int ttDepth = (inCheck || qdepth == 0) ? 0 : -1;
    TTResult tte = rule50Draw ? TTResult() : shared.tt.probe(pos.key(), ply);
    if (!pvNode && tte.hit && tte.depth >= ttDepth && tte.score != VALUE_NONE &&
        (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && tte.score >= beta) || (tte.bound == BOUND_UPPER && tte.score <= alpha)))
        return tte.score;

    MoveList ml;
    pos.generate_legal(ml);
    if (ml.size == 0) return inCheck ? mate_in(ply) : VALUE_DRAW;   // mated: we win; stalemate: draw
    if (rule50Draw) return VALUE_DRAW;
    // Depth bound: beyond QS_MAX_PLY quiescence plies a node that is not in check stands pat.  Capture, check
    // and recapture sequences are otherwise unbounded, and with several queens and exposed kings they run 40
    // plies deep and explode.  (A node in check still searches its evasions, which branch little.)
    if (!inCheck && qdepth >= QS_MAX_PLY) return evaluate(ply, &ml);

    int best, staticEval = VALUE_NONE;
    if (inCheck) {
        best = -VALUE_INFINITE;
    } else {
        staticEval = (tte.hit && tte.eval != VALUE_NONE) ? tte.eval : evaluate(ply, &ml);
        // A hash table bound on the score that is a better estimate replaces the evaluation as the stand-pat score,
        // and is stored as the position's evaluation - but not a mate score, whose distance depends on the ply.
        if (tte.hit && tte.score != VALUE_NONE && !is_mate_score(tte.score) &&
            ((tte.bound == BOUND_LOWER && tte.score > staticEval) || (tte.bound == BOUND_UPPER && tte.score < staticEval)))
            staticEval = tte.score;
        best = staticEval;
        if (best >= beta) {
            if (!tte.hit || tte.bound == BOUND_NONE) shared.tt.store(pos.key(), best, staticEval, MOVE_NONE, -1, BOUND_LOWER, ply);
            return best;
        }
        if (best > alpha) alpha = best;
    }

    // Score and select moves.
    int scores[256];
    bool givesCheck[256];
    int n = 0;
    Move moves[256];
    const CheckInfo ci = pos.check_info();
    for (Move m : ml) {
        bool capture = pos.is_capture(m);
        bool promo = type_of_move(m) == PROMOTION;
        if (!inCheck && !capture && !promo && qdepth > 0) continue;   // quiet moves: only checks, and only at the first ply
        bool gc = pos.gives_check(m, ci);
        if (!inCheck && !capture && !promo && !gc) continue;
        int s;
        if (m == tte.move) s = 1 << 30;
        else if (capture || promo) {
            Piece cap = pos.captured_piece(m);
            s = (1 << 20) + 10 * PieceValue[cap == NO_PIECE ? NO_PIECE_TYPE : type_of(cap)] - PieceValue[type_of(pos.moved_piece(m))] / 10;
            if (promo) s += PieceValue[promotion_type(m)];
            if (cap != NO_PIECE) s += capt_hist(pos.moved_piece(m), to_sq(m), type_of(cap)) / 4;
            if (gc) s += 500;
        } else {
            s = (gc ? (1 << 19) : 0) + main_hist(pos.side_to_move(), m);
        }
        moves[n] = m; scores[n] = s; givesCheck[n] = gc; n++;
    }

    Move bestMove = MOVE_NONE;
    int searched = 0;
    for (int i = 0; i < n; ++i) {
        // selection sort
        int bi = i;
        for (int j = i + 1; j < n; ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(moves[i], moves[bi]); std::swap(scores[i], scores[bi]); std::swap(givesCheck[i], givesCheck[bi]);
        Move m = moves[i];

        // Futility: a capture that cannot raise the score enough is skipped (not when in check).
        if (!inCheck && !givesCheck[i] && type_of_move(m) != PROMOTION && staticEval != VALUE_NONE) {
            Piece cap = pos.captured_piece(m);
            int gain = cap == NO_PIECE ? 0 : PieceValue[type_of(cap)];
            if (staticEval + gain + 160 <= alpha) {
                if (best < staticEval + gain + 160) best = staticEval + gain + 160;
                continue;
            }
        }
        // Two prunings of captures that neither give check nor promote (nor answer one: evasions are all
        // searched).  A capture that loses material on its square (static exchange evaluation below 0) is
        // skipped.  And after QS_RECAPTURES quiescence plies only recaptures - captures on the square the last
        // move went to - are searched: without that, positions with many pieces en prise on both sides branch
        // about threefold per quiescence ply up to QS_MAX_PLY.  Checks stay in at every ply.
        if (!inCheck && !givesCheck[i] && type_of_move(m) != PROMOTION) {
            if (qdepth >= QS_RECAPTURES && to_sq(m) != to_sq(ss[ply - 1].currentMove)) continue;
            if (!pos.see_ge(m, 0)) continue;
        }

        Position& child = stack[ply + 1];
        child = pos;
        child.do_move(m, &dirty[ply + 1]);
        accs.mark_dirty(ply + 1);
        ss[ply].currentMove = m;
        ss[ply].movedPiece = pos.moved_piece(m);
        ss[ply].contHist = cont_hist_slice(ss[ply].movedPiece, to_sq(m));
        keys[historySize + ply + 1] = child.key();
        shared.tt.prefetch(child.key());

        // Principal variation search, as in the main search: only the first move gets the full window.
        // Without it every node below a PV move keeps a wide window, which disables TT cutoffs and most
        // stand-pat cuts.
        int score;
        if (pvNode && searched > 0) {
            score = -qsearch(-alpha - 1, -alpha, ply + 1, qdepth + 1);
            if (score > alpha && score < beta) score = -qsearch(-beta, -alpha, ply + 1, qdepth + 1);
        } else {
            score = -qsearch(-beta, -alpha, ply + 1, qdepth + 1);
        }
        if (shared.stop.load(std::memory_order_relaxed)) return VALUE_DRAW;   // the child's score is a placeholder
        searched++;
        if (score > best) {
            best = score;
            if (score > alpha) {
                bestMove = m;
                alpha = score;
                if (pvNode) update_pv(ply, m);
                if (score >= beta) break;
            }
        }
    }

    Bound bound = best >= beta ? BOUND_LOWER : BOUND_UPPER;
    shared.tt.store(pos.key(), best, staticEval, bestMove, ttDepth, bound, ply);
    return best;
}

// ---------------------------------------------------------------------------
// Main alpha-beta search (principal variation search).
// ---------------------------------------------------------------------------
int Thread::search(int alpha, int beta, int depth, int ply, bool cutNode) {
    const bool pvNode = beta - alpha > 1;
    const bool rootNode = ply == 0;
    Position& pos = stack[ply];
    Stack* s = &ss[ply];

    if (depth <= 0) return qsearch(alpha, beta, ply, 0);

    nodes++;
    if ((nodes & 1023) == 0) check_time();
    if (shared.stop.load(std::memory_order_relaxed) && !rootNode) return VALUE_DRAW;
    if (ply > selDepth) selDepth = ply;
    pvLength[ply] = 0;
    if (ply >= MAX_PLY) return pos.in_check() ? VALUE_DRAW : evaluate(ply);

    const bool inCheck = pos.in_check();
    ss[ply + 1].killers[0] = ss[ply + 1].killers[1] = MOVE_NONE;

    const bool rule50Draw = !rootNode && pos.rule50_count() >= 100;
    if (!rootNode) {
        // Repetitions and dead positions cannot be checkmates, so they are settled before the moves are
        // generated; the fifty-move rule waits for the mate check (a mate on the hundredth half-move is a mate).
        if (!rule50Draw) {
            if (pos.is_dead_position()) return VALUE_DRAW;
            if (is_repetition(ply)) return repetition_value(ply);
        }
        // Mate distance pruning, with selfmate's bounds: the best this node can be worth is being mated right here,
        // the worst having to mate the opponent with the next move.
        alpha = std::max(alpha, mated_in(ply + 1));
        beta = std::min(beta, mate_in(ply));
        if (alpha >= beta) return alpha;
    }

    // The transposition table is probed before the moves are generated: a position with no legal moves is
    // never stored (the search returns before storing), so a hit at a sufficient depth ends the node without
    // generating them.  The single-move extension below needs the number of moves, so until they are known
    // an entry has to cover one more ply.
    const bool mayExtend = !rootNode && ply < 2 * rootDepth + 6;
    TTResult tte = rule50Draw ? TTResult() : shared.tt.probe(pos.key(), ply);
    const bool ttCut = !pvNode && tte.hit && tte.score != VALUE_NONE &&
        (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && tte.score >= beta) || (tte.bound == BOUND_UPPER && tte.score <= alpha));
    if (ttCut && tte.depth >= depth + mayExtend) return tte.score;

    // Generate all legal moves up front: this variant needs exact mate/stalemate knowledge and the
    // number of legal moves is itself an important search signal.
    MoveList ml;
    pos.generate_legal(ml);
    if (ml.size == 0) return inCheck ? mate_in(ply) : VALUE_DRAW;     // checkmated: we WIN
    if (rule50Draw) return VALUE_DRAW;

    // Extension: a node with a single legal move is forced, and forcing lines are the whole game in selfmate
    // chess.  Capped by ply to prevent explosions.  (Checks as such are not extended: with several queens
    // about, half of all moves give check.)
    if (mayExtend && ml.size == 1) depth++;
    if (ttCut && tte.depth >= depth) return tte.score;

    Move ttMove = tte.hit && ml.contains(tte.move) ? tte.move : MOVE_NONE;
    if (rootNode && (ttMove == MOVE_NONE || !root_move_allowed(ttMove))) ttMove = rootMove != MOVE_NONE && ml.contains(rootMove) && root_move_allowed(rootMove) ? rootMove : MOVE_NONE;

    // Static evaluation
    int staticEval = VALUE_NONE;
    if (!inCheck) {
        staticEval = (tte.hit && tte.eval != VALUE_NONE) ? tte.eval : evaluate(ply, &ml);
        if (!tte.hit) shared.tt.store(pos.key(), VALUE_NONE, staticEval, MOVE_NONE, -8, BOUND_NONE, ply);
    }
    s->staticEval = staticEval;
    const bool improving = !inCheck && ply >= 2 && ss[ply - 2].staticEval != VALUE_NONE && staticEval > ss[ply - 2].staticEval;

    // Reverse futility pruning: only when we have several moves to choose from (a side with very
    // few legal moves may be about to be forced).  No null-move pruning: zugzwang is everywhere.
    if (!pvNode && !inCheck && depth <= 6 && ml.size >= 4 && !is_mate_score(beta) &&
        staticEval - (85 * depth - 40 * improving) >= beta)
        return staticEval;

    // Internal iterative reduction
    if (depth >= 4 && ttMove == MOVE_NONE && !inCheck) depth--;

    // Move ordering
    int scores[256];
    bool givesCheck[256];
    Move moves[256];
    const int n = ml.size;
    const Color us = pos.side_to_move();
    const Move counter = ply > 0 && ss[ply - 1].currentMove != MOVE_NONE ? counterMoves[ss[ply - 1].movedPiece][to_sq(ss[ply - 1].currentMove)] : MOVE_NONE;
    const int16_t* ch1 = ply >= 1 ? ss[ply - 1].contHist : nullptr;
    const int16_t* ch2 = ply >= 2 ? ss[ply - 2].contHist : nullptr;
    const CheckInfo ci = pos.check_info();
    for (int i = 0; i < n; ++i) {
        Move m = ml.moves[i];
        moves[i] = m;
        bool gc = pos.gives_check(m, ci);
        givesCheck[i] = gc;
        Piece pc = pos.moved_piece(m);
        int sc;
        if (m == ttMove) sc = 1 << 30;
        else if (pos.is_capture(m) || type_of_move(m) == PROMOTION) {
            Piece cap = pos.captured_piece(m);
            sc = (1 << 28) + 10 * PieceValue[cap == NO_PIECE ? NO_PIECE_TYPE : type_of(cap)] - PieceValue[type_of(pc)] / 10;
            if (type_of_move(m) == PROMOTION) sc += PieceValue[promotion_type(m)];
            if (cap != NO_PIECE) sc += capt_hist(pc, to_sq(m), type_of(cap)) / 2;
            if (gc) sc += 1 << 27;
        } else {
            int h = main_hist(us, m);
            if (ch1) h += ch1[pc * 64 + to_sq(m)];
            if (ch2) h += ch2[pc * 64 + to_sq(m)];
            if (m == s->killers[0]) sc = (1 << 26) + 1;
            else if (m == s->killers[1]) sc = 1 << 26;
            else if (m == counter) sc = 1 << 25;
            else sc = h;
            if (gc) sc += 1 << 27;   // forcing moves first
        }
        scores[i] = sc;
    }

    int best = -VALUE_INFINITE;
    Move bestMove = MOVE_NONE;
    Move quiets[256];
    int quietCount = 0;
    int moveCount = 0;

    for (int i = 0; i < n; ++i) {
        int bi = i;
        for (int j = i + 1; j < n; ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(moves[i], moves[bi]); std::swap(scores[i], scores[bi]); std::swap(givesCheck[i], givesCheck[bi]);
        const Move m = moves[i];
        if (rootNode && !root_move_allowed(m)) continue;
        const bool gc = givesCheck[i];
        const bool capture = pos.is_capture(m);
        const bool promo = type_of_move(m) == PROMOTION;
        const bool quiet = !capture && !promo;
        const Piece pc = pos.moved_piece(m);
        moveCount++;

        int hist = 0;
        if (quiet) {
            hist = main_hist(us, m);
            if (ch1) hist += ch1[pc * 64 + to_sq(m)];
            if (ch2) hist += ch2[pc * 64 + to_sq(m)];
        }

        // Pruning of late quiet non-forcing moves at shallow depth.
        if (!rootNode && !inCheck && quiet && !gc && best > -VALUE_MATE_IN_MAX_PLY) {
            if (depth <= 8 && moveCount >= LMPTable[std::min(depth, 15)] + 2 * improving) continue;
            if (depth <= 6 && staticEval != VALUE_NONE && staticEval + 120 + 100 * depth <= alpha && std::abs(alpha) < VALUE_MAX_EVAL) continue;
            if (depth <= 3 && hist < -3000 * depth) continue;
        }

        Position& child = stack[ply + 1];
        child = pos;
        child.do_move(m, &dirty[ply + 1]);
        accs.mark_dirty(ply + 1);
        s->currentMove = m;
        s->movedPiece = pc;
        s->contHist = cont_hist_slice(pc, to_sq(m));
        keys[historySize + ply + 1] = child.key();
        shared.tt.prefetch(child.key());
        const uint64_t nodesBefore = nodes;

        int newDepth = depth - 1;
        int score;

        // Late move reductions.  Checking moves are ordered first (after the captures) and reduced only from the
        // LMR_CHECK_MOVES-th move on, by a ply less than a quiet move: they seldom cause a cutoff, and searching
        // every one of them at full depth would cost most of the nodes.
        if (depth >= 3 && moveCount > 1 + 2 * rootNode && (!gc || moveCount >= LMR_CHECK_MOVES) && (quiet || moveCount > 6)) {
            int r = LMRTable[std::min(depth, 63)][std::min(moveCount, 63)];
            if (gc) r--;
            if (pvNode) r--;
            if (cutNode) r++;
            if (!improving) r++;
            if (quiet) r -= hist / 6000;
            if (!quiet) r--;
            if (m == s->killers[0] || m == s->killers[1] || m == counter) r--;
            r = std::max(0, std::min(r, newDepth - 1));
            score = -search(-alpha - 1, -alpha, newDepth - r, ply + 1, true);
            if (score > alpha && r > 0) score = -search(-alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        } else if (!pvNode || moveCount > 1) {
            score = -search(-alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        } else {
            score = -VALUE_INFINITE;  // placeholder so the PV search below runs
        }
        if (pvNode && (moveCount == 1 || score > alpha))
            score = -search(-beta, -alpha, newDepth, ply + 1, false);

        if (rootNode) rootEffort[root_index(m)] += nodes - nodesBefore;
        if (shared.stop.load(std::memory_order_relaxed)) return VALUE_DRAW;

        // Against a bare king the side with material can at best draw, and a stalemate on the board now is that
        // draw for certain.  A draw the search sees further ahead also scores 0, and a draw that stays a few moves
        // away is never reached: so at the root a move that stalemates on the spot scores a millipawn above 0,
        // which no other move can score here (the side with material never scores above 0 against a bare king).
        if (rootNode && score == VALUE_DRAW && !inCheck && pos.bare_king(~us) && !pos.bare_king(us)) {
            MoveList replies;
            child.generate_legal(replies);
            if (replies.size == 0 && !child.in_check()) score = VALUE_DRAW + 1;
        }

        if (score > best) {
            best = score;
            if (score > alpha) {
                bestMove = m;
                alpha = score;
                if (pvNode) update_pv(ply, m);
                if (rootNode) {
                    if (moveCount > 1) bestMoveChanges++;
                    partialBest = m;
                    partialScore = score;
                }
                if (score >= beta) {
                    if (quiet) update_quiet_stats(s, ply, m, quiets, quietCount, depth);
                    else if (capture) {
                        Piece cap = pos.captured_piece(m);
                        history_update(capt_hist(pc, to_sq(m), type_of(cap)), stat_bonus(depth));
                    }
                    break;
                }
            }
        }
        if (quiet && quietCount < 256) quiets[quietCount++] = m;
        else if (capture) {
            Piece cap = pos.captured_piece(m);
            history_update(capt_hist(pc, to_sq(m), type_of(cap)), -stat_bonus(depth) / 2);
        }
    }

    if (best == -VALUE_INFINITE) {
        // Every move was pruned: treat as a fail-low with the static evaluation as an upper bound.
        best = staticEval != VALUE_NONE ? std::min(staticEval, alpha) : alpha;
    }

    // Not stored at the root when the search was restricted to some of the moves (the later MultiPV lines,
    // searchmoves): its result is not the position's.
    if (!(rootNode && (!rootExcluded.empty() || !shared.limits.searchMoves.empty()))) {
        Bound bound = best >= beta ? BOUND_LOWER : (pvNode && bestMove != MOVE_NONE) ? BOUND_EXACT : BOUND_UPPER;
        shared.tt.store(pos.key(), best, staticEval, bestMove, depth, bound, ply);
    }
    return best;
}

// ---------------------------------------------------------------------------
// Iterative deepening with aspiration windows.
// ---------------------------------------------------------------------------
bool Thread::root_move_allowed(Move m) const {
    for (Move x : rootExcluded) if (x == m) return false;
    const std::vector<Move>& sm = shared.limits.searchMoves;
    if (!sm.empty() && std::find(sm.begin(), sm.end(), m) == sm.end()) return false;
    return true;
}

void Thread::report_lines(const std::vector<RootLine>& lines, int searched) {
    if (id != 0 || shared.params.silent || !shared.output) return;
    int64_t ms = elapsed_ms(shared.startTime);
    uint64_t total = shared.total_nodes();
    // All MultiPV slots are printed every time so the GUI never keeps a stale line: the slots already
    // searched at this depth come first, sorted by score; the others still show their previous values.
    std::vector<const RootLine*> order;
    for (int i = 0; i < searched; ++i) order.push_back(&lines[i]);
    std::stable_sort(order.begin(), order.end(), [](const RootLine* a, const RootLine* b) { return a->score > b->score; });
    for (size_t i = searched; i < lines.size(); ++i) order.push_back(&lines[i]);
    int idx = 0;
    for (const RootLine* lp : order) {
        const RootLine& l = *lp;
        if (l.move == MOVE_NONE) continue;
        std::ostringstream o;
        o << "info depth " << l.depth << " seldepth " << selDepth << " multipv " << ++idx << " score " << score_to_uci(l.score)
          << " nodes " << total << " nps " << (ms > 0 ? total * 1000 / ms : total)
          << " hashfull " << shared.tt.hashfull() << " time " << ms << " pv";
        for (Move m : l.pv) o << " " << move_to_string(m);
        shared.output(o.str());
    }
}

SearchResult Thread::iterative_deepening() {
    const Limits& lim = shared.limits;
    const int maxDepth = lim.depth > 0 ? std::min(lim.depth, MAX_PLY - 1) : MAX_PLY - 1;
    for (auto& e : ss) { e.killers[0] = e.killers[1] = MOVE_NONE; e.currentMove = MOVE_NONE; e.movedPiece = NO_PIECE; e.staticEval = VALUE_NONE; e.contHist = nullptr; }
    std::memset(pvLength, 0, sizeof(pvLength));
    std::memset(rootEffort, 0, sizeof(rootEffort));
    rootMove = MOVE_NONE;
    rootExcluded.clear();
    bestMoveChanges = 0;

    stack[0].generate_legal(rootMoves);
    int allowedRootMoves = 0;
    for (Move m : rootMoves) allowedRootMoves += root_move_allowed(m);
    if (rootMoves.size == 0 || allowedRootMoves == 0) {
        result.bestMove = MOVE_NONE;
        result.score = stack[0].in_check() && rootMoves.size == 0 ? VALUE_MATE : VALUE_DRAW;
        return result;
    }
    for (Move m : rootMoves) if (root_move_allowed(m)) { result.bestMove = m; break; }
    result.score = -VALUE_INFINITE;

    const int multiPV = std::max(1, std::min(shared.params.multiPV, allowedRootMoves));
    std::vector<RootLine> lines(multiPV);

    // Time management (main thread, game clock only): the search stops at the first iteration boundary past
    // the optimum scaled by what this search has shown (TimeManager::scale).  Once half of that is used,
    // further iterations search again at a reduced depth instead of starting a full new one that would
    // rarely finish.
    double totBestMoveChanges = 0;
    int lastBestMoveDepth = 0;
    int iterValue[4];
    for (int& v : iterValue) v = shared.tm.last_score() == VALUE_NONE ? 0 : shared.tm.last_score();
    int iterIdx = 0, searchAgainCounter = 0;
    bool increaseDepth = true;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        // Helper threads search slightly different depths to diversify the shared hash table.
        if (id > 0 && depth > 1 && ((depth + id) % 3 == 0)) continue;
        if (!increaseDepth) searchAgainCounter++;
        const int searchDepth = std::max(1, depth - 3 * (searchAgainCounter + 1) / 4);
        rootDepth = searchDepth;
        selDepth = 0;
        rootExcluded.clear();
        totBestMoveChanges /= 2;
        partialBest = MOVE_NONE;
        const Move previousBest = lines[0].move;
        bool depthCompleted = true;

        for (int pvIdx = 0; pvIdx < multiPV; ++pvIdx) {
            rootMove = lines[pvIdx].move;   // previous occupant of this slot: searched first
            const int centre = lines[pvIdx].prevScore;
            int window = 25;
            int alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
            if (depth >= 4) { alpha = std::max(centre - window, -VALUE_INFINITE); beta = std::min(centre + window, VALUE_INFINITE); }
            int score;
            while (true) {
                score = search(alpha, beta, searchDepth, 0, false);
                if (shared.stop) break;
                if (score <= alpha) { beta = (alpha + beta) / 2; alpha = std::max(score - window, -VALUE_INFINITE); window *= 2; }
                else if (score >= beta) { beta = std::min(score + window, VALUE_INFINITE); window *= 2; }
                else break;
                if (window > 2000) { alpha = -VALUE_INFINITE; beta = VALUE_INFINITE; }
            }
            // A stop inside an iteration leaves it incomplete - the first one included: the node and time limits
            // wait for depth 1, so a first iteration is only ever cut short by the runaway-search safety limit
            // (check_time) or by the GUI, and then its root search returned the stop's placeholder score, not a
            // result.
            if (shared.stop) { depthCompleted = false; break; }
            if (pvLength[0] == 0) { depthCompleted = false; break; }   // nothing searched (all moves excluded)

            // Keep the slots distinct: if the move found now still sits in a later slot from the previous
            // depth, swap that slot here first, so no root move is ever reported in two lines at once.
            Move found = pvTable[0][0];
            for (int j = pvIdx + 1; j < multiPV; ++j)
                if (lines[j].move == found) { std::swap(lines[j], lines[pvIdx]); break; }
            RootLine& line = lines[pvIdx];
            line.score = score;
            line.prevScore = score;
            line.depth = depth;
            line.pv.assign(pvTable[0], pvTable[0] + pvLength[0]);
            line.move = found;
            rootExcluded.push_back(found);
            if (multiPV > 1) report_lines(lines, pvIdx + 1);
        }
        if (!depthCompleted) {
            // Cut short.  A root move that was searched to completion in this iteration and beat the previous
            // best is the search's current opinion: play it rather than the previous iteration's choice.
            if (multiPV == 1 && partialBest != MOVE_NONE && partialBest != lines[0].move && pvLength[0] > 0 && pvTable[0][0] == partialBest) {
                lines[0].move = partialBest;
                lines[0].score = lines[0].prevScore = partialScore;
                lines[0].depth = depth;
                lines[0].pv.assign(pvTable[0], pvTable[0] + pvLength[0]);
                result.score = lines[0].score;
                result.pv = lines[0].pv;
                result.bestMove = lines[0].move;
                result.lines = lines;
                result.nodes = shared.total_nodes();
                report_lines(lines, 1);
            }
            break;
        }

        completedDepth = depth;
        std::stable_sort(lines.begin(), lines.end(), [](const RootLine& a, const RootLine& b) { return a.score > b.score; });
        result.score = lines[0].score;
        result.depth = depth;
        result.pv = lines[0].pv;
        result.bestMove = lines[0].move;
        result.lines = lines;
        result.nodes = shared.total_nodes();
        if (multiPV == 1) report_lines(lines, 1);

        if (id == 0) {
            const int score = result.score;
            const int64_t ms = elapsed_ms(shared.startTime);
            totBestMoveChanges += bestMoveChanges;
            bestMoveChanges = 0;
            if (lines[0].move != previousBest) lastBestMoveDepth = depth;

            if (lim.mate > 0 && score >= VALUE_MATE - 2 * lim.mate) { shared.stop = true; break; }
            if (shared.tm.active() && !lim.infinite) {
                const double effortShare = 100.0 * double(rootEffort[root_index(lines[0].move)]) / double(std::max<uint64_t>(1, nodes));
                double totalTime = double(shared.tm.optimum()) * shared.tm.scale(score, iterValue[iterIdx], depth - lastBestMoveDepth, totBestMoveChanges, effortShare);
                if (allowedRootMoves == 1) totalTime = std::min(totalTime, 50.0);   // nothing to choose
                // A selfmate within three plies, or being forced to mate within two, needs no more thought.
                if (ms > std::min(totalTime, double(shared.tm.maximum())) || score >= mate_in(3) || score <= mated_in(2)) { shared.stop = true; break; }
                increaseDepth = ms <= totalTime * 0.5;
            } else if (lim.movetime > 0 && ms >= shared.tm.optimum()) { shared.stop = true; break; }
            iterValue[iterIdx] = score;
            iterIdx = (iterIdx + 1) & 3;
            if (const uint64_t total = shared.total_nodes(); (lim.nodes && total >= lim.nodes) || (lim.softNodes && total >= lim.softNodes)) {
                shared.stop = true;
                break;
            }
            // A proven mate needs no deeper search.
            if (multiPV == 1 && is_mate_score(score) && depth >= 10 && !lim.infinite && lim.depth == 0) { shared.stop = true; break; }
        }
        if (shared.stop) break;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
Engine::Engine() {
    shared.tt.resize(16);
    set_threads(1);
    shared.output = print_line;
    shared.rootPos.set(START_FEN);
}

Engine::~Engine() {
    stop();
    wait();
}

void Engine::set_threads(int n) {
    wait();
    n = std::max(1, std::min(n, 512));
    threads.clear();
    shared.clear_threads();
    for (int i = 0; i < n; ++i) {
        threads.emplace_back(std::make_unique<Thread>(shared, i));
        shared.register_thread(threads.back().get());
    }
}

void Engine::set_hash(size_t mb) {
    wait();
    shared.tt.resize(mb);
}

void Engine::new_game() {
    wait();
    shared.tt.clear();
    for (auto& t : threads) t->clear_history();
    shared.tm.new_game();
}

void Engine::set_position(const Position& pos, const std::vector<Key>& history) {
    shared.rootPos = pos;
    shared.rootHistory = history;
}

SearchResult Engine::search(const Limits& limits) {
    shared.stop = false;
    return run(limits);
}

SearchResult Engine::run(const Limits& limits) {
    shared.startTime = now();
    shared.limits = limits;
    shared.tt.new_search();

    shared.tm.init(limits, shared.rootPos.side_to_move(), shared.rootPos.game_ply(), shared.params.moveOverhead);

    for (auto& t : threads) t->set_position(shared.rootPos, shared.rootHistory);

    std::vector<std::thread> helpers;
    for (size_t i = 1; i < threads.size(); ++i) helpers.emplace_back([this, i] { threads[i]->iterative_deepening(); });
    SearchResult r = threads[0]->iterative_deepening();
    shared.stop = true;
    for (auto& h : helpers) h.join();
    r.nodes = shared.total_nodes();
    shared.tm.search_finished(r.score);
    return r;
}

void Engine::start_search_async(const Limits& limits) {
    wait();
    // Cleared here, before the search thread exists, so that a stop sent right after go is never lost.
    shared.stop = false;
    stopRequested = false;
    mainThread = std::thread([this, limits] {
        SearchResult r = run(limits);
        // An infinite search reports its move only when told to stop, even if it has finished (UCI).
        while (limits.infinite && !stopRequested) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (shared.output && !shared.params.silent) {   // the nodes actually spent (the info lines above report completed iterations)
            const int64_t ms = elapsed_ms(shared.startTime);
            shared.output("info nodes " + std::to_string(r.nodes) + " nps " + std::to_string(ms > 0 ? r.nodes * 1000 / ms : r.nodes) + " time " + std::to_string(ms));
        }
        if (shared.output) shared.output("bestmove " + (r.bestMove == MOVE_NONE ? std::string("0000") : move_to_string(r.bestMove)));
    });
}

void Engine::stop() {
    stopRequested = true;
    shared.stop = true;
}

void print_line(const std::string& line) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void Engine::wait() {
    if (mainThread.joinable()) mainThread.join();
}
