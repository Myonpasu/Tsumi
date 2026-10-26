#pragma once
#include "position.h"
#include "tt.h"
#include "nnue.h"
#include "timeman.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <functional>

typedef std::chrono::steady_clock::time_point TimePoint;
inline TimePoint now() { return std::chrono::steady_clock::now(); }
inline int64_t elapsed_ms(TimePoint since) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(now() - since).count();
}

struct Limits {
    int depth = 0;              // 0 = unlimited
    uint64_t nodes = 0;         // 0 = unlimited (the total of all threads)
    int64_t movetime = 0;       // ms
    int64_t time[2] = {0, 0};   // ms remaining per side
    bool clock = false;         // a game clock was given (wtime or btime), even one at 0
    int64_t inc[2] = {0, 0};
    int movestogo = 0;
    int mate = 0;               // stop when a forced selfmate in <= mate moves is found
    bool infinite = false;
    std::vector<Move> searchMoves;   // "go searchmoves": restrict the root to these moves (empty = all)
};

struct SearchParams {
    int moveOverhead = 30;   // ms
    bool silent = false;     // suppress UCI info output (datagen, rescore, bench)
    int multiPV = 1;         // number of principal variations to report
};

struct RootLine {
    Move move = MOVE_NONE;
    int score = -VALUE_INFINITE;
    int prevScore = 0;             // score at the previous depth (aspiration window centre)
    int depth = 0;
    std::vector<Move> pv;
};

struct SearchResult {
    Move bestMove = MOVE_NONE;
    int score = -VALUE_INFINITE;   // from the side to move's perspective
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
    std::vector<RootLine> lines;   // all principal variations (MultiPV), best first
};

// Per-ply search state that later plies read back (killers, the move played, its history slice).
struct Stack {
    Move killers[2];
    Move currentMove;
    Piece movedPiece;
    int staticEval;
    int16_t* contHist;   // continuation history slice for the move played at this ply
};

class SharedState;

// One search thread: owns its position stack, histories and NNUE accumulators.
class Thread {
public:
    Thread(SharedState& shared, int id);
    void set_position(const Position& root, const std::vector<Key>& history);
    SearchResult iterative_deepening();   // runs until stopped or limits reached
    void clear_history();

    uint64_t nodes = 0;
    int selDepth = 0;
    int completedDepth = 0;
    int id;
    SearchResult result;

private:
    int search(int alpha, int beta, int depth, int ply, bool cutNode);
    int qsearch(int alpha, int beta, int ply, int qdepth);
    int evaluate(int ply, const MoveList* ml = nullptr);   // ml: the position's legal moves when the caller has them
    bool is_repetition(int ply) const;
    int repetitions(int ply) const;        // earlier occurrences of the position at ply (game and search path)
    int repetition_value(int ply);         // the score of a repeated position: a draw, except against a bare king
    void check_time();
    void update_quiet_stats(Stack* ss, int ply, Move best, Move* quiets, int quietCount, int depth);
    void update_pv(int ply, Move m);
    int16_t& main_hist(Color c, Move m) { return mainHistory[c][from_sq(m) * 64 + to_sq(m)]; }
    int16_t& capt_hist(Piece pc, Square to, PieceType captured) { return captureHistory[pc][to][captured]; }
    int16_t* cont_hist_slice(Piece pc, Square to) { return &continuationHistory[pc][to][0][0]; }

    SharedState& shared;
    Position stack[MAX_PLY + 8];
    DirtyPiece dirty[MAX_PLY + 8];
    NNUE::AccumulatorStack accs;
    std::shared_ptr<const NNUE::Network> net;   // network snapshot used for this search
    Stack ss[MAX_PLY + 8];
    Move pvTable[MAX_PLY + 8][MAX_PLY + 8];
    int pvLength[MAX_PLY + 8];
    std::vector<Key> keys;      // game history followed by the search path
    size_t historySize = 0;
    int rootDepth = 0;
    Move rootMove = MOVE_NONE;
    std::vector<Move> rootExcluded;   // root moves already reported as earlier PV lines at this depth
    bool root_move_allowed(Move m) const;
    // Time-management statistics gathered at the root (see iterative_deepening): nodes spent below each root
    // move, how often the best move changed in the current iteration, and the best move found so far in an
    // iteration that may be cut short.
    MoveList rootMoves;
    uint64_t rootEffort[256];
    int bestMoveChanges = 0;
    Move partialBest = MOVE_NONE;
    int partialScore = 0;
    int root_index(Move m) const { for (int i = 0; i < rootMoves.size; ++i) if (rootMoves.moves[i] == m) return i; return 0; }
    void report_lines(const std::vector<RootLine>& lines, int searched);

    int16_t mainHistory[COLOR_NB][64 * 64];
    int16_t captureHistory[PIECE_NB][64][PIECE_TYPE_NB + 1];
    int16_t continuationHistory[PIECE_NB + 1][64][PIECE_NB][64];
    Move counterMoves[PIECE_NB + 1][64];
};

// State shared between the threads of one search.
class SharedState {
public:
    TranspositionTable tt;
    std::atomic<bool> stop{false};
    Limits limits;
    SearchParams params;
    TimePoint startTime;
    TimeManager tm;   // budget for this move; the search stops at an iteration boundary past its scaled optimum, anywhere past its maximum
    Position rootPos;
    std::vector<Key> rootHistory;
    std::function<void(const std::string&)> output;   // where to print info/bestmove lines

    uint64_t total_nodes() const;
    void register_thread(Thread* t) { threads.push_back(t); }
    void clear_threads() { threads.clear(); }
    std::vector<Thread*> threads;
};

// A complete engine instance: thread pool + shared state.  Multiple instances may coexist (datagen).
class Engine {
public:
    Engine();
    ~Engine();
    void set_threads(int n);
    void set_hash(size_t mb);
    void new_game();
    void set_position(const Position& pos, const std::vector<Key>& history);
    // Blocking search; returns the main thread's result.  Prints UCI output through shared.output.
    SearchResult search(const Limits& limits);
    // Non-blocking search for UCI: searches in a background thread and prints bestmove when done (an infinite
    // search only once stop() has been called).
    void start_search_async(const Limits& limits);
    void stop();
    void wait();

    SharedState shared;

private:
    SearchResult run(const Limits& limits);   // the search, with shared.stop cleared by the caller
    std::vector<std::unique_ptr<Thread>> threads;
    std::thread mainThread;
    std::atomic<bool> stopRequested{false};   // stop() was called: the GUI's stop, which an infinite search waits for
};

// Prints a line to standard output and flushes it.  All of the engine's UCI output goes through here, so that the
// lines of the search thread and of the command loop never interleave.
void print_line(const std::string& line);

std::string score_to_uci(int score);
void init_search_tables();
