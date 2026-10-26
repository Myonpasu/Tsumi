// Time management for games on a clock, after Stockfish's timeman.cpp and the stop rule of its iterative
// deepening.  init() turns the clock into a budget for this move: optimum, the share of
// the pool to spend when nothing unusual happens, and maximum, which an iteration in progress may run up
// to.  scale() then stretches or shrinks the optimum by what the search has shown so far; the search
// stops at the first iteration boundary past the scaled budget (Thread::iterative_deepening).  The state
// that carries from move to move within a game is reset by new_game().
#pragma once
#include "types.h"

struct Limits;

class TimeManager {
public:
    void new_game();
    void init(const Limits& limits, Color us, int gamePly, int moveOverhead);
    bool active() const { return useTimeManagement; }   // a game clock (not movetime, depth or nodes)
    int64_t optimum() const { return optimumTime; }
    int64_t maximum() const { return maximumTime; }
    int last_score() const { return previousScore; }     // final score of the previous search of this game (VALUE_NONE at first)

    // Multiplier for the optimum after a completed iteration: the current score, the score four iterations
    // earlier, the iterations since the best move last changed, the decayed count of best-move changes,
    // and the share of all nodes spent below the best move (per cent).
    double scale(int score, int scoreFourIterationsAgo, int iterationsSinceBestMoveChange, double bestMoveChanges, double effortShare);
    void search_finished(int score);   // remembers the score and stability for the next move

private:
    int64_t optimumTime = 0, maximumTime = 0;
    bool useTimeManagement = false;
    int previousScore = VALUE_NONE;
    double timeReduction = 1.0, previousTimeReduction = 1.0;   // best-move stability factor, this search and the last
    double originalTimeAdjust = -1.0;                          // multiplier fixed at the first move of the game
};
