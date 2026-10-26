#include "timeman.h"
#include "search.h"
#include <algorithm>
#include <cmath>

namespace {
double interpolate(double x, double x1, double x2, double y1, double y2) { return y1 + (x - x1) * (y2 - y1) / (x2 - x1); }
}

void TimeManager::new_game() {
    previousScore = VALUE_NONE;
    previousTimeReduction = 1.0;
    originalTimeAdjust = -1.0;
}

// The pool is the remaining time plus the increments over a 50-move horizon; this move gets a share of it that
// grows with the game ply (later positions decide more and fewer remain), and may run up to several times that
// share when an iteration is in progress, never beyond 81% of the clock less the move overhead.  The overhead is
// a safety margin for the delays of the GUI and the system, and it comes off the pool twice rather than once per
// move of the horizon as in Stockfish, so that a generous margin does not starve the search at fast time
// controls.  Selfmate games run about twice as long as chess games, so the ply is halved in the formulas tuned
// for chess.
void TimeManager::init(const Limits& limits, Color us, int gamePly, int moveOverhead) {
    optimumTime = maximumTime = 0;
    useTimeManagement = false;
    timeReduction = 1.0;
    if (limits.movetime > 0) {
        optimumTime = maximumTime = std::max<int64_t>(1, limits.movetime - moveOverhead);
        return;
    }
    if (!limits.clock) return;

    useTimeManagement = true;
    const double time = std::max(1.0, double(limits.time[us])), inc = double(limits.inc[us]);   // an empty clock: move at once
    const double ply = gamePly / 2.0;
    int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 50) : 50;
    if (time < 1000 && limits.movestogo == 0) mtg = std::max(1, int(time * 0.05));   // under a second: short horizon
    const double timeLeft = std::max(1.0, time + inc * (mtg - 1) - 2.0 * moveOverhead);
    double optScale, maxScale;
    if (limits.movestogo == 0) {
        if (originalTimeAdjust < 0) originalTimeAdjust = 0.3272 * std::log10(timeLeft) - 0.4141;   // fixed at the first move
        const double logTimeInSec = std::log10(time / 1000.0);
        const double optConstant = std::min(0.0029869 + 0.00033554 * logTimeInSec, 0.004905);
        const double maxConstant = std::max(3.3744 + 3.0608 * logTimeInSec, 3.1441);
        optScale = std::min(0.012112 + std::pow(ply + 3.22713, 0.46866) * optConstant, 0.19404 * time / timeLeft) * originalTimeAdjust;
        maxScale = std::min(6.873, maxConstant + ply / 12.352);
    } else {   // x moves in y seconds
        optScale = std::min((0.88 + ply / 116.4) / mtg, 0.88 * time / timeLeft);
        maxScale = 1.3 + 0.11 * mtg;
    }
    // Behind on the clock: spend less, by up to 90% of the relative deficit.
    if (limits.movestogo != 1) {
        const double advantage = (time - double(limits.time[~us])) / (1.0 + time + double(limits.time[~us]));
        optScale *= 1 + 0.9 * std::min(advantage, 0.0);
    }
    const double optimum = std::max(1.0, optScale * timeLeft);
    maximumTime = int64_t(std::max(1.0, std::min(0.8097 * time - moveOverhead, maxScale * optimum)));
    optimumTime = std::min(int64_t(optimum), maximumTime);
}

double TimeManager::scale(int score, int scoreFourIterationsAgo, int iterationsSinceBestMoveChange, double bestMoveChanges, double effortShare) {
    // Score fell since the previous move or over the last four iterations: spend more (0.58-1.73x).
    const int reference = previousScore == VALUE_NONE ? score : previousScore;
    const double fallingEval = std::clamp((11.48 + 2.30 * (reference - score) + 1.1 * (scoreFourIterationsAgo - score)) / 100.0, 0.576, 1.728);
    // Best move stable for many iterations: stop earlier; time saved on a stable move is available on the next.
    timeReduction = std::clamp(interpolate(iterationsSinceBestMoveChange, 4.96, 18.79, 0.639, 1.712), 0.629, 1.544);
    const double reduction = (1.468 + previousTimeReduction) / (2.284 * timeReduction);
    // Best move changed within recent iterations: spend more.
    const double instability = 1.077 + 2.229 * bestMoveChanges;
    // The best move absorbed nearly all the effort, so the alternatives were never close: stop earlier.
    const double effort = std::clamp(interpolate(effortShare * 1000.0, 75800, 104510, 0.969, 0.714), 0.693, 0.838);
    return fallingEval * reduction * instability * effort;
}

void TimeManager::search_finished(int score) {
    previousScore = score;
    previousTimeReduction = timeReduction;
}
