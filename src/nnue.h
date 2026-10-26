// NNUE evaluation.
//
// Each perspective (the side to move and its opponent) has H units, each the sum of a bias and the rows of the
// active piece-square and threat features (nnue_features.h).  The engine keeps the two sums in separate halves of
// a perspective's accumulator, so that a move updates each with its own rows, and adds them when it evaluates.
// The units are clipped to 0..1 and, as in Stockfish, each unit of the first half is multiplied with its partner
// in the second (unit j times unit j + H/2).  The side to move's H/2 products, the opponent's, and the legal-move
// inputs (nnue_features.h) feed the head, which has its own weights for each output bucket:
//
//     H + 32 -> L2 -> L3 -> 1
//
// Each hidden layer's clipped and squared clipped activations go side by side to the next layer, the output
// layer reads both hidden layers, and a skip connection - the difference of the first layer's last two outputs -
// carries the large evaluations that the clipped layers cannot.  H, L2 and L3 are read from the network file.
#pragma once
#include "position.h"
#include "cpu.h"
#include "nnue_features.h"
#include "memory.h"
#include <vector>
#include <string>
#include <memory>

namespace NNUE {

// The network this version of the engine is built for: the default EvalFile, which `make net` downloads.  A
// published network is named after the first 12 hex digits of its SHA-256 (tools/release_net.py).
constexpr char DEFAULT_NET[] = "tsumi-a9ab90c98ff7.nnue";

// Quantisation.  Accumulator values are int16 with 1.0 = QA; a unit is clipped to 0..QA, and a product of two,
// (a * b + 256) >> 9, is 0..127.  The head's inputs and activations are 0..127 (1.0 = HEAD_ACT).  The first
// layer's weights are int8 at scale 128 (over a thousand inputs they are small) and the second layer's at scale
// 64, so their outputs are 127 * 128 and 127 * 64 per 1.0.  The output layer's weights are int16 at scale 4096
// (they are few and small), so its sum is 127 * 4096 per 1.0; the skip joins it multiplied by 32.
constexpr int QA = 255;
constexpr int HEAD_ACT = 127;
constexpr int HEAD_W0SCALE = 128;
constexpr int HEAD_WSCALE = 64;
constexpr int HEAD_OUT_WSCALE = 4096;
constexpr int HEAD_UNIT0 = HEAD_ACT * HEAD_W0SCALE;
constexpr int HEAD_OUT_UNIT = HEAD_ACT * HEAD_OUT_WSCALE;
static_assert(HEAD_OUT_UNIT % HEAD_UNIT0 == 0, "the skip must join the output sum exactly");

struct Network {
    int H = 0;                    // units per perspective
    int NB = 0;                   // king buckets
    int L2 = 0, L3 = 0;           // widths of the head's hidden layers
    int SCALE = 0;                // internal units (millipawns) per 1.0 of network output
    uint8_t kingBucket[64];       // the king bucket of each oriented king square
    LargeVector<int8_t> psqW;     // [NB * 768][H]
    std::vector<int16_t> bias;    // [H]
    LargeVector<int8_t> threatW;  // [THREAT_FEATURES][H]
    std::vector<int8_t> fc0W;     // [OUTPUT_BUCKETS][L2][H + HEAD_EXTRAS]
    std::vector<int8_t> fc0Wc;    // fc0W in chunk-major order, [OUTPUT_BUCKETS][(H + HEAD_EXTRAS) / 4][L2][4] (nnue.cpp)
    std::vector<int32_t> fc0B;    // [OUTPUT_BUCKETS][L2]
    std::vector<int8_t> fc1W;     // [OUTPUT_BUCKETS][L3][2 * L2]
    std::vector<int32_t> fc1B;    // [OUTPUT_BUCKETS][L3]
    std::vector<int16_t> fc2W;    // [OUTPUT_BUCKETS][2 * L2 + 2 * L3]
    std::vector<int32_t> fc2B;    // [OUTPUT_BUCKETS]
    std::string path;

    bool load(const std::string& file, std::string* err = nullptr);
    int acc_width() const { return 2 * H; }             // one perspective's accumulator: piece-square half, threat half
    int head_inputs() const { return H + HEAD_EXTRAS; }
    std::string describe() const;   // one line for the "loaded network" message
};

// The loaded network (nullptr if none).  A function-local static rather than a global object: the engine has no
// dynamically initialised globals, so nothing runs before main()'s CPU check (see main.cpp).
std::shared_ptr<const Network>& current();
bool load(const std::string& file, std::string* err = nullptr);
inline bool available() { return current() != nullptr; }

// Feature index of a piece as seen from `perspective`, whose king stands on `ksq` (both unflipped).
inline int feature_index(const Network& net, Color perspective, Square ksq, Piece pc, Square sq) {
    return psq_feature(net.kingBucket, perspective, ksq, pc, sq);
}

// The piece-square half of an accumulator per (perspective, king square), with the pieces it was computed for:
// after a move of the perspective's king, the new accumulator is the cached one plus the difference between the
// two placements, usually a few pieces, rather than a sum over every piece.
struct AccumulatorCache {
    struct Entry {
        Bitboard pieces[COLOR_NB][PIECE_TYPE_NB];
        bool valid = false;
    };
    Entry entries[COLOR_NB][SQUARE_NB];
    std::vector<int16_t> buf;   // [2][64][H]
    int16_t* acc(Color perspective, Square ksq, int H) { return buf.data() + (size_t)(perspective * 64 + ksq) * H; }
    void reset(int H);
};

// The threat features a move creates and destroys, as feature indices for both perspectives: the search computes
// the list once per move and applies it to both accumulators.
struct ThreatDiff {
    static constexpr int MAX = 128;
    struct Entry { uint16_t idx[COLOR_NB]; uint8_t lost, pad; };   // lost: the threat disappeared (else it appeared)
    int n = 0;
    bool valid = false;   // computed for the move at hand
    bool full = false;    // more changes than fit: the threat half is recomputed instead
    Entry e[MAX];         // one short array, so that a typical diff (~15 entries) touches one or two cache lines
};

// A search thread's accumulators, one per (ply, perspective): the piece-square half (H values), then the threat
// half (H values).
class AccumulatorStack {
public:
    void resize(const Network& net);  // allocate for this network
    int16_t* acc(int ply, Color perspective) { return buf.data() + (size_t)(2 * ply + perspective) * W; }
    bool computed[MAX_PLY + 8][2];
    ThreatDiff diff[MAX_PLY + 8];     // the threat changes of the move that led to each ply
    void mark_dirty(int ply) { computed[ply][0] = computed[ply][1] = false; diff[ply].valid = false; }
    int width() const { return W; }
    AccumulatorCache cache;
private:
    std::vector<int16_t> buf;
    int W = 0;
};

// The accumulator and evaluation kernels exist in four versions inside one binary (the build's baseline, AVX2,
// AVX-512 and AVX-512 VNNI; see nnue.cpp).  `g_kernels` holds the set selected at startup by select_kernels(),
// which init_engine() calls after main() has examined the CPU.
struct Kernels {
    void (*refresh)(const Network&, const Position&, Color, int16_t*);
    void (*refresh_cached)(const Network&, const Position&, Color, int16_t*, AccumulatorCache&,
                           const Position*, const DirtyPiece*, const int16_t*, ThreatDiff*);
    void (*update)(const Network&, const Position&, const Position&, Color, const DirtyPiece&, const int16_t*, int16_t*, ThreatDiff&);
    int (*evaluate)(const Network&, const Position&, const int16_t*, const int16_t*, int);
};
extern Kernels g_kernels;
void select_kernels(CPU::Level level);

// Compute an accumulator from scratch.
inline void refresh(const Network& net, const Position& pos, Color perspective, int16_t* acc) {
    g_kernels.refresh(net, pos, perspective, acc);
}
// Compute an accumulator after a move of the perspective's own king: the piece-square half comes from the cache;
// the threat half is derived from the parent's when the move is given (posBefore, dp, parent and diff, as for
// update()), and computed from scratch otherwise.
inline void refresh_cached(const Network& net, const Position& pos, Color perspective, int16_t* acc, AccumulatorCache& cache,
                           const Position* posBefore = nullptr, const DirtyPiece* dp = nullptr, const int16_t* parent = nullptr,
                           ThreatDiff* diff = nullptr) {
    g_kernels.refresh_cached(net, pos, perspective, acc, cache, posBefore, dp, parent, diff);
}
// Derive child = parent + the changes of the move described by dp, which must not be a move of the perspective's
// king (posBefore is the parent's position, posAfter the child's).  `diff` is filled with the move's threat changes
// if it is not valid yet, and used as it is otherwise (one diff serves both perspectives).
inline void update(const Network& net, const Position& posBefore, const Position& posAfter, Color perspective,
                   const DirtyPiece& dp, const int16_t* parent, int16_t* child, ThreatDiff& diff) {
    g_kernels.update(net, posBefore, posAfter, perspective, dp, parent, child, diff);
}
// The evaluation in internal units from the side to move's point of view (positive: the side to move is winning,
// that is, heading for being mated).  accUs and accThem are the accumulators of the side to move and of its
// opponent; mobility is the side to move's legal moves as mobility() counts them (nnue_features.h).
inline int evaluate(const Network& net, const Position& pos, const int16_t* accUs, const int16_t* accThem, int mobility) {
    return g_kernels.evaluate(net, pos, accUs, accThem, mobility);
}

}  // namespace NNUE
