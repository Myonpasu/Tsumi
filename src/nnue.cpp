#include "nnue.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>   // all intrinsics are declared regardless of -m flags; each kernel enables its own set
#define TSUMI_X86 1
#endif

namespace NNUE {

std::shared_ptr<const Network>& current() {
    static std::shared_ptr<const Network> net;
    return net;
}

namespace {

constexpr int MAX_UNITS = 4096;   // the largest H the evaluation's buffers hold
constexpr int MAX_LAYER = 256;    // the widest hidden layer they hold

template <class V>
bool read_vec(std::ifstream& in, V& v, size_t n) {
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), n * sizeof(typename V::value_type));
    return bool(in);
}

}  // namespace

// The network file, little-endian:
//   magic      8 bytes "TSUMINET" ("ICARUSNN", the project's working name, in earlier files)
//   header     13 x uint32: format version (4), H, NB (king buckets), output buckets (16), QA (255), 64 (unused),
//              SCALE, L2, L3, then the fixed values 1, 32 (HEAD_EXTRAS), 1, 1
//   kingBucket 64 bytes
//   psqW       int8  [NB * 768][H]
//   bias       int16 [H]
//   threatW    int8  [THREAT_FEATURES][H]
//   fc0W       int8  [16][L2][H + 32]       fc0B  int32 [16][L2]
//   fc1W       int8  [16][L3][2 * L2]       fc1B  int32 [16][L3]
//   fc2W       int16 [16][2 * L2 + 2 * L3]  fc2B  int32 [16]
bool Network::load(const std::string& file, std::string* err) {
    auto fail = [&](const std::string& why) { if (err) *err = why; return false; };
    std::ifstream in(file, std::ios::binary);
    if (!in) return fail("cannot open " + file);
    char magic[8];
    uint32_t hdr[13];
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!in || (std::memcmp(magic, "TSUMINET", 8) != 0 && std::memcmp(magic, "ICARUSNN", 8) != 0))
        return fail(file + " is not a Tsumi network");
    if (hdr[0] != 4) return fail(file + " has format version " + std::to_string(hdr[0]) + " (this engine reads version 4)");
    if (hdr[3] != OUTPUT_BUCKETS || hdr[4] != QA || hdr[9] != 1 || hdr[10] != HEAD_EXTRAS || hdr[11] != 1 || hdr[12] != 1)
        return fail(file + " has an unexpected header (not a Tsumi version 4 network)");
    H = int(hdr[1]); NB = int(hdr[2]); SCALE = int(hdr[6]); L2 = int(hdr[7]); L3 = int(hdr[8]);
    // H a multiple of 64 gives each perspective a multiple of 32 products, and layer widths that are multiples of
    // 16 keep every input count of the head a multiple of 32, as the vector kernels need.
    if (H <= 0 || H > MAX_UNITS || H % 64 != 0 || NB <= 0 || NB > 64 || SCALE <= 0
        || L2 < 16 || L2 > MAX_LAYER || L2 % 16 != 0 || L3 < 16 || L3 > MAX_LAYER || L3 % 16 != 0)
        return fail(file + " has implausible dimensions");
    in.read(reinterpret_cast<char*>(kingBucket), 64);
    for (int s = 0; s < 64; ++s)
        if (kingBucket[s] >= NB) return fail(file + " has a king bucket out of range");

    const size_t N0 = head_inputs(), OB = OUTPUT_BUCKETS;
    const bool ok = read_vec(in, psqW, size_t(NB) * FEATURES_PER_BUCKET * H) && read_vec(in, bias, H)
                 && read_vec(in, threatW, size_t(THREAT_FEATURES) * H)
                 && read_vec(in, fc0W, OB * L2 * N0) && read_vec(in, fc0B, OB * L2)
                 && read_vec(in, fc1W, OB * L3 * 2 * L2) && read_vec(in, fc1B, OB * L3)
                 && read_vec(in, fc2W, OB * (2 * L2 + 2 * L3)) && read_vec(in, fc2B, OB);
    if (!ok) return fail(file + " is truncated");
    if (in.peek() != std::char_traits<char>::eof()) return fail(file + " has trailing data");
    // The sparse first layer's copy of fc0W: [bucket][chunk of 4 inputs][output][4].
    fc0Wc.resize(fc0W.size());
    for (size_t b = 0; b < OB; ++b)
        for (size_t o = 0; o < size_t(L2); ++o)
            for (size_t i = 0; i < N0; ++i)
                fc0Wc[b * L2 * N0 + ((i / 4) * L2 + o) * 4 + i % 4] = fc0W[(b * L2 + o) * N0 + i];
    path = file;
    return true;
}

std::string Network::describe() const {
    return std::to_string(H) + "x2 units -> " + std::to_string(L2) + " -> " + std::to_string(L3) + " -> 1, "
         + std::to_string(NB) + " king buckets, " + std::to_string(OUTPUT_BUCKETS) + " output buckets";
}

bool load(const std::string& file, std::string* err) {
    auto net = std::make_shared<Network>();
    if (!net->load(file, err)) return false;
    current() = net;
    return true;
}

void AccumulatorStack::resize(const Network& net) {
    W = net.acc_width();
    buf.assign(size_t(MAX_PLY + 8) * 2 * W, 0);
    for (auto& c : computed) c[0] = c[1] = false;
    cache.reset(net.H);
}

void AccumulatorCache::reset(int H) {
    buf.assign(size_t(2) * 64 * H, 0);
    for (auto& row : entries)
        for (auto& e : row) e.valid = false;
}

// =================================================================================================================
// Kernels.  The code below is written once and compiled four times: for the build's baseline instruction set, for
// AVX2, for AVX-512 and for AVX-512 with VNNI (GCC's `target` attribute).  The best set the running CPU supports
// is selected at startup, so one executable runs at full speed on every CPU it supports - the idea of Stockfish's
// universal binaries, confined to the code where newer instructions make a difference.  The helpers are plain
// `inline` or carry their own target attribute; the kernel entry points are `flatten`ed, which inlines everything
// into them under the entry point's instruction set.
// =================================================================================================================
namespace {

alignas(64) const int16_t NO_BIAS[MAX_UNITS] = {};   // the threat half's starting point: it has no bias of its own

// ---- row accumulation ---------------------------------------------------------------------------------------
// child = parent + the sum of the `adds` rows - the sum of the `subs` rows: n int16 values from int8 rows, the one
// operation behind every accumulator refresh and update.  A block of the accumulator stays in registers while the
// rows stream past (a few rows for a move, dozens for a refresh), so the cost is the row loads.  One functor per
// instruction-set level; the accumulator routines below take it as a template parameter.
struct AccumScalar {
    inline void operator()(const int16_t* __restrict parent, int16_t* __restrict child, int n,
                           const int8_t* const* adds, int na, const int8_t* const* subs, int ns) const {
        for (int i = 0; i < n; i += 16) {
            int32_t v[16];
            for (int j = 0; j < 16; ++j) v[j] = parent[i + j];
            for (int k = 0; k < na; ++k) for (int j = 0; j < 16; ++j) v[j] += adds[k][i + j];
            for (int k = 0; k < ns; ++k) for (int j = 0; j < 16; ++j) v[j] -= subs[k][i + j];
            for (int j = 0; j < 16; ++j) child[i + j] = int16_t(v[j]);
        }
    }
};

#if defined(TSUMI_X86)
// Loads as int16 vectors: accumulators are int16, rows int8 (sign-extended).
TSUMI_INLINE __m128i load8x16(const int16_t* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
TSUMI_INLINE __m128i load8x16(const int8_t* p) {
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    return _mm_unpacklo_epi8(b, _mm_cmpgt_epi8(_mm_setzero_si128(), b));
}
__attribute__((target("avx2"))) TSUMI_INLINE __m256i load16x16(const int16_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
__attribute__((target("avx2"))) TSUMI_INLINE __m256i load16x16(const int8_t* p) { return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
__attribute__((target("avx512f,avx512bw"))) TSUMI_INLINE __m512i load32x16(const int16_t* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
__attribute__((target("avx512f,avx512bw"))) TSUMI_INLINE __m512i load32x16(const int8_t* p) { return _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))); }

struct AccumSSE2 {   // 16 values (two registers) per block; n is a multiple of 16
    inline void operator()(const int16_t* __restrict parent, int16_t* __restrict child, int n,
                           const int8_t* const* adds, int na, const int8_t* const* subs, int ns) const {
        for (int i = 0; i < n; i += 16) {
            __m128i v0 = load8x16(parent + i), v1 = load8x16(parent + i + 8);
            for (int k = 0; k < na; ++k) { v0 = _mm_add_epi16(v0, load8x16(adds[k] + i)); v1 = _mm_add_epi16(v1, load8x16(adds[k] + i + 8)); }
            for (int k = 0; k < ns; ++k) { v0 = _mm_sub_epi16(v0, load8x16(subs[k] + i)); v1 = _mm_sub_epi16(v1, load8x16(subs[k] + i + 8)); }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(child + i), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(child + i + 8), v1);
        }
    }
};

struct AccumAVX2 {   // 128 values (eight registers) per block, then 32, then 16; n is a multiple of 16
    __attribute__((target("avx2")))
    inline void operator()(const int16_t* __restrict parent, int16_t* __restrict child, int n,
                           const int8_t* const* adds, int na, const int8_t* const* subs, int ns) const {
        int i = 0;
        for (; i + 128 <= n; i += 128) {
            __m256i v[8];
            for (int j = 0; j < 8; ++j) v[j] = load16x16(parent + i + 16 * j);
            for (int k = 0; k < na; ++k) {
                const int8_t* r = adds[k] + i;
                for (int j = 0; j < 8; ++j) v[j] = _mm256_add_epi16(v[j], load16x16(r + 16 * j));
            }
            for (int k = 0; k < ns; ++k) {
                const int8_t* r = subs[k] + i;
                for (int j = 0; j < 8; ++j) v[j] = _mm256_sub_epi16(v[j], load16x16(r + 16 * j));
            }
            for (int j = 0; j < 8; ++j) _mm256_storeu_si256(reinterpret_cast<__m256i*>(child + i + 16 * j), v[j]);
        }
        for (; i + 32 <= n; i += 32) {
            __m256i v0 = load16x16(parent + i), v1 = load16x16(parent + i + 16);
            for (int k = 0; k < na; ++k) { v0 = _mm256_add_epi16(v0, load16x16(adds[k] + i)); v1 = _mm256_add_epi16(v1, load16x16(adds[k] + i + 16)); }
            for (int k = 0; k < ns; ++k) { v0 = _mm256_sub_epi16(v0, load16x16(subs[k] + i)); v1 = _mm256_sub_epi16(v1, load16x16(subs[k] + i + 16)); }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(child + i), v0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(child + i + 16), v1);
        }
        if (i < n) {
            __m256i v0 = load16x16(parent + i);
            for (int k = 0; k < na; ++k) v0 = _mm256_add_epi16(v0, load16x16(adds[k] + i));
            for (int k = 0; k < ns; ++k) v0 = _mm256_sub_epi16(v0, load16x16(subs[k] + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(child + i), v0);
        }
    }
};

struct AccumAVX512 {   // 256 values (eight registers) per block, then 64, 32, 16; n is a multiple of 16
    __attribute__((target("avx512f,avx512bw")))
    inline void operator()(const int16_t* __restrict parent, int16_t* __restrict child, int n,
                           const int8_t* const* adds, int na, const int8_t* const* subs, int ns) const {
        int i = 0;
        for (; i + 256 <= n; i += 256) {
            __m512i v[8];
            for (int j = 0; j < 8; ++j) v[j] = load32x16(parent + i + 32 * j);
            for (int k = 0; k < na; ++k) {
                const int8_t* r = adds[k] + i;
                for (int j = 0; j < 8; ++j) v[j] = _mm512_add_epi16(v[j], load32x16(r + 32 * j));
            }
            for (int k = 0; k < ns; ++k) {
                const int8_t* r = subs[k] + i;
                for (int j = 0; j < 8; ++j) v[j] = _mm512_sub_epi16(v[j], load32x16(r + 32 * j));
            }
            for (int j = 0; j < 8; ++j) _mm512_storeu_si512(reinterpret_cast<void*>(child + i + 32 * j), v[j]);
        }
        for (; i + 64 <= n; i += 64) {
            __m512i v0 = load32x16(parent + i), v1 = load32x16(parent + i + 32);
            for (int k = 0; k < na; ++k) { v0 = _mm512_add_epi16(v0, load32x16(adds[k] + i)); v1 = _mm512_add_epi16(v1, load32x16(adds[k] + i + 32)); }
            for (int k = 0; k < ns; ++k) { v0 = _mm512_sub_epi16(v0, load32x16(subs[k] + i)); v1 = _mm512_sub_epi16(v1, load32x16(subs[k] + i + 32)); }
            _mm512_storeu_si512(reinterpret_cast<void*>(child + i), v0);
            _mm512_storeu_si512(reinterpret_cast<void*>(child + i + 32), v1);
        }
        if (i + 32 <= n) {
            __m512i v0 = load32x16(parent + i);
            for (int k = 0; k < na; ++k) v0 = _mm512_add_epi16(v0, load32x16(adds[k] + i));
            for (int k = 0; k < ns; ++k) v0 = _mm512_sub_epi16(v0, load32x16(subs[k] + i));
            _mm512_storeu_si512(reinterpret_cast<void*>(child + i), v0);
            i += 32;
        }
        if (i < n) {
            __m256i v0 = load16x16(parent + i);
            for (int k = 0; k < na; ++k) v0 = _mm256_add_epi16(v0, load16x16(adds[k] + i));
            for (int k = 0; k < ns; ++k) v0 = _mm256_sub_epi16(v0, load16x16(subs[k] + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(child + i), v0);
        }
    }
};
#endif

// ---- threat half ------------------------------------------------------------------------------------------------

template <class Accum>
TSUMI_INLINE void refresh_threats_impl(const Network& net, const Position& pos, Color perspective, int16_t* __restrict acc, Accum accum) {
    const int H = net.H;
    Threat threats[MAX_THREATS];
    const int8_t* rows[MAX_THREATS];
    const int n = collect_threats(pos, threats);
    const Orientation o = orientation(perspective, pos.king_square(perspective));
    for (int i = 0; i < n; ++i) {
        const int f = threat_feature(o, Piece(threats[i].attacker), Square(threats[i].target), Piece(threats[i].content));
        rows[i] = net.threatW.data() + size_t(f) * H;
    }
    accum(NO_BIAS, acc, H, rows, n, rows, 0);
}

// The part of a threat feature index that does not depend on the target square, per perspective:
// (attacker code * THREAT_CONTENTS + content code) * 64.  A perspective's orientation contributes only its colour
// swap here; its square mask is applied to the target.  Indexed by the raw piece codes, with NO_PIECE (= PIECE_NB)
// standing for an empty flight square.  (Kings attack nothing: their entries are never used.)
struct ThreatBases {
    uint16_t v[COLOR_NB][PIECE_NB][PIECE_NB + 1];
    constexpr ThreatBases() : v() {
        for (int p = 0; p < COLOR_NB; ++p)
            for (int a = 0; a < PIECE_NB; ++a)
                for (int c = 0; c <= PIECE_NB; ++c) {
                    const int oa = p == BLACK ? (a + 6) % 12 : a;
                    const int oc = c == PIECE_NB ? 0 : 1 + (p == BLACK ? (c + 6) % 12 : c);
                    const int att = oa % 6 == KING ? 0 : threat_attacker(oa);
                    v[p][a][c] = uint16_t((att * THREAT_CONTENTS + oc) * 64);
                }
    }
};
constexpr ThreatBases THREAT_BASES{};

// How many of a threat row's cache lines collect_threat_diff prefetches.  The first lines cover the latency of the
// first block of the row's adds and the hardware prefetcher follows the rest; prefetching every line of every row
// of a move would only choke the line-fill buffers.
constexpr int PREFETCH_LINES = 2;

// The threats a move creates and destroys.  Only the pieces standing on, moving to or from, or attacking a square
// the move touched, or an empty square that entered or left the kings' neighbourhoods, can have gained or lost
// threats, and each such piece's targets are compared before and after the move.  (The attackers of those squares
// before the move suffice: a piece that attacks one of them only after the move either moved itself or was
// unblocked by a piece leaving a touched square that it did attack before.)  Each change is stored as its feature
// index for both perspectives (with their orientations after the move), and the first lines of both rows are
// prefetched here, as the two accumulators are updated one after the other.  Plain code shared by all kernel sets.
void collect_threat_diff(const Network& net, const Position& before, const Position& after, const DirtyPiece& dp, ThreatDiff& d) {
    d.n = 0;
    d.full = false;
    d.valid = true;
    const int maskW = orientation(WHITE, after.king_square(WHITE)).sqMask;
    const int maskB = orientation(BLACK, after.king_square(BLACK)).sqMask;
    const auto& baseW = THREAT_BASES.v[WHITE];
    const auto& baseB = THREAT_BASES.v[BLACK];
    const int H = net.H;
    const int8_t* W = net.threatW.data();
    Bitboard D = 0;
    for (int i = 0; i < dp.count; ++i) {
        if (dp.from[i] != SQ_NONE) D |= square_bb(dp.from[i]);
        if (dp.to[i] != SQ_NONE) D |= square_bb(dp.to[i]);
    }
    const Bitboard occB = before.pieces(), occA = after.pieces();
    const Bitboard zoneB = king_zone(before.king_square(WHITE), before.king_square(BLACK));
    const Bitboard zoneA = king_zone(after.king_square(WHITE), after.king_square(BLACK));
    const Bitboard targetsB = occB | zoneB, targetsA = occA | zoneA;
    // An occupied square is a target whether or not it is next to a king, so of the squares that entered or left
    // the kings' neighbourhoods only the empty ones matter.
    const Bitboard touched = D | ((zoneB ^ zoneA) & ~(occB | occA));
    Bitboard affected = touched;
    for (Bitboard s = touched; s;) affected |= before.attackers_to(pop_lsb(s));
    // The squares whose contents the move changed.
    Bitboard changedSq = 0;
    for (Bitboard s = D; s;) {
        const Square q = pop_lsb(s);
        if (before.piece_on(q) != after.piece_on(q)) changedSq |= square_bb(q);
    }
    // A piece's targets (as threat_targets in nnue_features.h): the occupied squares and the flight squares it
    // attacks, less its own king's square; a king has none.  Only the sliders' attacks depend on the occupancy.
    const Bitboard ownKingB[COLOR_NB] = {square_bb(before.king_square(WHITE)), square_bb(before.king_square(BLACK))};
    const Bitboard ownKingA[COLOR_NB] = {square_bb(after.king_square(WHITE)), square_bb(after.king_square(BLACK))};
    auto targets = [](Piece pc, Square s, Bitboard occ, Bitboard targetSquares, const Bitboard* ownKing) -> Bitboard {
        const PieceType pt = type_of(pc);
        if (pt == KING) return 0;
        const Bitboard att = pt == PAWN   ? Bitboards::PawnAttacks[color_of(pc)][s]
                           : pt == KNIGHT ? Bitboards::KnightAttacks[s]
                                          : attacks_bb(pt, s, occ);
        return att & targetSquares & ~ownKing[color_of(pc)];
    };
    // A piece that moves keeps the threats that its old and new squares have in common on targets holding the same
    // thing before and after: the feature does not record the attacker's square, so the threat lost from the old
    // square and the one gained from the new are the same feature and cancel.  They are left out (a sliding piece
    // moving along a line keeps most of its targets).  The two target sets are kept for the main loop.
    Square movedFrom[2] = {SQ_NONE, SQ_NONE}, movedTo[2] = {SQ_NONE, SQ_NONE};
    Bitboard fromTargets[2] = {0, 0}, toTargets[2] = {0, 0};
    int moved = 0;
    for (int i = 0; i < dp.count && moved < 2; ++i) {
        if (dp.from[i] == SQ_NONE || dp.to[i] == SQ_NONE) continue;   // a captured piece, or a promotion
        const Bitboard tf = targets(dp.piece[i], dp.from[i], occB, targetsB, ownKingB);
        const Bitboard tt = targets(dp.piece[i], dp.to[i], occA, targetsA, ownKingA);
        const Bitboard kept = tf & tt & ~changedSq;
        movedFrom[moved] = dp.from[i];
        movedTo[moved] = dp.to[i];
        fromTargets[moved] = tf & ~kept;
        toTargets[moved++] = tt & ~kept;
    }
    // Writes an entry and keeps it if `keep`: the loops below write a (lost, gained) pair per target square and keep
    // one or both, which spares them a data-dependent branch per entry.  The entry's rows are prefetched either way.
    auto put = [&](Piece attacker, Square t, Piece content, bool lost, bool keep) {
        ThreatDiff::Entry& e = d.e[d.n];
        e.idx[WHITE] = uint16_t(baseW[attacker][content] + (t ^ maskW));
        e.idx[BLACK] = uint16_t(baseB[attacker][content] + (t ^ maskB));
        e.lost = lost;
        d.n += keep;
        for (int l = 0; l < PREFETCH_LINES && l * 64 < H; ++l) {
            __builtin_prefetch(W + size_t(e.idx[WHITE]) * H + l * 64);
            __builtin_prefetch(W + size_t(e.idx[BLACK]) * H + l * 64);
        }
    };
    while (affected) {
        const Square s = pop_lsb(affected);
        const Piece pb = before.piece_on(s), pa = after.piece_on(s);
        if (pb == NO_PIECE && pa == NO_PIECE) continue;   // a flight square that came or went: its attackers are handled
        // A piece has at most 27 targets, so a square adds at most 54 entries.
        if (d.n + 54 > ThreatDiff::MAX) { d.full = true; return; }
        int fromK = -1, toK = -1;   // the moving pieces' old and new squares
        for (int k = 0; k < moved; ++k) {
            if (s == movedFrom[k]) fromK = k;
            if (s == movedTo[k]) toK = k;
        }
        const Bitboard tb = fromK >= 0 ? fromTargets[fromK] : pb != NO_PIECE ? targets(pb, s, occB, targetsB, ownKingB) : 0;
        const Bitboard ta = toK >= 0 ? toTargets[toK] : pa != NO_PIECE ? targets(pa, s, occA, targetsA, ownKingA) : 0;
        if (pb == pa) {
            // The targets the piece lost or gained, and those whose contents changed: a lost and a gained threat.
            Bitboard changed = (tb ^ ta) | (tb & ta & changedSq);
            while (changed) {
                const Square t = pop_lsb(changed);
                const Bitboard bit = square_bb(t);
                put(pb, t, before.piece_on(t), true, (tb & bit) != 0);
                put(pa, t, after.piece_on(t), false, (ta & bit) != 0);
            }
        } else {
            Bitboard lost = tb, gained = ta;
            while (lost) { const Square t = pop_lsb(lost); put(pb, t, before.piece_on(t), true, true); }
            while (gained) { const Square t = pop_lsb(gained); put(pa, t, after.piece_on(t), false, true); }
        }
    }
}

template <class Accum>
TSUMI_INLINE void apply_threat_diff(const Network& net, Color perspective, const ThreatDiff& d,
                                     const int16_t* __restrict parent, int16_t* __restrict child, Accum accum) {
    const int H = net.H;
    const int8_t* W = net.threatW.data();
    const int8_t* adds[ThreatDiff::MAX];
    const int8_t* subs[ThreatDiff::MAX];
    int na = 0, ns = 0;
    for (int i = 0; i < d.n; ++i) {
        const ThreatDiff::Entry& e = d.e[i];
        const int8_t* row = W + size_t(e.idx[perspective]) * H;
        if (e.lost) subs[ns++] = row; else adds[na++] = row;
    }
    accum(parent, child, H, adds, na, subs, ns);
}

// The threat half of a child accumulator from its parent's: incremental (any move, king moves included) unless the
// perspective's orientation changed - its king crossed between the d- and e-files, which re-indexes every
// feature - or the move changed more threats than a ThreatDiff holds.
template <class Accum>
TSUMI_INLINE void child_threats(const Network& net, const Position& before, const Position& after, Color perspective,
                                 const DirtyPiece& dp, const int16_t* __restrict parent, int16_t* __restrict child, ThreatDiff& diff,
                                 Accum accum) {
    const Orientation ob = orientation(perspective, before.king_square(perspective));
    const Orientation oa = orientation(perspective, after.king_square(perspective));
    if (ob.sqMask == oa.sqMask) {
        if (!diff.valid) collect_threat_diff(net, before, after, dp, diff);
        if (!diff.full) { apply_threat_diff(net, perspective, diff, parent, child, accum); return; }
    }
    refresh_threats_impl(net, after, perspective, child, accum);
}

// ---- piece-square half ------------------------------------------------------------------------------------------

template <class Accum>
TSUMI_INLINE void refresh_cached_impl(const Network& net, const Position& pos, Color perspective, int16_t* __restrict acc,
                                       AccumulatorCache& cache, const Position* before, const DirtyPiece* dp,
                                       const int16_t* __restrict parent, ThreatDiff* diff, Accum accum) {
    const int H = net.H;
    const int8_t* W = net.psqW.data();
    Square ksq = pos.king_square(perspective);
    AccumulatorCache::Entry& e = cache.entries[perspective][ksq];
    int16_t* __restrict cached = cache.acc(perspective, ksq, H);
    if (!e.valid) {
        std::memcpy(cached, net.bias.data(), H * sizeof(int16_t));
        for (int c = 0; c < COLOR_NB; ++c)
            for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) e.pieces[c][pt] = 0;
        e.valid = true;
    }
    // Apply the difference between the cached piece placement and the current one.
    const int8_t* adds[32];
    const int8_t* subs[32];
    int na = 0, ns = 0;
    for (int c = 0; c < COLOR_NB; ++c) {
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            Bitboard now = pos.pieces(Color(c), PieceType(pt));
            Bitboard was = e.pieces[c][pt];
            Bitboard add = now & ~was, sub = was & ~now;
            Piece pc = make_piece(Color(c), PieceType(pt));
            while (add) {
                Square s = pop_lsb(add);
                adds[na++] = W + size_t(feature_index(net, perspective, ksq, pc, s)) * H;
            }
            while (sub) {
                Square s = pop_lsb(sub);
                subs[ns++] = W + size_t(feature_index(net, perspective, ksq, pc, s)) * H;
            }
            e.pieces[c][pt] = now;
        }
    }
    accum(cached, acc, H, adds, na, subs, ns);                    // the current accumulator ...
    if (na + ns) std::memcpy(cached, acc, H * sizeof(int16_t));   // ... and the cache entry, now at this position
    if (before && dp && parent && diff) child_threats(net, *before, pos, perspective, *dp, parent + H, acc + H, *diff, accum);
    else refresh_threats_impl(net, pos, perspective, acc + H, accum);
}

template <class Accum>
TSUMI_INLINE void refresh_impl(const Network& net, const Position& pos, Color perspective, int16_t* __restrict acc, Accum accum) {
    const int H = net.H;
    const int8_t* W = net.psqW.data();
    const Square ksq = pos.king_square(perspective);
    const int8_t* rows[32];
    int n = 0;
    Bitboard occ = pos.pieces();
    while (occ) {
        Square s = pop_lsb(occ);
        rows[n++] = W + size_t(feature_index(net, perspective, ksq, pos.piece_on(s), s)) * H;
    }
    accum(net.bias.data(), acc, H, rows, n, rows, 0);
    refresh_threats_impl(net, pos, perspective, acc + H, accum);
}

template <class Accum>
TSUMI_INLINE void update_impl(const Network& net, const Position& posBefore, const Position& posAfter, Color perspective,
                               const DirtyPiece& dp, const int16_t* __restrict parent, int16_t* __restrict child, ThreatDiff& diff,
                               Accum accum) {
    const int H = net.H;
    const int8_t* W = net.psqW.data();
    Square ksq = posAfter.king_square(perspective);
    if (ksq != posBefore.king_square(perspective)) {   // the king bucket changed: not for this routine
        refresh_impl(net, posAfter, perspective, child, accum);
        return;
    }
    const int8_t* adds[3];
    const int8_t* subs[3];
    int na = 0, ns = 0;
    for (int i = 0; i < dp.count; ++i) {
        if (dp.from[i] != SQ_NONE) subs[ns++] = W + size_t(feature_index(net, perspective, ksq, dp.piece[i], dp.from[i])) * H;
        if (dp.to[i] != SQ_NONE) adds[na++] = W + size_t(feature_index(net, perspective, ksq, dp.piece[i], dp.to[i])) * H;
    }
    accum(parent, child, H, adds, na, subs, ns);
    child_threats(net, posBefore, posAfter, perspective, dp, parent + H, child + H, diff, accum);
}

// ---- head inputs ------------------------------------------------------------------------------------------------
// A unit is the sum of its piece-square and threat halves (acc[j] + acc[H + j]) clipped to 0..QA, and the products
// of the first half of the units with the second, u[j] = (a[j] * a[j + H/2] + 256) >> 9, are 0..127.  Summing in
// int32 here and with saturation in the vector versions gives the same clipped value.
TSUMI_INLINE void pair_inputs_scalar(const int16_t* __restrict acc, uint8_t* __restrict u, int H) {
    const int half = H / 2;
    const int16_t* thr = acc + H;
    for (int j = 0; j < half; ++j) {
        int32_t a = int32_t(acc[j]) + thr[j];
        int32_t b = int32_t(acc[j + half]) + thr[j + half];
        a = a < 0 ? 0 : (a > QA ? QA : a);
        b = b < 0 ? 0 : (b > QA ? QA : b);
        u[j] = uint8_t((a * b + 256) >> 9);
    }
}

// Dense int8 layer: out[o] = bias[o] + sum_i u[i] * w[o][i], inputs 0..127, weights within +-127.
TSUMI_INLINE void dense_scalar(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int32_t* __restrict bias,
                                int32_t* __restrict out, int outputs) {
    for (int o = 0; o < outputs; ++o) {
        int32_t sum = bias[o];
        const int8_t* row = w + size_t(o) * n;
        for (int i = 0; i < n; ++i) sum += int32_t(u[i]) * row[i];
        out[o] = sum;
    }
}

#if defined(TSUMI_X86)
// AVX2 products: the two halves of a unit added with saturation and clipped, and (a*b + 256) >> 9 computed as
// mulhrs(a << 3, b << 3) = (64ab + 2^14) >> 15, exactly; packed to bytes in order (packus works within 128-bit
// lanes, hence the permute).  H/2 is a multiple of 32.
__attribute__((target("avx2")))
TSUMI_INLINE __m256i clipped_units_avx2(const int16_t* psq, const int16_t* thr, __m256i zero, __m256i qa) {   // 16 units
    __m256i v = _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(psq)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(thr)));
    return _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
}

__attribute__((target("avx2")))
TSUMI_INLINE void pair_inputs_avx2(const int16_t* __restrict acc, uint8_t* __restrict u, int H) {
    const int half = H / 2;
    const int16_t* thr = acc + H;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(int16_t(QA));
    for (int j = 0; j < half; j += 32) {
        __m256i a0 = clipped_units_avx2(acc + j, thr + j, zero, qa), a1 = clipped_units_avx2(acc + j + 16, thr + j + 16, zero, qa);
        __m256i b0 = clipped_units_avx2(acc + half + j, thr + half + j, zero, qa), b1 = clipped_units_avx2(acc + half + j + 16, thr + half + j + 16, zero, qa);
        __m256i p0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a0, 3), _mm256_slli_epi16(b0, 3));
        __m256i p1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a1, 3), _mm256_slli_epi16(b1, 3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
    }
}

// AVX-512 products: 64 per step (a tail of 32 with AVX2); one permute puts the eight 64-bit pieces back in order.
__attribute__((target("avx512f,avx512bw")))
TSUMI_INLINE __m512i clipped_units_avx512(const int16_t* psq, const int16_t* thr, __m512i zero, __m512i qa) {   // 32 units
    __m512i v = _mm512_adds_epi16(_mm512_loadu_si512(reinterpret_cast<const void*>(psq)), _mm512_loadu_si512(reinterpret_cast<const void*>(thr)));
    return _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
}

__attribute__((target("avx512f,avx512bw")))
TSUMI_INLINE void pair_inputs_avx512(const int16_t* __restrict acc, uint8_t* __restrict u, int H) {
    const int half = H / 2;
    const int16_t* thr = acc + H;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i qa = _mm512_set1_epi16(int16_t(QA));
    const __m512i order = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    int j = 0;
    for (; j + 64 <= half; j += 64) {
        __m512i a0 = clipped_units_avx512(acc + j, thr + j, zero, qa), a1 = clipped_units_avx512(acc + j + 32, thr + j + 32, zero, qa);
        __m512i b0 = clipped_units_avx512(acc + half + j, thr + half + j, zero, qa), b1 = clipped_units_avx512(acc + half + j + 32, thr + half + j + 32, zero, qa);
        __m512i p0 = _mm512_mulhrs_epi16(_mm512_slli_epi16(a0, 3), _mm512_slli_epi16(b0, 3));
        __m512i p1 = _mm512_mulhrs_epi16(_mm512_slli_epi16(a1, 3), _mm512_slli_epi16(b1, 3));
        _mm512_storeu_si512(reinterpret_cast<void*>(u + j), _mm512_permutexvar_epi64(order, _mm512_packus_epi16(p0, p1)));
    }
    if (j < half) {   // 32 left
        const __m256i z = _mm256_setzero_si256(), q = _mm256_set1_epi16(int16_t(QA));
        __m256i a0 = clipped_units_avx2(acc + j, thr + j, z, q), a1 = clipped_units_avx2(acc + j + 16, thr + j + 16, z, q);
        __m256i b0 = clipped_units_avx2(acc + half + j, thr + half + j, z, q), b1 = clipped_units_avx2(acc + half + j + 16, thr + half + j + 16, z, q);
        __m256i p0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a0, 3), _mm256_slli_epi16(b0, 3));
        __m256i p1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a1, 3), _mm256_slli_epi16(b1, 3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
    }
}

// ---- dense layers -----------------------------------------------------------------------------------------------
// AVX2: maddubs sums pairs of u8*i8 products into int16 (at most 2*127*127, no saturation) and madd with ones widens
// them to int32.  Four output rows go together so that their horizontal sums share one hadd tree.  n is a multiple
// of 32.
__attribute__((target("avx2")))
TSUMI_INLINE __m128i hsum4_avx2(__m256i s0, __m256i s1, __m256i s2, __m256i s3) {
    __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));   // [s0 s1 s2 s3 | s0 s1 s2 s3] half-sums
    return _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
}

__attribute__((target("avx2")))
TSUMI_INLINE int32_t hsum_avx2(__m256i s) {
    __m128i h = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4E));
    h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xB1));
    return _mm_cvtsi128_si32(h);
}

__attribute__((target("avx2")))
TSUMI_INLINE void dense_avx2(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int32_t* __restrict bias,
                              int32_t* __restrict out, int outputs) {
    const __m256i ones = _mm256_set1_epi16(1);
    int o = 0;
    for (; o + 4 <= outputs; o += 4) {
        const int8_t* r0 = w + size_t(o) * n;
        const int8_t* r1 = r0 + n;
        const int8_t* r2 = r1 + n;
        const int8_t* r3 = r2 + n;
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (int i = 0; i < n; i += 32) {
            __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + i))), ones));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + i))), ones));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + i))), ones));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r3 + i))), ones));
        }
        __m128i r = _mm_add_epi32(hsum4_avx2(s0, s1, s2, s3), _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + o)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + o), r);
    }
    for (; o < outputs; ++o) {
        const int8_t* row = w + size_t(o) * n;
        __m256i sum = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 32) {
            __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, wv), ones));
        }
        out[o] = bias[o] + hsum_avx2(sum);
    }
}

__attribute__((target("avx512f")))
TSUMI_INLINE __m256i fold_avx512(__m512i s) { return _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64(s, 1)); }

// AVX-512: 64 inputs per step, the remaining 32 (n is a multiple of 32) with AVX2.
__attribute__((target("avx512f,avx512bw")))
TSUMI_INLINE void dense_avx512(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int32_t* __restrict bias,
                                int32_t* __restrict out, int outputs) {
    if (n < 64) { dense_avx2(u, n, w, bias, out, outputs); return; }   // a small layer: 256-bit steps only
    const __m512i ones = _mm512_set1_epi16(1);
    const __m256i ones256 = _mm256_set1_epi16(1);
    const int n64 = n & ~63;
    int o = 0;
    for (; o + 4 <= outputs; o += 4) {
        const int8_t* r0 = w + size_t(o) * n;
        const int8_t* r1 = r0 + n;
        const int8_t* r2 = r1 + n;
        const int8_t* r3 = r2 + n;
        __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
        for (int i = 0; i < n64; i += 64) {
            __m512i uv = _mm512_loadu_si512(reinterpret_cast<const void*>(u + i));
            s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_maddubs_epi16(uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r0 + i))), ones));
            s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_maddubs_epi16(uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r1 + i))), ones));
            s2 = _mm512_add_epi32(s2, _mm512_madd_epi16(_mm512_maddubs_epi16(uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r2 + i))), ones));
            s3 = _mm512_add_epi32(s3, _mm512_madd_epi16(_mm512_maddubs_epi16(uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r3 + i))), ones));
        }
        __m256i t0 = fold_avx512(s0), t1 = fold_avx512(s1), t2 = fold_avx512(s2), t3 = fold_avx512(s3);
        if (n64 < n) {
            __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + n64));
            t0 = _mm256_add_epi32(t0, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + n64))), ones256));
            t1 = _mm256_add_epi32(t1, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + n64))), ones256));
            t2 = _mm256_add_epi32(t2, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + n64))), ones256));
            t3 = _mm256_add_epi32(t3, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r3 + n64))), ones256));
        }
        __m128i r = _mm_add_epi32(hsum4_avx2(t0, t1, t2, t3), _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + o)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + o), r);
    }
    for (; o < outputs; ++o) {
        const int8_t* row = w + size_t(o) * n;
        __m512i sum = _mm512_setzero_si512();
        for (int i = 0; i < n64; i += 64) {
            __m512i uv = _mm512_loadu_si512(reinterpret_cast<const void*>(u + i));
            __m512i wv = _mm512_loadu_si512(reinterpret_cast<const void*>(row + i));
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_maddubs_epi16(uv, wv), ones));
        }
        __m256i t = fold_avx512(sum);
        if (n64 < n) {
            __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + n64));
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + n64));
            t = _mm256_add_epi32(t, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, wv), ones256));
        }
        out[o] = bias[o] + hsum_avx2(t);
    }
}

// AVX-512 VNNI: vpdpbusd does maddubs + madd + add in one instruction (without the intermediate int16, which never
// saturates with inputs <= 127 and weights within +-127, so the results are the same).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
TSUMI_INLINE void dense_avx512vnni(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int32_t* __restrict bias,
                                    int32_t* __restrict out, int outputs) {
    if (n < 64) {   // a small layer: 256-bit steps only
        int o = 0;
        for (; o + 4 <= outputs; o += 4) {
            const int8_t* r0 = w + size_t(o) * n;
            __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
            for (int i = 0; i < n; i += 32) {
                __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
                s0 = _mm256_dpbusd_epi32(s0, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + i)));
                s1 = _mm256_dpbusd_epi32(s1, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + n + i)));
                s2 = _mm256_dpbusd_epi32(s2, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * n + i)));
                s3 = _mm256_dpbusd_epi32(s3, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 3 * n + i)));
            }
            __m128i r = _mm_add_epi32(hsum4_avx2(s0, s1, s2, s3), _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + o)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + o), r);
        }
        for (; o < outputs; ++o) {
            const int8_t* row = w + size_t(o) * n;
            __m256i sum = _mm256_setzero_si256();
            for (int i = 0; i < n; i += 32)
                sum = _mm256_dpbusd_epi32(sum, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)));
            out[o] = bias[o] + hsum_avx2(sum);
        }
        return;
    }
    const int n64 = n & ~63;
    int o = 0;
    for (; o + 4 <= outputs; o += 4) {
        const int8_t* r0 = w + size_t(o) * n;
        const int8_t* r1 = r0 + n;
        const int8_t* r2 = r1 + n;
        const int8_t* r3 = r2 + n;
        __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
        for (int i = 0; i < n64; i += 64) {
            __m512i uv = _mm512_loadu_si512(reinterpret_cast<const void*>(u + i));
            s0 = _mm512_dpbusd_epi32(s0, uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r0 + i)));
            s1 = _mm512_dpbusd_epi32(s1, uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r1 + i)));
            s2 = _mm512_dpbusd_epi32(s2, uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r2 + i)));
            s3 = _mm512_dpbusd_epi32(s3, uv, _mm512_loadu_si512(reinterpret_cast<const void*>(r3 + i)));
        }
        __m256i t0 = fold_avx512(s0), t1 = fold_avx512(s1), t2 = fold_avx512(s2), t3 = fold_avx512(s3);
        if (n64 < n) {
            __m256i uv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + n64));
            t0 = _mm256_dpbusd_epi32(t0, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + n64)));
            t1 = _mm256_dpbusd_epi32(t1, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + n64)));
            t2 = _mm256_dpbusd_epi32(t2, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + n64)));
            t3 = _mm256_dpbusd_epi32(t3, uv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r3 + n64)));
        }
        __m128i r = _mm_add_epi32(hsum4_avx2(t0, t1, t2, t3), _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + o)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + o), r);
    }
    for (; o < outputs; ++o) {
        const int8_t* row = w + size_t(o) * n;
        __m512i sum = _mm512_setzero_si512();
        for (int i = 0; i < n64; i += 64)
            sum = _mm512_dpbusd_epi32(sum, _mm512_loadu_si512(reinterpret_cast<const void*>(u + i)), _mm512_loadu_si512(reinterpret_cast<const void*>(row + i)));
        __m256i t = fold_avx512(sum);
        if (n64 < n)
            t = _mm256_dpbusd_epi32(t, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + n64)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + n64)));
        out[o] = bias[o] + hsum_avx2(t);
    }
}

// ---- sparse first layer -----------------------------------------------------------------------------------------
// Many products are zero (a unit clipped at 0 zeroes its product), so the first layer multiplies in only the
// 4-input chunks of u that hold a non-zero byte, as Stockfish's AffineTransformSparseInput does: a chunk's four
// inputs are broadcast to every lane and multiplied with a block of weights stored chunk-major (Network::fc0Wc,
// [chunk][output][4]), so the outputs accumulate in whole registers, without horizontal sums.  The integer sums
// come out the same as the dense layer's.  Two passes: the indices of the non-zero chunks are listed first (eight
// at a time, through a table of the positions of the set bits of a byte), then the listed chunks are multiplied
// in.  With VNNI the multiply-add reads its own accumulator, so four independent sets of accumulators take turns:
// a single set would wait out the instruction's latency on every chunk.  Specialised for 16, 32 and 64 outputs;
// other widths use the dense routine.  The scan for non-zero chunks reads up to 64 bytes past the n inputs, which
// must be zeros.
struct NnzTable { alignas(16) uint16_t idx[256][8]; };
constexpr NnzTable make_nnz_table() {
    NnzTable t{};
    for (int m = 0; m < 256; ++m) {
        int k = 0;
        for (int b = 0; b < 8; ++b)
            if ((m >> b) & 1) t.idx[m][k++] = uint16_t(b);
    }
    return t;
}
alignas(64) constexpr NnzTable NNZ_TABLE = make_nnz_table();
constexpr int MAX_CHUNKS = (MAX_UNITS + HEAD_EXTRAS) / 4 + 16;   // n/4 indices plus the slack of the last scan steps

TSUMI_INLINE void append_chunks(uint16_t* __restrict nz, int& count, uint32_t mask8, int base) {
    const __m128i idx = _mm_add_epi16(_mm_set1_epi16(int16_t(base)), _mm_load_si128(reinterpret_cast<const __m128i*>(NNZ_TABLE.idx[mask8])));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(nz + count), idx);
    count += __builtin_popcount(mask8);
}

__attribute__((target("avx512f,avx512bw")))
TSUMI_INLINE int nonzero_chunks_avx512(const uint8_t* __restrict u, int n, uint16_t* __restrict nz) {
    int count = 0;
    for (int c0 = 0; c0 < n / 4; c0 += 16) {
        const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(u + 4 * c0));
        const uint32_t m = _mm512_test_epi32_mask(v, v);
        append_chunks(nz, count, m & 0xFF, c0);
        append_chunks(nz, count, m >> 8, c0 + 8);
    }
    return count;
}

__attribute__((target("avx2")))
TSUMI_INLINE int nonzero_chunks_avx2(const uint8_t* __restrict u, int n, uint16_t* __restrict nz) {
    int count = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (int c0 = 0; c0 < n / 4; c0 += 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + 4 * c0));
        const uint32_t m = ~uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)))) & 0xFFu;
        append_chunks(nz, count, m, c0);
    }
    return count;
}

TSUMI_INLINE int32_t chunk_of(const uint8_t* u, int c) {   // the four inputs of chunk c as one 32-bit value
    int32_t x4;
    std::memcpy(&x4, u + 4 * c, 4);
    return x4;
}

template <int R>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void fc0_vnni_chunk(__m512i* __restrict acc, const uint8_t* __restrict u, const int8_t* __restrict wc, int c) {
    const __m512i x = _mm512_set1_epi32(chunk_of(u, c));
    const int8_t* w = wc + size_t(c) * 64 * R;
    for (int r = 0; r < R; ++r) acc[r] = _mm512_dpbusd_epi32(acc[r], x, _mm512_loadu_si512(reinterpret_cast<const void*>(w + 64 * r)));
}

template <int R>   // R = outputs / 16 registers
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
inline void fc0_sparse_vnni_impl(const uint8_t* __restrict u, int n, const int8_t* __restrict wc, const int32_t* __restrict bias,
                                 int32_t* __restrict out) {
    alignas(64) uint16_t nz[MAX_CHUNKS];
    const int count = nonzero_chunks_avx512(u, n, nz);
    __m512i a0[R], a1[R], a2[R], a3[R];
    for (int r = 0; r < R; ++r) {
        a0[r] = _mm512_loadu_si512(reinterpret_cast<const void*>(bias + 16 * r));
        a1[r] = a2[r] = a3[r] = _mm512_setzero_si512();
    }
    int k = 0;
    for (; k + 4 <= count; k += 4) {
        fc0_vnni_chunk<R>(a0, u, wc, nz[k]);
        fc0_vnni_chunk<R>(a1, u, wc, nz[k + 1]);
        fc0_vnni_chunk<R>(a2, u, wc, nz[k + 2]);
        fc0_vnni_chunk<R>(a3, u, wc, nz[k + 3]);
    }
    for (; k < count; ++k) fc0_vnni_chunk<R>(a0, u, wc, nz[k]);
    for (int r = 0; r < R; ++r)
        _mm512_storeu_si512(reinterpret_cast<void*>(out + 16 * r),
                            _mm512_add_epi32(_mm512_add_epi32(a0[r], a1[r]), _mm512_add_epi32(a2[r], a3[r])));
}

// Without VNNI: maddubs and madd, then an add, whose latency of one cycle needs no second set of accumulators.
template <int R>   // R = outputs / 16 registers
__attribute__((target("avx512f,avx512bw")))
inline void fc0_sparse_avx512_impl(const uint8_t* __restrict u, int n, const int8_t* __restrict wc, const int32_t* __restrict bias,
                                   int32_t* __restrict out) {
    alignas(64) uint16_t nz[MAX_CHUNKS];
    const int count = nonzero_chunks_avx512(u, n, nz);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm512_loadu_si512(reinterpret_cast<const void*>(bias + 16 * r));
    for (int k = 0; k < count; ++k) {
        const int c = nz[k];
        const __m512i x = _mm512_set1_epi32(chunk_of(u, c));
        const int8_t* w = wc + size_t(c) * 64 * R;
        for (int r = 0; r < R; ++r)
            acc[r] = _mm512_add_epi32(acc[r], _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_loadu_si512(reinterpret_cast<const void*>(w + 64 * r))), ones));
    }
    for (int r = 0; r < R; ++r) _mm512_storeu_si512(reinterpret_cast<void*>(out + 16 * r), acc[r]);
}

template <int R>   // R = outputs / 8 registers
__attribute__((target("avx2")))
inline void fc0_sparse_avx2_impl(const uint8_t* __restrict u, int n, const int8_t* __restrict wc, const int32_t* __restrict bias,
                                 int32_t* __restrict out) {
    alignas(64) uint16_t nz[MAX_CHUNKS];
    const int count = nonzero_chunks_avx2(u, n, nz);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + 8 * r));
    for (int k = 0; k < count; ++k) {
        const int c = nz[k];
        const __m256i x = _mm256_set1_epi32(chunk_of(u, c));
        const int8_t* w = wc + size_t(c) * 32 * R;
        for (int r = 0; r < R; ++r)
            acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * r))), ones));
    }
    for (int r = 0; r < R; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8 * r), acc[r]);
}

// Dispatch by output count: w is the row-major weight block (for the dense fallback), wc the chunk-major one.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
TSUMI_INLINE void fc0_sparse_avx512vnni(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int8_t* __restrict wc,
                                         const int32_t* __restrict bias, int32_t* __restrict out, int outputs) {
    if (outputs == 16) fc0_sparse_vnni_impl<1>(u, n, wc, bias, out);
    else if (outputs == 32) fc0_sparse_vnni_impl<2>(u, n, wc, bias, out);
    else if (outputs == 64) fc0_sparse_vnni_impl<4>(u, n, wc, bias, out);
    else dense_avx512vnni(u, n, w, bias, out, outputs);
}

__attribute__((target("avx512f,avx512bw")))
TSUMI_INLINE void fc0_sparse_avx512(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int8_t* __restrict wc,
                                     const int32_t* __restrict bias, int32_t* __restrict out, int outputs) {
    if (outputs == 16) fc0_sparse_avx512_impl<1>(u, n, wc, bias, out);
    else if (outputs == 32) fc0_sparse_avx512_impl<2>(u, n, wc, bias, out);
    else if (outputs == 64) fc0_sparse_avx512_impl<4>(u, n, wc, bias, out);
    else dense_avx512(u, n, w, bias, out, outputs);
}

__attribute__((target("avx2")))
TSUMI_INLINE void fc0_sparse_avx2(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int8_t* __restrict wc,
                                   const int32_t* __restrict bias, int32_t* __restrict out, int outputs) {
    if (outputs == 16) fc0_sparse_avx2_impl<2>(u, n, wc, bias, out);
    else if (outputs == 32) fc0_sparse_avx2_impl<4>(u, n, wc, bias, out);
    else if (outputs == 64) fc0_sparse_avx2_impl<8>(u, n, wc, bias, out);
    else dense_avx2(u, n, w, bias, out, outputs);
}
#endif

// The baseline kernel set's first layer: the dense routine (the chunk-major copy unused).
TSUMI_INLINE void fc0_dense_scalar(const uint8_t* __restrict u, int n, const int8_t* __restrict w, const int8_t* __restrict,
                                    const int32_t* __restrict bias, int32_t* __restrict out, int outputs) {
    dense_scalar(u, n, w, bias, out, outputs);
}

// The clipped (a) and squared clipped (s) activations of a layer's integer outputs, side by side: a = clamp(z >>
// shift, 0, 127) is the activation times 127 (z is 127 * 128 per 1.0 in the first layer, shift 7, and 127 * 64 in
// the second, shift 6), and s = a^2 / 127 is its square at the same scale.  Plain loops; the compiler vectorises them.
TSUMI_INLINE void activate(const int32_t* __restrict z, int n, uint8_t* __restrict h, int shift) {
    for (int o = 0; o < n; ++o) {
        int32_t a = z[o] >> shift;
        a = a < 0 ? 0 : (a > HEAD_ACT ? HEAD_ACT : a);
        h[o] = uint8_t(a);
        h[n + o] = uint8_t(uint32_t(a * a) / HEAD_ACT);
    }
}

// One complete kernel set per instruction-set level.  The routines above are inlined into these functions
// (`flatten`: everything they call, recursively), so the compiler compiles them all with the instruction set
// named in ATTR.  PAIRIN computes the products, SPARSE the first layer and DENSE the second.
#define TSUMI_KERNEL_SET(SUFFIX, ATTR, ACCUM, PAIRIN, DENSE, SPARSE)                                                    \
    ATTR __attribute__((flatten)) void refresh_##SUFFIX(const Network& net, const Position& pos, Color perspective, int16_t* acc) { \
        refresh_impl(net, pos, perspective, acc, ACCUM{});                                                                \
    }                                                                                                                     \
    ATTR __attribute__((flatten)) void refresh_cached_##SUFFIX(const Network& net, const Position& pos, Color perspective, int16_t* acc, \
                                                               AccumulatorCache& cache, const Position* before, const DirtyPiece* dp, \
                                                               const int16_t* parent, ThreatDiff* diff) {                \
        refresh_cached_impl(net, pos, perspective, acc, cache, before, dp, parent, diff, ACCUM{});                        \
    }                                                                                                                     \
    ATTR __attribute__((flatten)) void update_##SUFFIX(const Network& net, const Position& posBefore, const Position& posAfter, \
                                                       Color perspective, const DirtyPiece& dp, const int16_t* parent,    \
                                                       int16_t* child, ThreatDiff& diff) {                                \
        update_impl(net, posBefore, posAfter, perspective, dp, parent, child, diff, ACCUM{});                             \
    }                                                                                                                     \
    ATTR __attribute__((flatten)) int evaluate_##SUFFIX(const Network& net, const Position& pos, const int16_t* accUs,    \
                                                        const int16_t* accThem, int mobility) {                           \
        const int H = net.H, N0 = net.head_inputs(), L2 = net.L2, L3 = net.L3;                                            \
        const int b = output_bucket(pos.piece_count(), mobility);                                                         \
        alignas(64) uint8_t u[MAX_UNITS + HEAD_EXTRAS + 64];                                                              \
        PAIRIN(accUs, u, H);                                                                                              \
        PAIRIN(accThem, u + H / 2, H);                                                                                    \
        head_extra_inputs(mobility, pos.in_check(), u + H);                                                               \
        std::memset(u + N0, 0, 64);   /* the zeros SPARSE's scan reads past the inputs */                                 \
        alignas(64) int32_t z0[MAX_LAYER], z1[MAX_LAYER];                                                                 \
        alignas(64) uint8_t h[4 * MAX_LAYER];                                                                             \
        SPARSE(u, N0, net.fc0W.data() + size_t(b) * L2 * N0, net.fc0Wc.data() + size_t(b) * L2 * N0,                     \
               net.fc0B.data() + size_t(b) * L2, z0, L2);                                                                 \
        activate(z0, L2, h, 7);                                                                                           \
        DENSE(h, 2 * L2, net.fc1W.data() + size_t(b) * L3 * 2 * L2, net.fc1B.data() + size_t(b) * L3, z1, L3);           \
        activate(z1, L3, h + 2 * L2, 6);                                                                                  \
        const int n2 = 2 * L2 + 2 * L3;                                                                                   \
        const int16_t* w2 = net.fc2W.data() + size_t(b) * n2;                                                             \
        int64_t z2 = net.fc2B[b];                                                                                         \
        for (int i = 0; i < n2; ++i) z2 += int32_t(h[i]) * w2[i];                                                         \
        const int64_t skip = int64_t(HEAD_OUT_UNIT / HEAD_UNIT0) * (int64_t(z0[L2 - 2]) - z0[L2 - 1]);                    \
        return int((z2 + skip) * net.SCALE / HEAD_OUT_UNIT);                                                              \
    }

#if defined(TSUMI_X86)
TSUMI_KERNEL_SET(base, , AccumSSE2, pair_inputs_scalar, dense_scalar, fc0_dense_scalar)
TSUMI_KERNEL_SET(avx2, __attribute__((target("avx2"))), AccumAVX2, pair_inputs_avx2, dense_avx2, fc0_sparse_avx2)
TSUMI_KERNEL_SET(avx512, __attribute__((target("avx512f,avx512bw"))), AccumAVX512, pair_inputs_avx512, dense_avx512, fc0_sparse_avx512)
TSUMI_KERNEL_SET(avx512vnni, __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))), AccumAVX512, pair_inputs_avx512, dense_avx512vnni, fc0_sparse_avx512vnni)
#else
TSUMI_KERNEL_SET(base, , AccumScalar, pair_inputs_scalar, dense_scalar, fc0_dense_scalar)
#endif

}  // namespace

Kernels g_kernels = {refresh_base, refresh_cached_base, update_base, evaluate_base};

void select_kernels(CPU::Level level) {
    g_kernels = {refresh_base, refresh_cached_base, update_base, evaluate_base};
#if defined(TSUMI_X86)
    if (level == CPU::AVX2) g_kernels = {refresh_avx2, refresh_cached_avx2, update_avx2, evaluate_avx2};
    if (level == CPU::AVX512) g_kernels = {refresh_avx512, refresh_cached_avx512, update_avx512, evaluate_avx512};
    if (level == CPU::AVX512VNNI) g_kernels = {refresh_avx512vnni, refresh_cached_avx512vnni, update_avx512vnni, evaluate_avx512vnni};
#else
    (void)level;
#endif
}

}  // namespace NNUE
