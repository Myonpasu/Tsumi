// The trainer's feature extraction: a shared library loaded by train/features.py (ctypes) that decodes training
// records (src/datagen.h) and derives the network's inputs with the engine's own code (src/nnue_features.h and
// its move generator), so that the trainer and the engine never disagree about an input.
//
//   make featlib   ->  train/libfeat.so
#include "../src/nnue_features.h"
#include "../src/position.h"
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Record {   // src/datagen.h TrainingRecord
    uint64_t occ;
    uint8_t pcs[16];
    int16_t score;
    int8_t result;
    uint8_t stm, ply, flags;
    int16_t next;
};
static_assert(sizeof(Record) == 32, "record layout");

// A record's board, with the interface collect_threats() needs.
struct Board {
    Bitboard occ = 0;
    Piece board[SQUARE_NB];
    Square king[COLOR_NB] = {SQ_NONE, SQ_NONE};
    Bitboard pieces() const { return occ; }
    Piece piece_on(Square s) const { return board[s]; }
    Square king_square(Color c) const { return king[c]; }

    bool decode(const Record& r) {
        occ = r.occ;
        king[WHITE] = king[BLACK] = SQ_NONE;
        for (int s = 0; s < 64; ++s) board[s] = NO_PIECE;
        Bitboard b = occ;
        int i = 0;
        while (b) {
            Square s = pop_lsb(b);
            uint8_t nib = (i & 1) ? (r.pcs[i >> 1] >> 4) : (r.pcs[i >> 1] & 15);
            if (nib >= 12 || i >= 32) return false;
            Piece pc = Piece(nib);
            board[s] = pc;
            if (type_of(pc) == KING) {
                if (king[color_of(pc)] != SQ_NONE) return false;
                king[color_of(pc)] = s;
            }
            i++;
        }
        return king[WHITE] != SQ_NONE && king[BLACK] != SQ_NONE;
    }
};

// The inputs of records [0, n): see tsumi_extract.
int extract_range(const uint8_t* records, int n, const uint8_t* kingBucket, int padPsq, int padThr, int maxThreats,
                  int64_t* psqUs, int64_t* psqThem, int64_t* thrUs, int64_t* thrThem, int64_t* buckets, uint8_t* extras) {
    int truncated = 0;
    Board bd;
    NNUE::Threat threats[NNUE::MAX_THREATS];
    Position pos;
    for (int k = 0; k < n; ++k) {
        Record r;
        std::memcpy(&r, records + size_t(k) * sizeof(Record), sizeof(Record));
        if (!bd.decode(r)) return -1;
        const Color stm = r.stm ? BLACK : WHITE;
        int64_t* psq[2] = {psqUs + size_t(k) * 32, psqThem + size_t(k) * 32};
        int64_t* thr[2] = {thrUs + size_t(k) * maxThreats, thrThem + size_t(k) * maxThreats};
        const int nt = NNUE::collect_threats(bd, threats);
        if (nt > maxThreats) truncated++;
        for (int p = 0; p < 2; ++p) {
            const Color perspective = p == 0 ? stm : ~stm;
            const Square ksq = bd.king_square(perspective);
            int m = 0;
            for (Bitboard b = bd.occ; b;) {
                const Square s = pop_lsb(b);
                psq[p][m++] = NNUE::psq_feature(kingBucket, perspective, ksq, bd.piece_on(s), s);
            }
            for (; m < 32; ++m) psq[p][m] = padPsq;
            const NNUE::Orientation o = NNUE::orientation(perspective, ksq);
            m = 0;
            for (int i = 0; i < nt && m < maxThreats; ++i)
                thr[p][m++] = NNUE::threat_feature(o, Piece(threats[i].attacker), Square(threats[i].target), Piece(threats[i].content));
            for (; m < maxThreats; ++m) thr[p][m] = padThr;
        }
        // The legal-move inputs and the output bucket, from the engine's move generator on the position as the
        // record holds it (no castling rights, no en-passant square).
        pos.set_board(bd.board, stm);
        MoveList ml;
        pos.generate_legal(ml);
        const int mobility = NNUE::mobility(ml);
        NNUE::head_extra_inputs(mobility, pos.in_check(), extras + size_t(k) * NNUE::HEAD_EXTRAS);
        buckets[k] = NNUE::output_bucket(pos.piece_count(), mobility);
    }
    return truncated;
}

bool g_init = false;

}  // namespace

extern "C" {

int tsumi_featlib_version() { return NNUE::FEATLIB_VERSION; }
int tsumi_threat_features() { return NNUE::THREAT_FEATURES; }
int tsumi_head_extras() { return NNUE::HEAD_EXTRAS; }
int tsumi_output_buckets() { return NNUE::OUTPUT_BUCKETS; }

// The inputs of n records, from the side to move's perspective (us) and the opponent's (them), all int64 (torch's
// index type) but the extras:
//   psqUs, psqThem   [n][32] piece-square indices, padPsq where unused
//   thrUs, thrThem   [n][maxThreats] threat indices, padThr where unused
//   buckets          [n] output buckets
//   extras           [n][HEAD_EXTRAS] the head's legal-move inputs (0..127)
// over `threads` threads.  Returns the number of records with more than maxThreats threats (their lists are cut
// short), or -1 for a malformed record.
int tsumi_extract(const uint8_t* records, int n, const uint8_t* kingBucket, int padPsq, int padThr, int maxThreats,
                  int64_t* psqUs, int64_t* psqThem, int64_t* thrUs, int64_t* thrThem, int64_t* buckets, uint8_t* extras,
                  int threads) {
    if (!g_init) { Bitboards::init(); Zobrist::init(); g_init = true; }
    threads = std::max(1, std::min(threads, 64));
    if (threads == 1 || n < 2 * threads)
        return extract_range(records, n, kingBucket, padPsq, padThr, maxThreats, psqUs, psqThem, thrUs, thrThem, buckets, extras);
    std::vector<int> results(threads, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        const int a = int(int64_t(n) * t / threads), b = int(int64_t(n) * (t + 1) / threads);
        pool.emplace_back([=, &results] {
            results[t] = extract_range(records + size_t(a) * sizeof(Record), b - a, kingBucket, padPsq, padThr, maxThreats,
                                       psqUs + size_t(a) * 32, psqThem + size_t(a) * 32, thrUs + size_t(a) * maxThreats,
                                       thrThem + size_t(a) * maxThreats, buckets + a, extras + size_t(a) * NNUE::HEAD_EXTRAS);
        });
    }
    for (auto& th : pool) th.join();
    int truncated = 0;
    for (int r : results) { if (r < 0) return -1; truncated += r; }
    return truncated;
}

}  // extern "C"
