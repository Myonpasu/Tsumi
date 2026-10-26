// The network's inputs, shared by the engine (nnue.cpp) and the trainer's feature library (train/featlib.cpp),
// so that both derive exactly the same inputs from a position.
//
// Piece-square features (HalfKA, as Stockfish's HalfKAv2_hm): for each perspective, every piece including both
// kings, indexed by the perspective's king-square bucket, the piece and its square, all seen from the
// perspective: for Black the ranks are flipped and the colours swapped, and the files are mirrored when the
// perspective's king stands on files e-h.
//
// Threat features (after Stockfish's FullThreats, adapted to selfmate, where checks and the squares around the
// kings are what the game is about): one feature per (attacking piece, target square, what the target holds).
// Any piece but a king is an attacker (what stands next to a king the piece-square features already see through
// the king buckets).  Its targets are the occupied squares it attacks - enemy pieces including the enemy king
// (a check), and its own pieces (defences) but not its own king - and the empty squares next to either king
// that it attacks (flight squares).  Oriented like the piece-square features.  The attacker's square is not
// part of the feature, so two rooks attacking the same square count the same feature twice.
#pragma once
#include "types.h"
#include "bitboard.h"

namespace NNUE {

constexpr int FEATLIB_VERSION = 5;                 // bumped whenever train/libfeat.so must be rebuilt (make featlib)
constexpr int FEATURES_PER_BUCKET = 768;           // 12 pieces x 64 squares
constexpr int THREAT_ATTACKERS = 10;               // pawn, knight, bishop, rook and queen of either colour
constexpr int THREAT_CONTENTS = 13;                // empty + 12 pieces
constexpr int THREAT_FEATURES = THREAT_ATTACKERS * THREAT_CONTENTS * 64;   // 8320

// How squares and piece colours are seen from a perspective whose (unflipped) king stands on ksq.
struct Orientation {
    int sqMask;        // oriented square = square ^ sqMask
    bool swapColours;  // Black's perspective: its own pieces become "white"
};
TSUMI_INLINE Orientation orientation(Color perspective, Square ksq) {
    int mask = perspective == BLACK ? 56 : 0;
    if (file_of(ksq ^ mask) >= 4) mask ^= 7;
    return {mask, perspective == BLACK};
}
TSUMI_INLINE int oriented_piece(Piece pc, const Orientation& o) { return o.swapColours ? (pc + 6) % 12 : int(pc); }

// Piece-square feature of piece pc on square sq (kingBucket is indexed by the oriented king square).
TSUMI_INLINE int psq_feature(const uint8_t* kingBucket, Color perspective, Square ksq, Piece pc, Square sq) {
    Orientation o = orientation(perspective, ksq);
    return kingBucket[ksq ^ o.sqMask] * FEATURES_PER_BUCKET + oriented_piece(pc, o) * 64 + (sq ^ o.sqMask);
}

// Attacker code (0..9) of an oriented piece code (0..11, kings excluded).
constexpr int threat_attacker(int orientedPiece) { return orientedPiece / 6 * 5 + orientedPiece % 6; }

// Threat feature of `attacker` hitting square `target`, which holds `content` (NO_PIECE for a flight square).
TSUMI_INLINE int threat_feature(const Orientation& o, Piece attacker, Square target, Piece content) {
    int c = content == NO_PIECE ? 0 : 1 + oriented_piece(content, o);
    return (threat_attacker(oriented_piece(attacker, o)) * THREAT_CONTENTS + c) * 64 + (target ^ o.sqMask);
}

// The squares next to either king.
TSUMI_INLINE Bitboard king_zone(Square wk, Square bk) {
    return Bitboards::KingAttacks[wk] | Bitboards::KingAttacks[bk];
}

// The targets of a piece on `from`: the occupied squares and the squares of `zone` (king_zone) it attacks, less
// its own king's square; a king has none.
TSUMI_INLINE Bitboard threat_targets(Piece pc, Square from, Bitboard occ, Bitboard zone, Bitboard ownKing) {
    const PieceType pt = type_of(pc);
    if (pt == KING) return 0;
    const Bitboard att = pt == PAWN ? Bitboards::PawnAttacks[color_of(pc)][from] : attacks_bb(pt, from, occ);
    return att & (occ | zone) & ~ownKing;
}

// One threat, independent of the perspective (threat_feature() turns it into an index).
struct Threat {
    uint8_t attacker, target, content;   // Piece, Square, Piece (NO_PIECE = empty)
};

// collect_threats() stops at this many threats.  Games come nowhere near it: positions from self-play have about
// 17 on average and rarely more than 60.
constexpr int MAX_THREATS = 256;

// Every threat on a board.  `board` needs pieces(), piece_on(Square) and king_square(Color), as Position has.
template <class Board>
TSUMI_INLINE int collect_threats(const Board& board, Threat* out) {
    const Bitboard occ = board.pieces();
    const Square wk = board.king_square(WHITE), bk = board.king_square(BLACK);
    const Bitboard zone = king_zone(wk, bk);
    const Bitboard kings[COLOR_NB] = {square_bb(wk), square_bb(bk)};
    int n = 0;
    Bitboard pieces = occ;
    while (pieces) {
        Square from = pop_lsb(pieces);
        Piece pc = board.piece_on(from);
        Bitboard targets = threat_targets(pc, from, occ, zone, kings[color_of(pc)]);
        while (targets && n < MAX_THREATS) {
            Square t = pop_lsb(targets);
            out[n++] = {uint8_t(pc), uint8_t(t), uint8_t(board.piece_on(t))};
        }
    }
    return n;
}

// ---- legal-move inputs -----------------------------------------------------------------------------------
// Besides the products of the accumulators, the head reads what the side to move can do.  Selfmate is a game of
// compulsion - the side to move wants to be left with nothing but moves that get it mated, and wants its
// opponent to have the moves it needs - so the number of legal moves is the signal the game turns on, and one
// the piece-square and threat features show only indirectly (whether a slider's path is blocked, which pieces
// are pinned).  The search generates the legal moves of every node before it evaluates, so they cost nothing
// there.  Castling and en passant are not counted (mobility()): the training records (src/datagen.h) hold neither
// castling rights nor an en-passant square, and the trainer must compute the same inputs.
//
// The inputs are a block of HEAD_EXTRAS bytes after the products, 0..127 like them (127 = 1.0):
//   [0..13]  the legal moves, one-hot over 14 buckets: 0, 1, 2, 3, 4, 5-6, 7-8, 9-11, 12-15, 16-20, 21-26,
//            27-33, 34-42, 43+
//   [14]     in check
//   [30]     the legal moves / 48, at most 1.0
// The other bytes are unused and always 0 (a block of 32 keeps the head's input count a multiple of 32).
constexpr int HEAD_EXTRAS = 32;
constexpr int MOBILITY_BOUNDS[13] = {1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 34, 43};

// The legal moves of a list that the inputs count.
template <class MoveRange>
TSUMI_INLINE int mobility(const MoveRange& moves) {
    int n = 0;
    for (Move m : moves) n += type_of_move(m) != CASTLING && type_of_move(m) != EN_PASSANT;
    return n;
}

template <int N>
TSUMI_INLINE int count_bucket(int n, const int (&bounds)[N]) {
    int b = 0;
    for (int x : bounds) b += n >= x;
    return b;
}

TSUMI_INLINE void head_extra_inputs(int mobility, bool inCheck, uint8_t* out) {
    for (int i = 0; i < HEAD_EXTRAS; ++i) out[i] = 0;
    out[count_bucket(mobility, MOBILITY_BOUNDS)] = 127;
    out[14] = inCheck ? 127 : 0;
    out[30] = uint8_t(std::min(127, mobility * 127 / 48));
}

// ---- output buckets --------------------------------------------------------------------------------------
// The head has its own weights for each of 4 piece-count classes (2-9, 10-17, 18-25, 26-32 pieces) x 4 classes
// of the side to move's legal moves (0-1, 2-4, 5-11, 12+), so a side that is forced and a side that is free
// are evaluated by different weights.
constexpr int OUTPUT_BUCKETS = 16;
constexpr int MOBILITY_CLASS_BOUNDS[3] = {2, 5, 12};

TSUMI_INLINE int output_bucket(int pieceCount, int mobility) {
    const int pieces = std::min(std::max((pieceCount - 2) * 4 / 31, 0), 3);
    return pieces * 4 + count_bucket(mobility, MOBILITY_CLASS_BOUNDS);
}

}  // namespace NNUE
