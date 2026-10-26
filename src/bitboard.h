#pragma once
#include "types.h"
#if defined(USE_PEXT)
#include <immintrin.h>
#endif

namespace Bitboards {

void init();

extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];   // squares strictly between two aligned squares
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];      // full line through two aligned squares (0 if not aligned)
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];  // attacks on an empty board

struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    unsigned shift;
    unsigned index(Bitboard occ) const {
#if defined(USE_PEXT)
        return unsigned(_pext_u64(occ, mask));
#else
        return unsigned(((occ & mask) * magic) >> shift);
#endif
    }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

TSUMI_INLINE Bitboard rook_attacks(Square s, Bitboard occ) {
    const Magic& m = RookMagics[s];
    return m.attacks[m.index(occ)];
}
TSUMI_INLINE Bitboard bishop_attacks(Square s, Bitboard occ) {
    const Magic& m = BishopMagics[s];
    return m.attacks[m.index(occ)];
}
TSUMI_INLINE Bitboard queen_attacks(Square s, Bitboard occ) {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

TSUMI_INLINE Bitboard attacks_bb(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
    case KNIGHT: return KnightAttacks[s];
    case BISHOP: return bishop_attacks(s, occ);
    case ROOK:   return rook_attacks(s, occ);
    case QUEEN:  return queen_attacks(s, occ);
    case KING:   return KingAttacks[s];
    default:     return 0;
    }
}

TSUMI_INLINE Bitboard pawn_attacks_bb(Color c, Bitboard pawns) {
    return c == WHITE ? shift<NORTH_EAST>(pawns) | shift<NORTH_WEST>(pawns)
                      : shift<SOUTH_EAST>(pawns) | shift<SOUTH_WEST>(pawns);
}

TSUMI_INLINE bool aligned(Square a, Square b, Square c) { return LineBB[a][b] & square_bb(c); }

}  // namespace Bitboards

using Bitboards::rook_attacks;
using Bitboards::bishop_attacks;
using Bitboards::queen_attacks;
using Bitboards::attacks_bb;
