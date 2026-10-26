#include "bitboard.h"

namespace Bitboards {

Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

static Bitboard RookTable[0x19000];
static Bitboard BishopTable[0x1480];

static Bitboard safe_destination(Square s, int step) {
    Square to = s + step;
    return (to >= 0 && to < 64 && distance(s, to) <= 2) ? square_bb(to) : 0;
}

// Slow ray attack generator used to fill the magic tables.
static Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
    Bitboard attacks = 0;
    const int rookDirs[4] = {NORTH, SOUTH, EAST, WEST};
    const int bishopDirs[4] = {NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST};
    for (int d : (pt == ROOK ? rookDirs : bishopDirs)) {
        Square s = sq;
        while (safe_destination(s, d) && !(occupied & square_bb(s))) {
            s += d;
            attacks |= square_bb(s);
        }
    }
    return attacks;
}

static void init_magics(PieceType pt, Bitboard table[], Magic magics[]) {
    Bitboard occupancy[4096], reference[4096];
    int epoch[4096] = {0}, cnt = 0, size = 0;
    PRNG rng(0x3C2E6B8E6B2A1F5DULL);

    for (Square s = 0; s < 64; ++s) {
        // Board edges are not relevant for the occupancy mask, except when the piece is on an edge.
        Bitboard edges = ((RANK_1_BB | RANK_8_BB) & ~rank_bb(rank_of(s))) | ((FILE_A_BB | FILE_H_BB) & ~file_bb(file_of(s)));
        Magic& m = magics[s];
        m.mask = sliding_attack(pt, s, 0) & ~edges;
        m.shift = 64 - popcount(m.mask);
        m.attacks = s == 0 ? table : magics[s - 1].attacks + size;

        // Enumerate all subsets of the mask (Carry-Rippler).
        Bitboard b = 0;
        size = 0;
        do {
            occupancy[size] = b;
            reference[size] = sliding_attack(pt, s, b);
#if defined(USE_PEXT)
            m.attacks[_pext_u64(b, m.mask)] = reference[size];
#endif
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

#if defined(USE_PEXT)
        continue;
#endif
        // Find a magic that maps every occupancy to its attack set without harmful collisions.
        for (int i = 0; i < size;) {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                m.magic = rng.sparse_rand();
            for (++cnt, i = 0; i < size; ++i) {
                unsigned idx = m.index(occupancy[i]);
                if (epoch[idx] < cnt) {
                    epoch[idx] = cnt;
                    m.attacks[idx] = reference[i];
                } else if (m.attacks[idx] != reference[i])
                    break;
            }
        }
    }
}

void init() {
    for (Square s = 0; s < 64; ++s) {
        PawnAttacks[WHITE][s] = pawn_attacks_bb(WHITE, square_bb(s));
        PawnAttacks[BLACK][s] = pawn_attacks_bb(BLACK, square_bb(s));
        for (int step : {-9, -8, -7, -1, 1, 7, 8, 9}) KingAttacks[s] |= safe_destination(s, step);
        for (int step : {-17, -15, -10, -6, 6, 10, 15, 17}) KnightAttacks[s] |= safe_destination(s, step);
        PseudoAttacks[KING][s] = KingAttacks[s];
        PseudoAttacks[KNIGHT][s] = KnightAttacks[s];
    }

    init_magics(ROOK, RookTable, RookMagics);
    init_magics(BISHOP, BishopTable, BishopMagics);

    for (Square s1 = 0; s1 < 64; ++s1) {
        PseudoAttacks[BISHOP][s1] = bishop_attacks(s1, 0);
        PseudoAttacks[ROOK][s1] = rook_attacks(s1, 0);
        PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] | PseudoAttacks[ROOK][s1];
        for (Square s2 = 0; s2 < 64; ++s2) {
            if (PseudoAttacks[BISHOP][s1] & square_bb(s2)) {
                LineBB[s1][s2] = (bishop_attacks(s1, 0) & bishop_attacks(s2, 0)) | square_bb(s1) | square_bb(s2);
                BetweenBB[s1][s2] = bishop_attacks(s1, square_bb(s2)) & bishop_attacks(s2, square_bb(s1));
            } else if (PseudoAttacks[ROOK][s1] & square_bb(s2)) {
                LineBB[s1][s2] = (rook_attacks(s1, 0) & rook_attacks(s2, 0)) | square_bb(s1) | square_bb(s2);
                BetweenBB[s1][s2] = rook_attacks(s1, square_bb(s2)) & rook_attacks(s2, square_bb(s1));
            }
        }
    }
}

}  // namespace Bitboards
