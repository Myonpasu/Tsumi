// Tsumi - a chess engine for "selfmate chess": the player who gets checkmated wins.
// Basic types shared by every module.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <string>
#include <algorithm>

#if defined(__GNUC__)
#define TSUMI_INLINE inline __attribute__((always_inline))
#else
#define TSUMI_INLINE inline
#endif

typedef uint64_t Bitboard;
typedef uint64_t Key;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE = 6, PIECE_TYPE_NB = 6 };

// Piece = color * 6 + type.  White pieces 0..5, black pieces 6..11.
enum Piece : int { NO_PIECE = 12, PIECE_NB = 12 };
constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }
constexpr Color color_of(Piece p) { return Color(p / 6); }

// Squares: a1 = 0, b1 = 1, ..., h8 = 63 (little-endian rank-file).
typedef int Square;
enum : int { SQ_NONE = 64, SQUARE_NB = 64 };
constexpr Square make_square(int file, int rank) { return rank * 8 + file; }
constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr int distance(Square a, Square b) {               // king moves between two squares
    int df = file_of(a) - file_of(b), dr = rank_of(a) - rank_of(b);
    return std::max(df < 0 ? -df : df, dr < 0 ? -dr : dr);
}

// Castling rights bits.
enum CastlingRight : int { WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8 };

// Moves: bits 0-5 from, 6-11 to, 12-13 promotion piece (0=N,1=B,2=R,3=Q), 14-15 type.
typedef uint16_t Move;
enum MoveType : int { NORMAL = 0, PROMOTION = 1 << 14, EN_PASSANT = 2 << 14, CASTLING = 3 << 14 };
constexpr Move MOVE_NONE = 0;

constexpr Move make_move(Square from, Square to) { return Move(from | (to << 6)); }
constexpr Move make_move(Square from, Square to, MoveType mt, PieceType promo = KNIGHT) {
    return Move(from | (to << 6) | ((promo - KNIGHT) << 12) | mt);
}
constexpr Square from_sq(Move m) { return m & 63; }
constexpr Square to_sq(Move m) { return (m >> 6) & 63; }
constexpr MoveType type_of_move(Move m) { return MoveType(m & (3 << 14)); }
constexpr PieceType promotion_type(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }

// Scores.  Positive = good for the side to move = the side to move is heading for being checkmated.  Internal
// units are millipawns, reported to the GUI in centipawns (score / UNITS_PER_CP).  The network's output is
// multiplied by its SCALE (1000) into internal units, and the trainer makes an output of 1.0 a 50% chance of
// winning, so +1.00 as reported is a 50% chance of winning.  The search's margins are in internal units too.
constexpr int UNITS_PER_CP = 10;
constexpr int VALUE_INFINITE = 32001;
constexpr int VALUE_NONE = 32002;
constexpr int VALUE_MATE = 32000;              // side to move is checkmated right now (and therefore wins)
constexpr int MAX_PLY = 128;
constexpr int VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr int VALUE_MAX_EVAL = 20000;          // static evaluations are clamped inside +-VALUE_MAX_EVAL; only mate scores lie beyond
constexpr int VALUE_DRAW = 0;
constexpr int BARE_KING_CONTEMPT = 2;   // see Thread::evaluate: a position with a bare king that is still being played, vs. a draw reached
constexpr int BARE_KING_ROOM = 2;       // ... and per square the bare king can still step to: confining it brings the stalemate nearer

constexpr int mate_in(int ply) { return VALUE_MATE - ply; }      // we get mated at this ply: we win
constexpr int mated_in(int ply) { return -VALUE_MATE + ply; }    // we are forced to mate the opponent: we lose
constexpr bool is_mate_score(int v) { return v >= VALUE_MATE_IN_MAX_PLY || v <= -VALUE_MATE_IN_MAX_PLY; }

// ---------------------------------------------------------------------------
// Bitboard helpers
// ---------------------------------------------------------------------------
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;
constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;
constexpr Bitboard LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;
constexpr Bitboard DARK_SQUARES = ~LIGHT_SQUARES;

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }
constexpr Bitboard file_bb(int f) { return FILE_A_BB << f; }
constexpr Bitboard rank_bb(int r) { return RANK_1_BB << (8 * r); }

TSUMI_INLINE int popcount(Bitboard b) { return __builtin_popcountll(b); }
TSUMI_INLINE Square lsb(Bitboard b) { assert(b); return Square(__builtin_ctzll(b)); }
TSUMI_INLINE Square pop_lsb(Bitboard& b) { Square s = lsb(b); b &= b - 1; return s; }
TSUMI_INLINE bool more_than_one(Bitboard b) { return b & (b - 1); }

enum Direction : int { NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
                       NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9 };

template <Direction D>
constexpr Bitboard shift(Bitboard b) {
    return D == NORTH      ? b << 8 : D == SOUTH ? b >> 8
         : D == EAST       ? (b & ~FILE_H_BB) << 1 : D == WEST ? (b & ~FILE_A_BB) >> 1
         : D == NORTH_EAST ? (b & ~FILE_H_BB) << 9 : D == NORTH_WEST ? (b & ~FILE_A_BB) << 7
         : D == SOUTH_EAST ? (b & ~FILE_H_BB) >> 7 : D == SOUTH_WEST ? (b & ~FILE_A_BB) >> 9 : 0;
}

inline std::string square_to_string(Square s) {
    if (s == SQ_NONE) return "-";
    std::string r;
    r += char('a' + file_of(s));
    r += char('1' + rank_of(s));
    return r;
}

inline std::string move_to_string(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = square_to_string(from_sq(m)) + square_to_string(to_sq(m));
    if (type_of_move(m) == PROMOTION) s += "nbrq"[promotion_type(m) - KNIGHT];
    return s;
}

// Fast deterministic PRNG (xorshift64*), used for zobrist keys, magics and data generation.
struct PRNG {
    uint64_t s;
    explicit PRNG(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t rand64() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    uint64_t sparse_rand() { return rand64() & rand64() & rand64(); }
    uint64_t rand_below(uint64_t n) { return rand64() % n; }   // uniform in [0, n)
};

constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
