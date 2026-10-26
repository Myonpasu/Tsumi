#pragma once
#include "types.h"
#include <string>

struct MoveList {
    Move moves[256];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + size; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + size; }
    bool contains(Move m) const { for (int i = 0; i < size; ++i) if (moves[i] == m) return true; return false; }
};

// Precomputed data for fast "does this move give check" tests in one position.
struct CheckInfo {
    Bitboard checkSquares[PIECE_TYPE_NB];   // squares from which a piece of each type checks the enemy king
    Bitboard discoverers;                   // our pieces whose move may uncover a check
    Square enemyKing;
};

// The pieces the last move changed, for the incremental NNUE updates: at most three entries - the moved piece (a
// promotion is two: the pawn disappears and the new piece appears), a captured piece and a castling rook.
struct DirtyPiece {
    int count = 0;
    Piece piece[3];
    Square from[3];  // SQ_NONE = the piece appears
    Square to[3];    // SQ_NONE = the piece disappears
};

namespace Zobrist {
extern Key psq[PIECE_NB][SQUARE_NB];
extern Key castling[16];
extern Key enpassant[8];
extern Key side;
void init();
}

class Position {
public:
    Position() { clear(); }
    void clear();
    bool set(const std::string& fen);  // returns false on malformed FEN
    // A position from a board alone (the trainer's records): no castling rights, no en-passant square.
    void set_board(const Piece (&pieces)[SQUARE_NB], Color sideToMove);
    std::string fen() const;
    std::string pretty() const;

    // Accessors
    Color side_to_move() const { return stm; }
    Bitboard pieces() const { return byColor[WHITE] | byColor[BLACK]; }
    Bitboard pieces(Color c) const { return byColor[c]; }
    Bitboard pieces(PieceType pt) const { return byType[pt]; }
    Bitboard pieces(Color c, PieceType pt) const { return byColor[c] & byType[pt]; }
    Bitboard pieces(PieceType a, PieceType b) const { return byType[a] | byType[b]; }
    Bitboard pieces(Color c, PieceType a, PieceType b) const { return byColor[c] & (byType[a] | byType[b]); }
    Piece piece_on(Square s) const { return board[s]; }
    bool empty(Square s) const { return board[s] == NO_PIECE; }
    Square king_square(Color c) const { return lsb(pieces(c, KING)); }
    int rule50_count() const { return halfmoveClock; }
    int game_ply() const { return gamePly; }
    Key key() const { return hash; }
    Bitboard checkers() const { return checkersBB; }
    bool in_check() const { return checkersBB != 0; }
    int piece_count() const { return popcount(pieces()); }
    bool bare_king(Color c) const { return pieces(c) == pieces(c, KING); }   // side c has nothing but its king

    Bitboard attackers_to(Square s, Bitboard occ) const;
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    Bitboard slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const;

    // Move properties
    bool is_capture(Move m) const { return board[to_sq(m)] != NO_PIECE || type_of_move(m) == EN_PASSANT; }
    Piece captured_piece(Move m) const;   // piece that would be captured (NO_PIECE if none)
    Piece moved_piece(Move m) const { return board[from_sq(m)]; }
    bool gives_check(Move m) const;
    CheckInfo check_info() const;
    bool gives_check(Move m, const CheckInfo& ci) const;
    bool see_ge(Move m, int threshold) const;   // static exchange evaluation of m is at least threshold

    // Move generation
    void generate_legal(MoveList& ml) const;

    // Making moves (copy-make: callers copy the Position first)
    void do_move(Move m, DirtyPiece* dp = nullptr);

    // Variant-independent draw detection
    bool is_dead_position() const;  // no sequence of legal moves can lead to checkmate

    // Debug
    bool consistent() const;

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void update_checkers();
    bool set_impl(const std::string& fen);
    template <Color Us> void generate_legal_impl(MoveList& ml) const;

    Bitboard byType[PIECE_TYPE_NB];
    Bitboard byColor[COLOR_NB];
    Piece board[SQUARE_NB];
    Bitboard checkersBB;
    Key hash;
    Color stm;
    int castling;
    Square epSquare;
    int halfmoveClock;
    int gamePly;
};

Move parse_uci_move(const Position& pos, const std::string& str);
