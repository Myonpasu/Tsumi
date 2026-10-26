#include "position.h"
#include "bitboard.h"
#include <array>
#include <sstream>
#include <cstring>

using namespace Bitboards;

namespace Zobrist {
Key psq[PIECE_NB][SQUARE_NB];
Key castling[16];
Key enpassant[8];
Key side;

void init() {
    PRNG rng(0x5EEDC0FFEE123456ULL);
    for (int p = 0; p < PIECE_NB; ++p)
        for (int s = 0; s < 64; ++s) psq[p][s] = rng.rand64();
    for (int i = 0; i < 16; ++i) castling[i] = rng.rand64();
    castling[0] = 0;
    for (int f = 0; f < 8; ++f) enpassant[f] = rng.rand64();
    side = rng.rand64();
}
}  // namespace Zobrist

namespace {
// Castling rights lost when a piece moves from / is captured on a square.  Computed at compile time:
// the engine must not have global objects with dynamic initialisation (see src/main.cpp).
constexpr std::array<int, 64> make_castling_mask() {
    std::array<int, 64> m{};
    m[0] = WHITE_OOO;  m[7] = WHITE_OO;  m[4] = WHITE_OO | WHITE_OOO;
    m[56] = BLACK_OOO; m[63] = BLACK_OO; m[60] = BLACK_OO | BLACK_OOO;
    return m;
}
constexpr std::array<int, 64> CastlingMask = make_castling_mask();

const char PieceChars[] = "PNBRQKpnbrqk";
}  // namespace

void Position::clear() {
    std::memset(byType, 0, sizeof(byType));
    std::memset(byColor, 0, sizeof(byColor));
    for (int s = 0; s < 64; ++s) board[s] = NO_PIECE;
    checkersBB = 0;
    hash = 0;
    stm = WHITE;
    castling = 0;
    epSquare = SQ_NONE;
    halfmoveClock = 0;
    gamePly = 0;
}

void Position::put_piece(Piece p, Square s) {
    board[s] = p;
    byType[type_of(p)] |= square_bb(s);
    byColor[color_of(p)] |= square_bb(s);
    hash ^= Zobrist::psq[p][s];
}

void Position::remove_piece(Square s) {
    Piece p = board[s];
    byType[type_of(p)] ^= square_bb(s);
    byColor[color_of(p)] ^= square_bb(s);
    board[s] = NO_PIECE;
    hash ^= Zobrist::psq[p][s];
}

void Position::move_piece(Square from, Square to) {
    Piece p = board[from];
    Bitboard ft = square_bb(from) | square_bb(to);
    byType[type_of(p)] ^= ft;
    byColor[color_of(p)] ^= ft;
    board[from] = NO_PIECE;
    board[to] = p;
    hash ^= Zobrist::psq[p][from] ^ Zobrist::psq[p][to];
}

void Position::update_checkers() {
    checkersBB = attackers_to(king_square(stm)) & pieces(~stm);
}

bool Position::set(const std::string& fenStr) {
    if (!set_impl(fenStr)) {
        set_impl(START_FEN);  // never leave an unusable position behind
        return false;
    }
    return true;
}

void Position::set_board(const Piece (&pieces)[SQUARE_NB], Color sideToMove) {
    clear();
    for (int s = 0; s < SQUARE_NB; ++s)
        if (pieces[s] != NO_PIECE) put_piece(pieces[s], Square(s));
    stm = sideToMove;
    if (stm == BLACK) hash ^= Zobrist::side;
    update_checkers();
}

bool Position::set_impl(const std::string& fenStr) {
    clear();
    std::istringstream ss(fenStr);
    std::string board_s, stm_s, castle_s, ep_s;
    int hm = 0, fm = 1;
    ss >> board_s >> stm_s >> castle_s >> ep_s;
    if (!(ss >> hm)) hm = 0;
    if (!(ss >> fm)) fm = 1;
    if (board_s.empty() || stm_s.empty()) return false;

    int rank = 7, file = 0;
    for (char c : board_s) {
        if (c == '/') { rank--; file = 0; if (rank < 0) return false; }
        else if (c >= '1' && c <= '8') file += c - '0';
        else {
            const char* p = c ? std::strchr(PieceChars, c) : nullptr;
            if (!p || file > 7) return false;
            put_piece(Piece(p - PieceChars), make_square(file, rank));
            file++;
        }
    }
    // The material must be possible in a game: a king each, no pawns on the first or last rank, and at most
    // sixteen pieces a side, with no more promoted pieces than missing pawns.  (Fixed-size buffers - the move
    // list, the network's feature lists, the training records - rely on it.)
    if (pieces(PAWN) & (RANK_1_BB | RANK_8_BB)) return false;
    for (Color c : {WHITE, BLACK}) {
        auto extra = [&](PieceType pt, int original) { return std::max(0, popcount(pieces(c, pt)) - original); };
        const int promoted = extra(KNIGHT, 2) + extra(BISHOP, 2) + extra(ROOK, 2) + extra(QUEEN, 1);
        if (popcount(pieces(c, KING)) != 1 || popcount(pieces(c, PAWN)) + promoted > 8 || popcount(pieces(c)) > 16) return false;
    }

    stm = stm_s == "w" ? WHITE : stm_s == "b" ? BLACK : COLOR_NB;
    if (stm == COLOR_NB) return false;
    if (stm == BLACK) hash ^= Zobrist::side;

    castling = 0;
    for (char c : castle_s) {
        if (c == 'K' && board[4] == make_piece(WHITE, KING) && board[7] == make_piece(WHITE, ROOK)) castling |= WHITE_OO;
        if (c == 'Q' && board[4] == make_piece(WHITE, KING) && board[0] == make_piece(WHITE, ROOK)) castling |= WHITE_OOO;
        if (c == 'k' && board[60] == make_piece(BLACK, KING) && board[63] == make_piece(BLACK, ROOK)) castling |= BLACK_OO;
        if (c == 'q' && board[60] == make_piece(BLACK, KING) && board[56] == make_piece(BLACK, ROOK)) castling |= BLACK_OOO;
    }
    hash ^= Zobrist::castling[castling];

    epSquare = SQ_NONE;
    if (ep_s.size() == 2 && ep_s[0] >= 'a' && ep_s[0] <= 'h' && ep_s[1] == (stm == WHITE ? '6' : '3')) {
        Square ep = make_square(ep_s[0] - 'a', ep_s[1] - '1');
        Square capsq = stm == WHITE ? ep - 8 : ep + 8;
        // Only keep the square if the capture is at least pseudo-legal.
        if (board[capsq] == make_piece(~stm, PAWN) && empty(ep) && (PawnAttacks[~stm][ep] & pieces(stm, PAWN))) {
            epSquare = ep;
            hash ^= Zobrist::enpassant[file_of(ep)];
        }
    }
    halfmoveClock = hm;
    gamePly = std::max(2 * (fm - 1), 0) + (stm == BLACK);
    update_checkers();

    // The side not to move must not be in check (otherwise the position is unreachable).
    if (attackers_to(king_square(~stm)) & pieces(stm)) return false;
    return true;
}

std::string Position::fen() const {
    std::ostringstream ss;
    for (int r = 7; r >= 0; --r) {
        int emptyCnt = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE) emptyCnt++;
            else {
                if (emptyCnt) { ss << emptyCnt; emptyCnt = 0; }
                ss << PieceChars[p];
            }
        }
        if (emptyCnt) ss << emptyCnt;
        if (r > 0) ss << '/';
    }
    ss << (stm == WHITE ? " w " : " b ");
    if (!castling) ss << '-';
    else {
        if (castling & WHITE_OO) ss << 'K';
        if (castling & WHITE_OOO) ss << 'Q';
        if (castling & BLACK_OO) ss << 'k';
        if (castling & BLACK_OOO) ss << 'q';
    }
    ss << ' ' << square_to_string(epSquare) << ' ' << halfmoveClock << ' ' << 1 + gamePly / 2;
    return ss.str();
}

std::string Position::pretty() const {
    std::string s = "\n +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(f, r)];
            s += " | ";
            s += p == NO_PIECE ? ' ' : PieceChars[p];
        }
        s += " | " + std::to_string(1 + r) + "\n +---+---+---+---+---+---+---+---+\n";
    }
    s += "   a   b   c   d   e   f   g   h\n\nFen: " + fen() + "\nKey: ";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)hash);
    s += buf;
    s += "\nCheckers:";
    Bitboard b = checkersBB;
    while (b) s += " " + square_to_string(pop_lsb(b));
    return s + "\n";
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (PawnAttacks[BLACK][s] & pieces(WHITE, PAWN))
         | (PawnAttacks[WHITE][s] & pieces(BLACK, PAWN))
         | (KnightAttacks[s] & pieces(KNIGHT))
         | (rook_attacks(s, occ) & pieces(ROOK, QUEEN))
         | (bishop_attacks(s, occ) & pieces(BISHOP, QUEEN))
         | (KingAttacks[s] & pieces(KING));
}

Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const {
    Bitboard blockers = 0;
    pinners = 0;
    Bitboard snipers = ((PseudoAttacks[ROOK][s] & pieces(ROOK, QUEEN)) | (PseudoAttacks[BISHOP][s] & pieces(BISHOP, QUEEN))) & sliders;
    Bitboard occupancy = pieces() ^ snipers;
    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        Bitboard b = BetweenBB[s][sniperSq] & occupancy;
        if (b && !more_than_one(b)) {
            blockers |= b;
            if (b & pieces(color_of(board[s]))) pinners |= square_bb(sniperSq);
        }
    }
    return blockers;
}

// Static exchange evaluation: does the exchange that move m starts on its destination square - each side
// recapturing with its least valuable attacker, and free to stop when that is better - gain at least
// `threshold` (pawn 100, minor 300, rook 500, queen 900)?  X-ray attackers behind the pieces that capture join
// in; pins and checks are ignored, and promotions, en passant and castling count as even.  (Stockfish's form.)
bool Position::see_ge(Move m, int threshold) const {
    static constexpr int V[PIECE_TYPE_NB] = {100, 300, 300, 500, 900, 0};
    if (type_of_move(m) != NORMAL) return 0 >= threshold;
    const Square from = from_sq(m), to = to_sq(m);
    int swap = (board[to] == NO_PIECE ? 0 : V[type_of(board[to])]) - threshold;
    if (swap < 0) return false;
    swap = V[type_of(board[from])] - swap;
    if (swap <= 0) return true;
    Bitboard occ = pieces() ^ square_bb(from) ^ square_bb(to);
    Bitboard attackers = attackers_to(to, occ);
    const Bitboard diag = pieces(BISHOP, QUEEN), orth = pieces(ROOK, QUEEN);
    Color side = stm;
    int res = 1;
    while (true) {
        side = ~side;
        attackers &= occ;
        const Bitboard mine = attackers & pieces(side);
        if (!mine) break;
        res ^= 1;
        Bitboard bb;
        if ((bb = mine & pieces(PAWN))) {
            if ((swap = V[PAWN] - swap) < res) break;
            occ ^= bb & (0 - bb);
            attackers |= bishop_attacks(to, occ) & diag;
        } else if ((bb = mine & pieces(KNIGHT))) {
            if ((swap = V[KNIGHT] - swap) < res) break;
            occ ^= bb & (0 - bb);
        } else if ((bb = mine & pieces(BISHOP))) {
            if ((swap = V[BISHOP] - swap) < res) break;
            occ ^= bb & (0 - bb);
            attackers |= bishop_attacks(to, occ) & diag;
        } else if ((bb = mine & pieces(ROOK))) {
            if ((swap = V[ROOK] - swap) < res) break;
            occ ^= bb & (0 - bb);
            attackers |= rook_attacks(to, occ) & orth;
        } else if ((bb = mine & pieces(QUEEN))) {
            if ((swap = V[QUEEN] - swap) < res) break;
            occ ^= bb & (0 - bb);
            attackers |= (bishop_attacks(to, occ) & diag) | (rook_attacks(to, occ) & orth);
        } else {
            // The king captures last, and only if the opponent has nothing left to recapture with.
            return (attackers & ~pieces(side)) ? res ^ 1 : res;
        }
    }
    return bool(res);
}

Piece Position::captured_piece(Move m) const {
    if (type_of_move(m) == EN_PASSANT) return make_piece(~stm, PAWN);
    if (type_of_move(m) == CASTLING) return NO_PIECE;
    return board[to_sq(m)];
}

bool Position::gives_check(Move m) const {
    Color us = stm, them = ~us;
    Square from = from_sq(m), to = to_sq(m);
    Square ksq = king_square(them);
    Bitboard kbb = square_bb(ksq);
    Bitboard occ = pieces();
    PieceType pt = type_of(board[from]);
    Bitboard occAfter, removed = square_bb(from);

    switch (type_of_move(m)) {
    case EN_PASSANT: {
        Square capsq = make_square(file_of(to), rank_of(from));
        occAfter = (occ ^ square_bb(from) ^ square_bb(capsq)) | square_bb(to);
        break;
    }
    case CASTLING: {
        Square rfrom = to > from ? to + 1 : to - 2;
        Square rto = to > from ? to - 1 : to + 1;
        occAfter = (occ ^ square_bb(from) ^ square_bb(rfrom)) | square_bb(to) | square_bb(rto);
        if (rook_attacks(rto, occAfter) & kbb) return true;
        removed |= square_bb(rfrom);
        break;
    }
    default:
        occAfter = (occ ^ square_bb(from)) | square_bb(to);
        if (type_of_move(m) == PROMOTION) pt = promotion_type(m);
        break;
    }

    switch (pt) {
    case PAWN:   if (PawnAttacks[us][to] & kbb) return true; break;
    case KNIGHT: if (KnightAttacks[to] & kbb) return true; break;
    case BISHOP: if (bishop_attacks(to, occAfter) & kbb) return true; break;
    case ROOK:   if (rook_attacks(to, occAfter) & kbb) return true; break;
    case QUEEN:  if (queen_attacks(to, occAfter) & kbb) return true; break;
    default: break;
    }

    Bitboard rq = pieces(us, ROOK, QUEEN) & ~removed;
    if (rq && (rook_attacks(ksq, occAfter) & rq)) return true;
    Bitboard bq = pieces(us, BISHOP, QUEEN) & ~removed;
    if (bq && (bishop_attacks(ksq, occAfter) & bq)) return true;
    return false;
}

CheckInfo Position::check_info() const {
    CheckInfo ci;
    Color us = stm, them = ~us;
    Square ksq = king_square(them);
    Bitboard occ = pieces();
    ci.enemyKing = ksq;
    ci.checkSquares[PAWN] = PawnAttacks[them][ksq];
    ci.checkSquares[KNIGHT] = KnightAttacks[ksq];
    ci.checkSquares[BISHOP] = bishop_attacks(ksq, occ);
    ci.checkSquares[ROOK] = rook_attacks(ksq, occ);
    ci.checkSquares[QUEEN] = ci.checkSquares[BISHOP] | ci.checkSquares[ROOK];
    ci.checkSquares[KING] = 0;
    Bitboard pinners;
    ci.discoverers = slider_blockers(pieces(us), ksq, pinners) & pieces(us);
    return ci;
}

bool Position::gives_check(Move m, const CheckInfo& ci) const {
    Square from = from_sq(m), to = to_sq(m);
    if (type_of_move(m) != NORMAL) return gives_check(m);   // promotions, en passant, castling: exact slow path
    PieceType pt = type_of(board[from]);
    if (ci.checkSquares[pt] & square_bb(to)) return true;
    // Discovered check: a discoverer leaving the line between its slider and the enemy king.
    return (ci.discoverers & square_bb(from)) && !aligned(from, to, ci.enemyKing);
}

template <Color Us>
void Position::generate_legal_impl(MoveList& ml) const {
    constexpr Color Them = ~Us;
    constexpr Direction Up = Us == WHITE ? NORTH : SOUTH;
    constexpr Direction UpRight = Us == WHITE ? NORTH_EAST : SOUTH_WEST;
    constexpr Direction UpLeft = Us == WHITE ? NORTH_WEST : SOUTH_EAST;
    constexpr Bitboard Rank7 = Us == WHITE ? RANK_7_BB : RANK_2_BB;
    constexpr Bitboard Rank3 = Us == WHITE ? (RANK_1_BB << 16) : (RANK_1_BB << 40);

    const Square ksq = king_square(Us);
    const Bitboard occ = pieces();
    const Bitboard ours = pieces(Us);
    const Bitboard theirs = pieces(Them);
    const Bitboard checkers = checkersBB;

    // King moves: the king's own square is removed from the occupancy so x-ray attacks count.
    {
        Bitboard occNoKing = occ ^ square_bb(ksq);
        Bitboard b = KingAttacks[ksq] & ~ours;
        while (b) {
            Square to = pop_lsb(b);
            if (!(attackers_to(to, occNoKing) & theirs)) ml.add(make_move(ksq, to));
        }
    }
    if (more_than_one(checkers)) return;  // double check: only king moves

    Bitboard target = checkers ? (BetweenBB[ksq][lsb(checkers)] | checkers) : ~ours;

    Bitboard pinners;
    const Bitboard pinned = slider_blockers(theirs, ksq, pinners) & ours;

    // Knights (a pinned knight can never move)
    Bitboard b = pieces(Us, KNIGHT) & ~pinned;
    while (b) {
        Square from = pop_lsb(b);
        Bitboard att = KnightAttacks[from] & target;
        while (att) ml.add(make_move(from, pop_lsb(att)));
    }
    // Diagonal sliders
    b = pieces(Us, BISHOP, QUEEN);
    while (b) {
        Square from = pop_lsb(b);
        Bitboard att = bishop_attacks(from, occ) & target;
        if (pinned & square_bb(from)) att &= LineBB[ksq][from];
        while (att) ml.add(make_move(from, pop_lsb(att)));
    }
    // Straight sliders
    b = pieces(Us, ROOK, QUEEN);
    while (b) {
        Square from = pop_lsb(b);
        Bitboard att = rook_attacks(from, occ) & target;
        if (pinned & square_bb(from)) att &= LineBB[ksq][from];
        while (att) ml.add(make_move(from, pop_lsb(att)));
    }

    // Pawns
    const Bitboard pawns = pieces(Us, PAWN);
    const Bitboard pawnsOn7 = pawns & Rank7;
    const Bitboard pawnsNotOn7 = pawns & ~Rank7;
    const Bitboard emptySq = ~occ;

    auto add_pawn = [&](Square from, Square to) {
        if (!(pinned & square_bb(from)) || (LineBB[ksq][from] & square_bb(to))) ml.add(make_move(from, to));
    };
    auto add_promo = [&](Square from, Square to) {
        if (!(pinned & square_bb(from)) || (LineBB[ksq][from] & square_bb(to)))
            for (PieceType pt : {QUEEN, KNIGHT, ROOK, BISHOP}) ml.add(make_move(from, to, PROMOTION, pt));
    };

    Bitboard single = shift<Up>(pawnsNotOn7) & emptySq;
    Bitboard dbl = shift<Up>(single & Rank3) & emptySq & target;
    single &= target;
    while (single) { Square to = pop_lsb(single); add_pawn(to - Up, to); }
    while (dbl) { Square to = pop_lsb(dbl); add_pawn(to - Up - Up, to); }

    Bitboard capR = shift<UpRight>(pawnsNotOn7) & theirs & target;
    Bitboard capL = shift<UpLeft>(pawnsNotOn7) & theirs & target;
    while (capR) { Square to = pop_lsb(capR); add_pawn(to - UpRight, to); }
    while (capL) { Square to = pop_lsb(capL); add_pawn(to - UpLeft, to); }

    if (pawnsOn7) {
        Bitboard p1 = shift<Up>(pawnsOn7) & emptySq & target;
        Bitboard pR = shift<UpRight>(pawnsOn7) & theirs & target;
        Bitboard pL = shift<UpLeft>(pawnsOn7) & theirs & target;
        while (p1) { Square to = pop_lsb(p1); add_promo(to - Up, to); }
        while (pR) { Square to = pop_lsb(pR); add_promo(to - UpRight, to); }
        while (pL) { Square to = pop_lsb(pL); add_promo(to - UpLeft, to); }
    }

    // En passant: verified by simulating the capture (covers the horizontal-pin and check cases).
    if (epSquare != SQ_NONE) {
        Bitboard cands = pawns & PawnAttacks[Them][epSquare];
        Square capsq = epSquare - Up;
        while (cands) {
            Square from = pop_lsb(cands);
            Bitboard occ2 = (occ ^ square_bb(from) ^ square_bb(capsq)) | square_bb(epSquare);
            Bitboard theirsAfter = theirs ^ square_bb(capsq);
            if (!(attackers_to(ksq, occ2) & theirsAfter)) ml.add(make_move(from, epSquare, EN_PASSANT));
        }
    }

    // Castling (never while in check)
    if (!checkers) {
        if (Us == WHITE) {
            if ((castling & WHITE_OO) && !(occ & 0x60ULL) && !(attackers_to(5, occ) & theirs) && !(attackers_to(6, occ) & theirs))
                ml.add(make_move(4, 6, CASTLING));
            if ((castling & WHITE_OOO) && !(occ & 0x0EULL) && !(attackers_to(3, occ) & theirs) && !(attackers_to(2, occ) & theirs))
                ml.add(make_move(4, 2, CASTLING));
        } else {
            if ((castling & BLACK_OO) && !(occ & (0x60ULL << 56)) && !(attackers_to(61, occ) & theirs) && !(attackers_to(62, occ) & theirs))
                ml.add(make_move(60, 62, CASTLING));
            if ((castling & BLACK_OOO) && !(occ & (0x0EULL << 56)) && !(attackers_to(59, occ) & theirs) && !(attackers_to(58, occ) & theirs))
                ml.add(make_move(60, 58, CASTLING));
        }
    }
}

void Position::generate_legal(MoveList& ml) const {
    ml.size = 0;
    if (stm == WHITE) generate_legal_impl<WHITE>(ml);
    else generate_legal_impl<BLACK>(ml);
}

void Position::do_move(Move m, DirtyPiece* dp) {
    const Color us = stm, them = ~us;
    const Square from = from_sq(m), to = to_sq(m);
    const Piece pc = board[from];
    const MoveType mt = type_of_move(m);
    Piece captured = mt == EN_PASSANT ? make_piece(them, PAWN) : board[to];

    halfmoveClock++;
    gamePly++;
    if (dp) dp->count = 0;

    // Castling rights
    if (castling) {
        int lost = CastlingMask[from] | CastlingMask[to];
        if (castling & lost) {
            hash ^= Zobrist::castling[castling];
            castling &= ~lost;
            hash ^= Zobrist::castling[castling];
        }
    }
    // Reset en passant square
    if (epSquare != SQ_NONE) {
        hash ^= Zobrist::enpassant[file_of(epSquare)];
        epSquare = SQ_NONE;
    }

    if (mt == CASTLING) {
        Square rfrom = to > from ? to + 1 : to - 2;
        Square rto = to > from ? to - 1 : to + 1;
        Piece rook = board[rfrom];
        remove_piece(from);
        remove_piece(rfrom);
        put_piece(pc, to);
        put_piece(rook, rto);
        if (dp) {
            dp->count = 2;
            dp->piece[0] = pc; dp->from[0] = from; dp->to[0] = to;
            dp->piece[1] = rook; dp->from[1] = rfrom; dp->to[1] = rto;
        }
        captured = NO_PIECE;
    } else {
        if (captured != NO_PIECE) {
            Square capsq = to;
            if (mt == EN_PASSANT) capsq = make_square(file_of(to), rank_of(from));
            remove_piece(capsq);
            halfmoveClock = 0;
            if (dp) {
                dp->piece[dp->count] = captured; dp->from[dp->count] = capsq; dp->to[dp->count] = SQ_NONE;
                dp->count++;
            }
        }
        move_piece(from, to);
        if (type_of(pc) == PAWN) {
            halfmoveClock = 0;
            if ((to ^ from) == 16) {
                Square ep = Square((from + to) / 2);
                if (PawnAttacks[us][ep] & pieces(them, PAWN)) {
                    epSquare = ep;
                    hash ^= Zobrist::enpassant[file_of(ep)];
                }
            }
            if (mt == PROMOTION) {
                Piece promo = make_piece(us, promotion_type(m));
                remove_piece(to);
                put_piece(promo, to);
                if (dp) {
                    dp->piece[dp->count] = pc; dp->from[dp->count] = from; dp->to[dp->count] = SQ_NONE; dp->count++;
                    dp->piece[dp->count] = promo; dp->from[dp->count] = SQ_NONE; dp->to[dp->count] = to; dp->count++;
                }
                goto done;
            }
        }
        if (dp) {
            dp->piece[dp->count] = pc; dp->from[dp->count] = from; dp->to[dp->count] = to; dp->count++;
        }
    }
done:
    stm = them;
    hash ^= Zobrist::side;
    update_checkers();
}

bool Position::is_dead_position() const {
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    Bitboard minors = pieces(KNIGHT) | pieces(BISHOP);
    if (popcount(minors) <= 1) return true;              // K vs K, KN vs K, KB vs K
    if (pieces(KNIGHT)) return false;                     // knights can take part in (help)mates
    Bitboard bishops = pieces(BISHOP);                     // only bishops: dead iff all on one square colour
    return !(bishops & LIGHT_SQUARES) || !(bishops & DARK_SQUARES);
}

bool Position::consistent() const {
    Bitboard all = 0;
    for (int pt = 0; pt < 6; ++pt) all |= byType[pt];
    if (all != (byColor[WHITE] | byColor[BLACK])) return false;
    if (byColor[WHITE] & byColor[BLACK]) return false;
    for (int s = 0; s < 64; ++s) {
        Piece p = board[s];
        if (p == NO_PIECE) { if (all & square_bb(s)) return false; }
        else if (!(byType[type_of(p)] & byColor[color_of(p)] & square_bb(s))) return false;
    }
    Position tmp;
    tmp.set(fen());
    return tmp.hash == hash;
}

Move parse_uci_move(const Position& pos, const std::string& str) {
    if (str.size() < 4) return MOVE_NONE;
    MoveList ml;
    pos.generate_legal(ml);
    for (Move m : ml)
        if (move_to_string(m) == str) return m;
    return MOVE_NONE;
}
