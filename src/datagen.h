#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Training record: 32 bytes, little-endian.
//   occ      occupancy bitboard (a1 = bit 0)
//   pcs      the pieces on the occupied squares in ascending square order, 4 bits each (colour * 6 + type, with
//            pawn, knight, bishop, rook, queen, king = 0..5), the first in the low half of pcs[0]
//   score    search score in internal units from the side to move's point of view (positive: the side to move is
//            winning, that is, heading for being mated)
//   result   game result from White's point of view: +1 White won (was mated), 0 draw, -1 Black won
//   stm      side to move: 0 White, 1 Black
//   ply      game ply (at most 255)
//   flags    bit 1 (value 2): the result is not a label for this position - a random move later in the game turned
//            out to cost more than the random-move window allowed (see datagen.cpp), so the game no longer says
//            what this position was worth; the trainer uses the score alone
//   next     unused (0)
struct TrainingRecord {
    uint64_t occ;
    uint8_t pcs[16];
    int16_t score;
    int8_t result;
    uint8_t stm;
    uint8_t ply;
    uint8_t flags;
    int16_t next;
};
static_assert(sizeof(TrainingRecord) == 32, "TrainingRecord must be 32 bytes");

// Self-play games for training data, after Stockfish's generator.  A game starts from a random-play position (a few
// uniformly random plies from the start position, kept when a search scores it within a window: one side has the
// upper hand, which makes the game more often decisive, but the position is not decided) or else from the next
// position of --book (in a random order, each position once) or the start position.  The engine plays at a fixed node
// budget; variety also comes from random moves on a share of the opening plies, each one of the best few moves of a
// MultiPV search (Stockfish's random_multi_pv).
// Since many games open alike, the first plies are not recorded (write_min_ply), nor is a position already
// recorded in this run (was_seen_before).
struct DatagenOptions {
    int threads = 1;
    uint64_t games = 1000;
    uint64_t nodes = 10000;      // search nodes per move: a search finishes the iteration in which it reaches 80-120% of
                                 // this (varied per search) and is cut off at NODES_HARD_FACTOR times that
    int maxPlies = 400;          // a game still going after this many plies is a draw
    uint64_t seed = 1;
    std::string out = "data.bin";
    std::string book;            // file of starting positions, a FEN or EPD line each, for the games that do not start
                                 // from a random-play position (default: the start position)
    int randomOpeningPct = 50;   // this percentage of the games start from a random-play position ...
    int randomPliesMin = 8;      // ... of randomPliesMin to randomPliesMax random legal plies from the start position ...
    int randomPliesMax = 12;
    int openingMinScore = 500;   // ... kept when a search at `nodes` scores it, for either side, within these bounds
    int openingMaxScore = 2000;  //     (internal units; openingMaxScore 0: no check)
    int adjDrawMinPly = 120;     // a game is adjudicated a draw from this ply on (from any ply with a bare king on the
    int adjDrawScore = 10;       // board) once adjDrawPlies searches in a row have scored within +-adjDrawScore
    int adjDrawPlies = 16;
    int hashMB = 16;             // hash table per thread
    int randomMovePct = 40;      // a random move on this percentage of the plies in [randomMoveMinPly, randomMoveMaxPly)
    int randomMoveMinPly = 0;    // (8 per game on average with the defaults), never when the search has found a mate
    int randomMoveMaxPly = 20;
    int randomMultiPV = 4;       // a random move is one of the top randomMultiPV moves of a MultiPV search ...
    int randomMultiPVDiff = 250; // ... that score within this many internal units of the best (0.25 pawns)
    int writeMinPly = 5;         // positions before this ply of the game are not recorded
    int dedupMB = 128;           // the table of recorded positions (16M keys in 128 MB; 0 = off)
    std::string log;             // optional: a line "fen|moves|result" per game
    bool resume = false;         // continue an interrupted run (needs --log): the games in the log count towards --games,
                                 // the positions already in --out go back into the dedup table, and the seed moves on,
                                 // so the new games are not the old ones again
};

// The node budget of a datagen or rescore search is soft: the search completes the iteration in which it reaches it,
// so that its score comes from a finished depth, and is cut off at this multiple of it.
constexpr uint64_t NODES_HARD_FACTOR = 5;

bool run_datagen(const DatagenOptions& opt);   // false, having said why, if it could not run

// `tsumi rescore`: the positions of existing training records searched again with the current network, written
// with the new scores - what Stockfish's training does to its data when the network changes: the positions are
// kept, and the labels come from the latest, strongest search.
struct RescoreOptions {
    std::vector<std::string> in;   // record files, read in the order given
    std::string out = "rescored.bin";
    uint64_t nodes = 1;            // search nodes per position: 1 stops after the first iteration (depth 1 with
                                   // quiescence), the network's own evaluation with the tactics resolved
    int threads = 1;
    int hashMB = 16;               // hash table per thread
};

bool run_rescore(const RescoreOptions& opt);   // false, having said why, if it could not run
