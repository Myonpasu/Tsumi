#include "perft.h"
#include <chrono>
#include <cstdio>

uint64_t perft(const Position& pos, int depth) {
    MoveList ml;
    pos.generate_legal(ml);
    if (depth == 1) return ml.size;
    uint64_t nodes = 0;
    for (Move m : ml) {
        Position next = pos;
        next.do_move(m);
        nodes += perft(next, depth - 1);
    }
    return nodes;
}

uint64_t perft_divide(const Position& pos, int depth) {
    MoveList ml;
    pos.generate_legal(ml);
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (Move m : ml) {
        Position next = pos;
        next.do_move(m);
        uint64_t n = depth > 1 ? perft(next, depth - 1) : 1;
        total += n;
        std::printf("%s: %llu\n", move_to_string(m).c_str(), (unsigned long long)n);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nNodes searched: %llu\nTime: %.3fs (%.1f Mnps)\n", (unsigned long long)total, secs, total / secs / 1e6);
    return total;
}

struct PerftCase { const char* fen; int depth; uint64_t nodes; };

bool run_perft_tests() {
    static const PerftCase cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292ULL},
        {"r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 5, 15833292ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487ULL},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594ULL},
    };
    bool ok = true;
    for (const PerftCase& c : cases) {
        Position pos;
        if (!pos.set(c.fen)) { std::printf("FEN parse failed: %s\n", c.fen); ok = false; continue; }
        auto t0 = std::chrono::steady_clock::now();
        uint64_t n = perft(pos, c.depth);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool pass = n == c.nodes;
        ok &= pass;
        std::printf("%-6s depth %d: %12llu (expected %12llu) %6.2fs %6.1f Mnps  %s\n", pass ? "ok" : "FAIL", c.depth,
                    (unsigned long long)n, (unsigned long long)c.nodes, secs, n / secs / 1e6, c.fen);
    }
    return ok;
}
