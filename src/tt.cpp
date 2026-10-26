#include "tt.h"
#include "memory.h"
#include <thread>
#include <algorithm>

TranspositionTable::~TranspositionTable() { large_free(table); }

void TranspositionTable::resize(size_t mb) {
    // The old table goes first, so that replacing a large table needs no room for two.
    large_free(table);
    table = nullptr;
    entries = 0;
    const size_t n = std::max<size_t>(mb, 1) * 1024 * 1024 / sizeof(TTEntry);
    table = static_cast<TTEntry*>(large_alloc(n * sizeof(TTEntry)));
    // Memory the system cannot provide - other programs hold it, or the other engine does when a GUI plays two
    // engines against each other with a large Hash each - ends the engine, with a message saying why, rather than
    // leaving it to play with a smaller table than the one asked for.
    if (!table) out_of_memory(n * sizeof(TTEntry), "the hash table");
    entries = n;
    clear();
}

void TranspositionTable::clear() {
    // Clear with several threads: large tables take a while otherwise (and this is when their pages are
    // first touched, which costs more than the zeroing).
    size_t n = entries;
    unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    if (n < (1u << 20)) nt = 1;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([this, t, nt, n] {
            size_t a = n * t / nt, b = n * (t + 1) / nt;
            std::memset(table + a, 0, (b - a) * sizeof(TTEntry));
        });
    for (auto& x : th) x.join();
    generation = 1;
}

static inline int score_to_tt(int v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    if (v >= VALUE_MATE_IN_MAX_PLY) return v + ply;
    if (v <= -VALUE_MATE_IN_MAX_PLY) return v - ply;
    return v;
}
static inline int score_from_tt(int v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    if (v >= VALUE_MATE_IN_MAX_PLY) return v - ply;
    if (v <= -VALUE_MATE_IN_MAX_PLY) return v + ply;
    return v;
}

// The slot is chosen by the key's high bits (index(): key * size >> 64) and the entry is verified by its low 32
// bits, which are independent of the slot.  (The high bits would mostly repeat what chose the slot, and another
// position's entry would pass for this one's far more often; where the scores that matter are a few millipawns
// apart, as against a bare king, such an entry decides the move.)
TTResult TranspositionTable::probe(Key key, int ply) const {
    const TTEntry& e = table[index(key)];
    TTResult r;
    if (e.key32 != uint32_t(key) || e.genBound == 0) return r;
    r.hit = true;   // possibly an evaluation-only entry: score VALUE_NONE, bound BOUND_NONE
    r.score = score_from_tt(e.score, ply);
    r.eval = e.eval;
    r.move = e.move;
    r.depth = e.depth();
    r.bound = e.bound();
    return r;
}

void TranspositionTable::store(Key key, int score, int eval, Move move, int depth, Bound bound, int ply) {
    TTEntry& e = table[index(key)];
    uint32_t k32 = uint32_t(key);
    bool sameKey = e.key32 == k32 && e.genBound != 0;
    // Keep an existing deeper entry of the same position unless it is stale or we have an exact score.
    if (sameKey && bound != BOUND_EXACT && e.generation() == generation && e.depth() > depth + 2) {
        if (move != MOVE_NONE) e.move = uint16_t(move);
        return;
    }
    // An evaluation-only entry is not worth evicting a live search result of another position for.
    if (bound == BOUND_NONE && !sameKey && e.genBound != 0 && e.generation() == generation && e.bound() != BOUND_NONE) return;
    if (move != MOVE_NONE || !sameKey) e.move = uint16_t(move);
    e.key32 = k32;
    e.score = int16_t(score_to_tt(score, ply));
    e.eval = int16_t(eval);
    e.depth8 = uint8_t(std::max(-8, std::min(depth, 247)) + 8);
    e.genBound = uint8_t((generation << 2) | bound);
}

int TranspositionTable::hashfull() const {
    int cnt = 0;
    size_t n = std::min<size_t>(1000, entries);
    for (size_t i = 0; i < n; ++i)
        if (table[i].genBound != 0 && table[i].generation() == generation) cnt++;
    return int(cnt * 1000 / n);
}
