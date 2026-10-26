#pragma once
#include "types.h"
#include <cstring>

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

// An entry holds a search result (score, bound, depth, move) and/or the static evaluation of the position.
// Entries written only to cache the evaluation have bound BOUND_NONE and score VALUE_NONE.
struct TTEntry {
    uint32_t key32;     // the key's low 32 bits: the slot is chosen by its high bits (index), so these are independent of it
    int16_t score;
    int16_t eval;
    uint16_t move;
    uint8_t depth8;     // depth + 8, so that depths below 1 fit: -8 for evaluation-only entries, -1 and 0 for quiescence
    uint8_t genBound;   // generation (6 bits, 1..63) | bound (2 bits); 0 = empty slot
    uint32_t pad;

    Bound bound() const { return Bound(genBound & 3); }
    int depth() const { return int(depth8) - 8; }
    uint8_t generation() const { return genBound >> 2; }
};
static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

struct TTResult {
    bool hit = false;
    int score = VALUE_NONE;
    int eval = VALUE_NONE;
    Move move = MOVE_NONE;
    int depth = -8;
    Bound bound = BOUND_NONE;
};

class TranspositionTable {
public:
    TranspositionTable() = default;
    ~TranspositionTable();
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    // Allocates and clears a table of mb megabytes.  If the system cannot provide them, says so and exits.
    void resize(size_t mb);
    void clear();
    void new_search() { generation = uint8_t(generation % 63 + 1); }   // 1..63: an entry's genBound is never 0
    TTResult probe(Key key, int ply) const;
    void store(Key key, int score, int eval, Move move, int depth, Bound bound, int ply);
    int hashfull() const;
    void prefetch(Key key) const { __builtin_prefetch(&table[index(key)]); }

private:
    size_t index(Key key) const { return size_t((unsigned __int128)key * entries >> 64); }
    TTEntry* table = nullptr;   // large pages where the system allows (memory.h): a probe costs no page walk
    size_t entries = 0;
    uint8_t generation = 1;
};
