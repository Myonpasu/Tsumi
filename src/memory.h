// Storage for the big tables that are read at random all over: the network's weight tables (tens of
// megabytes, a row at a time) and the hash table (up to gigabytes, an entry at a time).  With 4 KB pages a
// 1 GB hash table spans 262,144 pages and the network's tables thousands, far more than a TLB holds, so every
// probe and most rows would also pay a page-table walk.  large_alloc returns memory aligned to 2 MB and
// backed by large pages where the system allows: on Linux through transparent huge pages (madvise, which
// needs no privilege), on Windows through VirtualAlloc with MEM_LARGE_PAGES, which works once the user
// has the "Lock pages in memory" right (as for Stockfish), and otherwise falls back to ordinary pages.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <vector>

void* large_alloc(size_t bytes);   // nullptr when the system cannot provide the memory
void large_free(void* p);
bool large_pages_used();   // did the last large_alloc get large pages (Windows; on Linux: was the request made)
// Says on standard output that `bytes` (for `what`, if given) could not be allocated - a GUI logs it - and exits.
[[noreturn]] void out_of_memory(size_t bytes, const char* what = nullptr);

template <class T>
struct LargePageAllocator {
    using value_type = T;
    LargePageAllocator() = default;
    template <class U> LargePageAllocator(const LargePageAllocator<U>&) {}
    T* allocate(size_t n) {
        void* p = large_alloc(n * sizeof(T));
        if (!p) out_of_memory(n * sizeof(T));   // the engine is built without exceptions
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { large_free(p); }
    template <class U> bool operator==(const LargePageAllocator<U>&) const { return true; }
    template <class U> bool operator!=(const LargePageAllocator<U>&) const { return false; }
};
template <class T> using LargeVector = std::vector<T, LargePageAllocator<T>>;
