#include "memory.h"
#include <algorithm>
#include <cstdio>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>   // madvise
#endif

namespace {
bool g_largePagesUsed = false;
}

bool large_pages_used() { return g_largePagesUsed; }

void out_of_memory(size_t bytes, const char* what) {
    std::printf("info string out of memory: could not allocate %zu MB%s%s\n", (bytes + (1 << 20) - 1) >> 20,
                what ? " for " : "", what ? what : "");
    std::fflush(stdout);
    std::_Exit(EXIT_FAILURE);   // not abort(): on Windows that can also raise a crash report
}

#if defined(_WIN32)

namespace {
// Large pages need the SeLockMemoryPrivilege ("Lock pages in memory" in the Local Security Policy, which
// takes effect at the next log-on).  The privilege is held by the account but enabled per process.
bool enable_lock_memory_privilege() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)
           && AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr)
           && GetLastError() == ERROR_SUCCESS;   // AdjustTokenPrivileges succeeds even when nothing was granted
    CloseHandle(token);
    return ok;
}
}

void* large_alloc(size_t bytes) {
    static const bool privileged = enable_lock_memory_privilege();
    static const size_t largePage = privileged ? GetLargePageMinimum() : 0;
    bytes = std::max<size_t>(bytes, 1);
    if (largePage) {
        const size_t size = (bytes + largePage - 1) / largePage * largePage;
        void* p = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p) { g_largePagesUsed = true; return p; }
    }
    g_largePagesUsed = false;
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void large_free(void* p) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

#else

void* large_alloc(size_t bytes) {
    constexpr size_t ALIGN = size_t(2) << 20;
    const size_t size = (std::max<size_t>(bytes, 1) + ALIGN - 1) / ALIGN * ALIGN;
    void* p = std::aligned_alloc(ALIGN, size);
    g_largePagesUsed = false;
#if defined(MADV_HUGEPAGE)
    if (p) g_largePagesUsed = madvise(p, size, MADV_HUGEPAGE) == 0;
#endif
    return p;
}

void large_free(void* p) { std::free(p); }

#endif
