#include "cpu.h"

namespace CPU {

// This file is compiled with the build's architecture flags, so the compiler's own macros say which
// instruction sets may appear anywhere in the engine.  main.cpp (compiled for plain x86-64) checks them
// against the CPU before anything else runs.
const unsigned BuildFeatures =
#if defined(__SSE4_2__)
    F_SSE42 |
#endif
#if defined(__POPCNT__)
    F_POPCNT |
#endif
#if defined(__AVX2__)
    F_AVX2 |
#endif
#if defined(__BMI2__)
    F_BMI2 |
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    F_AVX512 |
#endif
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    F_AVX512VNNI |
#endif
    0;

namespace {
Level g_level = BASE;
bool g_forced = false;
}

void set_level(Level level, bool forced) { g_level = level; g_forced = forced; }
Level level() { return g_level; }

const char* level_name(Level level) {
    switch (level) {
    case AVX512VNNI: return "AVX-512 VNNI";
    case AVX512: return "AVX-512";
    case AVX2:   return "AVX2";
    default:
#if defined(__x86_64__) || defined(__i386__)
        return "SSE4.2";
#else
        return "portable";
#endif
    }
}

std::string description() {
    std::string s = std::string(level_name(g_level)) + " NNUE kernels";
    if (g_forced) s += " (forced by TSUMI_ISA)";
    return s;
}

}  // namespace CPU
