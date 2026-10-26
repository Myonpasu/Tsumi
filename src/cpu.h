// Runtime CPU dispatch.  One Tsumi binary contains the NNUE kernels compiled for four instruction-set levels
// (SSE4.2, AVX2, AVX-512 and AVX-512 with VNNI), and the best one the running CPU supports is chosen at startup
// (main.cpp), the way Stockfish's universal binaries pick a build.  Nearly all of the gain from newer
// instructions is in the NNUE kernels, so the rest of the engine is compiled once, for the build's baseline.
#pragma once
#include <string>

namespace CPU {

enum Level { BASE = 0, AVX2 = 1, AVX512 = 2, AVX512VNNI = 3 };

// Instruction-set features as a bitmask.
enum Feature : unsigned { F_SSE42 = 1, F_POPCNT = 2, F_AVX2 = 4, F_BMI2 = 8, F_AVX512 = 16, F_AVX512VNNI = 32 };

// Features the compiler was allowed to use throughout this binary (its -m flags).  The running CPU must
// have all of them: a universal build needs only SSE4.2 and POPCNT (Intel CPUs since 2008, AMD since 2011),
// a dedicated -mavx2 or -march=native build needs more.  Defined in cpu.cpp from the compiler's macros.
extern const unsigned BuildFeatures;

void set_level(Level level, bool forced);
Level level();
const char* level_name(Level level);   // "SSE4.2", "AVX2", "AVX-512", "AVX-512 VNNI"
std::string description();             // for the start-up banner, e.g. "AVX-512 NNUE kernels"

}  // namespace CPU
