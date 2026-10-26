// Tsumi: an engine for selfmate chess (ordinary chess rules, but the player who gets checkmated wins).
//
// This file is compiled for plain x86-64 (see BASEARCH in the Makefile) while the rest of the engine uses
// the build's architecture flags, so that the CPU check below runs on any processor before a single newer
// instruction executes - the same arrangement as the entry stub of Stockfish's universal binaries.  For
// that reason it must not include engine headers: inline engine code instantiated here would be compiled
// without the build's instruction set.
#include "cpu.h"
#include "uci.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#if defined(__x86_64__) || defined(__i386__)
unsigned cpu_features() {
    __builtin_cpu_init();
    unsigned f = 0;
    if (__builtin_cpu_supports("sse4.2")) f |= CPU::F_SSE42;
    if (__builtin_cpu_supports("popcnt")) f |= CPU::F_POPCNT;
    if (__builtin_cpu_supports("avx2")) f |= CPU::F_AVX2;
    if (__builtin_cpu_supports("bmi2")) f |= CPU::F_BMI2;
    // libgcc only reports AVX/AVX-512 when the operating system has enabled their register state as well.
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) f |= CPU::F_AVX512;
    if ((f & CPU::F_AVX512) && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) f |= CPU::F_AVX512VNNI;
    return f;
}
#else
unsigned cpu_features() { return 0; }
#endif

const char* feature_name(unsigned f) {
    if (f & CPU::F_AVX512VNNI) return "AVX-512 VNNI";
    if (f & CPU::F_AVX512) return "AVX-512";
    if (f & CPU::F_AVX2) return "AVX2";
    if (f & CPU::F_BMI2) return "BMI2";
    if (f & CPU::F_POPCNT) return "POPCNT";
    return "SSE4.2";
}

// The NNUE kernel level: the best the CPU supports, unless TSUMI_ISA=sse4.2|avx2|avx512|avx512vnni asks for a lower
// one (a testing aid; it cannot enable instructions the CPU lacks).
CPU::Level choose_level(unsigned have) {
    CPU::Level best = (have & CPU::F_AVX512VNNI) ? CPU::AVX512VNNI : (have & CPU::F_AVX512) ? CPU::AVX512 : (have & CPU::F_AVX2) ? CPU::AVX2 : CPU::BASE;
    const char* want = std::getenv("TSUMI_ISA");
    if (!want || !*want || std::strcmp(want, "auto") == 0) {
        CPU::set_level(best, false);
        return best;
    }
    CPU::Level asked = std::strcmp(want, "avx512vnni") == 0 ? CPU::AVX512VNNI : std::strcmp(want, "avx512") == 0 ? CPU::AVX512
                     : std::strcmp(want, "avx2") == 0 ? CPU::AVX2 : CPU::BASE;
    if (asked > best) {
        std::fprintf(stderr, "TSUMI_ISA=%s asks for instructions this CPU lacks; using %s\n", want, CPU::level_name(best));
        CPU::set_level(best, false);
        return best;
    }
    CPU::set_level(asked, true);
    return asked;
}

}  // namespace

int main(int argc, char** argv) {
    unsigned have = cpu_features();
    if (unsigned missing = CPU::BuildFeatures & ~have) {
        std::fprintf(stderr, "This Tsumi build requires %s, which this CPU does not support.\n", feature_name(missing));
        if (!(missing & (CPU::F_SSE42 | CPU::F_POPCNT)))   // all that the universal build needs
            std::fprintf(stderr, "Use the universal build instead (make universal / make windows): it selects the best\n"
                                 "instructions this CPU has at startup.\n");
        return 1;
    }
    choose_level(have);
    init_engine();

    if (argc >= 2 && std::strcmp(argv[1], "datagen") == 0) return datagen_main(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "rescore") == 0) return rescore_main(argc, argv);
    if (argc >= 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::printf("Tsumi - selfmate chess engine\n\n"
                    "  tsumi                 start the UCI loop (scores are from the side to move's point of view, and\n"
                    "                         positive means winning: heading for being checkmated)\n"
                    "  tsumi bench [depth]   run the benchmark\n"
                    "  tsumi test            run the perft test suite\n"
                    "  tsumi datagen ...     generate self-play training data (tsumi datagen --help)\n"
                    "  tsumi rescore ...     search the positions of existing training data again with the current net (tsumi rescore --help)\n"
                    "  tsumi \"<uci cmd>\" ... execute UCI commands then exit\n\n"
                    "The NNUE kernels run with the best instruction set the CPU supports (SSE4.2, AVX2, AVX-512 or AVX-512\n"
                    "with VNNI); set TSUMI_ISA=sse4.2|avx2|avx512 to force a lower one.\n");
        return 0;
    }
    return uci_loop(argc, argv);
}
