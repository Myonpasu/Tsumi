The engine builds on x86-64 with g++ or clang (C++17):

```bash
make -j                 # ./tsumi, optimised for this machine (add PEXT=1 on Intel, or AMD Zen 3 and later)
make -j universal       # ./tsumi-universal: runs on any x86-64 CPU with SSE4.2 and POPCNT
make -j windows         # tsumi.exe, the universal build for Windows: cross-compiled from Linux with mingw-w64
make -j windows-avx512  # tsumi-avx512.exe, a dedicated AVX-512 Windows build
make debug              # tsumi-debug, with sanitizers
make featlib            # train/libfeat.so, the trainer's feature extraction (the engine's own code)
```

On Windows itself, build in an MSYS2 MINGW64 shell with `make` or `make universal`. MSVC is not supported
(the code uses GCC built-ins).

## CPU support

Only the network code benefits measurably from newer instruction sets, so the universal build compiles just
that code four times, for SSE4.2, AVX2, AVX-512 and AVX-512 VNNI, and picks the best set the CPU supports at
startup (the banner says which). All four give identical results. `TSUMI_ISA=avx2` (or `sse4.2`, `avx512`)
forces a lower set for testing. A build for a specific CPU (`make`, `make windows-avx512`) prints what is
missing and exits on a CPU that lacks its instructions.

## The network

The network is Tsumi's only evaluation, and it is not part of the repository:

```bash
make net                # downloads the network this version is built for, and checks its hash
make checkpoint         # ... and its training checkpoint (.pt), to train from
```

Networks are named after their contents, `tsumi-<the first 12 hex digits of the file's SHA-256>.nnue`, and
every published one is an asset of the
[networks release](https://github.com/Myonpasu/Tsumi/releases/tag/networks), with its training checkpoint
beside it. The engine's own network is `NNUE::DEFAULT_NET` in `src/nnue.h`; the release archives include it.
Tsumi loads it from its working directory or from the executable's directory, and the `EvalFile` option
names another file. Without a network the engine still answers `uci`, `perft` and the like, but exits when
asked to search.

## Checking the build

```bash
./tsumi bench           # searches a set of positions: nodes and speed
./tsumi test            # perft suite
tests/run_all.sh        # the test battery (needs python-chess: pip install chess)
tests/universal.sh      # the universal build's kernel sets, and emulated CPUs with qemu-user
```

[Tools and tests](tools-and-tests.md) describes the tests.
