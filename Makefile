# Tsumi build.
#   make                 optimised native binary for this machine (-march=native; add PEXT=1 on Intel / Zen 3+)
#   make universal       tsumi-universal: runs on any x86-64 CPU with SSE4.2 and POPCNT, and selects the
#                        AVX2, AVX-512 or AVX-512 VNNI NNUE kernels at startup when the CPU has them
#   make windows         tsumi.exe: the universal build as a statically linked Windows executable
#   make windows-avx512  tsumi-avx512.exe: a Windows build for AVX-512 CPUs, with PEXT (about as fast as
#                        the universal build on the same CPU)
#   make debug           tsumi-debug with sanitizers
#   make featlib         train/libfeat.so: the trainer's feature extraction (the engine's own code)
#   make net             downloads the network this version is built for (src/nnue.h); make checkpoint its .pt
CXX      ?= g++
EXE      ?= tsumi
SRC      := $(wildcard src/*.cpp)
BUILDDIR ?= build
OBJ      := $(SRC:src/%.cpp=$(BUILDDIR)/%.o)
DEP      := $(OBJ:.o=.d)

CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -DNDEBUG -flto=auto -fno-exceptions -pthread
LDFLAGS  ?= -pthread -flto=auto

ARCHFLAGS ?= -march=native
# Use PEXT bit-extraction for sliding attacks when the CPU supports BMI2 (fast on Intel and Zen3+).
ifeq ($(PEXT),1)
  ARCHFLAGS += -DUSE_PEXT -mbmi2
endif
# main.cpp is always compiled for plain x86-64: it checks the CPU against the build's requirements
# before any other engine code runs (see the comment at the top of src/main.cpp).
BASEARCH ?= -march=x86-64
# Baseline of the universal builds: SSE4.2 and POPCNT (Intel CPUs since 2008, AMD since 2011).  Newer
# instructions are selected at runtime.
UNIVERSAL := -msse4.2 -mpopcnt

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(OBJ) -o $@ $(LDFLAGS)

$(BUILDDIR)/main.o: src/main.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(BASEARCH) -MMD -MP -c $< -o $@

$(BUILDDIR)/%.o: src/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -MMD -MP -c $< -o $@

universal:
	$(MAKE) ARCHFLAGS="$(UNIVERSAL)" EXE=tsumi-universal BUILDDIR=build-universal

# Cross-compile Windows executables from Linux (sudo apt install mingw-w64).  Statically linked, so no
# DLLs are needed next to the .exe.
MINGW ?= x86_64-w64-mingw32-g++-posix
WINLD := -static -pthread -flto=auto
windows:
	$(MAKE) CXX=$(MINGW) EXE=tsumi.exe ARCHFLAGS="$(UNIVERSAL)" LDFLAGS="$(WINLD)" BUILDDIR=build-win
windows-avx512:
	$(MAKE) CXX=$(MINGW) EXE=tsumi-avx512.exe ARCHFLAGS="-mavx2 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mbmi -mbmi2 -mpopcnt -mfma -mtune=znver4 -DUSE_PEXT" LDFLAGS="$(WINLD)" BUILDDIR=build-win-avx512

debug:
	$(MAKE) CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -pthread -fsanitize=address,undefined" LDFLAGS="-pthread -fsanitize=address,undefined" EXE=tsumi-debug BUILDDIR=build-debug

# Feature-extraction library for the trainer (train/features.py loads it with ctypes): the engine's own
# feature code and move generator, so the trainer and the engine never disagree about an input.
featlib: train/libfeat.so
train/libfeat.so: train/featlib.cpp src/nnue_features.h src/bitboard.cpp src/bitboard.h src/position.cpp src/position.h src/types.h
	$(CXX) -std=c++17 -O2 -DNDEBUG -shared -fPIC -Wall train/featlib.cpp src/bitboard.cpp src/position.cpp -o $@

# The network this version is built for (NNUE::DEFAULT_NET in src/nnue.h), from the "networks" release of the
# repository, where every published network is an asset.  Its name is the first 12 hex digits of its SHA-256,
# which the download is checked against; the checkpoint takes the network's name.  NET_FETCH downloads $@ to
# $@.tmp (the workflow uses the GitHub CLI instead, which works while the repository is private).
REPO      ?= Myonpasu/Tsumi
NET_URL   ?= https://github.com/$(REPO)/releases/download/networks
NET_FETCH ?= curl -fL --progress-bar -o $@.tmp $(NET_URL)/$@
NET       := $(shell sed -n 's/.*DEFAULT_NET\[\] = "\([^"]*\)".*/\1/p' src/nnue.h)
net: $(NET)
checkpoint: $(NET).pt
$(NET) $(NET).pt:
	$(NET_FETCH)
	@if [ "$@" = "$(NET)" ] && [ "$$(sha256sum $@.tmp | cut -c1-12)" != "$(patsubst tsumi-%.nnue,%,$(NET))" ]; then \
	    echo "$@: the download's SHA-256 does not start with the name's digits"; rm -f $@.tmp; exit 1; fi
	mv $@.tmp $@

# (tests/universal.sh makes build-avx2 and tsumi-avx2.)
clean:
	rm -rf build build-universal build-debug build-win build-win-avx512 build-avx2 tsumi tsumi-universal tsumi-debug tsumi.exe tsumi-avx512.exe tsumi-avx2 train/libfeat.so

-include $(DEP)
.PHONY: all universal debug windows windows-avx512 featlib net checkpoint clean
