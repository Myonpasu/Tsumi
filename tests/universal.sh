#!/usr/bin/env bash
# Checks the runtime CPU dispatch of the universal build (see src/main.cpp, src/cpu.h, src/nnue.cpp):
#   * every NNUE kernel level (SSE4.2 / AVX2 / AVX-512 / AVX-512 VNNI) gives bit-identical results;
#   * with qemu-user installed (sudo apt install qemu-user), the universal binary runs on emulated CPUs
#     without AVX (Nehalem) and without AVX-512 (Haswell) and selects the right kernels, and a dedicated
#     AVX2 build refuses cleanly on Nehalem instead of crashing - which also proves that nothing runs
#     before main()'s CPU check (a global object with a dynamic initialiser would).
# Run from the repository root; builds tsumi-universal and a dedicated AVX2 build (build-avx2/).
set -euo pipefail
cd "$(dirname "$0")/.."
net=${NET:-$( { find nets -maxdepth 1 -name '*.nnue' 2>/dev/null | sort; find . -maxdepth 1 -name 'tsumi-*.nnue' | sort; } | head -n 1 || true)}
[[ -n "$net" ]] || { echo "no network found (NET=file, a .nnue in nets/, or make net): the searches compared need one" >&2; exit 1; }
bench=("setoption name EvalFile value $net" "bench 6")

make -s universal >/dev/null
make -s ARCHFLAGS="-mavx2 -mbmi -mbmi2 -mpopcnt" EXE=tsumi-avx2 BUILDDIR=build-avx2 >/dev/null

nodes() { "$@" "${bench[@]}" 2>/dev/null | awk '/Nodes searched/ {print $NF}'; }
level() { "$@" "${bench[@]}" 2>/dev/null | sed -n 's/.*(\(.*\) NNUE kernels.*/\1/p' | head -1; }

# The kernel levels in order, as the engine names them and as TSUMI_ISA asks for them.
names=("SSE4.2" "AVX2" "AVX-512" "AVX-512 VNNI"); isas=(sse4.2 avx2 avx512 avx512vnni)
rank() { local i; for i in "${!names[@]}"; do [[ "${names[$i]}" == "$1" ]] && { echo "$i"; return; }; done; echo -1; }

ref=$(nodes ./tsumi-universal)
host=$(level ./tsumi-universal)
printf "%-18s %s kernels, %s nodes\n" "host:" "$host" "$ref"
[[ $(rank "$host") -ge 0 ]] || { echo "FAIL: unknown kernel level '$host'"; exit 1; }
for ((i = $(rank "$host") - 1; i >= 0; i--)); do   # every level below the host's, which TSUMI_ISA must select
    isa=${isas[$i]}
    n=$(TSUMI_ISA=$isa nodes ./tsumi-universal)
    got=$(TSUMI_ISA=$isa level ./tsumi-universal)
    printf "%-18s %s kernels, %s nodes\n" "TSUMI_ISA=$isa:" "$got" "$n"
    [[ "$got" == "${names[$i]}" ]] || { echo "FAIL: TSUMI_ISA=$isa selected $got kernels"; exit 1; }
    [[ "$n" == "$ref" ]] || { echo "FAIL: node count differs"; exit 1; }
done

if command -v qemu-x86_64 >/dev/null; then
    # qemu emulates AVX only from version 7.2: an older one presents Haswell without it, so SSE4.2 is right there.
    qemu=$(qemu-x86_64 --version | sed -n 's/.*version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
    haswell=AVX2
    if [[ -n "$qemu" ]] && (( ${qemu% *} < 7 || (${qemu% *} == 7 && ${qemu#* } < 2) )); then
        haswell=SSE4.2; echo "(qemu ${qemu% *}.${qemu#* } has no AVX: Haswell is emulated without it)"
    fi
    for cpu in Nehalem:SSE4.2 Haswell:$haswell; do
        n=$(nodes qemu-x86_64 -cpu "${cpu%%:*}" ./tsumi-universal)
        got=$(level qemu-x86_64 -cpu "${cpu%%:*}" ./tsumi-universal)
        printf "%-18s %s kernels, %s nodes\n" "qemu ${cpu%%:*}:" "$got" "$n"
        [[ "$got" == "${cpu#*:}" ]] || { echo "FAIL: expected ${cpu#*:} kernels"; exit 1; }
        [[ "$n" == "$ref" ]] || { echo "FAIL: node count differs"; exit 1; }
    done
    msg=$(qemu-x86_64 -cpu Nehalem ./tsumi-avx2 bench 2 2>&1 | head -1 || true)
    echo "qemu Nehalem, dedicated AVX2 build: $msg"
    [[ "$msg" == *"requires AVX2"* ]] || { echo "FAIL: expected a clean refusal"; exit 1; }
else
    echo "(qemu-x86_64 not installed: emulated-CPU checks skipped)"
fi
echo "universal build OK"
