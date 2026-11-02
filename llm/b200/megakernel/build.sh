#!/usr/bin/env bash
# build.sh : build the SM-count-parameterised open-flapping-airplanes megakernel (training + decode + Doom, one launch).  Run it ON the GPU box (or any box with nvcc).
#   ./build.sh <arch> [out=ofa_b200_<arch>] [nvcc flags...]
#   arch: sm_80 (A100, tested) | sm_90a | sm_100a (B200, tested) | sm_103a (B300)
#   env : MODEL=S|F|M (default S)     DOOM_OBJ=<path to a doom_dev object for THIS arch>  (default: ~/doom-gpu/doom_dev.o for sm_80, ./doom_dev_<arch>.o otherwise;
#         NODOOM=1 builds without Doom (stubs the 5 extern symbols so the link works)
#   knobs (pass as nvcc -D flags): -DOFA_GEMM_BACKEND=1 (tcgen05, sm_100a only; -DGEMM100_STAGES=2|3)  -DGEMM_STAGES=6 (mma.sync backend, 96 KB/block)
#                                  -DOFA_BLOCKS_PER_SM=1 -DGEMM100_STAGES=3 -DGEMM100_LAYOUT_MASK=2 -DCLAIM_FETCHADD=0
# Prints the ptxas line (registers, stack, spills) for ofa_kernel.
set -euo pipefail
ARCH=${1:?usage: build.sh sm_80|sm_100a [out] [flags]}; shift
OUT=${1:-ofa_b200_${ARCH}}; [ $# -gt 0 ] && shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
case "${MODEL:-S}" in S) MF="";; F) MF="-DMODEL_F";; M) MF="-DMODEL_M";; *) echo "MODEL must be S|F|M"; exit 2;; esac
D=${DOOM_OBJ:-}
if [ -z "$D" ]; then if [ "$ARCH" = sm_80 ]; then D=$HOME/doom-gpu/doom_dev.o; else D=$HERE/doom_dev_${ARCH}.o; fi; fi
SRC=ofa.cu; EXTRA_OBJ="$D"
if [ "${NODOOM:-0}" = 1 ]; then
  cat > doom_stub.cu <<'STUB'
extern "C" {
__device__ void gpu_doom_setup(const unsigned char*, long, volatile char*, long) {}
__device__ void gpu_doom_create(unsigned long long) {}
__device__ void gpu_doom_frame(unsigned long long) {}
__device__ void gpu_doom_worker_always(int) {}
__device__ void gpu_doom_enable_par(int) {}
__device__ unsigned char* I_VideoBuffer; __device__ unsigned int colors[256];
}
STUB
  EXTRA_OBJ=doom_stub.cu
elif [ ! -f "$D" ]; then echo "missing Doom object $D (run ./build_doom_b200.sh, or NODOOM=1)"; exit 1; fi
echo "== nvcc $(nvcc --version | tail -1)  arch=$ARCH model=${MODEL:-S}  doom=$EXTRA_OBJ  flags: $MF $*"
nvcc -arch=$ARCH -O3 -std=c++17 --extended-lambda -rdc=true -maxrregcount=128 -Xptxas -v $MF "$@" $SRC $EXTRA_OBJ -o "$OUT" -lpthread > "$OUT.buildlog" 2>&1 || { grep -E "error" "$OUT.buildlog" | head; exit 1; }
# ptxas stats of the persistent kernel only (registers / stack / spills):
awk '/Compiling entry function .*ofa_kernel/ {p=1; n=0} p {print "  " $0; if (++n >= 4) p=0}' "$OUT.buildlog" | grep -E "Compiling|stack frame|Used" | sed 's/ptxas info    : //'
grep -c "warning" "$OUT.buildlog" | sed 's/^/  nvcc warnings: /'
ls -la "$OUT" | awk '{print "built", $5, $9}'
