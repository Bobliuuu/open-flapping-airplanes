#!/usr/bin/env bash
# build_doom_b200.sh : relink the Doom engine for another arch from the PTX that doom-gpu/build_doom.sh already produced (clang NVPTX, .target sm_80, ptx75).
# PTX is forward compatible, so nvcc can assemble it for sm_100a (verified to assemble with nvcc 12.8 on the A100 box; NOT run on B200).
#   ./build_doom_b200.sh [arch=sm_100a] [doom-gpu dir=~/doom-gpu]     -> ./doom_dev_<arch>.o
# If ~/doom-gpu/doom.ptx does not exist yet, run doom-gpu/setup_box.sh and build_doom.sh first (needs clang/llc with the NVPTX backend).
set -e
ARCH=${1:-sm_100a}; DG=${2:-$HOME/doom-gpu}; HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DG/doom.ptx" ] || { echo "no $DG/doom.ptx : run $DG/build_doom.sh first"; exit 1; }
nvcc -arch=$ARCH -dc -maxrregcount=112 "$DG/doom.ptx" -o "$HERE/doom_dev_${ARCH}.o" 2>&1 | grep -v 'Potential out of bound' || true
ls -la "$HERE/doom_dev_${ARCH}.o"
