#!/usr/bin/env bash
# One-time setup of the GPU box for the Doom megakernel demo (run from your Mac):  ./setup_box.sh ubuntu@<ip>
# Installs clang/LLVM with sudo apt, clones the doomgeneric engine source, and fetches the freely distributable shareware
# doom1.wad (kept only if its md5 matches the known shareware 1.9 checksum). Asks nothing; read it before running it.
set -euo pipefail
HOST="${1:?usage: setup_box.sh user@host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude doomgeneric --exclude obj --exclude src_gpu --exclude '*.o' --exclude '*.ptx' --exclude '*.bc' --exclude '*.ll' --exclude '*.cubin' "$DIR/" "$HOST:~/doom-gpu/"
ssh "$HOST" 'set -e; cd ~/doom-gpu
  command -v clang >/dev/null || { sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang llvm lld; }
  [ -d doomgeneric ] || git clone -q https://github.com/ozkl/doomgeneric.git
  want=f0cefca49926d00903cf57551d901abe
  if [ ! -s doom1.wad ] || [ "$(md5sum doom1.wad | cut -d" " -f1)" != "$want" ]; then
    for u in https://github.com/Akbar30Bill/DOOM_wads/raw/master/doom1.wad; do
      curl -fsSL -m 90 -o doom1.wad "$u" || true
      [ "$(md5sum doom1.wad 2>/dev/null | cut -d" " -f1)" = "$want" ] && break || rm -f doom1.wad
    done
  fi
  [ -s doom1.wad ] || { echo "doom1.wad missing: copy a DOOM1.WAD to ~/doom-gpu/doom1.wad yourself"; exit 1; }
  ./build_doom.sh | tail -3; echo setup done'
