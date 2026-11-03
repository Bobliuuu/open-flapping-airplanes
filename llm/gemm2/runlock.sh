#!/bin/bash
# on the box: ./runlock.sh OUT name1 name2 ...  -- one GPU-lock hold, times --steady for prebuilt gemm2_s_<name> binaries
out=$1; shift
flock -w 7200 ~/.gpu.lock bash -c "for n in $*; do echo == \$n; timeout 50 ./gemm2_s_\$n --steady --nopersist | grep -v 'kernel L' | cut -c1-24,90-; done > $out 2>&1"
