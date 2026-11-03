#!/usr/bin/env bash
# build_merged.sh <out-binary> [nvcc flags]   (run on the box, in ~/llm/ofa_fast)
# Whole-program (non-rdc) build of ofa.cu with Doom's PTX merged into the same PTX module before ptxas.
set -e
echo "nvcc: $(nvcc --version | tail -1)" >&2
OUT=$1; shift
DOOM=${DOOM_PTX:-$HOME/doom-gpu/doom.ptx}
W=$(mktemp -d /tmp/tfm.XXXXXX)
nvcc -arch=sm_80 -O3 -std=c++17 --extended-lambda -maxrregcount=128 "$@" ofa.cu -o $OUT -lpthread --dryrun 2>&1 | sed -n 's/^#\$ //p' | sed -E 's/^([A-Z_]+)=(.*)$/export \1="\2"/' > $W/cmds.sh
[ -f "$DOOM" ] || { echo "build_merged: $DOOM missing (run doom-gpu/build_doom.sh)" >&2; exit 1; }
python3 - "$W" "$DOOM" <<'PY'
import sys,re
W,DOOM=sys.argv[1],sys.argv[2]
L=open(W+'/cmds.sh').read().split('\n'); out=[]
for l in L:
    out.append(l)
    if 'cicc' in l.split()[0] if l.split() else False:
        m=re.search(r'"(/tmp/[^"]*\.ptx)"',l) or re.search(r'(/tmp/\S*\.ptx)',l)
        p=m.group(1)
        out.append('mv %s %s.main && python3 ptxmerge.py %s.main %s %s'%(p,p,p,DOOM,p))
open(W+'/cmds.sh','w').write('\n'.join(out))
PY
grep -n "ptxas" $W/cmds.sh | cut -c1-150 >&2
sed -i -e 's/^ptxas /ptxas -v /' -e 's/^rm /rm -f /' $W/cmds.sh
grep -q 'ptxmerge.py' $W/cmds.sh || { echo 'build_merged: no cicc step found in nvcc --dryrun output (layout changed for this nvcc)' >&2; exit 1; }
bash -e $W/cmds.sh > $W/log 2>&1 || { grep -v 'out of bound' $W/log | tail -20 >&2; echo 'build_merged: a step failed (see above); the nvcc dryrun/ptxmerge assumptions may not hold for this nvcc' >&2; exit 1; }
grep -v 'out of bound' $W/log | grep -A2 'Function properties for _Z12ofa_kernel' || true
[ -x $OUT ] || { echo 'build_merged: no binary produced' >&2; exit 1; }
