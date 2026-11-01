#!/usr/bin/env bash
# count_nsys.sh : independent kernel-launch count with Nsight Systems as a cross-check of the CUPTI/torch.profiler numbers. UNTESTED on B200.
# Counts GPU kernel executions and host launch API calls for N training steps, then extrapolates per step (setup kernels excluded by differencing two lengths).
# Usage: MODEL=S|F ./count_nsys.sh [megatron|pytorch]
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; M=${MODEL:-S}; WHICH=${1:-megatron}; T=${PY_TRAIN:-$HOME/venv-train/bin/python}
command -v nsys >/dev/null || { echo "nsys not found (install Nsight Systems or use the torch.profiler counts from run_all.sh)"; exit 1; }
SCRIPT=megatron_train.py; ARGS="--impl te"; [ "$WHICH" = pytorch ] && { SCRIPT=train_pytorch.py; ARGS=""; }
cnt() { # cnt STEPS -> "kernels launches"
  rm -f /tmp/nsys_$$.nsys-rep /tmp/nsys_$$.sqlite
  MODEL=$M nsys profile -t cuda -s none --cuda-graph-trace=node -o /tmp/nsys_$$ -f true $T $SCRIPT $ARGS --steps $1 --out /tmp/nsys_$$.json >/dev/null 2>&1
  nsys export --type sqlite -o /tmp/nsys_$$.sqlite /tmp/nsys_$$.nsys-rep >/dev/null 2>&1
  python3 -I - <<PY
import sqlite3
c = sqlite3.connect("/tmp/nsys_$$.sqlite")
k = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
try: r = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaLaunch%' or s.value like 'cudaGraphLaunch%'").fetchone()[0]
except Exception as e: r = -1
print(k, r)
PY
}
read k1 r1 < <(cnt 60); read k2 r2 < <(cnt 120)
echo "model=$M $WHICH: kernels/step=$(( (k2 - k1) / 60 ))  host_launch_calls/step=$(( (r2 - r1) / 60 ))   (raw: 60 steps: $k1/$r1, 120 steps: $k2/$r2)"
echo "FA published: 514,562 launches for 4000 steps + served tokens (see llm/baseline README for the arithmetic)."
