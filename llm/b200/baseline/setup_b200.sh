#!/usr/bin/env bash
# setup_b200.sh : build the baseline environment on a B200 box (sm_100, driver >= 570, CUDA 12.8+).
# UNTESTED ON B200 (written without access to B200 hardware). Every step is idempotent and logged to ~/b200_setup/*.log. Run steps individually
# with STEPS="check venv_train venv_serve data sms" if one fails.  Usage: ./setup_b200.sh
#
# WHY TWO VENVS: Megatron + Transformer Engine + flash-attn need a torch we pin (cu128, built against the box toolkit), while vLLM wheels pin their own
# torch. On the A100 box one venv worked (torch 2.13+cu130 from vLLM, TE cu13 wheels); on B200 we keep them apart so a vLLM upgrade can not break TE.
# INSTALL ORDER (matters): (1) check driver/nvcc (2) torch cu128 (3) numpy/tiktoken/transformers/onnx stubs (4) megatron-core (5) Transformer Engine,
# built from source against torch, NVTE_CUDA_ARCHS=100 (6) flash-attn (best effort; may not ship sm_100 kernels, then TE uses cuDNN fused attention)
# (7) separate venv for vLLM (8) data. Never `pip install vllm` into the train venv after TE: it replaces torch and breaks TE's compiled extension.
set -uo pipefail
STEPS=${STEPS:-"check venv_train venv_serve data sms"}
VENV_TRAIN=${VENV_TRAIN:-$HOME/venv-train}; VENV_SERVE=${VENV_SERVE:-$HOME/venv-serve}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
TORCH_SPEC=${TORCH_SPEC:-"torch"}          # torch >= 2.7 is the first release with sm_100 (cu128) kernels. Pin e.g. TORCH_SPEC="torch==2.8.0" for reproducibility.
PY=${PY:-python3}                           # 3.10-3.12
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$HOME/b200_setup; mkdir -p "$LOG" "$HOME/data"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "== $*"; }

if has check; then
  say "check: GPU / driver / toolkit"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv | tee "$LOG/gpu.txt"
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "$CC" = "10.0" ] || echo "WARNING: compute_cap is $CC, expected 10.0 for B200 (GB200 is also 10.0; B300 is 10.3 -> use sm_103a, see README)."
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${DRV:-0}" -ge 570 ] || echo "WARNING: driver $DRV < 570: CUDA 12.8 / sm_100 will not work. Upgrade the driver first."
  command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "WARNING: no nvcc on PATH. Needed for TE source build and for the megakernel. Install CUDA toolkit 12.8+ (12.9/13.0 fine) and export PATH=/usr/local/cuda/bin:\$PATH"
  nvidia-smi -q | grep -i -E "MIG Mode|Current" | head -4
  command -v nvidia-cuda-mps-control >/dev/null && echo "MPS: ok" || echo "WARNING: nvidia-cuda-mps-control missing (needed for SM restriction)"
  command -v nsys >/dev/null && echo "nsys: ok" || echo "note: nsys not found (only needed for count_nsys.sh)"
fi

if has venv_train; then
  say "venv_train: $VENV_TRAIN"
  [ -d "$VENV_TRAIN" ] || $PY -m venv "$VENV_TRAIN"
  P="$VENV_TRAIN/bin/python"; PIP="$P -m pip"
  $PIP install -U pip wheel setuptools packaging ninja psutil einops pybind11 2>&1 | tail -1
  $PIP install $TORCH_SPEC --index-url "$TORCH_INDEX" 2>&1 | tail -2 | tee "$LOG/torch.log"
  $P - <<'PY' || { echo "FATAL: torch does not see sm_100; fix before continuing"; exit 1; }
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch_list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "no CUDA"
assert any(a.startswith(("sm_100", "compute_100")) for a in torch.cuda.get_arch_list()), "this torch build has no sm_100 kernels (need a cu128+ wheel)"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", (x @ x).float().abs().mean().item())
PY
  $PIP install numpy tiktoken "transformers" onnx onnxscript 2>&1 | tail -1
  $PIP install megatron-core 2>&1 | tail -1 | tee "$LOG/megatron.log"       # A100 run used 0.16.1; newer is fine if megatron_train.py's imports still resolve
  # Transformer Engine from source: the PyPI wheel set follows the A100 box (cu13); for a cu128 torch build against the local toolkit.
  # Needs cudnn dev headers (pip nvidia-cudnn-cu12 is found via CUDNN_PATH) and ~30-60 min. MAX_JOBS limits RAM use (each nvcc job ~4-8 GB).
  export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-100} MAX_JOBS=${MAX_JOBS:-16}
  $PIP install nvidia-cudnn-cu12 2>&1 | tail -1
  export CUDNN_PATH=$($P -c "import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null || true)
  $PIP install --no-build-isolation "transformer_engine[pytorch]" 2>&1 | tail -5 | tee "$LOG/te.log"
  # flash-attn: best effort. FA2 builds sm_80/90 and (>=2.8, nvcc>=12.8) sm_100 gencode, but its kernels are not tuned for Blackwell.
  # If this fails or is slow, skip it: TE falls back to cuDNN fused attention (NVTE_FUSED_ATTN=1), which is the Blackwell-native path. Record which backend ran.
  MAX_JOBS=${MAX_JOBS:-16} $PIP install flash-attn --no-build-isolation 2>&1 | tail -3 | tee "$LOG/flashattn.log" || echo "flash-attn install failed (non-fatal)"
  ( cd "$HERE" && $P - <<'PY'
import te_env, torch
import transformer_engine.pytorch as te
print("TE", te.__version__ if hasattr(te, "__version__") else "?")
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("flash_attn NOT importable:", e)
import megatron.core as m; print("megatron-core", m.__version__)
PY
  ) 2>&1 | tee "$LOG/verify_train.txt"
fi

if has venv_serve; then
  say "venv_serve: $VENV_SERVE (vLLM brings its own torch; do not install anything else that pins torch)"
  [ -d "$VENV_SERVE" ] || $PY -m venv "$VENV_SERVE"
  S="$VENV_SERVE/bin/python"
  $S -m pip install -U pip 2>&1 | tail -1
  $S -m pip install vllm tiktoken "transformers" numpy 2>&1 | tail -2 | tee "$LOG/vllm.log"
  $S - <<'PY' 2>&1 | tee "$LOG/verify_serve.txt"
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())
PY
  echo "NOTE: if this vLLM wheel has no sm_100 support, install a newer vLLM or the cu128 wheel per docs.vllm.ai; Blackwell needs a recent vLLM (FlashInfer/Triton paths)."
fi

if has data; then
  say "data: TinyStories (valid + first 300 MB of train) -> ~/data, cl100k uint32"
  D=$HOME/data
  [ -s $D/TinyStories-valid.txt ] || curl -sL -o $D/TinyStories-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
  [ -s $D/TinyStories-train-prefix.txt ] || curl -sL -r 0-314572799 -o $D/TinyStories-train-prefix.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
  [ -s $D/tinystories_cl100k_train.u32 ] || (cd "$HERE" && "$VENV_TRAIN/bin/python" prep_data.py)
  ls -la $D | head
fi

if has sms; then
  say "sms: build check_sms for sm_100a and scan MPS percentages (verifies 116/24 SMs)"
  "$HERE/calibrate_sms.sh" 2>&1 | tee "$LOG/sms_calibration.txt"
fi
say "setup finished; next: ./run_all.sh  (or PHASES=\"count\" ./run_all.sh for the quick launch count first)"
