# B200 kit: UNTESTED on B200. Generalised from the A100 version (cu13-only preload) to also handle cu128/cu12 wheels.
"""Import this FIRST in any script that uses Transformer Engine (BASELINE v2).
1. TE 2.20 imports `onnx`/`onnxscript` unconditionally at import time (only needed for ONNX export, which we never use). They are NOT installed in
   ~/venv-baseline (install of anything beyond TE/flash-attn/apex was not permitted), so inert stand-ins are put in sys.modules when the real ones are missing.
2. The venv's CUDA-13 pip libs (libcublas.so.13 ...) are not on the default dlopen path for TE's core lib: prepend them to LD_LIBRARY_PATH via ctypes preload."""
import sys, os, glob, ctypes, importlib.util
import types, importlib.machinery
class _Stub:
    def __call__(self, *a, **k): return a[0] if len(a) == 1 and callable(a[0]) and not k else _Stub()
    def __getattr__(self, n):
        if n.startswith('__'): raise AttributeError(n)
        return _Stub()
def _modattr(n):
    if n.startswith('__'): raise AttributeError(n)
    return _Stub()
for m in ("onnx", "onnxscript"):
    if importlib.util.find_spec(m) is None:
        mod = types.ModuleType(m); mod.__spec__ = importlib.machinery.ModuleSpec(m, None); mod.__getattr__ = _modattr
        sys.modules[m] = mod
import torch  # noqa: E402  (loads torch's own CUDA libs first)
_sp = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia")
# Preload whichever CUDA pip libs exist (cu12 wheels: nvidia/cublas/lib/libcublas.so.12 ; cu13 wheels: nvidia/cu13/lib/libcublas.so.13). Harmless if absent.
import glob as _g
for pat in ("cu13/lib/libcublas.so.13", "cu13/lib/libcublasLt.so.13", "cu13/lib/libnvrtc.so.13", "cu13/lib/libcudart.so.13",
            "cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12", "cuda_nvrtc/lib/libnvrtc.so.12", "cuda_runtime/lib/libcudart.so.12"):
    for p in _g.glob(os.path.join(_sp, pat)):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
