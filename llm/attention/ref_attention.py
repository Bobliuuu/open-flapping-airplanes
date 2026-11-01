#!/usr/bin/env python3
"""Compare attn_test dumps against fp32 PyTorch SDPA (causal), and time PyTorch flash SDPA.  usage: ref_attention.py [dir]"""
import sys, numpy as np, torch, torch.nn.functional as F
d = sys.argv[1] if len(sys.argv) > 1 else "attn_data"
B, T, H = map(int, open(f"{d}/shape.txt").read().split()); D = H * 64; N = B * T
def bf(name, shape):
    a = np.fromfile(f"{d}/{name}", dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(a.view(np.float32).reshape(shape).copy()).cuda()
qkv = bf("qkv.bin", (N, 3 * D)); do = bf("do.bin", (N, D)); o_k = bf("o.bin", (N, D)); dqkv_k = bf("dqkv.bin", (N, 3 * D))
lse_k = torch.from_numpy(np.fromfile(f"{d}/lse.bin", dtype=np.float32).reshape(B, H, T)).cuda()
def heads(x): return x.view(B, T, H, 64).permute(0, 2, 1, 3)  # [B,H,T,64]
q, k, v = [heads(qkv[:, i * D:(i + 1) * D]).clone().requires_grad_() for i in range(3)]
o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # fp32 math
o.backward(heads(do))
with torch.no_grad():
    s = (q @ k.transpose(-1, -2)) * 0.125
    s = s.masked_fill(torch.triu(torch.ones(T, T, device='cuda', dtype=torch.bool), 1), float('-inf'))
    lse = torch.logsumexp(s, -1)
def flat(x): return x.permute(0, 2, 1, 3).reshape(N, D)
refs = {"o": (flat(o.detach()), o_k), "lse": (lse, lse_k),
        "dq": (flat(q.grad), dqkv_k[:, :D]), "dk": (flat(k.grad), dqkv_k[:, D:2 * D]), "dv": (flat(v.grad), dqkv_k[:, 2 * D:])}
ok = True
for n, (r, x) in refs.items():
    err = (r - x).abs(); rel_max = (err.max() / r.abs().max()).item(); rel_mean = (err.mean() / r.abs().mean()).item()
    tol = 2e-2 if n in ("o", "lse") else 5e-2
    good = rel_max <= tol and not torch.isnan(x).any(); ok &= good
    print(f"{n:4s} max_err/max|ref| {rel_max:.3e}  mean_err/mean|ref| {rel_mean:.3e}  tol {tol:g}  {'ok' if good else 'BAD'}")
print("ACCURACY", "PASS" if ok else "FAIL")
# PyTorch flash SDPA timing (bf16, same inputs)
qb, kb, vb = [heads(qkv[:, i * D:(i + 1) * D]).bfloat16().contiguous().requires_grad_() for i in range(3)]
g = heads(do).bfloat16().contiguous()
from torch.nn.attention import sdpa_kernel, SDPBackend
def tm(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    fwd = lambda: F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
    out = fwd()
    def fb(): 
        o_ = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True); o_.backward(g)
    tf = tm(fwd); tfb = tm(fb); tb = tfb - tf
ff = 4 * B * H * T * T * 64 / 2
print(f"TORCH_FLASH fwd_ms {tf:.4f} fwd_TFLOPs {ff/tf/1e9:.1f} bwd_ms(fwd+bwd minus fwd) {tb:.4f} bwd_TFLOPs(2.5x) {2.5*ff/tb/1e9:.1f}")
