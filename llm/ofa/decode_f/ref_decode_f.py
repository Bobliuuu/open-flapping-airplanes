#!/usr/bin/env python3
"""PyTorch fp32 reference for the decode kernels.
  python3 ref_decode.py gen   <dir> [ntok=32]   -> <dir>/weights.bin (bf16 raw), <dir>/tokens.bin (int32)
  python3 ref_decode.py check <dir>             -> compares <dir>/logits_cu.bin (ntok x V fp32) with a full-sequence fp32 forward
Layout of weights.bin: wte[V,d] wpe[T,d] gf[d], then per layer: g1[d] g2[d] Wqkv[d,3d] Wo[d,d] Wup[d,4d] Wdn[4d,d]  (bf16, row-major)
"""
import sys, os, numpy as np, torch
import torch.nn.functional as F
L, D, H, T, V = 4, 2048, 32, 512, 100352
VALID = 100277

def shapes():
    s = [("wte", (V, D)), ("wpe", (T, D)), ("gf", (D,))]
    for l in range(L):
        s += [(f"g1_{l}", (D,)), (f"g2_{l}", (D,)), (f"wqkv_{l}", (D, 3 * D)), (f"wo_{l}", (D, D)),
              (f"wup_{l}", (D, 4 * D)), (f"wdn_{l}", (4 * D, D))]
    return s

def gen(d, ntok):
    os.makedirs(d, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(1234)
    with open(os.path.join(d, "weights.bin"), "wb") as f:
        for name, shp in shapes():
            if name.startswith("g"):
                w = 1.0 + 0.1 * torch.randn(shp, generator=g)
            else:
                # embeddings 0.02 (SPEC); layer matrices larger than SPEC init so every layer matters in the test
                std = 0.02 if name in ("wte", "wpe") else (0.025 if name.startswith(("wqkv", "wup")) else 0.025 / (2 * L) ** 0.5 * 4)
                w = std * torch.randn(shp, generator=g)
            f.write(w.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
    toks = torch.randint(0, VALID, (ntok,), generator=g).numpy().astype(np.int32)
    toks.tofile(os.path.join(d, "tokens.bin"))
    print("wrote", d, "ntok", ntok)

def load(d):
    raw = np.fromfile(os.path.join(d, "weights.bin"), dtype=np.int16)
    W, o = {}, 0
    for name, shp in shapes():
        n = int(np.prod(shp))
        W[name] = torch.from_numpy(raw[o:o + n].copy()).view(torch.bfloat16).reshape(shp).float().cuda()
        o += n
    assert o == raw.size
    return W

@torch.no_grad()
def forward(W, toks):
    n = len(toks)
    t = torch.tensor(toks, dtype=torch.long, device="cuda")
    x = W["wte"][t] + W["wpe"][:n]
    for l in range(L):
        h = F.layer_norm(x, (D,), W[f"g1_{l}"], None, 1e-5)
        qkv = h @ W[f"wqkv_{l}"]
        q, k, v = qkv.split(D, dim=1)
        sp = lambda a: a.view(n, H, 64).transpose(0, 1)
        a = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)
        x = x + a.transpose(0, 1).reshape(n, D) @ W[f"wo_{l}"]
        h = F.layer_norm(x, (D,), W[f"g2_{l}"], None, 1e-5)
        x = x + F.gelu(h @ W[f"wup_{l}"], approximate="tanh") @ W[f"wdn_{l}"]
    x = F.layer_norm(x, (D,), W["gf"], None, 1e-5)
    return x @ W["wte"].T

def check(d):
    torch.backends.cuda.matmul.allow_tf32 = False
    toks = np.fromfile(os.path.join(d, "tokens.bin"), dtype=np.int32)
    n = len(toks)
    cu = torch.from_numpy(np.fromfile(os.path.join(d, "logits_cu.bin"), dtype=np.float32).reshape(n, V)).cuda()
    ref = forward(load(d), toks.tolist())
    rel = (cu - ref).abs().amax(1) / ref.abs().amax(1)
    mean_rel = ((cu - ref).abs().mean(1) / ref.abs().mean(1))
    am = (cu[:, :VALID].argmax(1) == ref[:, :VALID].argmax(1)).float().mean().item()
    top2 = ref[:, :VALID].topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])
    print(f"steps={n}  max rel err (max|d|/max|ref|) = {rel.max().item():.3e}  mean-over-steps = {rel.mean().item():.3e}")
    print(f"mean |d|/mean|ref| = {mean_rel.max().item():.3e} (max over steps)   ref logit std {ref.std().item():.3f}, top1-top2 margin median {margin.median().item():.3f} min {margin.min().item():.3f}")
    print(f"argmax agreement = {am*100:.1f}%")
    ok = rel.max().item() <= 3e-2 and am >= 0.99
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if sys.argv[1] == "gen": gen(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32)
    else: check(sys.argv[2])
