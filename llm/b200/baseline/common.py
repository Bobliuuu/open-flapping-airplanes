"""Shared config/model/sampler for the open-flapping-airplanes baseline (model "S", see llm/SPEC.md)."""
import os, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# MODEL env selects the config: S (SPEC: L=12 d=768 H=12 MLP 3072) or F (L=4 d=2048 H=32 MLP 8192, ~408M params). Default S.
MODEL = os.environ.get("MODEL", "S")
if MODEL == "S": V, D, L, H, T, B = 100352, 768, 12, 12, 512, 8
elif MODEL == "F": V, D, L, H, T, B = 100352, 2048, 4, 32, 512, 8
else: raise ValueError(MODEL)
FFN = 4 * D
DH = D // H
STEPS = 4000
LR_MAX, LR_MIN, WARMUP, WD, BETAS = 6e-4, 6e-5, 100, 0.1, (0.9, 0.95)
SEED = 1234
# B200 DEFAULTS (UNTESTED on B200): 148 SMs, FA split 116/24/8. Override with env for other GPUs.
TOTAL_SMS = int(os.environ.get("TOTAL_SMS", 148))
TRAIN_SMS = int(os.environ.get("TRAIN_SMS", 116))
INFER_SMS = int(os.environ.get("INFER_SMS", 24))
DATA = os.path.expanduser(os.environ.get("DATA_DIR", "~/data"))
TRAIN_BIN = f"{DATA}/tinystories_cl100k_train.u32"
VALID_BIN = f"{DATA}/tinystories_cl100k_valid.u32"

def sm_pct(sms):  # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for a given SM count
    return max(1, round(100 * sms / TOTAL_SMS))

def lr_at(s):
    """LR used by optimizer step s (0-indexed): linear warmup (s+1)/WARMUP, then cosine to LR_MIN at s=STEPS."""
    if s < WARMUP:
        return LR_MAX * (s + 1) / WARMUP
    p = (s - WARMUP) / (STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * p))

def batch_starts(ntok, nsteps=STEPS):
    """starts[s, b] for step s, sequence b: np.random.default_rng(1234).integers(0, ntok-T-1, size=(nsteps, B)).
    Sequence = tok[st:st+T] (inputs), targets = tok[st+1:st+T+1]."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, ntok - T - 1, size=(nsteps, B), dtype=np.int64)

def init_weights():
    """Returns dict name->fp32 CPU tensor in SPEC [in,out] layout. torch.Generator('cpu').manual_seed(1234),
    torch.randn order: wte, wpe, then per layer Wqkv, Wo, Wup, Wdown. std 0.02 (Wo, Wdown * 1/sqrt(2L)). gammas = 1."""
    g = torch.Generator("cpu").manual_seed(SEED)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32) * 0.02
    w = {"wte": r(V, D), "wpe": r(T, D)}
    sc = 1 / math.sqrt(2 * L)
    for l in range(L):
        w[f"L{l:02d}_Wqkv"] = r(D, 3 * D)
        w[f"L{l:02d}_Wo"] = r(D, D) * sc
        w[f"L{l:02d}_Wup"] = r(D, 4 * D)
        w[f"L{l:02d}_Wdown"] = r(4 * D, D) * sc
        w[f"L{l:02d}_gamma1"] = torch.ones(D)
        w[f"L{l:02d}_gamma2"] = torch.ones(D)
    w["gamma_f"] = torch.ones(D)
    return w

class LN(nn.Module):
    def __init__(s):
        super().__init__(); s.weight = nn.Parameter(torch.ones(D))
    def forward(s, x): return F.layer_norm(x, (D,), s.weight, None, 1e-5)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = LN(), LN()
        s.qkv = nn.Linear(D, 3 * D, bias=False); s.proj = nn.Linear(D, D, bias=False)
        s.up = nn.Linear(D, 4 * D, bias=False); s.down = nn.Linear(4 * D, D, bias=False)
    def forward(s, x):
        b, t, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(b, t, 3, H, DH).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + s.down(F.gelu(s.up(s.ln2(x)), approximate="tanh"))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.wte = nn.Embedding(V, D); s.wpe = nn.Embedding(T, D)
        s.blocks = nn.ModuleList([Block() for _ in range(L)]); s.lnf = LN()
    def load_spec(s, w):
        with torch.no_grad():
            s.wte.weight.copy_(w["wte"]); s.wpe.weight.copy_(w["wpe"])
            for l, b in enumerate(s.blocks):
                p = f"L{l:02d}_"
                b.qkv.weight.copy_(w[p + "Wqkv"].t()); b.proj.weight.copy_(w[p + "Wo"].t())
                b.up.weight.copy_(w[p + "Wup"].t()); b.down.weight.copy_(w[p + "Wdown"].t())
                b.ln1.weight.copy_(w[p + "gamma1"]); b.ln2.weight.copy_(w[p + "gamma2"])
            s.lnf.weight.copy_(w["gamma_f"])
    def spec_tensors(s):  # name -> tensor view in SPEC layout (for grads/weights export); order = golden README order
        out = {"wte": s.wte.weight, "wpe": s.wpe.weight}
        for l, b in enumerate(s.blocks):
            p = f"L{l:02d}_"
            out[p + "Wqkv"] = b.qkv.weight; out[p + "Wo"] = b.proj.weight
            out[p + "Wup"] = b.up.weight; out[p + "Wdown"] = b.down.weight
            out[p + "gamma1"] = b.ln1.weight; out[p + "gamma2"] = b.ln2.weight
        out["gamma_f"] = s.lnf.weight
        return out
    def forward(s, idx, targets=None):
        b, t = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(t, device=idx.device))
        for blk in s.blocks: x = blk(x)
        logits = F.linear(s.lnf(x), s.wte.weight)  # tied head
        if targets is None: return logits
        return F.cross_entropy(logits.float().view(-1, V), targets.view(-1))

def spec_layout(name, t):  # torch tensor -> SPEC layout fp32 numpy (transpose Linear [out,in] -> [in,out])
    t = t.detach().float().cpu()
    if name.endswith(("Wqkv", "Wo", "Wup", "Wdown")): t = t.t()
    return t.contiguous().numpy()

def make_optimizer(model, capturable=False):
    mats = [p for p in model.parameters() if p.dim() >= 2]   # all matrices incl. wte/wpe
    vecs = [p for p in model.parameters() if p.dim() < 2]    # gammas
    return torch.optim.AdamW([{"params": mats, "weight_decay": WD}, {"params": vecs, "weight_decay": 0.0}],
                             lr=(torch.tensor(LR_MAX, device="cuda") if capturable else LR_MAX), betas=BETAS, eps=1e-8,
                             fused=True, capturable=capturable)
