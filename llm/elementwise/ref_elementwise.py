#!/usr/bin/env python3
"""Driver/reference. usage: ref_elementwise.py gen DIR | check DIR | all DIR (needs ./ew_test built)."""
import sys, os, subprocess, numpy as np, torch, torch.nn.functional as F
D, T, V, N, NCE = 768, 512, 100352, 4096, 256
n_ad = 3 * 65536 + 1000
ND = [(1000, 1768), (70000, 70768), (197000, 197608)]
def w(d, name, t):
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16: t = t.view(torch.int16)
    t.numpy().tofile(os.path.join(d, name))
def r(d, name, dt, shape=None):
    a = np.fromfile(os.path.join(d, name), dtype=np.int16 if dt == torch.bfloat16 else {torch.float32: np.float32, torch.int32: np.int32}[dt])
    t = torch.from_numpy(a)
    if dt == torch.bfloat16: t = t.view(torch.bfloat16)
    return t.reshape(shape) if shape else t
def gen(d):
    os.makedirs(d, exist_ok=True); g = torch.Generator().manual_seed(0)
    bf = lambda *s, sc=1.0, off=0.0: (torch.randn(*s, generator=g) * sc + off).bfloat16()
    tok = torch.randint(0, V, (N,), generator=g, dtype=torch.int32); tok[::2] = torch.randint(0, 3000, (N // 2,), generator=g, dtype=torch.int32); tok[:64] = 5
    w(d, 'tok.bin', tok); w(d, 'wte.bin', bf(V, D, sc=0.02)); w(d, 'wpe.bin', bf(T, D, sc=0.02)); w(d, 'dx0.bin', bf(N, D))
    w(d, 'ln_x.bin', bf(N, D, sc=2, off=0.5)); w(d, 'ln_delta.bin', bf(N, D)); w(d, 'ln_gamma.bin', bf(D, sc=0.2, off=1.0))
    w(d, 'ln_dy.bin', bf(N, D)); w(d, 'ln_dres.bin', bf(N, D))
    lg = torch.randn(NCE, V, generator=g) * 3; tg = torch.randint(0, V, (NCE,), generator=g, dtype=torch.int32)
    lg[torch.arange(NCE), tg.long()] += torch.rand(NCE, generator=g) * 12  # peaked rows
    lg[:4] *= 5  # very peaked
    w(d, 'ce_logits.bin', lg.bfloat16()); w(d, 'ce_tgt.bin', tg)
    w(d, 'ad_w0.bin', torch.randn(n_ad, generator=g) * 0.05)
    for s in range(3): w(d, f'ad_g{s}.bin', torch.randn(n_ad, generator=g) * (0.1 if s else 1.0))
def err(name, a, b, tol):
    a, b = a.float(), b.float(); sc = b.abs().max().item()
    mx = (a - b).abs().max().item() / sc; mean = (a - b).abs().mean().item() / b.abs().mean().item()
    ok = mx <= tol and torch.isfinite(a).all().item()
    print(f"  {name:24s} max_err/max|ref| {mx:.3e}  mean_err/mean|ref| {mean:.3e}  tol {tol:g}  {'PASS' if ok else 'FAIL'}"); return ok
def adam_ref(W0, Gs, clip):
    segs, decay, pos = [], [], 0
    for (a, b) in ND + [(n_ad, n_ad)]:
        if a > pos: segs.append((pos, a, True))
        if b > a: segs.append((a, b, False))
        pos = b
    ps = [torch.nn.Parameter(W0[a:b].clone().double().float()) for a, b, _ in segs]
    opt = torch.optim.AdamW([{'params': [p for p, s in zip(ps, segs) if s[2]], 'weight_decay': 0.1}, {'params': [p for p, s in zip(ps, segs) if not s[2]], 'weight_decay': 0.0}], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    for G in Gs:
        for p, (a, b, _) in zip(ps, segs): p.grad = G[a:b].clone()
        if clip: torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
    return torch.cat([p.detach() for p in ps])
def check(d):
    ok = True
    tok = r(d, 'tok.bin', torch.int32).long(); wte = r(d, 'wte.bin', torch.bfloat16, (V, D)); wpe = r(d, 'wpe.bin', torch.bfloat16, (T, D)); dx0 = r(d, 'dx0.bin', torch.bfloat16, (N, D))
    print("embedding"); pos = torch.arange(N) % T
    ok &= err('fwd x', r(d, 'out_x_emb', torch.bfloat16, (N, D)), (wte[tok].float() + wpe[pos].float()), 2e-2)
    dwte = torch.ones(V, D); dwte.index_add_(0, tok, dx0.float()); ok &= err('dwte (init 1, +=)', r(d, 'out_dwte', torch.float32, (V, D)), dwte, 1e-5)
    dwpe = torch.ones(T, D) + dx0.float().reshape(N // T, T, D).sum(0); ok &= err('dwpe (init 1, +=)', r(d, 'out_dwpe', torch.float32, (T, D)), dwpe, 1e-5)
    print("layernorm"); x = r(d, 'ln_x.bin', torch.bfloat16, (N, D)); dl = r(d, 'ln_delta.bin', torch.bfloat16, (N, D)); ga = r(d, 'ln_gamma.bin', torch.bfloat16)
    dy = r(d, 'ln_dy.bin', torch.bfloat16, (N, D)); dres = r(d, 'ln_dres.bin', torch.bfloat16, (N, D))
    xf = x.float().requires_grad_(); gf = ga.float().requires_grad_(); y = F.layer_norm(xf, (D,), gf, None, 1e-5); y.backward(dy.float())
    ok &= err('fwd y', r(d, 'out_ln_y', torch.bfloat16, (N, D)), y.detach(), 2e-2)
    xx = x.float(); mu = xx.mean(1); rs = torch.rsqrt(xx.var(1, unbiased=False) + 1e-5)
    ok &= err('mean', r(d, 'out_ln_mean', torch.float32), mu, 1e-4); ok &= err('rstd', r(d, 'out_ln_rstd', torch.float32), rs, 1e-5)
    h = (x.float() + dl.float()).bfloat16(); ok &= err('add: h', r(d, 'out_ln_h', torch.bfloat16, (N, D)), h, 1e-6)
    ok &= err('add: y', r(d, 'out_ln_y2', torch.bfloat16, (N, D)), F.layer_norm(h.float(), (D,), ga.float(), None, 1e-5), 2e-2)
    ok &= err('bwd dx', r(d, 'out_ln_dx', torch.bfloat16, (N, D)), xf.grad, 5e-2)
    ok &= err('bwd dgamma (init 1, +=)', r(d, 'out_ln_dgamma', torch.float32), gf.grad + 1, 5e-3)
    ok &= err('bwd dx + dres', r(d, 'out_ln_dx_res', torch.bfloat16, (N, D)), xf.grad + dres.float(), 5e-2)
    print("cross-entropy"); lg = r(d, 'ce_logits.bin', torch.bfloat16, (NCE, V)).float().requires_grad_(); tg = r(d, 'ce_tgt.bin', torch.int32).long()
    l = F.cross_entropy(lg, tg, reduction='none'); lm = l.mean(); lm.backward()
    ok &= err('loss per token', r(d, 'out_ce_loss', torch.float32)[:NCE], l.detach(), 1e-5)
    ok &= err('mean loss', r(d, 'out_ce_mean', torch.float32), lm.detach().reshape(1), 1e-5)
    ok &= err('dlogits', r(d, 'out_ce_dlogits', torch.bfloat16, (NCE, V)), lg.grad, 2e-2)
    print("adamw"); W0 = r(d, 'ad_w0.bin', torch.float32); Gs = [r(d, f'ad_g{s}.bin', torch.float32) for s in range(3)]
    Wr = adam_ref(W0, Gs, False); Wc = r(d, 'out_ad_w', torch.float32)
    ok &= err('W32 after 3 steps', Wc, Wr, 1e-5); ok &= err('Wbf working copy', r(d, 'out_ad_wbf', torch.bfloat16), Wr.bfloat16(), 1e-2)
    ok &= err('W32 vs W0 moved', Wc - W0, Wr - W0, 1e-3)
    Wr = adam_ref(W0, Gs[:1], True); ok &= err('W32 1 step + clip', r(d, 'out_ad_wclip', torch.float32), Wr, 1e-5)
    cs = r(d, 'out_ad_clipscale', torch.float32); n = Gs[0].norm().item(); print(f"  clip scale {cs[0].item():.6e} vs {min(1, 1/(n+1e-6)):.6e}, norm {cs[1].item():.4f} vs {n:.4f}")
    ok &= abs(cs[1].item() - n) / n < 1e-5
    print("PYTORCH-REF OVERALL", "PASS" if ok else "FAIL"); return ok
if __name__ == '__main__':
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd in ('gen', 'all'): gen(d)
    if cmd == 'all':
        rc = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ew_test'), d]).returncode
    if cmd in ('check', 'all'): sys.exit(0 if check(d) and (cmd != 'all' or rc == 0) else 1)
