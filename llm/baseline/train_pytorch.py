"""Model S, plain PyTorch eager training (bf16 autocast, fused AdamW). One JSON line of results per run.
  python train_pytorch.py --out r.json [--steps 4000] [--golden] [--count_launches] [--save ckpt.pt]
"""
import argparse, json, time, os, sys
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=STEPS)
ap.add_argument("--out", default=None)
ap.add_argument("--golden", action="store_true")
ap.add_argument("--count_launches", action="store_true")
ap.add_argument("--save", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--cuda_graph", action="store_true", help="whole training step (data gather, LR, fwd, bwd, AdamW) replayed from ONE CUDA graph")
ap.add_argument("--count3", action="store_true", help="launch counting (3 definitions) only; writes res['launch3'], skips the timed run")
a = ap.parse_args()
dev = "cuda"
here = os.path.dirname(os.path.abspath(__file__))

tok_np = np.memmap(TRAIN_BIN, dtype=np.uint32, mode="r")
starts_np = batch_starts(len(tok_np))
tok = torch.from_numpy(tok_np.astype(np.int32)).to(dev)
starts = torch.from_numpy(starts_np).to(dev)
ar = torch.arange(T + 1, device=dev)
def get_batch(s):
    seq = tok[(starts[s][:, None] + ar).view(-1)].view(B, T + 1).long()
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

w0 = init_weights()
model = GPT().to(dev); model.load_spec(w0)
opt = make_optimizer(model)
nparams = sum(p.numel() for p in model.parameters())

def step(s, grab_grads=False):
    for g in opt.param_groups: g["lr"] = lr_at(s)
    x, y = get_batch(s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    grads = None
    if grab_grads:
        grads = {n: p.grad.detach().clone() for n, p in model.spec_tensors().items()}
    opt.step(); opt.zero_grad(set_to_none=True)
    return loss.detach(), grads

res = {"model": MODEL, "model_params": nparams, "tag": a.tag, "env": {"TRAIN_SMS": TRAIN_SMS, "TOTAL_SMS": TOTAL_SMS,
       "MPS_PCT": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0)}}

if a.golden:
    gd = f"{here}/golden" + ("" if MODEL == "S" else f"/{MODEL}"); os.makedirs(gd, exist_ok=True)
    for n, t in w0.items(): t.numpy().astype("<f4").tofile(f"{gd}/w_{n}.f32")
    x0, y0 = get_batch(0)
    allb = np.stack([np.stack([tok_np[st:st + T + 1] for st in starts_np[s]]) for s in range(8)]).astype("<u4")
    allb.tofile(f"{gd}/batches_0_7.u32"); starts_np[:8].astype("<u4").tofile(f"{gd}/starts_0_7.u32")
    # logits for batch 0, seq 0 (no grad, same weights as step 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(x0)[0].float()
    logits = {str(p): lg[p, :8].cpu().tolist() for p in (0, 255, 511)}
    # step 0 fwd/bwd, grads in fp32 (autocast keeps fp32 master params)
    for g in opt.param_groups: g["lr"] = lr_at(0)
    x, y = get_batch(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss0 = model(x, y)
    loss0.backward()
    gr = {n: p.grad.detach().float().cpu() for n, p in model.spec_tensors().items()}
    # note: with tied head wte.grad includes the head contribution; shapes are PyTorch-native, flat idx used for samples
    gnorm = torch.sqrt(sum((g.double() ** 2).sum() for g in gr.values())).item()
    per = {}
    for n, g in gr.items():
        sg = spec_layout(n, g)  # SPEC [in,out] layout
        flat = sg.reshape(-1); idx = np.linspace(0, flat.size - 1, 5).astype(np.int64)
        per[n] = {"shape": list(sg.shape), "l2": float(np.linalg.norm(flat.astype(np.float64))),
                  "sample_flat_idx": idx.tolist(), "sample_vals": flat[idx].tolist()}
    json.dump({"loss_step0": loss0.item(), "global_grad_norm": gnorm, "logits_seq0_first8vocab": logits,
               "grads": per, "note": "grads are d(mean CE)/d(param) for batch 0 at init; layouts are SPEC [in,out]; flat idx = row-major in that layout; bf16 autocast forward so expect ~1e-2 relative noise"},
              open(f"{gd}/step0.json", "w"), indent=1)
    opt.zero_grad(set_to_none=True)
    # steps 0..20 with fresh model for the loss list
    model.load_spec(w0); opt = make_optimizer(model)
    ls = [step(s)[0] for s in range(21)]
    json.dump({"losses_step0_to_20": [l.item() for l in ls]}, open(f"{gd}/losses_0_20.json", "w"), indent=1)
    print("golden written", loss0.item(), gnorm, [round(l.item(), 4) for l in ls]); sys.exit()

# ---- CUDA-graph version of the step: step counter, LR schedule, batch gather, fwd, bwd, fused AdamW all inside the graph ----
def build_graph_step():
    global opt
    opt = make_optimizer(model, capturable=True)
    lr_t = opt.param_groups[0]["lr"]; opt.param_groups[1]["lr"] = lr_t
    s_t = torch.zeros((), dtype=torch.long, device=dev); losses_g = torch.zeros(STEPS + 8, device=dev)
    def gstep():
        sf = s_t.float()
        lr_t.copy_(torch.where(sf < WARMUP, LR_MAX * (sf + 1) / WARMUP,
                               LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + torch.cos(math.pi * (sf - WARMUP) / (STEPS - WARMUP)))))
        seq = tok[(starts.index_select(0, s_t.reshape(1))[0][:, None] + ar).view(-1)].view(B, T + 1).long()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss = model(seq[:, :-1].contiguous(), seq[:, 1:].contiguous())
        loss.backward(); opt.step()
        losses_g.index_copy_(0, s_t.reshape(1), loss.detach().reshape(1)); s_t.add_(1)
    def warm(n=3):   # n eager steps (they are real steps 0..n-1), on a side stream as required before capture
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(n): opt.zero_grad(set_to_none=True); gstep()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    def capture():
        torch.autograd.graph.set_override_stale_capture_stream(True)
        opt.zero_grad(set_to_none=True); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): gstep()
        return g
    return gstep, warm, capture, losses_g, s_t
import math

if a.count3:
    from torch.profiler import profile, ProfilerActivity
    from launchcount import trace_counts
    NS = 5
    for s in range(20): step(s)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for s in range(100, 100 + NS): step(s)
        torch.cuda.synchronize()
    res["launch3"] = {"eager": trace_counts(prof, NS)}
    model.load_spec(w0)
    gstep, warm, capture, losses_g, s_t = build_graph_step(); warm(3); g = capture()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(NS): g.replay()
        torch.cuda.synchronize()
    res["launch3"]["graph"] = trace_counts(prof, NS)
    print(json.dumps(res["launch3"], indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    sys.exit()

if a.cuda_graph:
    gstep, warm, capture, losses_g, s_t = build_graph_step()
    torch.cuda.synchronize(); t0 = time.time()
    warm(3); g = capture()
    for _ in range(a.steps - 3): g.replay()
    torch.cuda.synchronize(); dt = time.time() - t0
    losses = losses_g[:a.steps].clone()
    res["cuda_graph"] = True
else:
  losses = torch.zeros(a.steps, device=dev)
  torch.cuda.synchronize(); t0 = time.time()
  for s in range(a.steps):
    losses[s] = step(s)[0]
  torch.cuda.synchronize(); dt = time.time() - t0
# warm-up not counted: none (timing includes cuDNN/autotune first-step costs: negligible vs 4000 steps)
lc = losses.cpu().numpy()
res.update(steps=a.steps, seconds=dt, tok_s=a.steps * B * T / dt, loss_initial=float(lc[0]), loss_final=float(lc[-1]),
           loss_final_mean_last25=float(lc[-25:].mean()),
           loss_curve_every25={int(i): float(lc[i]) for i in range(0, a.steps, 25)} | {a.steps - 1: float(lc[-1])},
           peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
if a.save:
    torch.save({n: p.detach().cpu() for n, p in model.state_dict().items()}, a.save)
if a.count_launches:
    from torch.profiler import profile, ProfilerActivity
    NS = 3
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for s in range(a.steps, a.steps + NS) if False else range(100, 100 + NS):
            step(s)
        torch.cuda.synchronize()
    ev = [e for e in prof.events()]
    kern = sum(1 for e in ev if e.device_type == torch.autograd.DeviceType.CUDA and e.name and
               not e.name.startswith(("Memcpy", "Memset")))
    mem = sum(1 for e in ev if e.device_type == torch.autograd.DeviceType.CUDA and e.name and e.name.startswith(("Memcpy", "Memset")))
    res["launches"] = {"method": "torch.profiler(CUPTI) over 3 steady-state steps; kernel activity records, memcpy/memset counted separately",
                       "kernels_per_step": kern / NS, "memops_per_step": mem / NS,
                       "kernels_total_extrapolated": kern / NS * a.steps}
print(json.dumps({k: v for k, v in res.items() if k != "loss_curve_every25"}))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
