"""fa_compare.py : print/write the B200 baseline numbers next to FA's published ones. UNTESTED on B200.
Reads the newest results/baseline_v2_<GPU_TAG>_*.json (aggregate_v2.py output) and ../../../benchmarks/fa_published.json. No numbers are invented:
missing measurements print n/a."""
import json, glob, os, sys
here = os.path.dirname(os.path.abspath(__file__)); tag = os.environ.get("GPU_TAG", "b200")
fa = json.load(open(os.path.join(here, "..", "..", "..", "benchmarks", "fa_published.json")))
files = sorted(glob.glob(f"{here}/results/baseline_v2_{tag}_*.json"))
if not files: sys.exit("no aggregate json found")
d = json.load(open(files[-1]))
f = lambda v, p=1: "n/a" if v is None else f"{v:.{p}f}"
rows = []
for sysname in ("megatron_te", "pytorch_eager", "pytorch_cuda_graph", "megatron_te_fwdbwd_cudagraph"):
    for m in ("S", "F"):
        blk = d["training"].get(sysname, {}).get(m)
        if not blk: continue
        for kind in ("sm_restricted", "full_gpu"):
            b = blk.get(kind)
            if b: rows.append((sysname, m, kind, b["seconds"]["mean"], b["seconds"]["sd"], b["tok_s"]["mean"], b["loss_final"]["mean"]))
out = ["| system | model | SMs | 4000 steps (s) | sd | tok/s | final loss | vs FA 70.1 s |", "|---|---|---|---|---|---|---|---|"]
for s, m, k, sec, sd, tok, lf in rows:
    out.append(f"| {s} | {m} | {'%d (MPS)' % d['sm_split']['training'] if k == 'sm_restricted' else 'full %d' % d['gpu_sms']} | {f(sec)} | {f(sd, 2)} | {f(tok, 0)} | {f(lf, 4)} | {f(sec / fa['training']['seconds_baseline'], 2)}x |")
out += ["", "| vLLM decode (batch 1) | model | SMs | tok/s | vs FA 710 |", "|---|---|---|---|---|"]
for m in ("S", "F"):
    for k in ("sm_restricted", "full_gpu"):
        b = d["inference"]["vllm"][m].get(k)
        if b: out.append(f"| vllm | {m} | {'%d (MPS)' % d['sm_split']['inference'] if k == 'sm_restricted' else 'full'} | {f(b['tok_s']['mean'], 0)} | {f(b['tok_s']['mean'] / fa['inference']['tok_s_baseline'], 2)}x |")
out += ["", f"FA published (B200): baseline {fa['training']['seconds_baseline']} s / 4000 steps, ~{fa['training']['tok_s_baseline_at_frame']} tok/s, vLLM {fa['inference']['tok_s_baseline']} tok/s; launches {fa['kernel_launches']['baseline']}.",
        "Launch counts (this machine): see launch_counts.demo_equivalent in " + os.path.basename(files[-1])]
txt = "\n".join(out); print(txt); open(f"{here}/results/fa_compare_{tag}.md", "w").write(txt + "\n")
