"""Aggregate results/runs/*.json into results/baseline_<gpu>_<date>.json (schema mirrors benchmarks/fa_published.json)."""
import json, glob, os, datetime, statistics as st
from common import TOTAL_SMS, TRAIN_SMS, INFER_SMS, STEPS, B, T
here = os.path.dirname(os.path.abspath(__file__)); R = f"{here}/results/runs"
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def ms(v):
    v = [x for x in v if x is not None]
    if not v: return None
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v), "values": v}
def runs(prefix):
    out = []
    for p in sorted(glob.glob(f"{R}/{prefix}_r*.json")):
        if p.endswith(".meta.json"): continue
        d = load(p)
        m = load(p.replace(".json", ".meta.json")) or {}
        if d: d["_noisy"] = m.get("noisy_not_idle_before_start"); d["_maxp"] = m.get("max_compute_procs_seen"); out.append(d)
    return out
def train_block(prefix):
    rs = runs(prefix)
    if not rs: return None
    g = lambda k: [r.get(k) for r in rs]
    b = {"steps": rs[0]["steps"], "seconds": ms(g("seconds")), "tok_s": ms(g("tok_s")), "loss_initial": ms(g("loss_initial")),
         "loss_final": ms(g("loss_final")), "loss_final_mean_last25": ms(g("loss_final_mean_last25")),
         "peak_mem_gb": ms(g("peak_mem_gb")), "runs_not_idle_at_start": sum(1 for r in rs if r["_noisy"]), "max_compute_procs_seen_per_run": [r["_maxp"] for r in rs],
         "loss_curve_every25_rep1": rs[0]["loss_curve_every25"], "env": rs[0]["env"], "model_params": rs[0]["model_params"]}
    for r in rs:
        if "launches" in r: b["launches"] = r["launches"]; break
    return b
def serve_block(prefix):
    rs = runs(prefix)
    if not rs: return None
    tps = [x for r in rs for x in r["tok_s"]]
    b = {"new_tokens": rs[0]["new_tokens"], "tok_s": ms(tps), "mps_pct": rs[0]["mps_pct"],
         "runs_not_idle_at_start": sum(1 for r in rs if r["_noisy"]), "max_compute_procs_seen_per_run": [r["_maxp"] for r in rs]}
    for r in rs:
        if "launches" in r: b["launches"] = r["launches"]; break
    return b
out = {"source": "Measured on this machine by llm/baseline/run_baseline.sh. Method follows FA: each system timed individually on the GPU, no Doom.",
       "gpu": None, "gpu_sms": TOTAL_SMS, "date": datetime.date.today().isoformat(),
       "sm_split": {"training": TRAIN_SMS, "inference": INFER_SMS, "total": TOTAL_SMS,
                    "sm_check_train": open(f"{R}/sms_train.txt").read().strip() if os.path.exists(f"{R}/sms_train.txt") else None,
                    "sm_check_infer": open(f"{R}/sms_infer.txt").read().strip() if os.path.exists(f"{R}/sms_infer.txt") else None},
       "training": {}, "inference": {}}
for impl, pre in (("megatron_core", "megatron"), ("pytorch_eager", "pytorch")):
    full, mps = train_block(f"{pre}_full"), train_block(f"{pre}_mps")
    if full or mps: out["training"][impl] = {"full_gpu": full, "sm_restricted": mps}
    if full and not out["gpu"]: out["gpu"] = full["env"]["gpu"]
out["inference"]["vllm"] = {"full_gpu": serve_block("vllm_full"), "sm_restricted": serve_block("vllm_mps")}
tx = load(f"{R}/text.json"); out["inference"]["text_sample"] = tx
sw = f"{R}/serve_weights.txt"; out["inference"]["weights"] = open(sw).read().strip() if os.path.exists(sw) else None
# headline: Megatron if it worked, else PyTorch eager
hd = "megatron_core" if "megatron_core" in out["training"] and out["training"]["megatron_core"]["full_gpu"] else "pytorch_eager"
out["headline_impl"] = hd
tb = out["training"].get(hd, {})
tr = tb.get("sm_restricted") or tb.get("full_gpu")
v = out["inference"]["vllm"]; sv = v["sm_restricted"] or v["full_gpu"]
kl = {"fa_published_baseline": 514562, "fa_published_kvm": 1}
if tr and "launches" not in (tb.get("full_gpu") or {}): pass
fg = tb.get("full_gpu") or {}
if "launches" in fg:
    kl["training_per_step_kernels"] = fg["launches"]["kernels_per_step"]
    kl["training_4000_steps"] = fg["launches"]["kernels_per_step"] * STEPS
    if sv and "launches" in (sv or {}) or (v["full_gpu"] and "launches" in v["full_gpu"]):
        L_ = (v["full_gpu"] or sv)["launches"]; kl["serving_per_token_kernel_records"] = L_["kernel_records_per_token"]
        kl["serving_per_token_host_launch_calls"] = L_["host_cudaLaunchKernel_calls_per_token"] + L_["host_cudaGraphLaunch_calls_per_token"]
        if tr and sv:
            toks = tr["seconds"]["mean"] * sv["tok_s"]["mean"]
            kl["demo_equiv_serving_tokens"] = toks
            kl["demo_equiv_total_kernel_records"] = kl["training_4000_steps"] + toks * L_["kernel_records_per_token"]
            kl["demo_equiv_total_host_launch_calls"] = kl["training_4000_steps"] + toks * kl["serving_per_token_host_launch_calls"]
            kl["demo_equiv_basis"] = f"{hd} training seconds ({'SM-restricted' if tb.get('sm_restricted') else 'full GPU'}) x vLLM tok/s ({'SM-restricted' if v['sm_restricted'] else 'full GPU'})"
out["kernel_launches"] = kl
out["notes"] = ["Training launch count = kernel activity records (CUPTI via torch.profiler) per steady-state step x 4000; includes optimizer and data-gather kernels.",
                "vLLM runs with CUDA graphs (default); kernel records count kernels executed inside graphs, host launch calls count cudaLaunchKernel+cudaGraphLaunch.",
                "SM restriction via CUDA MPS CUDA_MPS_ACTIVE_THREAD_PERCENTAGE; actual SMs used verified by sm_check_* (MPS rounds to hardware granularity).",
                "GPU shared with other jobs; runs_not_idle_at_start counts runs where util stayed >=5% for the whole wait window."]
p = f"{here}/results/baseline_a100_{datetime.date.today():%Y%m%d}.json"
json.dump(out, open(p, "w"), indent=1); print("wrote", p)
