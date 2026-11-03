"""Aggregate results/runs_v2/*.json -> results/baseline_v2_a100_<date>.json (mean/sd/n), incl. launch counts under 3 definitions and whole-demo equivalents."""
import json, glob, os, datetime, statistics as st
from common import TOTAL_SMS, TRAIN_SMS, INFER_SMS, STEPS, B, T
here = os.path.dirname(os.path.abspath(__file__)); R = f"{here}/results/runs_v2"
FA = 514562
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def ms(v):
    v = [x for x in v if x is not None]
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v), "values": v} if v else None
def runs(prefix):
    out = []
    for p in sorted(glob.glob(f"{R}/{prefix}_r[0-9].json")):
        d = load(p); m = load(p.replace(".json", ".meta.json")) or {}
        if d: d["_noisy"] = m.get("noisy_not_idle_before_start"); d["_maxp"] = m.get("max_compute_procs_seen"); out.append(d)
    return out
def train_block(prefix):
    rs = runs(prefix)
    if not rs: return None
    g = lambda k: [r.get(k) for r in rs]
    return {"steps": rs[0]["steps"], "seconds": ms(g("seconds")), "tok_s": ms(g("tok_s")), "loss_initial": ms(g("loss_initial")),
            "loss_final": ms(g("loss_final")), "loss_final_mean_last25": ms(g("loss_final_mean_last25")), "peak_mem_gb": ms(g("peak_mem_gb")),
            "runs_not_idle_at_start": sum(1 for r in rs if r["_noisy"]), "max_compute_procs_seen_per_run": [r["_maxp"] for r in rs],
            "loss_curve_every25_rep1": rs[0]["loss_curve_every25"], "env": rs[0]["env"], "model_params": rs[0]["model_params"],
            "model_params_trainable": rs[0].get("model_params_trainable"), "impl": rs[0].get("impl"), "cuda_graph": rs[0].get("cuda_graph")}
def serve_block(prefix):
    rs = runs(prefix)
    if not rs: return None
    return {"new_tokens": rs[0]["new_tokens"], "tok_s": ms([x for r in rs for x in r["tok_s"]]), "mps_pct": rs[0]["mps_pct"],
            "runs_not_idle_at_start": sum(1 for r in rs if r["_noisy"]), "max_compute_procs_seen_per_run": [r["_maxp"] for r in rs]}
out = {"source": "Measured by llm/baseline/run_v2.sh (BASELINE v2). Each system timed individually, one job on the GPU at a time (shared GPU lock), no Doom.",
       "date": datetime.date.today().isoformat(), "gpu_sms": TOTAL_SMS,
       "sm_split": {"training": TRAIN_SMS, "inference": INFER_SMS, "total": TOTAL_SMS}, "training": {}, "inference": {"vllm": {}}, "launch_counts": {}}
systems = {"megatron_te": "megatron_te_{m}", "megatron_te_fwdbwd_cudagraph": "megatron_te_graph_{m}", "pytorch_eager": "pytorch_{m}",
           "pytorch_cuda_graph": "pytorch_graph_{m}", "megatron_te_flash_attn_n1": "megatron_te_flashattn_{m}"}
for sysname, pat in systems.items():
    for m in ("S", "F"):
        pre = pat.format(m=m)
        if sysname == "pytorch_eager" and m == "S": continue   # S eager: BASELINE v1 (results/baseline_a100_20261101.json)
        full, mps = train_block(f"{pre}_full"), train_block(f"{pre}_mps")
        if full or mps: out["training"].setdefault(sysname, {})[m] = {"full_gpu": full, "sm_restricted": mps}
for m in ("S", "F"):
    out["inference"]["vllm"][m] = {"full_gpu": serve_block(f"vllm_{m}_full"), "sm_restricted": serve_block(f"vllm_{m}_mps")}
sw = f"{R}/serve_weights.txt"; out["inference"]["weights"] = open(sw).read().strip() if os.path.exists(sw) else None
v1 = load(f"{here}/results/baseline_a100_20261101.json")
if v1: out["training"].setdefault("pytorch_eager", {})["S"] = {"full_gpu": v1["training"]["pytorch_eager"]["full_gpu"], "sm_restricted": v1["training"]["pytorch_eager"]["sm_restricted"], "from": "baseline_a100_20261101.json (BASELINE v1)"}
# ---- launch counts ----
def c3(name): 
    d = load(f"{R}/{name}.json"); return d.get("launch3") if d else None
def vl(m, cg):
    d = load(f"{R}/vllmc3_{m}_{cg}.json"); return d.get("launches") if d else None
lc = {}
for sysname, pre in (("megatron_te", "c3_megatron_te"), ("pytorch_eager", "c3_pytorch")):
    for m in ("S", "F"):
        d = c3(f"{pre}_{m}")
        if not d: continue
        g = d.get("graph") or d.get("graph_fwd_bwd_only")
        lc.setdefault(sysname, {})[m] = {
            "a_kernels_per_step": d["eager"]["kernels_per_unit"], "memops_per_step": d["eager"]["memops_per_unit"],
            "b_host_launch_calls_per_step": d["eager"]["host_launch_calls_per_unit"], "b_breakdown": d["eager"]["host_launch_breakdown_per_unit"],
            "c_host_launch_calls_per_step_one_graph": g["host_launch_calls_per_unit"], "c_breakdown": g["host_launch_breakdown_per_unit"],
            "c_kernels_per_step_inside_graph": g["kernels_per_unit"],
            "c_note": "whole step (data, LR, fwd, bwd, AdamW) in one CUDA graph" if "graph" in d else "fwd+bwd(+data+grad zero) in one CUDA graph; Megatron's optimizer (host-synchronising inf/nan check) stays eager",
            "top_kernels_per_step": d["eager"]["top_kernels_per_unit"]}
out["launch_counts"]["training"] = lc
vc = {}
for m in ("S", "F"):
    vd, ve, vf = vl(m, "default"), vl(m, "eager"), vl(m, "full")
    if vd: vc[m] = {"a_kernels_per_token": vd["kernel_records_per_token"], "b_host_launch_calls_per_token_default_vllm": vd["host_launch_calls_per_token"],
                    "b_breakdown_default": vd["host_launch_breakdown_per_token"],
                    "b_host_launch_calls_per_token_enforce_eager": ve["host_launch_calls_per_token"] if ve else None,
                    "c_host_launch_calls_per_token_full_cudagraph": vf["host_launch_calls_per_token"] if vf else None,
                    "c_breakdown": vf["host_launch_breakdown_per_token"] if vf else None,
                    "a_kernels_per_token_full_cudagraph": vf["kernel_records_per_token"] if vf else None}
out["launch_counts"]["serving"] = vc
# ---- whole-demo equivalents ----
demo = {"fa_published_baseline": FA, "fa_published_kvm": 1,
        "fa_arithmetic": "514,562 ~ 4000 steps x ~116 launches + ~50k served tokens x ~1 launch (CUDA graph)",
        "basis": "training seconds (SM-restricted, mean) x vLLM tok/s (SM-restricted, mean) = tokens served concurrently; launches = per_step x 4000 + per_token x tokens", "rows": []}
for sysname in ("megatron_te", "pytorch_eager"):
    for m in ("S", "F"):
        t = out["training"].get(sysname, {}).get(m); l = lc.get(sysname, {}).get(m); s = vc.get(m); sv = out["inference"]["vllm"][m]
        if not (t and l and s): continue
        tr = t["sm_restricted"] or t["full_gpu"]; se = sv["sm_restricted"] or sv["full_gpu"]
        if not (tr and se): continue
        FA_TOK = 46.1 * 1076   # tokens FA's megakernel served in its run (concurrent serving), = 49,604; also 70.1 s x 710 tok/s = 49,771 for their baseline
        row = {"system": sysname, "model": m, "train_seconds": tr["seconds"]["mean"], "serve_tok_s": se["tok_s"]["mean"],
               "per_step": {"a": l["a_kernels_per_step"], "b": l["b_host_launch_calls_per_step"], "c": l["c_host_launch_calls_per_step_one_graph"]},
               "per_token": {"a": s["a_kernels_per_token"], "b": s["b_host_launch_calls_per_token_default_vllm"],
                             "c": s["c_host_launch_calls_per_token_full_cudagraph"], "c_ideal_floor": 1}}
        for tag, toks in (("walltime_matched", tr["seconds"]["mean"] * se["tok_s"]["mean"]), ("fixed_fa_tokens_49604", FA_TOK)):
            a = l["a_kernels_per_step"] * STEPS + s["a_kernels_per_token"] * toks
            b = l["b_host_launch_calls_per_step"] * STEPS + s["b_host_launch_calls_per_token_default_vllm"] * toks
            c = l["c_host_launch_calls_per_step_one_graph"] * STEPS + (s["c_host_launch_calls_per_token_full_cudagraph"] or 0) * toks
            c_ideal = l["c_host_launch_calls_per_step_one_graph"] * STEPS + 1 * toks
            row[tag] = {"served_tokens": toks, "a_gpu_kernel_executions": a, "b_host_launch_calls": b, "c_host_launch_calls_graphed": c,
                        "c_with_ideal_1_launch_per_token": c_ideal, "training_only_a": l["a_kernels_per_step"] * STEPS,
                        "training_only_b": l["b_host_launch_calls_per_step"] * STEPS, "training_only_c": l["c_host_launch_calls_per_step_one_graph"] * STEPS,
                        "ratio_to_fa": {"a": a / FA, "b": b / FA, "c": c / FA, "c_ideal": c_ideal / FA}}
        demo["rows"].append(row)
out["launch_counts"]["demo_equivalent"] = demo
p = f"{here}/results/baseline_v2_a100_{datetime.date.today():%Y%m%d}.json"
json.dump(out, open(p, "w"), indent=1); print("wrote", p)
