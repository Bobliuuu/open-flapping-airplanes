"""vLLM single-stream decode benchmark. Run with VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in-process).
  python serve_vllm.py HF_DIR --out r.json [--reps 3] [--new 256] [--text] [--count_launches]
SM restriction is applied externally (MPS env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)."""
import os, sys, time, json, argparse
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import tiktoken
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
ap = argparse.ArgumentParser(); ap.add_argument("hf"); ap.add_argument("--out"); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--new", type=int, default=256); ap.add_argument("--text", action="store_true")
ap.add_argument("--prompt", default="Once upon a time", help="timed prompt (FA's demo serves 'What is a megakernel?')")
ap.add_argument("--count_launches", action="store_true"); ap.add_argument("--eager", action="store_true")
ap.add_argument("--cg", default="default", choices=["default", "eager", "full"], help="default = vLLM defaults; eager = enforce_eager; full = FULL cudagraph mode (decode step = ONE graph)")
a = ap.parse_args()
enc = tiktoken.get_encoding("cl100k_base")
llm = LLM(a.hf, skip_tokenizer_init=True, dtype="bfloat16", max_model_len=512, kv_cache_memory_bytes=2 * 2**30, gpu_memory_utilization=0.2,
          enforce_eager=a.eager or a.cg == "eager", max_num_seqs=1, seed=1234,
          **({"compilation_config": {"cudagraph_mode": "FULL"}} if a.cg == "full" else {}))
res = {"cg": a.cg, "mps_pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "enforce_eager": a.eager}
def gen(ids, n, **kw):
    sp = SamplingParams(temperature=kw.get("temperature", 0.0), max_tokens=n, min_tokens=n if kw.get("fixed", True) else 0,
                        ignore_eos=kw.get("fixed", True), seed=1234)
    return llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].outputs[0]
if a.text:
    ids = enc.encode("What is a megakernel?")
    o = gen(ids, 200, fixed=False)
    res["text_prompt"] = "What is a megakernel?"
    res["text_greedy"] = enc.decode([t for t in o.token_ids if t < 100277] )
    o = gen(ids, 200, fixed=False, temperature=0.8)
    res["text_temp0.8"] = enc.decode([t for t in o.token_ids if t < 100277])
    print(res["text_greedy"]); print("----"); print(res["text_temp0.8"])
else:
    ids = enc.encode(a.prompt)  # a few prompt tokens + 256 new <= 512
    res["prompt"] = a.prompt
    gen(ids, a.new); gen(ids, a.new)  # warm-up (cuda graphs, autotune)
    ts = []
    for _ in range(a.reps):
        t0 = time.time(); o = gen(ids, a.new); dt = time.time() - t0
        assert len(o.token_ids) == a.new
        ts.append(dt)
    res.update(new_tokens=a.new, prompt_tokens=len(ids), seconds=ts, tok_s=[a.new / t for t in ts])
    print(json.dumps(res))
    if a.count_launches:
        import torch
        from torch.profiler import profile, ProfilerActivity
        from launchcount import trace_counts
        cs = {}
        for n in (32, 64):   # steady-state per-token count = (64-token run - 32-token run) / 32, removes prefill/setup kernels
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                gen(ids, n); torch.cuda.synchronize()
            cs[n] = trace_counts(prof, 1)
        d = lambda k: (cs[64][k] - cs[32][k]) / 32
        bd = {k: (cs[64]["host_launch_breakdown_per_unit"].get(k, 0) - cs[32]["host_launch_breakdown_per_unit"].get(k, 0)) / 32
              for k in cs[64]["host_launch_breakdown_per_unit"]}
        res["launches"] = {"method": "torch.profiler chrome trace (CUPTI); per-token = (64-new-token generate minus 32-new-token generate)/32, steady-state decode only",
                           "cg_mode": a.cg, "kernel_records_per_token": d("kernels_per_unit"), "memops_per_token": d("memops_per_unit"),
                           "host_launch_calls_per_token": d("host_launch_calls_per_unit"), "host_launch_breakdown_per_token": bd,
                           "top_kernels": cs[64]["top_kernels_per_unit"], "whole_64tok_run": {k: cs[64][k] for k in ("kernels_per_unit", "host_launch_calls_per_unit")}}
        print(json.dumps(res["launches"], indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
