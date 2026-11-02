#!/usr/bin/env python3
"""Builds video/b200/data.json for the demo video from the repository's raw results. Every number the video shows comes from here, with
the file it was read from; nothing is typed into the page by hand.

  python3 video/b200/make_data.py [--build triad_tc2] [--recording video/out/b200_llm_dashboard_triad_tc2]

Sources (relative to the repository root):
  Triad:      b200-run/results/megakernel_<tc2|tc3>_fa/split_r{1,2,3}.txt (3 full runs; kernel time, sampled tok/s, Doom fps, loss)
  Recording:  b200-run/results/video_<build>/b200_llm_dashboard_<build>.json / .log / _loss.txt, written by
              `modal run b200-run/modal_app.py --step video --binary <build>` (the MP4 itself stays on the Modal Volume and in video/out/)
  Training:   b200-run/results/runs_v2/{megatron_te,pytorch}_S_mps_r{1,2,3}.json (each alone on 116 SMs, eager), and the CUDA-graph
              variants in runs_v3_graph/ and runs_v3_graph_cpu16/ when present; per system the faster container's mean is used
  Serving:    b200-run/results/runs_v2_cpu16/vllmfa_S_mps_r{1,2,3}.json (vLLM alone on 24 SMs, FA's prompt, 16 CPU cores)
  Launches:   b200-run/results/baseline_v2_b200_20261102.json (launch_counts, measured with torch.profiler / CUPTI)
  FA:         benchmarks/fa_published.json (read from their video; not measured by us)
"""
import argparse, json, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "b200-run" / "results"


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def triad_runs(build):
    d = R / f"megakernel_{build.replace('triad_', '')}_fa"
    runs = []
    for i in (1, 2, 3):
        o = (d / f"split_r{i}.txt").read_text()
        runs.append({
            "kernel_s": float(re.search(r"kernel time ([\d.]+) s \(event\)", o).group(1)),
            "serve_tok_s": int(re.search(r"-> (\d+) tok/s sampled", o).group(1)),
            "sampled_tokens": int(re.search(r"(\d+) sampled tokens", o).group(1)),
            "answers": int(re.search(r"decode answers: (\d+) answers", o).group(1)),
            "doom_fps": float(re.search(r"([\d.]+) fps over the run", o).group(1)),
            "loss_final": float(re.search(r"s3999 ([\d.]+)", o).group(1)),
            "phases": int(re.search(r"training graph: (\d+) phases", o).group(1)),
            "tasks": int(re.search(r"(\d+) tasks per step", o).group(1)),
            "decode_phases": int(re.search(r"decode job: (\d+) phases", o).group(1)),
            "decode_tasks": int(re.search(r"(\d+) tasks per token", o).group(1)),
        })
    loss = [(int(a), float(b)) for a, b in (l.split() for l in (d / "loss_split_r1.txt").read_text().split("\n") if l.strip())]
    return d, runs, loss


def json_runs(sub, stem, reps=3):
    out = []
    for i in range(1, reps + 1):
        f = R / sub / f"{stem}_r{i}.json"
        if f.exists():
            out.append((f, json.loads(f.read_text())))
    return out


def train_system(stems_by_sub, label, launches_per_step, kernels_per_step, mode, short, full):
    """Mean of the 3 runs in each container; the faster container counts (it favours the baseline)."""
    best = None
    for sub, stem in stems_by_sub:
        rs = json_runs(sub, stem)
        if len(rs) < 3:
            continue
        secs = [r["seconds"] for _, r in rs]
        cand = {"label": label, "short": short, "full": full, "mode": mode, "train_s": [round(x, 2) for x in secs], "train_s_mean": round(st.mean(secs), 2),
                "loss_final_mean": round(st.mean(r["loss_final"] for _, r in rs), 4),
                "loss_curve": sorted((int(k), round(v, 4)) for k, v in rs[0][1]["loss_curve_every25"].items()),
                "launches_per_step": launches_per_step, "kernels_per_step": kernels_per_step,
                "source": f"b200-run/results/{sub}/{stem}_r{{1,2,3}}.json", "containers": []}
        if best is None or cand["train_s_mean"] < best["train_s_mean"]:
            if best is not None:
                cand["containers"] = best["containers"]
            best = cand
        best["containers"].append({"results": sub, "train_s_mean": round(st.mean(secs), 2)})
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="triad_tc2")
    ap.add_argument("--recording", default=None, help="path of the recording's files without extension (default: b200-run/results/video_<build>/...)")
    a = ap.parse_args()
    rec_stem = Path(a.recording) if a.recording else R / f"video_{a.build}" / f"b200_llm_dashboard_{a.build}"   # the recorded run's json/log/loss

    tdir, truns, tloss = triad_runs(a.build)
    lc = json.loads((R / "baseline_v2_b200_20261102.json").read_text())["launch_counts"]
    meg_lc, pt_lc, vl_lc = lc["training"]["megatron_te"]["S"], lc["training"]["pytorch_eager"]["S"], lc["serving"]["S"]
    fa = json.loads((ROOT / "benchmarks" / "fa_published.json").read_text())

    triad = {
        "build": a.build,
        "label": {"triad_tc2": "Triad (tcgen05 GEMMs, 2 stages)", "triad_tc3": "Triad (tcgen05 GEMMs, 3 stages)"}.get(a.build, a.build),
        "train_s": [r["kernel_s"] for r in truns], "train_s_mean": round(st.mean(r["kernel_s"] for r in truns), 2),
        "serve_tok_s": [r["serve_tok_s"] for r in truns], "serve_tok_s_mean": round(st.mean(r["serve_tok_s"] for r in truns)),
        "sampled_tokens_mean": round(st.mean(r["sampled_tokens"] for r in truns)),
        "answers_mean": round(st.mean(r["answers"] for r in truns)),
        "doom_fps": min(r["doom_fps"] for r in truns), "loss_final": truns[0]["loss_final"],
        "phases_per_step": truns[0]["phases"], "tasks_per_step": truns[0]["tasks"],
        "decode_phases": truns[0]["decode_phases"], "decode_tasks_per_token": truns[0]["decode_tasks"],
        "loss_curve": [p for p in tloss if p[0] % 25 == 0 or p[0] == 3999], "launches": 1,
        "source": rel(tdir) + "/split_r{1,2,3}.txt",
    }

    trains = {
        "megatron_te": train_system([("runs_v2", "megatron_te_S_mps"), ("runs_v2_cpu16", "megatron_te_S_mps")],
                                    "Megatron-core + TE", meg_lc["b_host_launch_calls_per_step"], meg_lc["a_kernels_per_step"],
                                    "eager (the frameworks' default)", "eager", "Megatron-core 0.16.1 + Transformer Engine 2.20"),
        "megatron_te_graph": train_system([("runs_v3_graph", "megatron_te_graph_S_mps"), ("runs_v3_graph_cpu16", "megatron_te_graph_S_mps")],
                                          "Megatron-core + TE", meg_lc["c_host_launch_calls_per_step_one_graph"],
                                          meg_lc["c_kernels_per_step_inside_graph"], "forward and backward in one CUDA graph, optimizer eager",
                                          "CUDA graphs", "Megatron-core 0.16.1 + Transformer Engine 2.20"),
        "pytorch": train_system([("runs_v2", "pytorch_S_mps"), ("runs_v2_cpu16", "pytorch_S_mps")],
                                "PyTorch + flash attention", pt_lc["b_host_launch_calls_per_step"], pt_lc["a_kernels_per_step"],
                                "eager (the frameworks' default)", "eager", "PyTorch 2.15 (NGC 26.10), flash attention"),
        "pytorch_graph": train_system([("runs_v3_graph", "pytorch_graph_S_mps"), ("runs_v3_graph_cpu16", "pytorch_graph_S_mps")],
                                      "PyTorch + flash attention", pt_lc["c_host_launch_calls_per_step_one_graph"],
                                      pt_lc["c_kernels_per_step_inside_graph"], "the whole step in one CUDA graph", "CUDA graph",
                                      "PyTorch 2.15 (NGC 26.10), flash attention"),
    }
    trains = {k: v for k, v in trains.items() if v}

    vl = json_runs("runs_v2_cpu16", "vllmfa_S_mps")
    vtok = [r["tok_s"][0] for _, r in vl]
    text = json.loads((R / "runs_v2" / "vllm_S_text.json").read_text())
    vllm = {"label": "vLLM 0.33.0 (CUDA graphs, 16 CPU cores)", "tok_s": [round(x) for x in vtok], "tok_s_mean": round(st.mean(vtok)),
            "launches_per_token": vl_lc["b_host_launch_calls_per_token_default_vllm"], "kernels_per_token": vl_lc["a_kernels_per_token"],
            "sample_text": text["text_temp0.8"], "source": "b200-run/results/runs_v2_cpu16/vllmfa_S_mps_r{1,2,3}.json"}

    rec = None
    if rec_stem.with_suffix(".json").exists():
        rj = json.loads(rec_stem.with_suffix(".json").read_text())
        logf = Path(str(rec_stem) + ".log")
        log = logf.read_text() if logf.exists() else ""      # *.log is not committed; the JSON's "run" block carries the numbers
        lossf = Path(str(rec_stem) + "_loss.txt")
        rloss = [(int(x), float(y)) for x, y in (l.split() for l in lossf.read_text().split("\n") if l.strip())] if lossf.exists() else tloss
        def grab(pat, conv, key=None):
            if key and (rj.get("run") or {}).get(key) is not None:
                return rj["run"][key]
            m = re.search(pat, log)
            return conv(m.group(1)) if m else None
        rec = dict(rj, file=rel(rec_stem.with_suffix(".mp4")),
                   kernel_s=grab(r"kernel time ([\d.]+) s \(event\)", float, "kernel_s"),
                   serve_tok_s=grab(r"-> (\d+) tok/s sampled", int, "serve_tok_s"),   # missing when the log was cut before the run's summary
                   doom_fps=grab(r"([\d.]+) fps over the run", float, "doom_fps"),
                   loss_final=grab(r"s3999 ([\d.]+)", float, "loss_final"),
                   loss_curve=[p for p in rloss if p[0] % 10 == 0 or p[0] == 3999])
        if rec["serve_tok_s"] is None:
            rec["serve_tok_s_note"] = "not in the recorded run's log (cut before its summary); the race uses the 3-run mean"
            rec["serve_tok_s"] = triad["serve_tok_s_mean"]

    # Comparisons, FA's metric: 1 - baseline rate / Triad rate (for times: 1 - Triad time / baseline time)
    def slower(t_triad, t_base):
        return round(100 * (1 - t_triad / t_base), 1)
    best_train = min(trains.values(), key=lambda v: v["train_s_mean"])
    meg = trains["megatron_te"]
    same_tokens = triad["sampled_tokens_mean"]
    derived = {
        "train_vs_megatron_pct": slower(triad["train_s_mean"], meg["train_s_mean"]),
        "train_vs_best_pct": slower(triad["train_s_mean"], best_train["train_s_mean"]),
        "best_train_key": [k for k, v in trains.items() if v is best_train][0],
        "train_vs_each_pct": {k: slower(triad["train_s_mean"], v["train_s_mean"]) for k, v in trains.items()},
        "serve_vs_vllm_pct": round(100 * (1 - vllm["tok_s_mean"] / triad["serve_tok_s_mean"]), 1),
        "same_work": f"4,000 training steps + the {same_tokens:,} tokens Triad served during them",
        "same_work_tokens": same_tokens,
        "launches_megatron_vllm_host": round(4000 * meg["launches_per_step"] + same_tokens * vllm["launches_per_token"]),
        "launches_megatron_graph_vllm_host": (round(4000 * trains["megatron_te_graph"]["launches_per_step"] + same_tokens * vllm["launches_per_token"])
                                              if "megatron_te_graph" in trains else None),
        "launches_megatron_vllm_kernels": round(4000 * meg["kernels_per_step"] + same_tokens * vllm["kernels_per_token"]),
        "fa_train_pct": fa["training"]["baseline_slower_pct"], "fa_serve_pct": fa["inference"]["baseline_slower_pct"],
    }
    # The code scene shows the persistent kernel's entry exactly as in the source (trailing comments trimmed) and its one launch.
    src = (ROOT / "llm" / "b200" / "megakernel" / "triad.cu").read_text()
    m = re.search(r"^__global__ void __launch_bounds__\(256, TRIAD_BLOCKS_PER_SM\) triad_kernel.*?^}", src, re.S | re.M)
    kernel = [re.sub(r"\s+//.*$", "", l).rstrip() for l in m.group(0).split("\n")]
    launch = re.search(r"^.*triad_kernel<<<.*$", src, re.M).group(0).strip()
    launch = re.search(r"triad_kernel<<<[^;]*;", launch).group(0)
    code = {"file": "llm/b200/megakernel/triad.cu", "kernel": kernel, "launch": launch}
    other = {"triad_tc2": "triad_tc3", "triad_tc3": "triad_tc2"}.get(a.build)
    alt = None
    if other and (R / f"megakernel_{other.replace('triad_', '')}_fa").exists():
        _, oruns, _ = triad_runs(other)
        alt = {"build": other, "train_s_mean": round(st.mean(r["kernel_s"] for r in oruns), 2),
               "serve_tok_s_mean": round(st.mean(r["serve_tok_s"] for r in oruns))}
    data = {"gpu": "NVIDIA B200 on Modal (gpu=\"B200\", 148 SMs)", "split": fa["sm_split"], "code": code, "other_build": alt,
            "model": {"params": "162.4M", "desc": "GPT, 12 layers, d 768, cl100k vocabulary (100,352)", "data": "TinyStories",
                      "tokens_per_step": 4096, "steps": 4000, "prompt": "What is a megakernel?", "answer_tokens": 256, "temperature": 0.8},
            "triad": triad, "recording": rec, "training_baselines": trains, "vllm": vllm, "fa": fa, "derived": derived,
            "generated_by": "video/b200/make_data.py"}
    out = Path(__file__).resolve().parent / "data.json"
    out.write_text(json.dumps(data, indent=1) + "\n")
    (out.parent / "data.js").write_text("window.__DATA = " + json.dumps(data) + ";\n")   # for opening demo.html directly in a browser
    print(f"wrote {rel(out)}: Triad {triad['train_s_mean']} s / {triad['serve_tok_s_mean']} tok/s; "
          + "; ".join(f"{k} {v['train_s_mean']} s ({derived['train_vs_each_pct'][k]:+.1f}%)" for k, v in trains.items())
          + f"; vLLM {vllm['tok_s_mean']} tok/s ({derived['serve_vs_vllm_pct']:+.1f}%); launches {derived['launches_megatron_vllm_host']:,} host / "
          f"{derived['launches_megatron_vllm_kernels']:,} kernels; recording {'yes' if rec else 'missing'}")


if __name__ == "__main__":
    main()
