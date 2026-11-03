"""Round-2 Modal driver (B300).  Generic: a plan JSON has  {"tag": str, "build": {name: shell_cmd}, "gpu": [[shell_cmd, timeout_s], ...], "gpu_budget_s": int}.
  modal run llm/b200/modal/app2.py --plan /path/plan.json
Build commands run on a CPU container (nvcc 13.0, sources under /src/llm/{b200,gemm,attention,elementwise,decode}); artifacts named in 'artifacts' are copied to the Volume
/vol/bin2/.  GPU commands run in ONE gpu container (one cold start) from /tmp/bin (copy of /vol/bin2 + /vol/bin); every command gets `timeout -s KILL`.
SPEND: every GPU function has an explicit timeout; the local entrypoint appends the measured GPU seconds to results/r2_spend.jsonl.  Rate $0.001736/GPU-s."""
import modal, os, subprocess, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
B200 = os.path.dirname(HERE)
LLM = os.path.dirname(B200)
GPU = os.environ.get("OFA_GPU", "B300")   # "B200+" landed on a real B200 (cc 10.0) in round 2 and on B300 (cc 10.3) in round 1: pin the model and record it
RATE = 0.001736
app = modal.App("ofa-b300-r2")
vol = modal.Volume.from_name("ofa-b300-data", create_if_missing=True)
V = "/vol"
IGN = ["results/**", "modal/**", "**/*.o", "**/*.buildlog", "**/__pycache__/**", "**/ofa_orig_reference.cu.txt", "**/t_*", "**/ofa_mma*", "**/ofa_tc*"]

base = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("curl", "git", "build-essential", "psmisc")
        .env({"CUDA_CACHE_PATH": f"{V}/cudacache"}))
src = base.add_local_dir(B200, "/src/b200", ignore=IGN)
for _d in ("gemm", "attention", "elementwise", "decode"):
    src = src.add_local_dir(f"{LLM}/{_d}", f"/src/{_d}", ignore=["**/*.f32", "**/*.u32", "**/*.bin", "**/__pycache__/**"])


def sh(c, tail=0, timeout=None):
    t = time.time()
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash", timeout=timeout)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout.decode() if e.stdout else "") + "\nTIMEOUT"
    o = f"$ {c}   [{time.time()-t:.1f}s rc={rc}]\n{out}"
    if tail: o = "\n".join(o.splitlines()[-tail:])
    return rc, o


@app.function(image=src, volumes={V: vol}, cpu=16, memory=32768, timeout=1500)
def cpu_build(jobs: dict, artifacts: dict):
    """jobs: name -> shell cmd (run in parallel, 4 at a time). artifacts: path (abs, in container) -> published name in /vol/bin2."""
    import concurrent.futures as cf, shutil
    sh("mkdir -p /src/llm && for d in b200 gemm attention elementwise decode; do ln -sfn /src/$d /src/llm/$d; done; mkdir -p /vol/bin2")
    res = {}
    def run(k):
        rc, o = sh(jobs[k], timeout=1300); return k, rc, o
    with cf.ThreadPoolExecutor(4) as ex:
        for k, rc, o in ex.map(run, list(jobs)): res[k] = {"rc": rc, "tail": o[-2500:]}
    for pth, name in artifacts.items():
        if os.path.exists(pth):
            shutil.copy(pth, f"/vol/bin2/{name}")
            rc, o = sh(f"cuobjdump -sass /vol/bin2/{name} 2>/dev/null | grep -c -E 'UTCHMMA|UTMALDG|UTCBAR|LDTM'; cuobjdump -lelf /vol/bin2/{name} | head -3; cuobjdump -lptx /vol/bin2/{name} | head -3")
            res[name] = {"published": True, "sass": o.strip()[-300:]}
        else:
            res[name] = {"published": False}
    vol.commit()
    return res


@app.function(gpu=GPU, image=src, volumes={V: vol}, timeout=1800)
def gpu_cmds(cmds: list, budget_s: int = 1500):
    t0 = time.time(); vol.reload(); outs = []
    sh("mkdir -p /tmp/bin && cp /vol/bin/* /tmp/bin/ 2>/dev/null; cp /vol/bin2/* /tmp/bin/ 2>/dev/null; chmod +x /tmp/bin/*")
    rc, o = sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm --format=csv,noheader"); outs.append(o); print(o, flush=True)
    for c, to in cmds:
        if time.time() - t0 + to > budget_s:
            outs.append(f"SKIPPED (budget): {c}"); continue
        rc, o = sh(f"cd /tmp && timeout -s KILL {to} bash -c {json.dumps(c)} 2>&1 | tail -400", timeout=to + 30)
        outs.append(o); print(o, flush=True)
    wall = time.time() - t0
    return {"wall_s": wall, "cost_usd": wall * RATE, "outs": outs}


@app.local_entrypoint()
def main(plan: str):
    P = json.load(open(plan)); tag = P["tag"]
    rd = os.path.normpath(os.path.join(B200, "..", "..", "b300-run")); os.makedirs(rd, exist_ok=True)
    if P.get("build"):
        r = cpu_build.remote(P["build"], P.get("artifacts", {}))
        txt = json.dumps(r, indent=1); open(os.path.join(rd, f"r2_{tag}_build.json"), "w").write(txt)
        for k, v in r.items():
            print("BUILD", k, v.get("rc", ""), (v.get("tail", "") or v.get("sass", ""))[-700:].replace("\\n", "\n"))
    if P.get("gpu"):
        g = gpu_cmds.remote(P["gpu"], P.get("gpu_budget_s", 1500))
        open(os.path.join(rd, f"r2_{tag}_gpu.txt"), "w").write("\n".join(g["outs"]))
        with open(os.path.join(rd, "r2_spend.jsonl"), "a") as f:
            f.write(json.dumps({"tag": tag, "gpu_s": g["wall_s"], "usd": g["cost_usd"], "t": time.time()}) + "\n")
        print("\n".join(g["outs"]))
        print(f"GPU wall {g['wall_s']:.1f}s = ${g['cost_usd']:.3f}")


# ---------------------------------------------------------------- baselines re-measured on the PINNED B300 (same image recipe as app.py -> cached layers; kit's one_run.sh harness)
def _install_bl():
    def run(c):
        print("$", c, flush=True); r = subprocess.run(c, shell=True, executable="/bin/bash"); print("rc", r.returncode, flush=True); return r.returncode
    run("pip install -U pip wheel setuptools packaging ninja psutil einops pybind11")
    run("pip install vllm")
    run("pip install tiktoken transformers numpy onnx onnxscript")
    run("pip install megatron-core")
    run("export NVTE_FRAMEWORK=pytorch MAX_JOBS=16; pip install --no-build-isolation 'transformer_engine[pytorch]' 2>&1 | tail -25")
    run("pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformer.engine|flash|megatron|nvidia-cudnn|triton|numpy|transformers) '")

bl_base = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
           .apt_install("curl", "git", "build-essential", "ninja-build", "psmisc", "numactl")
           .env({"CUDA_CACHE_PATH": f"{V}/cudacache", "CUDA_CACHE_MAXSIZE": "4294967296", "CUDA_HOME": "/usr/local/cuda"})
           .run_function(_install_bl, cpu=16, memory=32768, timeout=3600))
bl_img = bl_base.add_local_dir(B200, "/src/b200", ignore=["results/**", "modal/**", "**/*.o", "**/*.buildlog", "**/__pycache__/**", "**/ofa_orig_reference.cu.txt"])

def _baseline_driver(jobs, tag):
    os.environ.update({"DATA_DIR": f"{V}/data", "GPU_TAG": "b300", "TOTAL_SMS": "148", "TRAIN_SMS": "116", "INFER_SMS": "24",
                       "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "1", "HOME": "/root"})
    os.makedirs("/root/data", exist_ok=True)
    bd = "/src/b200/baseline"; out = f"{V}/bl_{tag}"; os.makedirs(out, exist_ok=True)
    rc, o = sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"); print(o, flush=True)
    res = {"gpu": o}
    for name, pct, cmd in jobs:
        t = time.time()
        rc, o = sh(f"cd {bd} && IDLE_WAIT_S=60 ./one_run.sh {name} {pct} bash -c \"{cmd}\"")
        print(o[-1500:], flush=True)
        res[name] = {"rc": rc, "wall_s": time.time() - t, "summary": o.strip().splitlines()[-1][:300] if o.strip() else ""}
        sh(f"mkdir -p {out} && cp -r {bd}/results/runs_v2/* {out}/ 2>/dev/null; true")
        vol.commit()
    return res

@app.function(gpu=GPU, image=bl_img, volumes={V: vol}, timeout=1700)
def bl_r2_a():
    t0 = time.time(); vol.reload(); T = "python"; jobs = []
    for i in (1, 2):
        jobs += [(f"pytorch_S_mps_r{i}", 79, f"MODEL=S {T} train_pytorch.py --steps 4000 --out \\$OUT --tag mps116"),
                 (f"pytorch_S_full_r{i}", 0, f"MODEL=S {T} train_pytorch.py --steps 4000 --out \\$OUT --tag full")]
    res = _baseline_driver(jobs, "r2a"); res["wall_s"] = time.time() - t0; res["cost_usd_est"] = res["wall_s"] * RATE
    json.dump(res, open(f"{V}/bl_r2a_summary.json", "w"), indent=1); vol.commit(); return res

@app.function(gpu=GPU, image=bl_img, volumes={V: vol}, timeout=1700)
def bl_r2_b():
    t0 = time.time(); vol.reload(); T = "python"
    jobs = [("pytorch_F_mps_r1", 79, f"MODEL=F {T} train_pytorch.py --steps 4000 --out \\$OUT --tag mps116"),
            ("pytorch_F_full_r1", 0, f"MODEL=F {T} train_pytorch.py --steps 4000 --out \\$OUT --tag full"),
            ("megatron_local_S_mps_r1", 79, f"MODEL=S {T} megatron_train.py --impl local --steps 4000 --out \\$OUT --tag mps116")]
    res = _baseline_driver(jobs, "r2b"); res["wall_s"] = time.time() - t0; res["cost_usd_est"] = res["wall_s"] * RATE
    json.dump(res, open(f"{V}/bl_r2b_summary.json", "w"), indent=1); vol.commit(); return res

@app.local_entrypoint()
def baselines(which: str = "a"):
    f = bl_r2_a if which == "a" else bl_r2_b
    g = f.remote()
    rd = os.path.normpath(os.path.join(B200, "..", "..", "b300-run")); json.dump(g, open(os.path.join(rd, f"r2_baselines_{which}.json"), "w"), indent=1)
    with open(os.path.join(rd, "r2_spend.jsonl"), "a") as fh: fh.write(json.dumps({"tag": "baselines_" + which, "gpu_s": g["wall_s"], "usd": g["cost_usd_est"], "t": time.time()}) + "\n")
    print(json.dumps(g, indent=1)[:3000])
