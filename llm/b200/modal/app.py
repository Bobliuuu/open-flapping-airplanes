"""Modal driver for the B300 (cc10.3) experiments.  gpu="B200+" landed on a B300 SXM6 (sm_103).  Everything here is labelled B300.
Run:  modal run llm/b200/modal/app.py::probe     (and ::prep_data, ::build_mk, ...)   Results land in the Volume 'ofa-b300-data' and are fetched by fetch.sh.
SPEND: B200 rate 0.001736 $/s; every GPU function has an explicit timeout."""
import modal, os, subprocess, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
B200 = os.path.dirname(HERE)                       # llm/b200
GPU = "B200+"
RATE = 0.001736
app = modal.App("ofa-b300")
vol = modal.Volume.from_name("ofa-b300-data", create_if_missing=True)
V = "/vol"
IGN = ["results/**", "modal/**", "**/*.o", "**/*.buildlog", "**/__pycache__/**", "**/ofa_orig_reference.cu.txt"]

base = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("curl", "git", "build-essential", "ninja-build", "psmisc")
        .env({"CUDA_CACHE_PATH": f"{V}/cudacache", "CUDA_CACHE_MAXSIZE": "4294967296"}))
LLM = os.path.dirname(B200)
src = base.add_local_dir(B200, "/src/b200", ignore=IGN)
for _d in ("gemm", "attention", "elementwise", "decode"):
    src = src.add_local_dir(f"{LLM}/{_d}", f"/src/{_d}", ignore=["**/*.f32", "**/*.u32", "**/*.bin", "**/__pycache__/**"])

def sh(c, check=False, tail=0):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash")
    o = f"$ {c}   [{time.time()-t:.1f}s rc={r.returncode}]\n{r.stdout}{r.stderr}"
    if tail: o = "\n".join(o.splitlines()[-tail:])
    print(o, flush=True)
    if check and r.returncode: raise RuntimeError(o)
    return r.returncode, r.stdout + r.stderr

PROBE_CU = r'''
#include <cstdio>
#include <cuda_runtime.h>
__global__ void cp(const float4* a, float4* b, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; for(;i<n;i+=s) b[i]=a[i]; }
__global__ void sm(unsigned* m){ if(threadIdx.x==0){unsigned id; asm volatile("mov.u32 %0, %%smid;":"=r"(id)); atomicOr(&m[id>>5],1u<<(id&31)); long long t=clock64(); while(clock64()-t<20000){} } }
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0); int clk=0,mclk=0,bus=0; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,0); cudaDeviceGetAttribute(&mclk,cudaDevAttrMemoryClockRate,0); cudaDeviceGetAttribute(&bus,cudaDevAttrGlobalMemoryBusWidth,0);
  printf("name=%s cc=%d.%d SMs=%d smemPerBlockOptin=%zu smemPerSM=%zu regsPerSM=%d L2=%d MB clk=%d MHz memclk=%d MHz bus=%d mem=%.1f GB maxThrPerSM=%d\n",p.name,p.major,p.minor,p.multiProcessorCount,p.sharedMemPerBlockOptin,p.sharedMemPerMultiprocessor,p.regsPerMultiprocessor,p.l2CacheSize>>20,clk/1000,mclk/1000,bus,p.totalGlobalMem/1e9,p.maxThreadsPerMultiProcessor);
  unsigned* m; cudaMallocManaged(&m,64*4); for(int i=0;i<64;i++)m[i]=0; sm<<<p.multiProcessorCount*8,64>>>(m); cudaDeviceSynchronize(); int n=0,mx=0; for(int i=0;i<64;i++){n+=__builtin_popcount(m[i]); for(int b=0;b<32;b++) if(m[i]>>b&1) mx=i*32+b;}
  printf("smid: distinct=%d max=%d\n",n,mx);
  size_t N=(size_t)1<<27; float4 *a,*b; cudaMalloc(&a,N*16); cudaMalloc(&b,N*16); cudaMemset(a,1,N*16);
  cp<<<p.multiProcessorCount*8,512>>>(a,b,N); cudaDeviceSynchronize(); cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  for(int i=0;i<10;i++) cp<<<p.multiProcessorCount*8,512>>>(a,b,N); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms,e0,e1);
  printf("copy_bw=%.0f GB/s (read+write, %.1f GB moved)\n", 10.0*2*N*16/ms/1e6, 10.0*2*N*16/1e9);
  cudaError_t er=cudaGetLastError(); printf("lasterr=%s\n",cudaGetErrorString(er)); return er!=cudaSuccess;
}
'''

@app.function(gpu=GPU, image=src, volumes={V: vol}, timeout=900)
def probe():
    t0 = time.time(); out = {}
    sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv")
    sh("nvcc --version | tail -2; which nvcc")
    os.makedirs("/tmp/p", exist_ok=True); open("/tmp/p/p.cu", "w").write(PROBE_CU)
    cands = {"sm_100a": "-arch=sm_100a", "sm_100f": "-arch=sm_100f", "sm_103a": "-arch=sm_103a", "sm_103f": "-arch=sm_103f",
             "ptx_compute_100": "-gencode arch=compute_100,code=compute_100", "ptx_compute_100a": "-gencode arch=compute_100a,code=compute_100a",
             "sm_80_cubin": "-arch=sm_80", "ptx_compute_80": "-gencode arch=compute_80,code=compute_80", "sm_90a": "-arch=sm_90a"}
    for k, fl in cands.items():
        rc, o = sh(f"cd /tmp/p && nvcc {fl} -O2 -o p_{k} p.cu 2>&1 | tail -3")
        if os.path.exists(f"/tmp/p/p_{k}"):
            rc, o = sh(f"cd /tmp/p && timeout 120 ./p_{k}")
            out[k] = {"runs": rc == 0, "out": o[-600:]}
        else:
            out[k] = {"compiles": False, "out": o[-200:]}
    work = [k for k, v in out.items() if v.get("runs")]
    print("RUNS ON THIS GPU:", work)
    # MPS availability + SM calibration scan with the first working native arch
    arch = next((a for a in ["sm_103a", "sm_103f", "sm_100f"] if a in work), None)
    if arch:
        sh(f"cd {B200_SRC}/baseline && nvcc -arch={arch} -O2 -o /tmp/chk check_sms.cu")
        sh("/tmp/chk")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/mps/pipe"; os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/mps/log"
        os.makedirs("/tmp/mps/pipe", exist_ok=True); os.makedirs("/tmp/mps/log", exist_ok=True)
        rc, o = sh("nvidia-cuda-mps-control -d; sleep 2; ps aux | grep -c [m]ps-control; tail -5 /tmp/mps/log/control.log")
        scan = {}
        for pct in [8, 10, 12, 14, 16, 18, 20, 25, 50, 70, 74, 76, 78, 79, 80, 82, 85, 90, 100]:
            rc, o = sh(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={pct} timeout 60 /tmp/chk")
            scan[pct] = o.strip().splitlines()[-1] if o.strip() else f"rc={rc}"
        sh("echo quit | nvidia-cuda-mps-control; sleep 1")
        out["mps_scan"] = scan
    out["wall_s"] = time.time() - t0; out["cost_usd_est"] = out["wall_s"] * RATE
    json.dump(out, open(f"{V}/probe.json", "w"), indent=1); vol.commit()
    return out


# ---------------------------------------------------------------- data prep (CPU only, cached in the Volume)
prep_img = (modal.Image.debian_slim(python_version="3.11").apt_install("curl")
            .pip_install("numpy", "tiktoken", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
            .add_local_dir(B200, "/src/b200", ignore=IGN))

@app.function(image=prep_img, volumes={V: vol}, cpu=16, memory=32768, timeout=2400)
def prep_data():
    D = f"{V}/data"; os.makedirs(D, exist_ok=True)
    HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
    if not os.path.exists(f"{D}/TinyStories-valid.txt") or os.path.getsize(f"{D}/TinyStories-valid.txt") < 1e7:
        sh(f"curl -sL -o {D}/TinyStories-valid.txt {HF}/TinyStories-valid.txt", check=True)
    if not os.path.exists(f"{D}/TinyStories-train-prefix.txt") or os.path.getsize(f"{D}/TinyStories-train-prefix.txt") < 3e8:
        sh(f"curl -sL -r 0-314572799 -o {D}/TinyStories-train-prefix.txt {HF}/TinyStories-train.txt", check=True)
    sh(f"ls -la {D}")
    if not os.path.exists(f"{D}/tinystories_cl100k_train.u32"):
        # prep_data.py reads ~/data
        sh(f"mkdir -p ~/data && ln -sfn {D}/* ~/data/ ; cd /src/b200/baseline && python prep_data.py", check=True)
        sh(f"ls -la ~/data; cp -n ~/data/*.u32 {D}/ 2>/dev/null; ls -la {D}")
    import numpy as np
    ntok = os.path.getsize(f"{D}/tinystories_cl100k_train.u32") // 4
    st = np.random.default_rng(1234).integers(0, ntok - 512 - 1, size=(4000, 8)).astype(np.uint32); st.tofile(f"{D}/starts_4000.u32")
    sh(f"cd /src/b200/megakernel && python make_vocab.py {D}/cl100k_vocab.bin {D}/prompt_ids.txt 'Once upon a time'", check=True)
    # flat init in ofa layout from the deterministic SPEC init (common.init_weights, seed 1234); same order as make_init_flat.py
    os.environ["DATA_DIR"] = D; os.environ["MODEL"] = "S"; import sys; sys.path.insert(0, "/src/b200/baseline")
    import common, torch
    w = common.init_weights()
    parts = [w["wte"], w["wpe"]]
    for l in range(common.L):
        p = f"L{l:02d}_"; parts += [w[p+"gamma1"], w[p+"Wqkv"], w[p+"Wo"], w[p+"gamma2"], w[p+"Wup"], w[p+"Wdown"]]
    parts.append(w["gamma_f"])
    flat = torch.cat([x.reshape(-1) for x in parts]).numpy().astype("float32"); flat.tofile(f"{D}/init_flat_S.f32")
    print("init_flat_S params", flat.size, "ntok", ntok)
    vol.commit(); sh(f"ls -la {D}")
    return {"train_tokens": int(ntok), "init_params": int(flat.size)}

B200_SRC = "/src/b200"

@app.local_entrypoint()
def main(fn: str = "probe"):
    r = globals()[fn].remote()
    print(json.dumps(r, indent=1) if isinstance(r, (dict, list)) else r)


# ---------------------------------------------------------------- builds (CPU only; nvcc 13.0 knows sm_103a; results into the Volume)
@app.function(image=src, volumes={V: vol}, cpu=8, memory=16384, timeout=2400)
def build_mk():
    import shutil
    os.makedirs(f"{V}/bin", exist_ok=True)
    # relative includes (../../gemm, ../../attention, ...) resolve if b200/ sits next to gemm/, attention/, ... : link /src/b200 into /src/llm
    sh("mkdir -p /src/llm && for d in b200 gemm attention elementwise decode; do ln -sfn /src/$d /src/llm/$d; done; ls -la /src/llm")
    mk = "/src/llm/b200/megakernel"; gt = "/src/llm/b200/gemm_tcgen05"
    jobs = {
      "ofa_mma_100a": f"cd {mk} && NODOOM=1 ./build.sh sm_100a ofa_mma_100a",
      "ofa_mma_103a": f"cd {mk} && NODOOM=1 ./build.sh sm_103a ofa_mma_103a",
      "ofa_mma6_100a": f"cd {mk} && NODOOM=1 ./build.sh sm_100a ofa_mma6_100a -DGEMM_STAGES=6",
      "ofa_tc_100a": f"cd {mk} && NODOOM=1 ./build.sh sm_100a ofa_tc_100a -DOFA_GEMM_BACKEND=1",
      "t_mma": f"cd {gt} && nvcc -arch=sm_100a -O3 -std=c++17 -DOFA_GEMM_BACKEND=0 -Xptxas -v -o t_mma test_gemm_sm100.cu -lcublas",
      "t_nt": f"cd {gt} && nvcc -arch=sm_100a -O3 -std=c++17 -DOFA_GEMM_BACKEND=1 -DGEMM100_LAYOUT_MASK=2 -o t_nt test_gemm_sm100.cu -lcublas",
      "t_nn": f"cd {gt} && nvcc -arch=sm_100a -O3 -std=c++17 -DOFA_GEMM_BACKEND=1 -DGEMM100_LAYOUT_MASK=3 -o t_nn test_gemm_sm100.cu -lcublas",
      "t_all": f"cd {gt} && nvcc -arch=sm_100a -O3 -std=c++17 -DOFA_GEMM_BACKEND=1 -o t_all test_gemm_sm100.cu -lcublas",
      "check_sms": "cd /src/b200/baseline && nvcc -arch=sm_100a -O2 -o check_sms check_sms.cu",
    }
    import concurrent.futures as cf
    res = {}
    def run(k):
        rc, o = sh(jobs[k]); return k, rc, o
    with cf.ThreadPoolExecutor(5) as ex:
        for k, rc, o in ex.map(run, list(jobs)): res[k] = {"rc": rc, "tail": o[-1500:]}
    for k, d in [("ofa_mma_100a", mk), ("ofa_mma_103a", mk), ("ofa_mma6_100a", mk), ("ofa_tc_100a", mk),
                 ("t_mma", gt), ("t_nt", gt), ("t_nn", gt), ("t_all", gt), ("check_sms", "/src/b200/baseline")]:
        if os.path.exists(f"{d}/{k}"): shutil.copy(f"{d}/{k}", f"{V}/bin/{k}")
        res[k]["built"] = os.path.exists(f"{d}/{k}")
    vol.commit(); return res


# ---------------------------------------------------------------- megakernel runs on the B300 (Doom OFF: stubs; its 8 SMs are reserved but idle)
@app.function(gpu=GPU, image=src, volumes={V: vol}, timeout=1500)
def run_mk(steps: int = 4000):
    t0 = time.time(); R = {}
    vol.reload()
    sh("mkdir -p /tmp/bin && cp /vol/bin/* /tmp/bin/ && chmod +x /tmp/bin/*; nvidia-smi --query-gpu=name,compute_cap,utilization.gpu --format=csv")
    D = f"{V}/data"
    base = f"--vocab {D}/cl100k_vocab.bin --prompt {D}/prompt_ids.txt --prompt-text 'Once upon a time' --data {D}/tinystories_cl100k_train.u32 --init {D}/init_flat_S.f32 --starts {D}/starts_4000.u32 --no-doom"
    def go(name, cmd, to=600):
        rc, o = sh(f"cd /tmp && timeout {to} {cmd} 2>&1 | tail -150")
        R[name] = {"rc": rc, "out": o[-6000:]}; open(f"{V}/mk_{name}.txt", "w").write(o)
        return o
    # (a) 20-step correctness: loss file vs A100 build (step0 11.710649, step19 8.821497)
    go("loss20_100a", f"/tmp/bin/ofa_mma_100a --steps 20 --data {D}/tinystories_cl100k_train.u32 --init {D}/init_flat_S.f32 --starts {D}/starts_4000.u32 --no-doom --loss-out /tmp/l20.txt", 120)
    sh("head -2 /tmp/l20.txt; tail -2 /tmp/l20.txt")
    R["loss20_file"] = open("/tmp/l20.txt").read() if os.path.exists("/tmp/l20.txt") else None
    # (b) FA split 116/24/8 (Doom idle), 4000 steps: training + live decode
    go("split_100a", f"/tmp/bin/ofa_mma_100a --split b200 {base} --steps {steps} --loss-out /tmp/l_split.txt", 900)
    # (c) training only, all 148 SMs
    go("trainonly_100a", f"/tmp/bin/ofa_mma_100a --data {D}/tinystories_cl100k_train.u32 --init {D}/init_flat_S.f32 --starts {D}/starts_4000.u32 --no-doom --steps {steps} --loss-out /tmp/l_to.txt", 900)
    # (d) GEMM_STAGES=6 and native sm_103a, 116/24/8 split, short (600 steps) for step-time comparison
    go("split_mma6_100a", f"/tmp/bin/ofa_mma6_100a --split b200 {base} --steps 600", 300)
    go("split_103a", f"/tmp/bin/ofa_mma_103a --split b200 {base} --steps 600", 300)
    R["wall_s"] = time.time() - t0; R["cost_usd_est"] = R["wall_s"] * RATE
    json.dump(R, open(f"{V}/mk_results.json", "w"), indent=1); vol.commit()
    return {k: (v if not isinstance(v, dict) else {"rc": v["rc"], "out_tail": v["out"][-1800:]}) for k, v in R.items()}


# ---------------------------------------------------------------- baseline image: vLLM first (brings torch cu130), then megatron-core + Transformer Engine (A100 box order)
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
bl_img = bl_base.add_local_dir(B200, "/src/b200", ignore=IGN)

@app.function(image=bl_img, cpu=2, timeout=300)
def bl_info():
    sh("pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformer.engine|transformer_engine|flash|megatron|nvidia-cudnn|triton|numpy|transformers) '")
    sh("python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())'")
    return "ok"


@app.function(image=bl_img, cpu=16, memory=32768, timeout=3000)
def te_debug():
    # find the real compile error of transformer_engine_torch, then retry with cuDNN headers from pip
    sh("pip list 2>/dev/null | grep -i -E 'nvidia|transformer'; ls /usr/local/cuda/include | grep -i cudnn | head -3; find / -name 'cudnn.h' 2>/dev/null | head -3")
    rc, o = sh("export NVTE_FRAMEWORK=pytorch MAX_JOBS=16; pip install --no-build-isolation -v 'transformer_engine[pytorch]==2.19.0' 2>&1 | grep -E 'error|Error|fatal|No such' | head -15")
    rc, o2 = sh("pip install nvidia-cudnn-cu13 2>&1 | tail -2; python - <<'PY'\nimport nvidia, os, glob\nprint(glob.glob(os.path.dirname(nvidia.__path__[0]) + '/nvidia/*cudnn*'), glob.glob(nvidia.__path__[0] + '/cudnn*'))\nPY")
    return o + o2


# ---------------------------------------------------------------- baselines on the B300 (FA method: each job ALONE on the GPU, MPS for the SM-restricted shares)
def _baseline_driver(jobs, tag):
    """jobs: list of (name, mps_pct, shell_cmd_using_$OUT).  Uses the kit's one_run.sh (idle-wait, MPS start/stop, proc sampler)."""
    os.environ.update({"DATA_DIR": f"{V}/data", "GPU_TAG": "b300", "TOTAL_SMS": "148", "TRAIN_SMS": "116", "INFER_SMS": "24",
                       "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "1", "HOME": "/root"})
    os.makedirs("/root/data", exist_ok=True)
    bd = "/src/b200/baseline"; out = f"{V}/bl_{tag}"; os.makedirs(out, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; pip list 2>/dev/null | grep -i -E '^(torch|vllm|transformer.engine|megatron|flash)'")
    res = {}
    for name, pct, cmd in jobs:
        t = time.time()
        rc, o = sh(f"cd {bd} && IDLE_WAIT_S=60 ./one_run.sh {name} {pct} bash -c \"{cmd}\"")
        res[name] = {"rc": rc, "wall_s": time.time() - t, "summary": o.strip().splitlines()[-1][:300] if o.strip() else ""}
        sh(f"mkdir -p {out} && cp -r {bd}/results/runs_v2/* {out}/ 2>/dev/null; true")
        vol.commit()
    return res

@app.function(gpu=GPU, image=bl_img, volumes={V: vol}, timeout=2700)
def bl_gpu1(mps_train: int = 79, mps_infer: int = 17):
    t0 = time.time(); vol.reload()
    os.makedirs("/tmp/mps_unused", exist_ok=True)
    T = "python"
    jobs = []
    for i in (1, 2):
        sv = "--save /root/data/ckpt_pytorch_step4000.pt" if i == 1 else ""
        jobs += [(f"pytorch_S_full_r{i}", 0, f"MODEL=S {T} train_pytorch.py --steps 4000 --out \\$OUT --tag full {sv}"),
                 (f"pytorch_S_mps_r{i}", mps_train, f"MODEL=S {T} train_pytorch.py --steps 4000 --out \\$OUT --tag mps116")]
    jobs += [("pytorch_graph_S_full_r1", 0, f"MODEL=S {T} train_pytorch.py --cuda_graph --steps 4000 --out \\$OUT --tag full"),
             ("megatron_local_S_full_r1", 0, f"MODEL=S {T} megatron_train.py --impl local --steps 4000 --out \\$OUT --tag full"),
             ("megatron_local_S_mps_r1", mps_train, f"MODEL=S {T} megatron_train.py --impl local --steps 4000 --out \\$OUT --tag mps116")]
    res = _baseline_driver(jobs, "g1")
    # vLLM decode (batch 1) full and restricted; real trained weights if the eager run saved them
    sh("cd /src/b200/baseline && ls /root/data; (test -f /root/data/ckpt_pytorch_step4000.pt && MODEL=S python make_hf.py /root/data/hf_trained /root/data/ckpt_pytorch_step4000.pt || MODEL=S python make_hf.py /root/data/hf_trained) 2>&1 | tail -3")
    sv = []
    for i in (1, 2):
        sv += [(f"vllm_S_full_r{i}", 0, f"MODEL=S python serve_vllm.py /root/data/hf_trained --reps 1 --out \\$OUT"),
               (f"vllm_S_mps_r{i}", mps_infer, f"MODEL=S python serve_vllm.py /root/data/hf_trained --reps 1 --out \\$OUT")]
    res.update(_baseline_driver(sv, "g1"))
    res["wall_s"] = time.time() - t0; res["cost_usd_est"] = res["wall_s"] * RATE
    json.dump(res, open(f"{V}/bl_g1_summary.json", "w"), indent=1); vol.commit()
    return res


# ---------------------------------------------------------------- TE layer: the sdist needs cudnn.h (pip nvidia-cudnn-cu13 ships it under site-packages/nvidia/cudnn)
def _install_te():
    cud = "/usr/local/lib/python3.11/site-packages/nvidia/cudnn"
    env = dict(os.environ, NVTE_FRAMEWORK="pytorch", MAX_JOBS="16", CUDNN_PATH=cud, CPATH=f"{cud}/include", CPLUS_INCLUDE_PATH=f"{cud}/include",
               LIBRARY_PATH=f"{cud}/lib", LD_LIBRARY_PATH=f"{cud}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run("pip install --no-build-isolation 'transformer_engine[pytorch]==2.19.0' 2>&1 | tail -15", shell=True, env=env, executable="/bin/bash")
    subprocess.run("pip list 2>/dev/null | grep -i -E 'transformer'", shell=True)
    subprocess.run("python -c \"import transformer_engine, transformer_engine.pytorch as te; print('TE import ok', te.__file__)\" 2>&1 | tail -3", shell=True, env=env, executable="/bin/bash")

te_base = bl_base.run_function(_install_te, cpu=16, memory=32768, timeout=3000)
te_img = te_base.add_local_dir(B200, "/src/b200", ignore=IGN)

@app.function(image=te_img, cpu=2, timeout=300)
def te_info():
    sh("pip list 2>/dev/null | grep -i -E 'transformer|flash|torch |megatron'")
    return "ok"


@app.function(image=bl_img, cpu=16, memory=32768, timeout=3000)
def te_debug2():
    cud = "/usr/local/lib/python3.11/site-packages/nvidia/cudnn"
    e = f"export NVTE_FRAMEWORK=pytorch MAX_JOBS=16 CUDNN_PATH={cud} CPATH={cud}/include CPLUS_INCLUDE_PATH={cud}/include LIBRARY_PATH={cud}/lib; "
    rc, o = sh(e + "pip install --no-build-isolation -v 'transformer_engine[pytorch]==2.19.0' 2>&1 | grep -E 'error|Error|fatal|No such|FAILED' | sort | uniq -c | sort -rn | head -15")
    return o


# ---------------------------------------------------------------- stand-alone GEMM tests (mma.sync control, then tcgen05 NT -> NN -> all); every run under timeout -s KILL
@app.function(gpu=GPU, image=src, volumes={V: vol}, timeout=1800)
def gemm_test():
    t0 = time.time(); vol.reload(); R = {}
    sh("mkdir -p /tmp/bin && cp /vol/bin/t_* /tmp/bin/ && chmod +x /tmp/bin/*; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    def go(name, cmd, to):
        rc, o = sh(f"cd /tmp && timeout -s KILL {to} {cmd} 2>&1 | tail -80")
        R[name] = o[-5000:]; open(f"{V}/gemm_{name}.txt", "w").write(o); vol.commit()
        return o
    go("t_mma", "/tmp/bin/t_mma", 400)
    o = go("t_nt_logits", "/tmp/bin/t_nt --only logits --nobench", 90)
    if "hang" not in o:
        go("t_nt_all", "/tmp/bin/t_nt --nobench", 180)
        go("t_nn_all", "/tmp/bin/t_nn --nobench", 180)
        go("t_all_nobench", "/tmp/bin/t_all --nobench", 180)
        go("t_all_bench", "/tmp/bin/t_all", 400)
    R["wall_s"] = time.time() - t0; R["cost_usd_est"] = R["wall_s"] * RATE
    json.dump(R, open(f"{V}/gemm_results.json", "w"), indent=1); vol.commit()
    return {k: (v if not isinstance(v, str) else v[-1200:]) for k, v in R.items()}
