"""Triad at LLM scale against Megatron and vLLM on ONE exact NVIDIA B200 (Modal gpu="B200"), with Flapping Airplanes' protocol.

The megakernel runs training + serving + Doom in one launch on a static 116 / 24 / 8 SM split (llm/b200/megakernel, --split b200).
Each baseline runs alone on the same GPU type, limited with CUDA MPS to the same share (116 SMs for training, 24 for serving; on a B200,
79% and 17% give exactly those counts). Both sides use the same model (162.4M GPT "S", llm/SPEC.md), data (TinyStories in cl100k tokens),
initial weights, batch order and AdamW schedule, so their loss curves can be compared. The baseline scripts are llm/b200/baseline; run names
follow its aggregate_v2.py, which, with fa_compare.py, writes the summary.

  modal run b200-run/modal_app.py --step prep      # CPU: TinyStories -> cl100k tokens, initial weights, batch starts, vocab -> Volume
  modal run b200-run/modal_app.py --step build     # CPU: build the megakernel image (prints ptxas registers / spills)
  modal run b200-run/modal_app.py --step env       # GPU, ~2 min: torch, Transformer Engine, Megatron-core and vLLM on the B200
  modal run b200-run/modal_app.py --step mk        # megakernel: 20-step loss check, then 4,000 steps x reps with split + Doom + serving
  modal run b200-run/modal_app.py --step gemm      # the tcgen05 GEMM backend's test (mma.sync control first); --tag picks results/<tag>
  modal run b200-run/modal_app.py --step tccmp     # the tcgen05 megakernel (2 and 3 stages) on FA's split with FA's prompt, two B200s
  modal run b200-run/modal_app.py --step train     # Megatron-core (TE and local) and PyTorch eager: 4,000 steps each, alone
  modal run b200-run/modal_app.py --step train_graphs   # the same two with CUDA graphs (--cuda_graph), default and 16-core containers
  modal run b200-run/modal_app.py --step serve     # vLLM batch-1 decode of the trained weights, alone
  modal run b200-run/modal_app.py --step mk --serving fa --tag megakernel_fa     # serving FA's prompt, asked again every 256 tokens
  modal run b200-run/modal_app.py --step serve --prompt 'What is a megakernel?' --sub runs_v2_fa   # vLLM on FA's prompt
  modal run b200-run/modal_app.py --step video     # screen recording of a full run's dashboard (FA's prompt, --binary triad_tc2) -> video/out/
  modal run b200-run/modal_app.py --step report    # aggregate, then copy the Volume's results/ to b200-run/results/
"""
import json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = "B200"                                         # exactly a B200: Modal's "B200+" may place the job on a B300
V, DATA, RES = "/vol", "/vol/data", "/vol/results"
TRAIN_PCT, INFER_PCT = 79, 17                        # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -> 116 / 24 SMs on a B200 (benchmarks/results/b200_20261101_mps_baselines.json)
FA_PROMPT, ANSWER_LEN = "What is a megakernel?", 256  # FA's demo serves answers to this prompt; vLLM's timed runs generate 256 tokens
GPU_INFO = ("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader;"
            " echo \"cpus=$(nproc)$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)\"")
CPUS = 16.0                                          # reserved host cores: the framework baselines are CPU-bound at batch 1 / small model
WAD_URL, WAD_MD5 = "https://github.com/Akbar30Bill/DOOM_wads/raw/master/doom1.wad", "f0cefca49926d00903cf57551d901abe"
SKIP = ["doomgeneric", "obj", "src_gpu", "**/*.o", "**/*.ptx", "**/*.bc", "**/*.ll", "**/*.cubin", "**/*.wad", "demo", "doomtest",
        "player/build_cpu", "**/*.so", "**/__pycache__"]
LLM_SKIP = ["**/results/**", "**/__pycache__/**", "**/*.o", "**/*.buildlog", "**/*.png", "**/triad_orig_reference.cu.txt"]
MK = "/opt/triad/llm/b200/megakernel"

app = modal.App("triad-b200-fair")
vol = modal.Volume.from_name("triad-b200-data", create_if_missing=True)

# Megakernel: the Doom engine is compiled to PTX by doom-gpu/build_doom.sh (clang NVPTX), re-assembled for sm_100a, linked into triad.cu.
# The first layers are the same as benchmarks/run_modal.py, so Modal reuses them.
mk_img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("clang", "llvm", "lld", "git", "curl", "ca-certificates")
    .run_commands(
        "git clone -q https://github.com/ozkl/doomgeneric.git /opt/src/doomgeneric && git -C /opt/src/doomgeneric checkout -q e257a09"
        " && git -C /opt/src/doomgeneric rev-parse --short HEAD > /opt/src/doomgeneric.rev",     # pinned: the player was checked on it
        f"curl -fsSL -m 120 -o /opt/src/doom1.wad {WAD_URL} && echo '{WAD_MD5}  /opt/src/doom1.wad' | md5sum -c -",
    )
    .add_local_dir(ROOT / "doom-gpu", "/opt/triad/doom-gpu", copy=True, ignore=SKIP)
    .run_commands(
        "ln -sfn /opt/src/doomgeneric /opt/triad/doom-gpu/doomgeneric && cp /opt/src/doom1.wad /opt/triad/doom-gpu/doom1.wad",
        "cd /opt/triad/doom-gpu && bash build_doom.sh",
    )
    .add_local_dir(ROOT / "llm", "/opt/triad/llm", copy=True, ignore=LLM_SKIP)
    .run_commands(f"cd {MK} && bash build_doom_b200.sh sm_100a /opt/triad/doom-gpu && MODEL=S bash build.sh sm_100a triad_s")
)
# The tcgen05 GEMM backend's own test (llm/b200/gemm_tcgen05): mma.sync control, then tcgen05 for NT, NT+NN, all layouts.
gemm_img = mk_img.run_commands(
    "cd /opt/triad/llm/b200/gemm_tcgen05 && F='-arch=sm_100a -O3 -std=c++17'"
    " && nvcc $F -DTRIAD_GEMM_BACKEND=0 -o t_mma test_gemm_sm100.cu -lcublas"
    " && nvcc $F -DTRIAD_GEMM_BACKEND=1 -DGEMM100_LAYOUT_MASK=2 -o t_nt test_gemm_sm100.cu -lcublas"
    " && nvcc $F -DTRIAD_GEMM_BACKEND=1 -DGEMM100_LAYOUT_MASK=3 -o t_nn test_gemm_sm100.cu -lcublas"
    " && nvcc $F -DTRIAD_GEMM_BACKEND=1 -o t_all test_gemm_sm100.cu -lcublas"
)
# The megakernel with the tcgen05 GEMM backend (every GEMM layout), 2 blocks per SM: triad_tc2 with 2 pipeline stages (66,432 B of shared
# memory per block, the same L1 as triad_s), triad_tc3 with 3 (99,200 B; faster training, less L1 for the serving SMs).
tc_img = mk_img.run_commands(
    f"cd {MK} && MODEL=S bash build.sh sm_100a triad_tc2 -DTRIAD_GEMM_BACKEND=1 -DGEMM100_STAGES=2"
    " && MODEL=S bash build.sh sm_100a triad_tc3 -DTRIAD_GEMM_BACKEND=1 -DGEMM100_STAGES=3"
)
# --step video: the megakernel's dashboard in Google Chrome on a virtual X display, recorded by ffmpeg. The page asks for macOS monospace
# fonts; on Linux the generic monospace family is used, which fontconfig points at JetBrains Mono.
FONTS_CONF = ('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><alias><family>monospace</family>'
              '<prefer><family>JetBrains Mono</family></prefer></alias></fontconfig>')
vid_img = tc_img.apt_install("xvfb", "ffmpeg", "wget", "fonts-dejavu-core", "fonts-jetbrains-mono").run_commands(
    "wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
    " && apt-get update -q && apt-get install -y -q /tmp/chrome.deb && rm /tmp/chrome.deb && google-chrome --version",
    f"echo '{FONTS_CONF}' > /etc/fonts/local.conf && fc-cache -f && fc-match monospace",
)
# Training baselines: NVIDIA's NGC PyTorch container (torch + Transformer Engine + cuDNN for Blackwell); megatron-core pinned to the
# version llm/baseline used, without dependencies so the container's torch and TE stay.
train_img = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.10-py3")
    .run_commands("pip install --no-deps megatron-core==0.16.1 && pip install tiktoken")
    .add_local_dir(ROOT / "llm" / "b200" / "baseline", "/src/bl", copy=True, ignore=LLM_SKIP)
)
# Serving baseline: vLLM from PyPI. vllm 0.33.0 pins torch 2.14.0, built for CUDA 13.0, so the base toolkit is 13.0 too (FlashInfer may
# compile kernels at start-up with it). Modal cannot read the Python version of the vllm/vllm-openai image, hence this route.
serve_img = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm==0.33.0", "tiktoken")
    .add_local_dir(ROOT / "llm" / "b200" / "baseline", "/src/bl", copy=True, ignore=LLM_SKIP)
)
prep_img = (
    modal.Image.debian_slim(python_version="3.12").apt_install("curl")
    .pip_install("numpy", "tiktoken")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(ROOT / "llm" / "b200", "/src/b200", copy=True, ignore=LLM_SKIP)
    .add_local_file(ROOT / "benchmarks" / "fa_published.json", "/src/fa_published.json", copy=True)
)


def sh(cmd, cwd=None, env=None):
    """Run a shell command, echo its output into the Modal log as it arrives, return (exit code, output)."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace")
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return p.wait(), "".join(lines)


@app.function(image=prep_img, volumes={V: vol}, cpu=16, memory=32768, timeout=3600)
def prep():
    """TinyStories (valid + first 300 MB of train) -> cl100k uint32, batch starts, cl100k vocab + prompt ids, initial weights in triad's layout.
    Same recipe as llm/baseline and llm/b200/modal/app.py, so the tokens and initial loss match the A100 runs."""
    import numpy as np

    os.makedirs(DATA, exist_ok=True)
    hf = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
    if not os.path.exists(f"{DATA}/TinyStories-valid.txt") or os.path.getsize(f"{DATA}/TinyStories-valid.txt") < 1e7:
        sh(f"curl -sL -o {DATA}/TinyStories-valid.txt {hf}/TinyStories-valid.txt")
    if not os.path.exists(f"{DATA}/TinyStories-train-prefix.txt") or os.path.getsize(f"{DATA}/TinyStories-train-prefix.txt") < 3e8:
        sh(f"curl -sL -r 0-314572799 -o {DATA}/TinyStories-train-prefix.txt {hf}/TinyStories-train.txt")
    if not os.path.exists(f"{DATA}/tinystories_cl100k_train.u32"):
        rc, _ = sh(f"mkdir -p ~/data && ln -sfn {DATA}/TinyStories-valid.txt {DATA}/TinyStories-train-prefix.txt ~/data/"
                   f" && cd /src/b200/baseline && python prep_data.py && cp ~/data/*.u32 {DATA}/")
        assert rc == 0, "prep_data.py failed"
    ntok = os.path.getsize(f"{DATA}/tinystories_cl100k_train.u32") // 4
    np.random.default_rng(1234).integers(0, ntok - 512 - 1, size=(4000, 8)).astype(np.uint32).tofile(f"{DATA}/starts_4000.u32")
    rc, _ = sh(f"cd /src/b200/megakernel && python make_vocab.py {DATA}/cl100k_vocab.bin {DATA}/prompt_ids.txt 'Once upon a time'"
               f" && python make_vocab.py /tmp/vocab_unused.bin {DATA}/prompt_ids_fa.txt '{FA_PROMPT}'")
    assert rc == 0, "make_vocab.py failed"
    os.environ.update(DATA_DIR=DATA, MODEL="S")
    sys.path.insert(0, "/src/b200/baseline")
    import common
    import torch

    w = common.init_weights()                         # same order as llm/b200/megakernel/make_init_flat.py
    parts = [w["wte"], w["wpe"]]
    for layer in range(common.L):
        p = f"L{layer:02d}_"
        parts += [w[p + "gamma1"], w[p + "Wqkv"], w[p + "Wo"], w[p + "gamma2"], w[p + "Wup"], w[p + "Wdown"]]
    parts.append(w["gamma_f"])
    flat = torch.cat([x.reshape(-1) for x in parts]).numpy().astype("float32")
    flat.tofile(f"{DATA}/init_flat_S.f32")
    vol.commit()
    sh(f"ls -la {DATA}")
    return {"train_tokens": int(ntok), "valid_tokens": os.path.getsize(f"{DATA}/tinystories_cl100k_valid.u32") // 4, "init_params": int(flat.size)}


@app.function(image=mk_img, timeout=600)
def build_info():
    """Builds the megakernel image (CPU) and reports the binary and its ptxas line."""
    sh(f"ls -la {MK}/triad_s {MK}/doom_dev_sm_100a.o; grep -E 'triad_kernel|registers|stack frame|spill' {MK}/triad_s.buildlog | head -8")
    return "ok"


@app.function(gpu=GPU, image=train_img, timeout=900)
def env_train():
    sh(GPU_INFO)
    rc, out = sh("cd /src/bl && python - <<'PY'\n"
                 "import te_env, torch\n"
                 "print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name())\n"
                 "import transformer_engine as te, transformer_engine.pytorch as tep\n"
                 "print('TE', te.__version__)\n"
                 "import megatron.core as m\n"
                 "print('megatron-core', m.__version__)\n"
                 "x = tep.Linear(768, 768, params_dtype=torch.bfloat16).cuda()\n"
                 "print('TE linear ok', x(torch.randn(8, 768, device='cuda', dtype=torch.bfloat16)).shape)\n"
                 "from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec\n"
                 "print('TE layer spec ok', type(get_gpt_layer_with_transformer_engine_spec(use_te_activation_func=True)).__name__)\n"
                 "PY")
    return out


@app.function(gpu=GPU, image=serve_img, timeout=900)
def env_serve():
    sh(GPU_INFO)
    rc, out = sh("python3 -c \"import vllm, torch, tiktoken, transformers; "
                 "print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())\"")
    return out


BASE_ARGS = f"--data {DATA}/tinystories_cl100k_train.u32 --init {DATA}/init_flat_S.f32 --starts {DATA}/starts_4000.u32"


def serve_args(serve):
    """story: 'Once upon a time' fed once, then continuous generation (the kit's default). fa: FA's prompt, asked again after every
    ANSWER_LEN sampled tokens from the live weights (--answer-len), like vLLM answering one request after another."""
    if serve == "fa":
        return f"--prompt {DATA}/prompt_ids_fa.txt --prompt-text '{FA_PROMPT}' --answer-len {ANSWER_LEN}"
    return f"--prompt {DATA}/prompt_ids.txt --prompt-text 'Once upon a time'"


def decode_answers(tokens_path, vocab_path, out_path, picks=(0, 0.25, 0.5, 0.75, 1.0)):
    """Turns a --text-out token dump of an answer-mode run into text: the answers at the given fractions of the run."""
    import struct
    raw = Path(vocab_path).read_bytes()
    n = struct.unpack_from("<I", raw, 0)[0]
    offs = struct.unpack_from(f"<{n + 1}I", raw, 4)
    blob = raw[4 + 4 * (n + 1):]
    ids = [int(x) for x in Path(tokens_path).read_text().split()]
    answers, cur = [], None
    for t in ids:
        if t == 1 << 24:                               # decode::OUT_RESTART: a new answer starts
            cur = []
            answers.append(cur)
        elif cur is not None and 0 <= t < n:
            cur.append(blob[offs[t]:offs[t + 1]])
    lines = [f"{len(answers)} answers to {FA_PROMPT!r}, {ANSWER_LEN} sampled tokens each, from the live weights during training\n"]
    for f in picks if answers else ():
        k = min(len(answers) - 1, int(f * (len(answers) - 1)))
        lines.append(f"--- answer {k + 1} of {len(answers)} ({f:.0%} into the run)\n{b''.join(answers[k]).decode('utf-8', 'replace')}\n")
    Path(out_path).write_text("\n".join(lines))


@app.function(gpu=GPU, image=tc_img, volumes={V: vol}, timeout=5400)
def mk(reps: int = 3, steps: int = 4000, binary: str = "triad_s", tag: str = "megakernel", trainonly: bool = True, serve: str = "story"):
    """The megakernel on one B200: a 20-step loss check (no Doom), then `reps` runs of training + serving + Doom on FA's split, then
    training alone on all 148 SMs for reference. binary: triad_s (mma.sync GEMMs), triad_tc2 or triad_tc3 (tcgen05). serve: story | fa."""
    vol.reload()
    out = f"{RES}/{tag}"
    os.makedirs(out, exist_ok=True)
    full = f"--split b200 --wad /opt/triad/doom-gpu/doom1.wad --vocab {DATA}/cl100k_vocab.bin {serve_args(serve)} {BASE_ARGS}"
    runs = [("loss20", f"./{binary} --steps 20 {BASE_ARGS} --no-doom --loss-out {out}/loss20_steps.txt", 300)]
    for i in range(1, reps + 1):
        dump = f" --text-out /tmp/tokens_r{i}.txt" if serve == "fa" and i == 1 else ""
        runs.append((f"split_r{i}", f"./{binary} {full} --steps {steps} --loss-out {out}/loss_split_r{i}.txt{dump}", 1200))
    if trainonly:
        runs.append(("trainonly", f"./{binary} {BASE_ARGS} --no-doom --steps {steps} --loss-out {out}/loss_trainonly.txt", 1200))
    keep = ("SM partition", "GPU ", "model:", "steady state", "loss:", "decode:", "decode answers:", "doom:", "SM utilisation", "kernel time")
    res = {"gpu": sh(GPU_INFO)[1].strip()}
    for name, cmd, limit in runs:
        rc, o = sh(f"timeout -k 10 {limit} {cmd}", cwd=MK)
        Path(f"{out}/{name}.txt").write_text(o)
        vol.commit()
        res[name] = {"rc": rc, "lines": [line for line in o.splitlines() if line.startswith(keep)]}
    if os.path.exists("/tmp/tokens_r1.txt"):
        decode_answers("/tmp/tokens_r1.txt", f"{DATA}/cl100k_vocab.bin", f"{out}/answers_split_r1.txt")
        res["answers"] = Path(f"{out}/answers_split_r1.txt").read_text()[:3000]
    Path(f"{out}/summary.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


@app.function(gpu=GPU, image=vid_img, volumes={V: vol}, cpu=8.0, memory=32768, timeout=2400)
def video(steps: int = 4000, width: int = 1920, height: int = 1080, scale: float = 4 / 3, fps: int = 30, binary: str = "triad_tc2") -> dict:
    """Screen recording of the megakernel's dashboard for a full run on FA's split: the training loss, answers to FA's prompt from the live
    weights, Doom. Google Chrome shows the page on a virtual X display and ffmpeg records it in real time; the dashboard is served
    --start-delay seconds before the launch, so the recording starts at step 0. The MP4 goes to the Volume's video/; stills come back."""
    import signal
    import urllib.request

    vol.reload()
    gpu, env, delay = sh(GPU_INFO)[1].strip(), dict(os.environ, DISPLAY=":99"), 15.0
    procs = [subprocess.Popen(["Xvfb", ":99", "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"])]
    cmd = (f"./{binary} --split b200 --wad /opt/triad/doom-gpu/doom1.wad --vocab {DATA}/cl100k_vocab.bin {serve_args('fa')} {BASE_ARGS}"
           f" --steps {steps} --port 8080 --start-delay {delay} --loss-out /tmp/video_loss.txt")
    demo = subprocess.Popen(f"timeout -k 5 {int(delay + 900)} stdbuf -oL -eL {cmd}", shell=True, executable="/bin/bash", cwd=MK,   # line-buffered: the log keeps the run's summary
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    procs.append(demo)
    log = []

    def pump():
        for line in demo.stdout:
            print(line, end="", flush=True)
            log.append(line)

    def state():
        try:
            return json.loads(urllib.request.urlopen("http://127.0.0.1:8080/state", timeout=5).read())
        except Exception:
            return None

    def wait_for(cond, limit):
        end = time.time() + limit
        while time.time() < end and demo.poll() is None:
            s = state()
            if s is not None and cond(s):
                return s
            time.sleep(0.05)
        return None

    threading.Thread(target=pump, daemon=True).start()
    started = final = None
    try:
        if wait_for(lambda s: True, 300) is None:
            return {"gpu": gpu, "log": "".join(log)[-3000:], "error": "the dashboard never came up"}
        procs.append(subprocess.Popen(
            ["google-chrome", "--app=http://127.0.0.1:8080/", "--kiosk", "--window-position=0,0",
             f"--window-size={round(width / scale)},{round(height / scale)}",      # in CSS pixels: Chrome multiplies by the scale factor
             f"--force-device-scale-factor={scale}", "--no-sandbox", "--test-type", "--no-first-run", "--no-default-browser-check",
             "--password-store=basic", "--disable-gpu", "--disable-dev-shm-usage", "--hide-scrollbars", "--user-data-dir=/tmp/chrome",
             "--disable-features=Translate,MediaRouter"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(6)                                  # Chrome start-up and first paint, inside the start delay
        ff = subprocess.Popen(["ffmpeg", "-y", "-hide_banner", "-f", "x11grab", "-video_size", f"{width}x{height}", "-framerate", str(fps),
                               "-draw_mouse", "0", "-i", ":99.0", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "/tmp/raw.mkv"],
                              env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        t_rec = time.time()
        started = wait_for(lambda s: s["phase"] >= 1, delay + 120)
        t_start = time.time()
        final = wait_for(lambda s: s["phase"] >= 3, 900)
        t_end = time.time()
        time.sleep(4)
        ff.send_signal(signal.SIGINT)                  # ffmpeg finishes the file and exits
        ff_log = ff.communicate(timeout=180)[1]
    finally:
        for p in reversed(procs):
            p.terminate()
    if started is None or final is None:
        return {"gpu": gpu, "log": "".join(log)[-3000:], "error": f"the run did not {'start' if started is None else 'finish'}"}
    lead, tail = 1.5, 3.5                              # keep 1.5 s before the launch and 3.5 s after the run ends
    ss, dur = max(0.0, t_start - t_rec - lead), (t_end - t_start) + lead + tail
    rc, out = sh(f"ffmpeg -y -loglevel error -ss {ss:.2f} -i /tmp/raw.mkv -t {dur:.2f} -c:v libx264 -preset slow -crf 16"
                 f" -pix_fmt yuv420p -r {fps} -movflags +faststart /tmp/llm_dashboard.mp4")
    if rc:
        return {"gpu": gpu, "error": "encoding failed: " + out[-500:]}
    os.makedirs(f"{V}/video", exist_ok=True)
    name = f"b200_llm_dashboard_{binary}"
    shutil.copy("/tmp/llm_dashboard.mp4", f"{V}/video/{name}.mp4")
    Path(f"{V}/video/{name}.log").write_text("".join(log))                       # the recorded run's own console output
    if os.path.exists("/tmp/video_loss.txt"):
        shutil.copy("/tmp/video_loss.txt", f"{V}/video/{name}_loss.txt")
    full = "".join(log)
    def grab(pat, conv):
        m = re.search(pat, full)
        return conv(m.group(1)) if m else None
    timing = {"binary": binary, "gpu": gpu, "launch_at_s": round(t_start - t_rec - ss, 3), "run_wall_s": round(t_end - t_start, 3),
              "video_s": round(dur, 3), "fps": fps, "size": [width, height],
              "note": "launch_at_s: seconds into the MP4 at which the dashboard first reported the kernel running (polled at 20 Hz)",
              "run": {"kernel_s": grab(r"kernel time ([\d.]+) s \(event\)", float), "serve_tok_s": grab(r"-> (\d+) tok/s sampled", int),
                      "doom_fps": grab(r"([\d.]+) fps over the run", float), "loss_final": grab(r"s3999 ([\d.]+)", float)}}
    Path(f"{V}/video/{name}.json").write_text(json.dumps(timing, indent=1))
    vol.commit()
    stills = {}
    for t in (0.5, lead + 5, lead + 30, dur / 2, dur - 1.0):
        rc, _ = sh(f"ffmpeg -y -loglevel error -ss {t:.2f} -i /tmp/llm_dashboard.mp4 -frames:v 1 -vf scale=960:-2 /tmp/still.png")
        if rc == 0:
            stills[round(t, 1)] = Path("/tmp/still.png").read_bytes()
    return {"gpu": gpu, "seconds": dur, "mb": os.path.getsize("/tmp/llm_dashboard.mp4") / 1e6, "stills": stills,
            "final": {k: final.get(k) for k in ("steps", "loss", "chars", "frames", "t")},
            "summary": [l for l in log if l.startswith(("steady state", "decode", "doom:", "loss:"))], "ffmpeg": ff_log[-600:]}


@app.function(gpu=GPU, image=gemm_img, volumes={V: vol}, timeout=2400)
def gemm(tag: str = "gemm"):
    """llm/b200/gemm_tcgen05's test plan: the mma.sync control, then tcgen05 for NT only, NT + NN, all layouts, then speed against
    cuBLAS. Every run is under timeout -s KILL (a wrong barrier phase hangs); stops at the first failure. Results go to results/<tag>."""
    vol.reload()
    out, d = f"{RES}/{tag}", "/opt/triad/llm/b200/gemm_tcgen05"
    os.makedirs(out, exist_ok=True)
    plan = [("t_mma", "./t_mma", 600), ("t_nt_logits", "./t_nt --only logits --nobench", 90), ("t_nt", "./t_nt --nobench", 300),
            ("t_nn", "./t_nn --nobench", 300), ("t_all", "./t_all --nobench", 300), ("t_all_bench", "./t_all", 900)]
    res = {"gpu": sh(GPU_INFO)[1].strip()}
    for name, cmd, limit in plan:
        rc, o = sh(f"timeout -s KILL {limit} {cmd}", cwd=d)
        Path(f"{out}/{name}.txt").write_text(o)
        vol.commit()
        lines = o.splitlines()
        res[name] = {"rc": rc, "overall": next((l for l in lines if "OVERALL" in l), "no OVERALL line: hang (killed) or crash"),
                     "case_fails": sum(1 for l in lines if l.rstrip().endswith("FAIL"))}
        if rc != 0 and name != "t_mma":
            break
    return res


def _run_jobs(jobs, py, sub="runs_v2"):
    """Each job ALONE on the GPU through the kit's one_run.sh (waits for an idle GPU, starts / stops MPS at the given percentage, samples the
    process count); raw JSON / logs go to the Volume's results/<sub>/."""
    env = dict(os.environ, DATA_DIR=DATA, MODEL="S", GPU_TAG="b200", TOTAL_SMS="148", TRAIN_SMS="116", INFER_SMS="24", HOME="/root",
               NVTE_DEBUG="1", NVTE_DEBUG_LEVEL="1", CUDA_DEVICE_MAX_CONNECTIONS="1", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    sh(f"rm -rf /tmp/bl && cp -r /src/bl /tmp/bl && mkdir -p /tmp/bl/results/runs_v2 {RES}/{sub}")
    res = {"gpu": sh(GPU_INFO)[1].strip()}
    for name, pct, cmd in jobs:
        t = time.time()
        rc, o = sh(f"IDLE_WAIT_S=30 bash one_run.sh {name} {pct} bash -c \"{py} {cmd} --out \\$OUT\"", cwd="/tmp/bl", env=env)
        sh(f"cp /tmp/bl/results/runs_v2/{name}.* {RES}/{sub}/ 2>/dev/null; true")
        vol.commit()
        res[name] = {"rc": rc, "wall_s": round(time.time() - t, 1), "summary": (o.strip().splitlines() or [""])[-1][:300]}
    return res


@app.function(gpu=GPU, image=train_img, volumes={V: vol}, cpu=CPUS, memory=65536, timeout=10800)
def train(reps: int = 3, steps: int = 4000, sub: str = "runs_v2"):
    """Training baselines, each alone: Megatron-core with Transformer Engine on 116 SMs (FA's baseline) and on the whole GPU; PyTorch eager
    (flash attention) the same; Megatron-core without TE (the A100 baseline's configuration) on 116 SMs; launch counts."""
    vol.reload()
    jobs = [("c3_megatron_te_S", 0, "megatron_train.py --impl te --count3"), ("c3_pytorch_S", 0, "train_pytorch.py --count3")]
    jobs += [(f"megatron_te_S_mps_r{i}", TRAIN_PCT, f"megatron_train.py --impl te --steps {steps} --tag mps116") for i in range(1, reps + 1)]
    jobs.append(("megatron_te_S_full_r1", 0, f"megatron_train.py --impl te --steps {steps} --tag full"))
    for i in range(1, reps + 1):                       # rep 1 saves the step-4000 weights that vLLM serves
        save = f" --save {DATA}/ckpt_pytorch_S_step{steps}.pt" if i == 1 and sub == "runs_v2" else ""
        jobs.append((f"pytorch_S_mps_r{i}", TRAIN_PCT, f"train_pytorch.py --steps {steps} --tag mps116{save}"))
    jobs.append(("pytorch_S_full_r1", 0, f"train_pytorch.py --steps {steps} --tag full"))
    jobs.append(("megatron_local_S_mps_r1", TRAIN_PCT, f"megatron_train.py --impl local --steps {steps} --tag mps116"))
    return _run_jobs(jobs, "python", sub)


def _graph_jobs(reps, steps):
    """The training baselines with CUDA graphs (their scripts' --cuda_graph): PyTorch replays the whole step from one graph, Megatron its
    forward and backward (its optimizer stays eager); each alone on 116 SMs."""
    jobs = [(f"pytorch_graph_S_mps_r{i}", TRAIN_PCT, f"train_pytorch.py --cuda_graph --steps {steps} --tag mps116") for i in range(1, reps + 1)]
    jobs += [(f"megatron_te_graph_S_mps_r{i}", TRAIN_PCT, f"megatron_train.py --impl te --cuda_graph --steps {steps} --tag mps116")
             for i in range(1, reps + 1)]
    return jobs


@app.function(gpu=GPU, image=train_img, volumes={V: vol}, memory=65536, timeout=7200)
def train_graphs(reps: int = 3, steps: int = 4000, sub: str = "runs_v3_graph"):
    """CUDA-graph training baselines in a container without reserved cores (where Megatron ran fastest, results/runs_v2/)."""
    vol.reload()
    return _run_jobs(_graph_jobs(reps, steps), "python", sub)


@app.function(gpu=GPU, image=train_img, volumes={V: vol}, cpu=CPUS, memory=65536, timeout=7200)
def train_graphs_cpu16(reps: int = 3, steps: int = 4000, sub: str = "runs_v3_graph_cpu16"):
    """The same with 16 reserved cores."""
    vol.reload()
    return _run_jobs(_graph_jobs(reps, steps), "python", sub)


@app.function(gpu=GPU, image=serve_img, volumes={V: vol}, cpu=CPUS, memory=65536, timeout=7200)
def serve_both(reps: int = 3, sub: str = "runs_v2_cpu16"):
    """vLLM, both prompts in ONE container (same host and CPU), alone, on 24 SMs and on the whole GPU: separates a prompt effect from a
    host effect. 'Once upon a time' runs are named vllm_S_*, FA's prompt vllmfa_S_*."""
    vol.reload()
    ck, hf = f"{DATA}/ckpt_pytorch_S_step4000.pt", f"{DATA}/hf_trained_S"
    sh(f"cd /src/bl && MODEL=S DATA_DIR={DATA} python3 make_hf.py {hf} {ck if os.path.exists(ck) else ''}")
    jobs = []
    for pre, p in (("vllm", ""), ("vllmfa", f" --prompt '{FA_PROMPT}'")):
        jobs += [(f"{pre}_S_mps_r{i}", INFER_PCT, f"serve_vllm.py {hf} --reps 1{p}") for i in range(1, reps + 1)]
        jobs += [(f"{pre}_S_full_r{i}", 0, f"serve_vllm.py {hf} --reps 1{p}") for i in (1, 2)]
    return _run_jobs(jobs, "python3", sub)


@app.function(gpu=GPU, image=serve_img, volumes={V: vol}, cpu=CPUS, memory=65536, timeout=7200)
def serve(reps: int = 3, prompt: str = "", sub: str = "runs_v2"):
    """vLLM batch-1 decode (256 new tokens) of the PyTorch run's step-4000 weights, alone: on 24 SMs (FA's baseline) and on the whole GPU.
    Without `prompt` (timed prompt 'Once upon a time'): also launch counts (default / eager / full CUDA graphs) and a text sample.
    With `prompt` (e.g. FA's 'What is a megakernel?'): the timed runs only, into results/<sub>/."""
    vol.reload()
    ck, hf = f"{DATA}/ckpt_pytorch_S_step4000.pt", f"{DATA}/hf_trained_S"
    sh(f"cd /src/bl && MODEL=S DATA_DIR={DATA} python3 make_hf.py {hf} {ck if os.path.exists(ck) else ''}")
    p = f" --prompt '{prompt}'" if prompt else ""
    jobs = [(f"vllm_S_mps_r{i}", INFER_PCT, f"serve_vllm.py {hf} --reps 1{p}") for i in range(1, reps + 1)]
    jobs += [(f"vllm_S_full_r{i}", 0, f"serve_vllm.py {hf} --reps 1{p}") for i in (1, 2)]
    if not prompt:
        jobs += [(f"vllmc3_S_{cg}", 0, f"serve_vllm.py {hf} --reps 1 --count_launches --cg {cg}") for cg in ("default", "eager", "full")]
        jobs.append(("vllm_S_text", 0, f"serve_vllm.py {hf} --reps 1 --text"))
    return _run_jobs(jobs, "python3", sub)


@app.function(image=prep_img, volumes={V: vol}, timeout=900)
def report():
    """The kit's aggregate_v2.py (mean / sd per configuration) and fa_compare.py (next to FA's numbers), on the Volume's raw results."""
    vol.reload()
    kit = "/tmp/r/llm/b200/baseline"                   # fa_compare.py finds benchmarks/fa_published.json three levels up
    sh(f"rm -rf /tmp/r && mkdir -p /tmp/r/llm/b200 /tmp/r/benchmarks && cp -r /src/b200/baseline {kit} && cp /src/fa_published.json /tmp/r/benchmarks/"
       f" && mkdir -p {kit}/results/runs_v2 && cp {RES}/runs_v2/* {kit}/results/runs_v2/")
    env = dict(os.environ, GPU_TAG="b200", TOTAL_SMS="148", TRAIN_SMS="116", INFER_SMS="24", MODEL="S", DATA_DIR=DATA)
    rc, out = sh("python aggregate_v2.py && python fa_compare.py", cwd=kit, env=env)
    sh(f"cp {kit}/results/baseline_v2_b200_*.json {kit}/results/fa_compare_b200.md {RES}/ 2>/dev/null; ls -la {RES}")
    vol.commit()
    return out


@app.local_entrypoint()
def main(step: str = "prep", reps: int = 3, steps: int = 4000, binary: str = "triad_s", tag: str = "megakernel", trainonly: bool = True,
         serving: str = "story", prompt: str = "", sub: str = "runs_v2"):
    if step == "prep":
        print(json.dumps(prep.remote(), indent=1))
    elif step == "build":
        build_info.remote()
    elif step == "env":
        print(env_train.remote())
        print(env_serve.remote())
    elif step == "mk":
        for name, r in mk.remote(reps, steps, binary, tag, trainonly, serving).items():
            print(name, r if isinstance(r, str) else f"rc={r['rc']}", *([] if isinstance(r, str) else r["lines"]), sep="\n  ")
    elif step == "video":                              # dashboard recording -> video/out/ (ignored by video/.gitignore); --binary picks the build
        b = binary if binary != "triad_s" else "triad_tc2"
        r = video.remote(steps, binary=b)
        if "error" in r:
            raise SystemExit(f"video failed: {r['error']}\n{r.get('log', '')}")
        dst = ROOT / "video" / "out"
        dst.mkdir(parents=True, exist_ok=True)
        for ext in (".mp4", ".json", ".log", "_loss.txt"):
            subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "triad-b200-data", f"video/b200_llm_dashboard_{b}{ext}",
                            str(dst)], check=True)
        for t, png in r["stills"].items():
            (dst / f"b200_llm_dashboard_{b}_t{t:05.1f}.png").write_bytes(png)
        keep = ROOT / "b200-run" / "results" / f"video_{b}"   # the recorded run's timing, log and losses are kept with the results
        keep.mkdir(parents=True, exist_ok=True)
        for ext in (".json", ".log", "_loss.txt"):
            shutil.copy(dst / f"b200_llm_dashboard_{b}{ext}", keep)
        print(f"saved {dst / f'b200_llm_dashboard_{b}.mp4'} ({r['mb']:.1f} MB, {r['seconds']:.1f} s); final {r['final']}", *r["summary"], sep="\n  ")
    elif step == "tccmp":                              # both tcgen05 builds on FA's split with FA's prompt, in parallel on two B200s
        names = ("triad_tc2", "triad_tc3")
        calls = [mk.spawn(reps, steps, b, f"megakernel_{b[6:]}_fa", trainonly, "fa") for b in names]
        for b, c in zip(names, calls):
            for name, r in c.get().items():
                print(b, name, r if isinstance(r, str) else f"rc={r['rc']}", *([] if isinstance(r, str) else r["lines"]), sep="\n  ")
    elif step == "gemm":
        print(json.dumps(gemm.remote(tag if tag != "megakernel" else "gemm"), indent=1))
    elif step == "train":
        print(json.dumps(train.remote(reps, steps, sub), indent=1))
    elif step == "train_graphs":                       # CUDA-graph baselines, both container kinds in parallel
        calls = [train_graphs.spawn(reps, steps), train_graphs_cpu16.spawn(reps, steps)]
        for c in calls:
            print(json.dumps(c.get(), indent=1))
    elif step == "serve_both":
        print(json.dumps(serve_both.remote(reps, sub if sub != "runs_v2" else "runs_v2_cpu16"), indent=1))
    elif step == "serve":
        print(json.dumps(serve.remote(reps, prompt, sub), indent=1))
    elif step == "report":
        print(report.remote())
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "triad-b200-data", "results", str(ROOT / "b200-run")], check=True)
        print("copied the Volume's results/ to", ROOT / "b200-run" / "results")
    else:
        raise SystemExit(f"unknown step {step}")
