"""Build and run open-flapping-airplanes on a Modal cloud GPU (default: one NVIDIA B200). No SSH box needed.

  modal run benchmarks/run_modal.py                    # smoke test: Doom alone, then all three jobs together for a few seconds
  modal run benchmarks/run_modal.py --mode live        # live demo without a browser: saves Doom frames, generated text, loss
  modal run benchmarks/run_modal.py --mode bench       # full benchmark (5 reps x 6 s) -> benchmarks/results/b200_<date>.json
  modal run benchmarks/run_modal.py --mode mps         # baselines limited to FA's SM shares with CUDA MPS (116 / 24 of 148 SMs)
  OFA_GPU=H100 OFA_ARCH=sm_90 modal run benchmarks/run_modal.py --mode bench    # another GPU type

Everything is compiled while Modal builds the image, on CPU (CUDA 12.8 + clang 14, doomgeneric, the shareware WAD checked
by md5), so the GPU is billed only while the programs run. A bench run writes its JSON next to the A100 results but does not
touch latest.json, summary.md or the site data; run compare.py on it yourself to publish it.
"""
import datetime, json, os, struct, subprocess, threading, time, zlib
from pathlib import Path

import modal

GPU = os.environ.get("OFA_GPU", "B200")          # any Modal GPU type: B200, H200, H100, A100-40GB, ...
ARCH = os.environ.get("OFA_ARCH", "sm_100")      # its SASS target: sm_100 B200, sm_90 H100/H200, sm_80 A100
TAG = GPU.split(":")[0].rstrip("!+").lower()
ROOT = Path(__file__).resolve().parent.parent
DOOM, DEMO = "/opt/ofa/doom-gpu", "/opt/ofa/a100-demo"
WAD = f"{DOOM}/doom1.wad"
WAD_URL, WAD_MD5 = "https://github.com/Akbar30Bill/DOOM_wads/raw/master/doom1.wad", "f0cefca49926d00903cf57551d901abe"
SKIP = ["doomgeneric", "obj", "src_gpu", "**/*.o", "**/*.ptx", "**/*.bc", "**/*.ll", "**/*.cubin", "**/*.wad", "demo", "doomtest"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("clang", "llvm", "lld", "git", "curl", "ca-certificates")
    .run_commands(
        "git clone -q https://github.com/ozkl/doomgeneric.git /opt/src/doomgeneric"
        " && git -C /opt/src/doomgeneric rev-parse --short HEAD > /opt/src/doomgeneric.rev",
        f"curl -fsSL -m 120 -o /opt/src/doom1.wad {WAD_URL} && echo '{WAD_MD5}  /opt/src/doom1.wad' | md5sum -c -",
    )
    .add_local_dir(ROOT / "doom-gpu", DOOM, copy=True, ignore=SKIP)
    .run_commands(
        f"ln -sfn /opt/src/doomgeneric {DOOM}/doomgeneric && cp /opt/src/doom1.wad {WAD}",
        f"cd {DOOM} && ARCH={ARCH} bash build_doom.sh",
        f"cd {DOOM} && nvcc -O3 -arch={ARCH} -std=c++17 -rdc=true doomtest.cu doom_dev.o -o doomtest",
    )
    .add_local_dir(ROOT / "a100-demo", DEMO, copy=True, ignore=SKIP)
    .run_commands(f"cd {DEMO} && nvcc -O3 -arch={ARCH} -std=c++17 -rdc=true demo.cu {DOOM}/doom_dev.o -o demo -lpthread")
)
app = modal.App(f"ofa-{TAG}", image=image)
GPU_INFO = "nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader"


def sh(cmd, cwd=None):
    """Run a shell command, echo its output into the Modal log as it arrives, return (exit code, output)."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return p.wait(), "".join(lines)


@app.function(gpu=GPU, timeout=900)
def smoke(secs: float = 4.0) -> dict:
    """Doom alone in one block (doomtest, 120 frames), then one megakernel launch with all three jobs."""
    r = {"gpu": sh(GPU_INFO)[1].strip(), "doomgeneric": Path("/opt/src/doomgeneric.rev").read_text().strip()}
    r["doomtest_rc"], r["doomtest"] = sh(f"timeout -k 5 150 ./doomtest {WAD} 120", cwd=DOOM)
    ppm = Path(DOOM, "doom_frame.ppm")
    r["ppm"] = ppm.read_bytes() if ppm.exists() else None
    r["mega_rc"], r["mega"] = sh(f"timeout -k 5 240 ./demo --port 0 --wad {WAD} --bench-one mega_all --bench-secs {secs}", cwd=DEMO)
    return r


@app.function(gpu=GPU, timeout=900)
def live(secs: float = 20.0, grabs: int = 4) -> dict:
    """The live demo for `secs` seconds; its dashboard endpoints are read from inside the container instead of a browser."""
    import urllib.request

    gpu, port = sh(GPU_INFO)[1].strip(), 8080
    p = subprocess.Popen(["timeout", "-k", "5", str(int(secs) + 180), "./demo", "--port", str(port), "--secs", str(secs), "--wad", WAD],
                         cwd=DEMO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    log = []

    def pump():
        for line in p.stdout:
            print(line, end="", flush=True)
            log.append(line)

    def get(path):
        return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10).read()

    def state():
        try:
            return json.loads(get("/state"))
        except Exception:
            return None

    def wait_phase(phase, limit):
        end = time.time() + limit
        while time.time() < end and p.poll() is None:
            s = state()
            if s and s["phase"] >= phase:
                return s
            time.sleep(0.3)
        return None

    threading.Thread(target=pump, daemon=True).start()
    if not wait_phase(1, 120):
        p.kill()
        return {"gpu": gpu, "log": "".join(log), "state": None, "frames": [], "error": "the demo never started"}
    t0, frames = time.time(), []
    for k in range(grabs):
        time.sleep(max(0.0, t0 + secs * (k + 1) / (grabs + 1) - time.time()))
        try:
            s = state()
            frames.append({"t": round(time.time() - t0, 1), "frame_no": s and s["frames"], "fb": get("/frame")})
        except Exception as e:
            print("frame grab failed:", e)
    final = wait_phase(3, secs + 120)
    p.terminate()
    return {"gpu": gpu, "log": "".join(log), "state": final, "frames": frames}


@app.function(gpu=GPU, timeout=3600)
def bench(reps: int = 5, secs: float = 6.0) -> dict:
    """The same benchmark run_benchmarks.sh runs over SSH."""
    gpu = sh(GPU_INFO)[1].strip()
    limit = int(reps * (15 * secs + 60) + 120)        # about twice the expected run time
    rc, log = sh(f"timeout -k 10 {limit} ./demo --port 0 --wad {WAD} --bench /tmp/bench.json --reps {reps} --bench-secs {secs}", cwd=DEMO)
    out = Path("/tmp/bench.json")
    return {"gpu": gpu, "rc": rc, "log": log, "json": out.read_text() if out.exists() else None}


@app.function(gpu=GPU, timeout=1800)
def mps_baselines(reps: int = 3, secs: float = 6.0, train_pct: int = 79, infer_pct: int = 17) -> dict:
    """FA's protocol for the baselines: each per-op program alone, limited with CUDA MPS to FA's SM share (on 148 SMs, 79% gives
    116 SMs for training and 17% gives 24 for serving). Runs at 100% under the same MPS daemon are the control."""
    import re

    os.environ.update(CUDA_MPS_PIPE_DIRECTORY="/tmp/mps/pipe", CUDA_MPS_LOG_DIRECTORY="/tmp/mps/log")
    os.makedirs("/tmp/mps/pipe", exist_ok=True)
    os.makedirs("/tmp/mps/log", exist_ok=True)
    gpu = sh(GPU_INFO)[1].strip()
    rc, _ = sh("nvidia-cuda-mps-control -d && sleep 2")
    runs = {}
    try:
        for _ in range(reps):
            for role, pct in (("train", train_pct), ("train", 100), ("infer", infer_pct), ("infer", 100)):
                for mode in ("eager", "graph"):
                    rc, out = sh(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={pct} timeout -k 5 120 ./demo --child {role} --child-mode {mode}"
                                 f" --cmp-secs {secs} --wad {WAD} --port 0", cwd=DEMO)
                    sms, res = re.search(r"\((\d+) SMs\)", out), re.search(r"RESULT \S+ (\d+) ([\d.]+)", out)
                    runs.setdefault(f"{role}_{mode}_mps{pct}", []).append(
                        {"sms": int(sms.group(1)) if sms else None, "rate": int(res.group(1)) / float(res.group(2)) if res else None, "rc": rc})
    finally:
        sh("echo quit | nvidia-cuda-mps-control; sleep 1")
    return {"gpu": gpu, "reps": reps, "secs": secs, "runs": runs}


def write_png(path, rgb, w=320, h=200, scale=2):
    """Minimal RGB PNG writer (no dependencies), nearest-neighbour upscaled."""
    rows = []
    for y in range(h):
        line = rgb[y * w * 3:(y + 1) * w * 3]
        line = b"".join(line[i:i + 3] * scale for i in range(0, len(line), 3))
        rows += [b"\0" + line] * scale
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w * scale, h * scale, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def frame_rgb(fb):
    """A frame from the demo's /frame endpoint: 320x200 palette indices, then 256 colours stored as 0x00RRGGBB."""
    lut = [bytes(((c >> 16) & 255, (c >> 8) & 255, c & 255)) for c in struct.unpack_from("<256I", fb, 64000)]
    return b"".join(lut[i] for i in fb[:64000])


@app.local_entrypoint()
def main(mode: str = "smoke", reps: int = 5, secs: float = 6.0, live_secs: float = 20.0, out: str = ""):
    dst = Path(out) if out else ROOT / "benchmarks" / "results"
    dst.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().strftime("%Y%m%d")
    print(f"open-flapping-airplanes on Modal: {GPU} ({ARCH}), mode {mode}")
    if mode in ("smoke", "all"):
        r = smoke.remote()
        print(f"\nGPU: {r['gpu']}   doomgeneric {r['doomgeneric']}")
        for line in r["doomtest"].splitlines():
            if " frames: " in line or "pixel sum" in line:
                print("doomtest:", line)
        for line in r["mega"].splitlines():
            if line.startswith(("GPU:", "METRIC")):
                print(line)
        if r["ppm"]:
            write_png(dst / f"{TAG}_{stamp}_doomtest_frame.png", r["ppm"].split(b"\n", 3)[3])
            print("saved", dst / f"{TAG}_{stamp}_doomtest_frame.png")
        if r["doomtest_rc"] or r["mega_rc"]:
            raise SystemExit(f"smoke test failed: doomtest exit {r['doomtest_rc']}, demo exit {r['mega_rc']}")
    if mode in ("live", "all"):
        r = live.remote(live_secs)
        for i, f in enumerate(r["frames"]):
            write_png(dst / f"{TAG}_{stamp}_live_frame{i}.png", frame_rgb(f["fb"]))
            print(f"saved {dst / f'{TAG}_{stamp}_live_frame{i}.png'}  (t={f['t']} s, Doom frame {f['frame_no']})")
        s = r["state"]
        if not s:
            raise SystemExit("live run produced no final state: " + r.get("error", "see the log above"))
        (dst / f"{TAG}_{stamp}_live_state.json").write_text(json.dumps(s, indent=1))
        print(f"\nlive: {s['t']:.1f} s, {s['steps']} training steps (loss {s['loss']:.3f}), {s['chars']} chars generated, {s['frames']} Doom frames")
        print("generated text (tail):\n" + s["text"][-400:])
    if mode in ("bench", "all"):
        r = bench.remote(reps, secs)
        (dst / f"{TAG}_{stamp}.log").write_text(f"# nvidia-smi: {r['gpu']}\n" + r["log"])
        if not r["json"]:
            raise SystemExit(f"benchmark failed (exit {r['rc']}); log: {dst / f'{TAG}_{stamp}.log'}")
        path = dst / f"{TAG}_{stamp}.json"
        path.write_text(r["json"])
        new = json.loads(r["json"])
        ref_path = ROOT / "benchmarks" / "results" / "latest.json"
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {"meta": {"gpu": "-"}, "metrics": {}}
        print(f"\nsaved {path}\n{'metric':30} {ref['meta']['gpu'][:22]:>22} {new['meta']['gpu'][:22]:>22} {'ratio':>6}")
        for k, v in new["metrics"].items():
            a = ref["metrics"].get(k, {}).get("mean")
            print(f"{k:30} {a if a is not None else float('nan'):22.1f} {v['mean']:22.1f} {v['mean'] / a if a else float('nan'):6.2f}")
    if mode in ("mps", "all"):
        import statistics
        r = mps_baselines.remote(3, secs)
        metrics = {}
        for k, v in r["runs"].items():
            rates = [x["rate"] for x in v if x["rate"]]
            metrics[k] = {"sms": sorted({x["sms"] for x in v if x["sms"]}), "mean": statistics.mean(rates) if rates else None,
                          "sd": statistics.stdev(rates) if len(rates) > 1 else 0.0, "n": len(rates), "values": rates}
        path = dst / f"{TAG}_{stamp}_mps_baselines.json"
        path.write_text(json.dumps({"meta": {"gpu": r["gpu"], "reps": r["reps"], "secs_per_measurement": r["secs"],
                                             "unit": "train: steps/s, infer: tok/s",
                                             "method": "demo --child <role> --child-mode <eager|graph>, each alone, under CUDA MPS at the given active-thread percentage"},
                                    "metrics": metrics}, indent=1))
        print(f"\nsaved {path}\n{'config':24} {'SMs':>8} {'mean':>10} {'sd':>8} {'vs 100%':>8}")
        for k, v in metrics.items():
            full = metrics.get(k.rsplit("_mps", 1)[0] + "_mps100", {}).get("mean")
            print(f"{k:24} {str(v['sms']):>8} {v['mean'] or float('nan'):10.0f} {v['sd']:8.0f} {(v['mean'] or 0) / full if full else float('nan'):8.3f}")
