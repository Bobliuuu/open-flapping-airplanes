"""Renders the demo video on a Modal CPU container (Google Chrome, node, ffmpeg, Inter and JetBrains Mono): no local browser needed.
The music is composed by make_music.py on the same timeline and muxed in as AAC.

  python3 video/b200/make_data.py --build ofa_tc2                  # data.json / data.js from the results and the recording
  modal run video/b200/modal_render.py --build ofa_tc2 --stills 3,13,20,30,40,48,56,64,70   # PNG stills -> video/out/demo_stills/
  modal run video/b200/modal_render.py --build ofa_tc2             # the MP4 -> video/out/ofa_demo_<build>.mp4

The dashboard recording comes from the Volume ofa-b200-data (video/b200_llm_dashboard_<build>.mp4, written by
`modal run b200-run/modal_app.py --step video --binary <build>`).
"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
app = modal.App("ofa-demo-video")
vol = modal.Volume.from_name("ofa-b200-data")
img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "xz-utils", "ca-certificates", "ffmpeg", "fonts-inter", "fonts-jetbrains-mono", "fontconfig")
    .run_commands(
        "wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
        " && apt-get update -q && apt-get install -y -q /tmp/chrome.deb && rm /tmp/chrome.deb && google-chrome --version",
        "curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar -xJ -C /opt"
        " && ln -s /opt/node-v22.11.0-linux-x64/bin/node /usr/local/bin/node && ln -s /opt/node-v22.11.0-linux-x64/bin/npm /usr/local/bin/npm",
        "mkdir -p /opt/r && cd /opt/r && npm init -y >/dev/null && npm install --no-audit --no-fund puppeteer-core@24 && fc-list | grep -i -E 'inter|jetbrains' | head -3",
    )
    .pip_install("numpy==2.1.3")                                    # make_music.py
)


@app.function(image=img, volumes={"/vol": vol}, cpu=16.0, memory=32768, timeout=3600)
def render(files: dict, build: str, stills: str = "", conc: int = 8, fa: bytes = b"") -> dict:
    w = Path("/tmp/demo")
    (w / "frames").mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (w / name).write_text(text)
    (w / "node_modules").symlink_to("/opt/r/node_modules")
    rec = f"/vol/video/b200_llm_dashboard_{build}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", rec, "-vf", "fps=30", "-q:v", "2", str(w / "frames" / "rec_%05d.jpg")], check=True)
    if fa:                                                          # Flapping Airplanes' demo video: the two stretches of their dashboard
        (w / "fa.mp4").write_bytes(fa)
        for name, start, length in (("fa_a", 0.0, 6.0), ("fa_b", 21.5, 12.0)):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(length), "-i", str(w / "fa.mp4"), "-vf", "fps=60",
                            "-q:v", "2", str(w / "frames" / f"{name}_%05d.jpg")], check=True)
    cmd = ["node", "render_demo.js", "--chrome", "/usr/bin/google-chrome", "--out", "/tmp/out/ofa_demo.mp4", "--conc", str(conc)]
    if stills:
        cmd += ["--stills", stills]
    p = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    res = {"rc": p.returncode, "log": (p.stdout + p.stderr)[-4000:], "frames": len(list((w / "frames").glob("rec_*.jpg")))}
    if stills:
        res["stills"] = {f.name: f.read_bytes() for f in sorted(Path("/tmp/out/demo_stills").glob("*.png"))}
    elif Path("/tmp/out/ofa_demo.mp4").exists():
        # the music, composed on the video's timeline, muxed in as AAC (the video stream is copied unchanged)
        m = subprocess.run(["python3", "make_music.py", "--info", "/tmp/out/demo_info.json", "--out", "/tmp/out/music.wav"], cwd=w,
                           capture_output=True, text=True)
        res["music_log"] = (m.stdout + m.stderr)[-1500:]
        if m.returncode == 0:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/out/ofa_demo.mp4", "-i", "/tmp/out/music.wav", "-map", "0:v",
                            "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                            "/tmp/out/ofa_demo_music.mp4"], check=True)
            res["mp4"] = Path("/tmp/out/ofa_demo_music.mp4").read_bytes()
        else:
            res["rc"] = m.returncode
    return res


@app.local_entrypoint()
def main(build: str = "ofa_tc2", stills: str = "", conc: int = 8, check_times: str = "", fa_video: str = ""):
    if check_times:                                     # inspect an already rendered MP4: --check-times 9.3,9.4,...
        f = OUT / f"ofa_demo_{build}.mp4"
        r = check.remote(f.read_bytes(), check_times)
        print(r["probe"])
        (OUT / f"ofa_demo_{build}_sheet.png").write_bytes(r["sheet"])
        print("contact sheet ->", OUT / f"ofa_demo_{build}_sheet.png")
        return
    files = {p.name: p.read_text() for p in HERE.iterdir() if p.name in ("demo.html", "data.js", "data.json", "render_demo.js", "make_music.py")}
    # --fa-video PATH: add the side-by-side scene with Flapping Airplanes' demo video (their footage: kept out of the repository; the MP4
    # goes to video/out/ofa_demo_<build>_with_fa.mp4, which git ignores)
    fa = Path(fa_video).read_bytes() if fa_video else b""
    r = render.remote(files, build, stills, conc, fa)
    print(r["log"])
    if r.get("music_log"):
        print("music:", r["music_log"].strip())
    OUT.mkdir(exist_ok=True)
    if "stills" in r:
        d = OUT / "demo_stills"
        d.mkdir(exist_ok=True)
        for name, b in r["stills"].items():
            (d / name).write_bytes(b)
        print(f"{len(r['stills'])} stills -> {d}")
    if "mp4" in r:
        f = OUT / (f"ofa_demo_{build}_with_fa.mp4" if fa else f"ofa_demo_{build}.mp4")
        f.write_bytes(r["mp4"])
        print(f"wrote {f} ({len(r['mp4']) / 1e6:.1f} MB)")
    if r["rc"]:
        raise SystemExit(f"render failed (rc {r['rc']})")


@app.function(image=img, cpu=4.0, timeout=600)
def check(mp4: bytes, times: str) -> dict:
    """ffprobe of a rendered MP4 and a contact sheet of frames at the given times (to inspect chunk boundaries)."""
    Path("/tmp/c.mp4").write_bytes(mp4)
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt:format=duration,size",
                         "-of", "json", "/tmp/c.mp4"], capture_output=True, text=True).stdout
    ts = [float(x) for x in times.split(",")]
    for i, t in enumerate(ts):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", "/tmp/c.mp4", "-frames:v", "1", "-vf", "scale=640:-2",
                        f"/tmp/f{i:02d}.png"], check=True)
    cols = 4
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "/tmp/f%02d.png", "-vf", f"tile={cols}x{(len(ts) + cols - 1) // cols}:padding=6:color=white",
                    "-frames:v", "1", "/tmp/sheet.png"], check=True)
    return {"probe": pr, "sheet": Path("/tmp/sheet.png").read_bytes()}
