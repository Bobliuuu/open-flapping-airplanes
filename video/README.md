# video/: Triad explainer (data-driven, rendered with headless Chrome + ffmpeg)

Scenes are one HTML/SVG/CSS page (`index.html`). `render.js` drives headless Chrome (puppeteer-core, system Google Chrome)
frame by frame at 1920x1080, 30 fps, pipes JPEG frames into ffmpeg (libx264, crf 16), one MP4 segment per scene, then concatenates.
Every number comes from `data.json`; `value: null` renders as "pending" (hatched bars).

## Render
```
cd video && npm install            # once: puppeteer-core only (no browser download; uses /Applications/Google Chrome.app)
node render.js                     # -> out/triad_explainer_preview.mp4 (86 s, ~2 min, 3 scenes in parallel; --conc N)
node render.js --scene results     # re-render one scene segment only (out/seg/results.mp4), then re-run plain `node render.js` or compose.sh
node render.js --stills 20,50,76   # PNGs at global times (s) into out/stills/ for a quick visual check
CHROME=/path/to/chrome node render.js   # other Chrome/Chromium
```
Needs ffmpeg and node on PATH (found: Homebrew ffmpeg, mise node 22).

## Put the live-demo recording in
```
./compose.sh recording.mp4 [out/triad_explainer.mp4]     # MAXS=45 ./compose.sh ...  to allow a longer recording (default cap 30 s)
```
Replaces the placeholder `slot` scene with the recording (scaled into the dashboard window, header + caption overlay) and re-concatenates.
Missing scene segments are rendered first. Final length = 78 s + recording length.

## Updating numbers
Edit `data.json` (a number is `{value, unit, status, source}`; set `status` to `measured` and fill `value` when a pending run finishes).
Pending now: `b300.launches_base`, `b300.train_s_base`, `b300.train_toks_base`, `b300.loss_end_base`, `b300.doom_fps`.
Scene lengths and burned-in captions are in `data.json` under `scenes` (`[start_s, end_s, text]` relative to the scene).
`tools/make_data.py` bootstraps data.json from the repo results (refuses to overwrite without --force, and does not carry the `scenes` block).
`python3 tools/refresh_numbers.py [--dry]` re-reads the repo result files (llm/triad/README.md, llm/b200/results/mk_split_100a.txt + bl_gpu1.log, llm/baseline/results, benchmarks/fa_published.json) and updates only value/status in data.json (scenes, captions, sources untouched), printing a diff and the still-pending list. B300 numbers carry status `interim` until the baselines are complete.
`compose.sh` accepts any input resolution/fps/pixel format (audio is dropped; tested 1282x721@25, 2560x1440@60, 1280x720 yuv444 with audio, MAXS=45 with a 40 s clip).
Percentages use FA's convention, 1 - faster/slower (70.1 s vs 46.1 s = 34%).

Large outputs (`out/`, `*.mp4`, `node_modules/`) are ignored by `video/.gitignore`. Not affiliated with Flapping Airplanes.

## Demo video on a B200, in Flapping Airplanes' format (`b200/`)

`release/triad_demo_b200.mp4` (74.6 s, 1920x1080, 30 fps, with music): the structure of Flapping Airplanes' demo video (live dashboard,
zoom out to the GPU, code, side by side with the baselines and their launch counts, result cards, end card) for Triad's `triad_tc2` build
on one B200, plus a card on how it was measured and what differs from their demo. Rules it follows:

- Every number comes from `b200/data.json`, which `b200/make_data.py` derives from the raw results (`b200-run/results/`), each with its
  source file. Nothing is typed into the page.
- The dashboard is a screen recording of a real run (`modal run b200-run/modal_app.py --step video --binary triad_tc2`; its timing and
  losses are in `b200-run/results/video_triad_tc2/`). That run, with the dashboard and screen capture, took 52.9 s; the cards use the
  mean of 3 runs without them (52.4 s), and the video says so. Doom in it is played by the scripted player in `doom-gpu/player/`.
- With `--fa-video PATH` (their demo video, which is not in this repository) the render adds a scene with their dashboard next to ours
  at the same run time (seconds since each kernel launch): their full-screen stretch (their run at 3.4-8.0 s) in real time, then the end
  of both runs at 2x, from the two real-time stretches their video shows. The clocks in their footage set the alignment (constants
  `FA_A`, `FA_B` in `demo.html`). The MP4 goes to `out/triad_demo_<build>_with_fa.mp4`, which git ignores.
- The music is composed by `b200/make_music.py` on the video's timeline (no samples, no outside audio; numpy only) and muxed in as AAC
  by the render job.
- The side-by-side runs at 4x speed (labelled). The baselines are their measured runs (each alone on its share of the same B200), replayed
  at their measured mean speed; their loss curves are their logged losses. Launch counters use the per-step and per-token counts measured
  with torch.profiler.
- Megatron-core + Transformer Engine and PyTorch are timed both eager and with CUDA graphs, each in two containers, and the faster
  counts; vLLM uses CUDA graphs and 16 CPU cores. With CUDA graphs Megatron + TE is faster than Triad at training, and the cards say so.

```bash
python3 video/b200/make_data.py --build triad_tc2                       # data.json + data.js from the results
modal run video/b200/modal_render.py --build triad_tc2 --stills 5,30,52  # PNG stills -> video/out/demo_stills/
modal run video/b200/modal_render.py --build triad_tc2                  # the MP4 -> video/out/triad_demo_triad_tc2.mp4 (about 2 minutes)
modal run video/b200/modal_render.py --build triad_tc2 --fa-video their_demo.mp4   # with their dashboard side by side (not committed)
```
Rendering runs on a Modal CPU container (Google Chrome, node, ffmpeg, Inter and JetBrains Mono): `render_demo.js` drives `demo.html`'s
`seek(t)` frame by frame in parallel chunks and pipes JPEG frames into ffmpeg. `--build triad_tc3` renders the 3-stage build instead
(record it first with `--binary triad_tc3`).
