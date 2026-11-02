# video/: open-flapping-airplanes explainer (data-driven, rendered with headless Chrome + ffmpeg)

Scenes are one HTML/SVG/CSS page (`index.html`). `render.js` drives headless Chrome (puppeteer-core, system Google Chrome)
frame by frame at 1920x1080, 30 fps, pipes JPEG frames into ffmpeg (libx264, crf 16), one MP4 segment per scene, then concatenates.
Every number comes from `data.json`; `value: null` renders as "pending" (hatched bars).

## Render
```
cd video && npm install            # once: puppeteer-core only (no browser download; uses /Applications/Google Chrome.app)
node render.js                     # -> out/ofa_explainer_preview.mp4 (86 s, ~2 min, 3 scenes in parallel; --conc N)
node render.js --scene results     # re-render one scene segment only (out/seg/results.mp4), then re-run plain `node render.js` or compose.sh
node render.js --stills 20,50,76   # PNGs at global times (s) into out/stills/ for a quick visual check
CHROME=/path/to/chrome node render.js   # other Chrome/Chromium
```
Needs ffmpeg and node on PATH (found: Homebrew ffmpeg, mise node 22).

## Put the live-demo recording in
```
./compose.sh recording.mp4 [out/ofa_explainer.mp4]     # MAXS=45 ./compose.sh ...  to allow a longer recording (default cap 30 s)
```
Replaces the placeholder `slot` scene with the recording (scaled into the dashboard window, header + caption overlay) and re-concatenates.
Missing scene segments are rendered first. Final length = 78 s + recording length.

## Updating numbers
Edit `data.json` (a number is `{value, unit, status, source}`; set `status` to `measured` and fill `value` when a pending run finishes).
Pending now: `b300.launches_base`, `b300.train_s_base`, `b300.train_toks_base`, `b300.loss_end_base`, `b300.doom_fps`.
Scene lengths and burned-in captions are in `data.json` under `scenes` (`[start_s, end_s, text]` relative to the scene).
`tools/make_data.py` bootstraps data.json from the repo results (refuses to overwrite without --force, and does not carry the `scenes` block).
`python3 tools/refresh_numbers.py [--dry]` re-reads the repo result files (llm/ofa/README.md, llm/b200/results/mk_split_100a.txt + bl_gpu1.log, llm/baseline/results, benchmarks/fa_published.json) and updates only value/status in data.json (scenes, captions, sources untouched), printing a diff and the still-pending list. B300 numbers carry status `interim` until the baselines are complete.
`compose.sh` accepts any input resolution/fps/pixel format (audio is dropped; tested 1282x721@25, 2560x1440@60, 1280x720 yuv444 with audio, MAXS=45 with a 40 s clip).
Percentages use FA's convention, 1 - faster/slower (70.1 s vs 46.1 s = 34%).

Large outputs (`out/`, `*.mp4`, `node_modules/`) are ignored by `video/.gitignore`. Not affiliated with Flapping Airplanes.

## Demo video on a B200, in Flapping Airplanes' format (`b200/`)

`release/ofa_demo_b200.mp4` (1920x1080, 30 fps) follows the structure of Flapping Airplanes' demo video for open-flapping-airplanes's `ofa_tc2` build on
one B200: the live dashboard, a zoom out to the GPU, the kernel's code, the Doom player, the run side by side with the baselines and their
launch counts, result cards and an end card. Type and colours are Magni's (`b200/brand/`: Hanken Grotesk and Fragment Mono under the SIL
Open Font License, and the Magni lockup).

- Every number comes from `b200/data.json`, which `b200/make_data.py` derives from the raw results (`b200-run/results/`), each with its
  source file. Nothing is typed into the page.
- The dashboard is a screen recording of a real run (`modal run b200-run/modal_app.py --step video --binary ofa_tc2`, the dashboard
  page with `?theme=magni`); its timing and losses are in `b200-run/results/video_ofa_tc2/`. The cards use the mean of 3 runs without
  the dashboard and screen capture.
- Doom is played by the scripted player in `doom-gpu/player/` (no cheats). The Doom scene is the recorded run's game panel at 3x speed.
- The side by side runs at 5x speed (labelled). The baselines are their measured runs, replayed at their measured mean speed; the loss
  curves are their logged losses; the launch counters use the host launch calls per step and per token measured with torch.profiler.
- The training card shows Megatron-core + Transformer Engine with CUDA graphs (43.1 s, faster than open-flapping-airplanes) and without (62.4 s).

```bash
python3 video/b200/make_data.py --build ofa_tc2                       # data.json + data.js from the results
modal run video/b200/modal_render.py --build ofa_tc2 --stills 5,14,22,30,45,57,66   # PNG stills -> video/out/demo_stills/
modal run video/b200/modal_render.py --build ofa_tc2                  # the MP4 -> video/out/ofa_demo_ofa_tc2.mp4
```
Rendering runs on a Modal CPU container (Google Chrome, node, ffmpeg and the brand fonts): `render_demo.js` drives `demo.html`'s
`seek(t)` frame by frame in parallel chunks and pipes JPEG frames into ffmpeg.
