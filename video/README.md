# video/: open-flapping-airplanes explainer (data-driven, rendered with headless Chrome + ffmpeg)

The explainer is one HTML/SVG/CSS page (`index.html`). `render.js` drives headless Chrome (puppeteer-core, system Google Chrome) frame by
frame at 1920x1080, 30 fps, pipes JPEG frames into ffmpeg (libx264, crf 16), writes one MP4 segment per scene, then concatenates them.
Every number in the video is read from `data.json`; each value carries a `source` (a result file in this repo, or Flapping Airplanes'
published figure). It covers one NVIDIA B200 only.

## Render
```
cd video && npm install            # once: puppeteer-core only (no browser download; uses /Applications/Google Chrome.app)
node render.js                     # -> out/ofa_explainer.mp4 (98 s, about 3 min, 3 scenes in parallel; --conc N)
node render.js --scene resA        # re-render one scene segment only (out/seg/resA.mp4), then run plain `node render.js` to concatenate
node render.js --stills 20,50,76   # PNGs at global times (s) into out/stills/ for a quick visual check
CHROME=/path/to/chrome node render.js   # other Chrome/Chromium
```
Needs ffmpeg and node on PATH. The release copy is `release/ofa_explainer.mp4` (copy `out/ofa_explainer.mp4` there after a render).

## Scenes (98 s)
title (4 s), problem (7), launches: which definition of "launch" (9), idea: one kernel, the 116 / 24 / 8 SM split (9),
arch (10), doom (10), method (8), dash: live dashboard still, `assets/dashboard.png` (9), resA: launches and training (14),
resB: inference, loss and "what it means" (12), end (6). Scene lengths and burned-in captions are in `data.json` under `scenes`
(`[start_s, end_s, text]` relative to the scene).

## Updating numbers
Edit `data.json`. A number is `{value, unit, status, source}`; the sources are `b200-run/README.md` and `b200-run/results/` for the B200
rows and `benchmarks/fa_published.json` for Flapping Airplanes' published column. Nothing is typed into the scenes, so change the value
in `data.json`, check the stills, and re-render. There is no script that rewrites `data.json` any more (the old helpers read result
files that were moved or replaced); keep it in step with the result files by hand.
Percentages use FA's convention, 1 - faster/slower (70.1 s vs 46.1 s = 34%).

## Other video
`release/ofa_demo_b200.mp4` is the demo in Flapping Airplanes' format (a recorded B200 run, the split, the code, the side by side, the
result cards, their published benchmark next to ours, the end card), with music. It is built separately from `b200/` (`b200/make_data.py`
reads `b200-run/results/` and `benchmarks/fa_published.json`, `b200/render_demo.js` renders it, `b200/make_music.py` composes the music on
the same timeline with numpy, no samples) and needs the recorded dashboard frames (`modal run b200-run/modal_app.py --step video`, which
draws the dashboard at 2x so it stays readable at half size). `modal run video/b200/modal_render.py --fa-video their_demo.mp4` adds a scene
with their dashboard next to ours at the same run time, until their run ends at 46.155 s; their footage is not committed to this repository
(that render goes to `out/`, which is not tracked). The README plays a copy of it uploaded to GitHub (H.264, 1080p, under 10 MB).

Large outputs (`out/`, `node_modules/`, any other `*.mp4`) are ignored by `video/.gitignore`; only `release/*.mp4` is tracked.
Not affiliated with Flapping Airplanes.
