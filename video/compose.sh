#!/usr/bin/env bash
# Insert the live-demo screen recording into the explainer and re-encode the final MP4.
# usage: ./compose.sh <recording.mp4> [output.mp4]
set -euo pipefail
cd "$(dirname "$0")"
REC="${1:?usage: compose.sh <recording.mp4> [output.mp4]}"
OUT="${2:-out/triad_explainer.mp4}"
MAXS="${MAXS:-30}"          # cap the recording at this many seconds (override with MAXS=45)
mkdir -p out/seg
# scenes other than the slot are reused if already rendered; render any missing one
need=0; for id in $(node -e 'const d=require("./data.json");console.log(d.scenes.map(s=>s.id).join(" "))'); do [ -f out/seg/$id.mp4 ] || need=1; done
[ $need = 1 ] && node render.js
# transparent header + caption overlay for the slot
node render.js --overlay slot
# recording: scaled to fit a 1760x860 window under the header, letterboxed on the dark background, caption on top
ffmpeg -y -loglevel error -i "$REC" -loop 1 -i out/overlay_slot.png -t "$MAXS" \
  -filter_complex "color=c=0x06080c:s=1920x1080:r=30[bg];[0:v]fps=30,scale=1760:860:force_original_aspect_ratio=decrease:force_divisible_by=2[v];[bg][v]overlay=(W-w)/2:110:shortest=1[b];[b][1:v]overlay=0:0:shortest=1,format=yuv420p" \
  -an -c:v libx264 -preset medium -crf 16 -r 30 -movflags +faststart out/seg/slot_rec.mp4
# concat in scene order, swapping the placeholder for the recording
list=out/concat_final.txt; : > $list
for id in $(node -e 'const d=require("./data.json");console.log(d.scenes.map(s=>s.id).join(" "))'); do
  if [ "$id" = slot ]; then echo "file '$PWD/out/seg/slot_rec.mp4'" >> $list; else echo "file '$PWD/out/seg/$id.mp4'" >> $list; fi
done
ffmpeg -y -loglevel error -f concat -safe 0 -i $list -c copy "$OUT"
echo "wrote $OUT"
