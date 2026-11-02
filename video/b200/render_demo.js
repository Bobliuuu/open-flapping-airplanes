#!/usr/bin/env node
// Renders demo.html frame by frame (headless Chrome through puppeteer-core) into an MP4 with ffmpeg. The page's seek(t) sets every
// element for time t, so frames are deterministic; the timeline is cut into chunks rendered in parallel pages, then concatenated.
//   node render_demo.js [--out ../out/ofa_demo.mp4] [--stills 2,20,40] [--conc 6] [--chrome path] [--from s --to s]
// Needs node, ffmpeg and Google Chrome; frames/rec_00001.jpg... (the dashboard recording at 30 fps) next to demo.html.
const fs = require("fs"), path = require("path"), {spawn, spawnSync} = require("child_process");
const puppeteer = require("puppeteer-core");
const argv = process.argv.slice(2);
const A = (k, d) => { const i = argv.indexOf("--" + k); return i < 0 ? d : argv[i + 1]; };
const ROOT = __dirname, FPS = 30, W = 1920, H = 1080;
const CHROME = A("chrome", process.env.CHROME || "google-chrome");
const OUT = path.resolve(A("out", path.join(ROOT, "..", "out", "ofa_demo.mp4")));
const FR = path.join(ROOT, "frames");
const NFR = fs.existsSync(FR) ? fs.readdirSync(FR).filter(f => /^rec_\d+\.jpg$/.test(f)).length : 0;

async function newPage(browser) {
  const p = await browser.newPage();
  await p.setViewport({width: W, height: H, deviceScaleFactor: 1});
  await p.evaluateOnNewDocument(n => { window.__NFRAMES = n; }, NFR);
  await p.goto("file://" + path.join(ROOT, "demo.html"));
  await p.waitForFunction("window.READY === true");
  await p.evaluate(() => document.fonts.ready);
  return p;
}
function ffmpegPipe(file) {
  return spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", String(FPS), "-movflags", "+faststart", file],
    {stdio: ["pipe", "inherit", "inherit"]});
}
const LAUNCH = {headless: true, protocolTimeout: 600000,
  args: ["--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1", "--font-render-hinting=none", "--allow-file-access-from-files",
         "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"]};
async function renderChunk(i0, i1, file) {          // one browser per chunk: a background tab may stop decoding images
  const browser = await puppeteer.launch({executablePath: CHROME, ...LAUNCH});
  const page = await newPage(browser), ff = ffmpegPipe(file), done = new Promise(r => ff.on("close", r));
  for (let i = i0; i < i1; i++) {
    await page.evaluate(t => window.seek(t), i / FPS);
    const buf = await page.screenshot({type: "jpeg", quality: 93});
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
  }
  ff.stdin.end(); await done; await page.close(); await browser.close();
}
(async () => {
  const browser = await puppeteer.launch({executablePath: CHROME, ...LAUNCH});
  const probe = await newPage(browser);
  const info = await probe.evaluate(() => window.getInfo());
  console.log(`timeline ${info.duration.toFixed(2)} s, recording frames ${NFR}`, JSON.stringify(info.scenes));
  if (A("stills")) {
    const dir = path.join(path.dirname(OUT), "demo_stills"); fs.mkdirSync(dir, {recursive: true});
    for (const t of String(A("stills")).split(",").map(Number)) {
      await probe.evaluate(t => window.seek(t), t);
      await probe.screenshot({path: path.join(dir, `t${t.toFixed(1).padStart(5, "0")}.png`)});
    }
    await browser.close(); console.log("stills in", dir); return;
  }
  await probe.close(); await browser.close();
  const n0 = Math.round(Number(A("from", 0)) * FPS), n1 = Math.round(Number(A("to", info.duration)) * FPS);
  const conc = Number(A("conc", 6)), per = Math.ceil((n1 - n0) / conc), seg = path.join(path.dirname(OUT), "demo_seg");
  fs.mkdirSync(seg, {recursive: true});
  const parts = [];
  for (let c = 0; c < conc; c++) { const a = n0 + c * per, b = Math.min(n1, a + per); if (a < b) parts.push([a, b, path.join(seg, `part${c}.mp4`)]); }
  const t0 = Date.now();
  await Promise.all(parts.map(([a, b, f]) => renderChunk(a, b, f)));
  const list = path.join(seg, "list.txt");
  fs.writeFileSync(list, parts.map(p => `file '${p[2]}'`).join("\n"));
  const r = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list, "-c", "copy", "-movflags", "+faststart", OUT], {stdio: "inherit"});
  console.log(`wrote ${OUT} (${n1 - n0} frames in ${((Date.now() - t0) / 1000).toFixed(0)} s), ffmpeg status ${r.status}`);
})();
