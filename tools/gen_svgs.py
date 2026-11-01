#!/usr/bin/env python3
"""Generates the animated SVGs used by the README and the stats site (site/assets/*.svg). Pure SMIL animation: no scripts, plays inside <img>."""
import os, math
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'site', 'assets')
BG, PANEL, LINE, DIM, TXT = '#07090d', '#0d121a', '#1d2633', '#7a8798', '#d6dee8'
CY, RD, YL, GR = '#27c8e8', '#ff5757', '#f2c14e', '#5be08a'
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif"

def orbit(cx, cy, r, color, dur, begin, size=7):
    p = f"M {cx},{cy - r} A {r},{r} 0 1,1 {cx - 0.01},{cy - r}"
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-opacity=".18" stroke-dasharray="3 7"/>'
            f'<circle r="{size}" fill="{color}"><animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" path="{p}"/>'
            f'<animate attributeName="r" values="{size};{size + 3};{size}" dur="2s" repeatCount="indefinite"/></circle>'
            f'<circle r="{size * 2.6}" fill="{color}" opacity=".16"><animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" path="{p}"/></circle>')

def hero():
    cx, cy = 930, 180
    s = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 360" width="1200" height="360" role="img" aria-label="open-flapping-airplanes: one kernel, three programs">
<defs>
<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{CY}"/><stop offset=".55" stop-color="{RD}"/><stop offset="1" stop-color="{YL}"/></linearGradient>
<radialGradient id="glow"><stop offset="0" stop-color="{CY}" stop-opacity=".45"/><stop offset="1" stop-color="{CY}" stop-opacity="0"/></radialGradient>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="{LINE}" stroke-width="1" opacity=".5"/></pattern>
<linearGradient id="sweep" x1="0" x2="1"><stop offset="0" stop-color="{CY}" stop-opacity="0"/><stop offset=".5" stop-color="{CY}" stop-opacity=".55"/><stop offset="1" stop-color="{CY}" stop-opacity="0"/></linearGradient>
</defs>
<rect width="1200" height="360" fill="{BG}"/><rect width="1200" height="360" fill="url(#grid)"/>
<rect y="0" width="260" height="360" fill="url(#sweep)" opacity=".5"><animateTransform attributeName="transform" type="translate" from="-300 0" to="1300 0" dur="7s" repeatCount="indefinite"/></rect>
<text x="64" y="188" font-family="{SANS}" font-weight="800" font-size="150" letter-spacing="-4" fill="url(#g)">OFA</text>
<text x="70" y="240" font-family="{FONT}" font-size="26" fill="{TXT}">one kernel <tspan fill="{DIM}">·</tspan> three programs</text>
<g font-family="{FONT}" font-size="17">
<text x="70" y="288" fill="{CY}">● train</text><text x="188" y="288" fill="{RD}">● serve</text><text x="306" y="288" fill="{YL}">● play Doom</text>
<text x="70" y="318" fill="{DIM}">all at once, in a single GPU launch</text></g>
<circle cx="{cx}" cy="{cy}" r="150" fill="url(#glow)"><animate attributeName="r" values="130;165;130" dur="4s" repeatCount="indefinite"/></circle>
{orbit(cx, cy, 70, CY, 5, 0)}{orbit(cx, cy, 108, RD, 7, -2.3)}{orbit(cx, cy, 146, YL, 10, -5.1)}
<rect x="{cx - 38}" y="{cy - 38}" width="76" height="76" rx="14" fill="{PANEL}" stroke="url(#g)" stroke-width="2.5"><animate attributeName="stroke-width" values="2.5;5;2.5" dur="2.4s" repeatCount="indefinite"/></rect>
<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-family="{FONT}" font-weight="700" font-size="30" fill="#fff">1</text>
<text x="{cx}" y="{cy + 24}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{DIM}">launch</text>
</svg>'''
    return s

def flow(path, color, dur, begin, size=5):
    return (f'<circle r="{size}" fill="{color}"><animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" path="{path}" keyPoints="0;1" keyTimes="0;1" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;.1;.85;1" dur="{dur}s" begin="{begin}s" repeatCount="indefinite"/></circle>')

def architecture():
    W, H = 1200, 850
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="open-flapping-airplanes architecture">',
         f'<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{DIM}"/></marker>'
         f'<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{CY}"/><stop offset=".55" stop-color="{RD}"/><stop offset="1" stop-color="{YL}"/></linearGradient></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    def box(x, y, w, h, title, sub='', color=LINE, tc=TXT, dash=False):
        d = ' stroke-dasharray="6 6"' if dash else ''
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{PANEL}" stroke="{color}" stroke-width="1.6"{d}/>'
        s += f'<text x="{x + 12}" y="{y + 22}" font-family="{FONT}" font-size="13" font-weight="700" fill="{tc}">{title}</text>'
        if sub: s += f'<text x="{x + 12}" y="{y + 42}" font-family="{FONT}" font-size="11.5" fill="{DIM}">{sub}</text>'
        return s
    # GPU frame
    o.append(f'<rect x="262" y="24" width="918" height="802" rx="16" fill="none" stroke="url(#g)" stroke-width="2" stroke-dasharray="10 8"><animate attributeName="stroke-dashoffset" from="0" to="-72" dur="3s" repeatCount="indefinite"/></rect>')
    o.append(f'<text x="282" y="52" font-family="{FONT}" font-size="15" font-weight="700" fill="#fff">ONE KERNEL LAUNCH</text><text x="470" y="52" font-family="{FONT}" font-size="13" fill="{DIM}">216 persistent blocks · 2 per SM · A100 (108 SMs)</text>')
    # host side
    o.append(box(20, 70, 210, 200, 'CPU host', '', LINE))
    o.append(f'<text x="32" y="112" font-family="{FONT}" font-size="12" fill="{TXT}">HTTP server thread</text><text x="32" y="134" font-family="{FONT}" font-size="12" fill="{TXT}">sampler (10 Hz)</text>'
             f'<text x="32" y="168" font-family="{FONT}" font-size="12" fill="{DIM}">reads pinned memory:</text><text x="32" y="188" font-family="{FONT}" font-size="12" fill="{GR}">· frame 65 KB (8-bit + palette)</text>'
             f'<text x="32" y="208" font-family="{FONT}" font-size="12" fill="{GR}">· text ring · loss · counters</text>')
    o.append(box(20, 560, 210, 110, 'Browser', 'three panels, 35 fps', LINE))
    o.append(f'<path d="M125 270 V560" stroke="{DIM}" stroke-width="1.6" stroke-dasharray="5 6" marker-end="url(#ar)"><animate attributeName="stroke-dashoffset" from="0" to="-22" dur="1.2s" repeatCount="indefinite"/></path>'
             f'<text x="136" y="420" font-family="{FONT}" font-size="11" fill="{DIM}">SSH tunnel</text><text x="136" y="436" font-family="{FONT}" font-size="11" fill="{DIM}">/stream, /state</text>')
    o.append(f'<path d="M262 170 H230" stroke="{GR}" stroke-width="1.6" stroke-dasharray="4 5" marker-end="url(#ar)"><animate attributeName="stroke-dashoffset" from="0" to="-18" dur="1s" repeatCount="indefinite"/></path><text x="236" y="160" font-family="{FONT}" font-size="10" fill="{GR}">pinned</text>')
    # queues
    lanes = [('TRAINING', CY, 'fwd → softmax → bwd + SGD', '144 tasks / step', 120), ('INFERENCE', RD, 'hidden → logits → sample', '33 tasks / token', 178), ('DOOM', YL, '1 tick + 5 copy tasks', '35 frames / s', 236)]
    o.append(f'<text x="282" y="76" font-family="{FONT}" font-size="11" fill="{DIM}">SHARDED TICKET QUEUES (claim with CAS, home shard per block)</text>')
    for name, col, desc, rate, y in lanes:
        o.append(f'<rect x="282" y="{y - 20}" width="560" height="48" rx="7" fill="{PANEL}" stroke="{col}" stroke-opacity=".55"/>')
        o.append(f'<text x="294" y="{y - 1}" font-family="{FONT}" font-size="12.5" font-weight="700" fill="{col}">{name}</text><text x="294" y="{y + 16}" font-family="{FONT}" font-size="10.5" fill="{DIM}">{desc} · {rate}</text>')
        for i in range(16):
            x = 628 + i * 13
            o.append(f'<rect x="{x}" y="{y - 10}" width="9" height="28" rx="2" fill="{col}" opacity=".18"><animate attributeName="opacity" values=".18;.9;.18" keyTimes="0;.15;1" dur="3.2s" begin="{(15 - i) * 0.2:.1f}s" repeatCount="indefinite"/></rect>')
    o.append('<g transform="translate(0,34)">')
    # counters
    o.append(box(282, 262, 280, 118, 'dependency counters', '', LINE))
    o.append(f'<text x="294" y="302" font-family="{FONT}" font-size="11" fill="{DIM}">task done → fence → atomicAdd</text><text x="294" y="320" font-family="{FONT}" font-size="11" fill="{DIM}">next phase: counter ≥ target</text>')
    for i, col in enumerate([RD, RD, CY, CY, CY, YL, YL]):
        x = 296 + i * 36
        o.append(f'<rect x="{x}" y="{366 - 30}" width="22" height="30" rx="3" fill="{LINE}"/><rect x="{x}" y="336" width="22" height="30" rx="3" fill="{col}"><animate attributeName="height" values="4;30;30;4" keyTimes="0;.5;.9;1" dur="{2.4 + i * 0.35:.2f}s" repeatCount="indefinite"/><animate attributeName="y" values="362;336;336;362" keyTimes="0;.5;.9;1" dur="{2.4 + i * 0.35:.2f}s" repeatCount="indefinite"/></rect>')
    # scheduler
    o.append(box(590, 262, 252, 118, 'warp 0: scheduler', '', CY, CY))
    o.append(f'<text x="602" y="302" font-family="{FONT}" font-size="11" fill="{TXT}">1 probe round, 32 lanes:</text><text x="602" y="320" font-family="{FONT}" font-size="11" fill="{DIM}">ticket → dependency → ready?</text><text x="602" y="338" font-family="{FONT}" font-size="11" fill="{DIM}">priority: Doom > infer > train</text><text x="602" y="356" font-family="{FONT}" font-size="11" fill="{DIM}">claim with CAS → run task</text>')
    o.append(f'<circle cx="822" cy="282" r="5" fill="{CY}"><animate attributeName="opacity" values="1;.2;1" dur="1s" repeatCount="indefinite"/></circle>')
    o.append('</g>')
    # flows: lanes -> scheduler -> workers
    for j, (name, col, desc, rate, y) in enumerate(lanes):
        for k in range(3):
            o.append(flow(f'M 842 {y + 4} C 900 {y + 4} 716 250 716 296', col, 3.4, j * 0.5 + k * 1.1))
    o.append('<g transform="translate(0,34)">')
    # workers
    o.append(f'<text x="282" y="412" font-family="{FONT}" font-size="11" fill="{DIM}">COMPUTE WARPS (8 per block, tasks run from registers + L2)</text>')
    for i in range(8):
        x = 282 + i * 70
        col = [CY, RD, YL, CY, CY, RD, CY, YL][i]
        o.append(f'<rect x="{x}" y="424" width="62" height="44" rx="6" fill="{PANEL}" stroke="{col}" stroke-opacity=".5"/><rect x="{x}" y="424" width="62" height="44" rx="6" fill="{col}" opacity="0"><animate attributeName="opacity" values="0;.55;0" keyTimes="0;.2;1" dur="2.6s" begin="{i * 0.33:.2f}s" repeatCount="indefinite"/></rect><text x="{x + 31}" y="451" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{TXT}">W{i}</text>')
    for k in range(3):
        o.append(flow('M 716 380 C 716 400 560 410 560 424', CY, 2.6, k * 0.9, 4)); o.append(flow('M 716 380 C 716 400 400 410 400 424', RD, 2.6, 0.4 + k * 0.9, 4)); o.append(flow('M 716 380 C 716 400 700 410 700 424', YL, 2.6, 0.7 + k * 0.9, 4))
    # Doom block detail
    o.append(f'<rect x="282" y="500" width="880" height="256" rx="12" fill="none" stroke="{YL}" stroke-opacity=".6"/>')
    o.append(f'<text x="298" y="524" font-family="{FONT}" font-size="13" font-weight="700" fill="{YL}">DOOM TICK: one block, all 256 threads</text><text x="690" y="524" font-family="{FONT}" font-size="11" fill="{DIM}">real engine (doomgeneric), compiled for the GPU by clang</text>')
    o.append(box(298, 540, 250, 200, 'thread 0', 'runs the game', YL, YL))
    o.append(f'<text x="310" y="596" font-family="{FONT}" font-size="11" fill="{TXT}">· AI, physics, BSP walk</text><text x="310" y="618" font-family="{FONT}" font-size="11" fill="{TXT}">· bot input (iddqd, idfa…)</text><text x="310" y="640" font-family="{FONT}" font-size="11" fill="{TXT}">· queue column + span draws</text><text x="310" y="658" font-family="{FONT}" font-size="11" fill="{DIM}">  (ordered per screen column)</text>'
             f'<text x="310" y="684" font-family="{FONT}" font-size="11" fill="{TXT}">· flush at planes / masked</text><text x="310" y="716" font-family="{FONT}" font-size="11" fill="{DIM}">6.7 ms / frame (was 24 ms)</text>')
    o.append(f'<rect x="596" y="540" width="14" height="200" rx="5" fill="{LINE}"/><rect x="596" y="540" width="14" height="200" rx="5" fill="{CY}"><animate attributeName="opacity" values=".15;1;.15" keyTimes="0;.12;1" dur="2s" repeatCount="indefinite"/></rect>'
             f'<text x="603" y="760" text-anchor="middle" font-family="{FONT}" font-size="10.5" fill="{CY}">bar.sync</text>')
    o.append(f'<path d="M548 640 H596" stroke="{YL}" stroke-width="2" marker-end="url(#ar)"/><path d="M610 640 H650" stroke="{CY}" stroke-width="2" marker-end="url(#ar)"/>')
    o.append(f'<text x="668" y="560" font-family="{FONT}" font-size="12" font-weight="700" fill="{TXT}">threads 1..255: workers</text><text x="668" y="578" font-family="{FONT}" font-size="11" fill="{DIM}">each owns screen columns x, x+256</text>')
    for r in range(8):
        for c in range(32):
            x, y = 668 + c * 14, 592 + r * 14
            o.append(f'<rect x="{x}" y="{y}" width="10" height="10" rx="2" fill="{YL}" opacity=".12"><animate attributeName="opacity" values=".12;.95;.12" keyTimes="0;.1;1" dur="2s" begin="{(c * 0.02 + r * 0.03):.2f}s" repeatCount="indefinite"/></rect>')
    o.append(f'<text x="668" y="728" font-family="{FONT}" font-size="11" fill="{TXT}">→ 8-bit frame + palette → pinned host memory → browser</text><text x="668" y="746" font-family="{FONT}" font-size="11" fill="{DIM}">2 flushes per frame, pixel-identical to serial drawing</text>')
    o.append('</g></svg>')
    return '\n'.join(o)

def doom_pipeline():
    steps = [('game tick', 'thread 0: AI, BSP', YL), ('queue draws', 'columns + spans', YL), ('bar.sync', 'wake 255 workers', CY), ('parallel draw', '256 threads', CY), ('bar.sync', 'join', CY), ('copy tasks', '5 blocks, 8-bit', GR), ('pinned mem', 'frame + palette', GR), ('browser', '65 KB stream', RD)]
    W, H, n, bw, gap = 1200, 250, 8, 124, 30
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Doom frame pipeline">', f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         f'<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{DIM}"/></marker></defs>',
         f'<text x="30" y="34" font-family="{FONT}" font-size="14" font-weight="700" fill="#fff">ONE DOOM FRAME</text><text x="190" y="34" font-family="{FONT}" font-size="12" fill="{DIM}">every 28.6 ms, inside the kernel</text>']
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    for i, (t, s, col) in enumerate(steps):
        x = x0 + i * (bw + gap); y = 80
        kt = f'0;{i / n:.4f};{(i + 0.5) / n:.4f};{(i + 1) / n:.4f};1'
        o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="96" rx="10" fill="{PANEL}" stroke="{LINE}" stroke-width="1.6"><animate attributeName="stroke" values="{LINE};{LINE};{col};{LINE};{LINE}" keyTimes="{kt}" dur="8s" repeatCount="indefinite"/><animate attributeName="stroke-width" values="1.6;1.6;4;1.6;1.6" keyTimes="{kt}" dur="8s" repeatCount="indefinite"/></rect>')
        o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="96" rx="10" fill="{col}" opacity="0"><animate attributeName="opacity" values="0;0;.22;0;0" keyTimes="{kt}" dur="8s" repeatCount="indefinite"/></rect>')
        o.append(f'<text x="{x + bw / 2}" y="{y + 40}" text-anchor="middle" font-family="{FONT}" font-size="13.5" font-weight="700" fill="{col}">{t}</text><text x="{x + bw / 2}" y="{y + 62}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{DIM}">{s}</text><text x="{x + bw / 2}" y="{y + 84}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{DIM}">{i + 1}</text>')
        if i < n - 1:
            o.append(f'<path d="M{x + bw + 3} {y + 48} H{x + bw + gap - 3}" stroke="{DIM}" stroke-width="1.8" marker-end="url(#ar)"/>')
    o.append(f'<circle r="6" fill="#fff"><animate attributeName="cx" values="{x0 + bw / 2};{x0 + 7 * (bw + gap) + bw / 2}" dur="8s" repeatCount="indefinite"/><animate attributeName="cy" values="205;205" dur="8s" repeatCount="indefinite"/></circle>'
             f'<rect x="{x0}" y="203" width="{7 * (bw + gap) + bw}" height="4" rx="2" fill="{LINE}"/>')
    o.append(f'<text x="30" y="238" font-family="{FONT}" font-size="11" fill="{DIM}">serial Doom frame: 24 ms on one GPU thread · with the parallel drawing queue: 6.7 ms · boot 0.9 s → 0.3 s</text>')
    o.append('</svg>')
    return '\n'.join(o)

os.makedirs(OUT, exist_ok=True)
for name, fn in (('hero.svg', hero), ('architecture.svg', architecture), ('doom-pipeline.svg', doom_pipeline)):
    open(os.path.join(OUT, name), 'w').write(fn()); print('wrote', name, os.path.getsize(os.path.join(OUT, name)) // 1024, 'KB')
