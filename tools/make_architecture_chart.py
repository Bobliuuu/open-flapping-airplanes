import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, FG, MUTE = "#0d1117", "#e6edf3", "#8b949e"
GREEN, BLUE, AMBER, GREY, LINE = "#3fb950", "#58a6ff", "#f2c14e", "#21262d", "#30363d"

fig = plt.figure(figsize=(12, 6.75), dpi=200, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(0, 67.5); ax.axis("off")

def box(x, y, w, h, fc, ec=None, r=1.2, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=lw))

def t(x, y, s, size=11, color=FG, ha="left", weight="normal", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight)

def arrow(x1, y1, x2, y2, color=MUTE, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.4, connectionstyle=f"arc3,rad={rad}"))

t(4, 63, "One launch. Three programs. 148 SMs.", 20, weight="bold")
t(4, 58.6, "A persistent kernel: blocks never exit; they claim work with an atomicAdd", 12, MUTE)

# --- GPU strip (SM split)
t(4, 52, "ONE persistent kernel launch  ·  B200", 10.5, MUTE, weight="bold")
x0, W = 4, 112
parts = [(116, GREEN, "TRAIN  ·  116 SMs"), (24, BLUE, "SERVE  ·  24 SMs"), (8, AMBER, "DOOM")]
x = x0
for n, c, lab in parts:
    w = W * n / 148
    box(x, 43, w - 0.4, 6.4, c, r=0.8)
    t(x + (w - 0.4) / 2, 46.2, lab, 10.5 if n > 10 else 8.5, "#0d1117", ha="center", weight="bold")
    x += w
t(4, 40.6, "static split by block index: no preemption, no host in the loop", 10, MUTE)

# --- three columns
cols = [
    (4, 36.5, GREEN, "TRAIN", ["fwd + bwd + AdamW for a GPT", "102 phases, 125,942 tasks / step", "tensor-core GEMMs (tcgen05 + TMA)", "loss 11.7 → 2.16 in 4,000 steps"]),
    (43.3, 36.5, BLUE, "SERVE", ["decodes from the LIVE weights", "14 phases, 2,705 tasks / token", "answers \"What is a megakernel?\"", "again and again while it trains"]),
    (82.6, 36.5, AMBER, "DOOM", ["real C engine, clang → NVPTX", "thread 0 runs the game tic", "255 threads draw walls + floors", "frames stream to the browser"]),
]
for cx, top, c, name, lines in cols:
    box(cx, 16.5, 33.4, 21, "#161b22", LINE)
    ax.add_patch(plt.Rectangle((cx + 1, 35.4), 1.2, 1.9, color=c))
    t(cx + 3.2, 36.4, name, 12.5, c, weight="bold")
    for i, s in enumerate(lines):
        t(cx + 2, 31.6 - i * 3.6, s, 9.6, FG)

# --- scheduler loop
box(4, 3, 112, 11.4, "#161b22", LINE)
t(6, 12.2, "HOW BLOCKS GET WORK", 10, MUTE, weight="bold")
steps = ["ticket counter\n(one per job)", "atomicAdd\nclaims a task", "wait: acquire-load\nof phase counters", "run the task\n(GEMM, norm, attn…)", "fence, then atomicAdd\nphase counter"]
sx, sw, gap = 6, 18.0, 3.5
for i, s in enumerate(steps):
    xx = sx + i * (sw + gap)
    box(xx, 4.2, sw, 6.2, GREY, LINE, r=0.8)
    t(xx + sw / 2, 7.3, s, 9.2, FG, ha="center")
    if i < len(steps) - 1:
        arrow(xx + sw + 0.3, 7.3, xx + sw + gap - 0.3, 7.3)

fig.savefig("4_architecture.png")
print("ok")
