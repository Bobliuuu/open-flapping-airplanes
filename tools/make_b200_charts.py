import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTE = "#0d1117", "#e6edf3", "#8b949e"
GREEN, BLUE, RED, GREY = "#3fb950", "#58a6ff", "#f85149", "#6e7681"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTE, "ytick.color": FG,
                     "font.family": "DejaVu Sans"})

def hbar(fname, title, sub, rows, unit, xmax, note=None):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    bars = ax.barh(labels, vals, color=cols, height=0.62)
    for b, v, r in zip(bars, vals, rows[::-1]):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2, r[3], va="center", fontsize=12, color=FG)
    ax.set_xlim(0, xmax)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.set_xlabel(unit, color=MUTE)
    fig.text(0.04, 0.95, title, fontsize=19, fontweight="bold", va="top")
    fig.text(0.04, 0.885, sub, fontsize=11.5, color=MUTE, va="top")
    if note:
        import textwrap
        fig.text(0.04, 0.025, "\n".join(textwrap.wrap(note, 105)), fontsize=9.5, color=MUTE)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.80, bottom=0.17)
    fig.savefig(fname)
    plt.close(fig)

# 1: training, B200, 4,000 steps, 116 SMs  (lower is better)
hbar("1_training.png", "Train 4,000 steps on 116 SMs",
     "B200 · 162M-param GPT · seconds, lower is better",
     [("open-flapping-airplanes (tcgen05)", 48.1, GREEN, "48.1 s"),
      ("Megatron + TE, CUDA graphs", 43.1, BLUE, "43.1 s"),
      ("PyTorch + flash-attn, graph", 50.9, GREY, "50.9 s"),
      ("PyTorch + flash-attn, eager", 52.5, GREY, "52.5 s"),
      ("Megatron + TE, eager", 62.4, BLUE, "62.4 s"),
      ("FA video: their kernel", 46.1, RED, "46.1 s"),
      ("FA video: their Megatron", 70.1, RED, "70.1 s")],
     "seconds", 80,
     "Each baseline ran alone on the same 116 SMs (CUDA MPS). Same data, init weights, batch order. Final loss 2.1575. Red = numbers read from the FA video.")

# 2: serving
hbar("2_serving.png", "Serve \"What is a megakernel?\" on 24 SMs",
     "B200 · tokens/s from the live training weights, higher is better",
     [("open-flapping-airplanes (2-stage)", 1741, GREEN, "1,741"),
      ("open-flapping-airplanes (3-stage)", 1660, GREEN, "1,660"),
      ("vLLM, batch 1, CUDA graphs", 1537, BLUE, "1,537"),
      ("FA video: their kernel", 1076, RED, "1,076"),
      ("FA video: their vLLM", 710, RED, "710")],
     "tokens / second", 2000,
     "vLLM got 16 reserved CPU cores. At batch 1 it is CPU-bound: with starved cores it did 684 tok/s. Red = read from the FA video.")

# 3: shared vs sequential
hbar("3_sharing.png", "Does one kernel save compute?",
     "B200 · 4,000 training steps + 91k served tokens, seconds, lower is better",
     [("One after the other (Megatron graphs, then vLLM)", 102, GREY, "~102 s"),
      ("Side by side on SM shares (estimate)", 59, BLUE, "~59 s, est."),
      ("open-flapping-airplanes, one kernel", 52.4, GREEN, "52.4 s")],
     "seconds", 125,
     "Most of the win is sharing the GPU (~2x); the single kernel itself adds ~12% at this scale.")
print("ok")
