# Benchmarks

Measures the megakernel (training + inference + Doom in **one kernel launch**) against the same work launched the usual way, and lines the
result up with the numbers Flapping Airplanes publish in their demo video.

```bash
./run_benchmarks.sh ubuntu@<ip> 5 6     # 5 repetitions x 6 s per measurement; pulls results/a100_<date>.json, runs compare.py
python3 compare.py results/latest.json   # re-derive results/summary.md and ../site/data/* from a saved run
```
Then open `../site/index.html` (see `../site/README.md`).

## What is measured

Every number is a mean over repetitions with its standard deviation (`results/*.json` keeps the raw values).

| config | what runs | why |
|---|---|---|
| `mega_all` | training + inference (unpaced) + Doom at 35 Hz, one kernel launch | the claim: all three at once |
| `mega_all_live` | same, inference paced at 1000 tok/s | the live demo configuration; gives the SM split |
| `mega_train`, `mega_infer`, `mega_doom` | each job alone in the megakernel | cost of sharing; Doom capacity (unpaced) |
| `eager_train`, `eager_infer` | the same task code as ordinary per-op kernels launched one by one, each **alone** on the GPU | the video's method: Megatron and vLLM "each timed individually" |
| `graph_train`, `graph_infer` | same, one CUDA graph per step/token, each alone | best case for the baseline |
| `streams_both` | both per-op programs at once in one process on two CUDA-graph streams | concurrent baseline |
| `procs_both` | both per-op programs as two OS processes time-sharing the GPU | what two separate frameworks on one GPU do |

Doom cannot be re-created inside one process (the engine keeps static state), so every Doom measurement runs in its own process.
Per-op launch counts are 4 kernels per training step and 3 per generated token.

## Derived metrics (`compare.py`)

- **baseline slower %** = `(1 - baseline_rate / megakernel_rate) * 100`; negative means the baseline is faster.
- **time for 4,000 steps**, the video's headline training metric.
- **GPU-equivalents** = `mega_train/solo_train + mega_infer/solo_infer`: how many GPUs of per-op baselines deliver what the megakernel delivers on one (Doom has no baseline).
- **SM busy** = summed task time per job divided by wall time, over two resident blocks per SM.

## What Flapping Airplanes publish

`fa_published.json`, read frame by frame from the HD video: 1 vs 514,562 kernel launches; 4,000 training steps in 46.1 s vs 70.1 s (baseline 34% slower);
inference about 1,076 vs 710 tok/s (baseline 34% slower); SM split 116 training / 24 inference / 8 Doom of 148 on one B200; Doom plays throughout.

## How to read the comparison honestly

- **Different hardware** (A100, 108 SMs vs B200, 148 SMs) and a **different model**: ours is a tiny character-level MLP language model (512 hidden units, 96-way softmax, about 3.5M training tokens/s); theirs is an LLM-sized vocabulary at about 355k tokens/s. Absolute numbers are not comparable; ratios are the meaningful part.
- Their baseline is Megatron/vLLM, whose per-step cost includes many more tiny kernels; our per-op baseline reuses the megakernel's own task code, so it is a *tight* baseline with few launches.
- Our scheduler is dynamic, so the "SM split" is measured busy time, not a static partition. Our workload leaves most of the GPU idle; theirs fills a B200.
