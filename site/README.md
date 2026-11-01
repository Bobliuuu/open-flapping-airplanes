# Stats site

A static page (no build step, no network) that renders the benchmark results. It reads `data/benchmarks.js`, which `benchmarks/compare.py` writes.

```bash
python3 -m http.server 8000 --directory site     # then open http://localhost:8000
# or just open site/index.html (the data loads as a script, so file:// works)
```

Sections: the three programs, the headline numbers, every benchmark configuration with mean and standard deviation, the baselines compared with the
megakernel, and a side-by-side table against the numbers Flapping Airplanes publish in their demo video. To refresh it, run
`benchmarks/run_benchmarks.sh ubuntu@<ip>` or `python3 benchmarks/compare.py benchmarks/results/latest.json`.
