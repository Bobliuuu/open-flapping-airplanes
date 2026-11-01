"""Launch counting helpers (BASELINE v2). Three definitions, all from a torch.profiler (CUPTI) chrome trace:
  (a) kernels      = GPU kernel executions (trace category "kernel"); memcpy/memset reported separately and NOT included.
  (b) host_launch  = host-side launch API calls: cudaLaunchKernel, cudaLaunchKernelExC, cudaLaunchCooperativeKernel, cudaGraphLaunch,
                     (+ driver cuLaunchKernel / cuLaunchKernelEx / cuGraphLaunch when they are recorded as separate non-nested events).
                     A graph launch counts as ONE call, however many kernels the graph contains.
  (c) host_launch while the unit of work (training step / decode token) is replayed from ONE CUDA graph = (b) measured in graph mode.
Usage: with profile(...) as prof: ...; counts = trace_counts(prof, nunits)."""
import json, os, tempfile, collections

LAUNCH_NAMES = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel", "cudaGraphLaunch",
                "cuLaunchKernel", "cuLaunchKernelEx", "cuLaunchCooperativeKernel", "cuGraphLaunch",
                "cudaLaunchKernel_ptsz", "cudaGraphLaunch_ptsz"}

def trace_counts(prof, nunits):
    f = tempfile.mktemp(suffix=".json"); prof.export_chrome_trace(f)
    ev = json.load(open(f))["traceEvents"]; os.remove(f)
    cat = collections.Counter(); launch = collections.Counter(); kern = collections.Counter()
    for e in ev:
        c = e.get("cat"); n = e.get("name", "")
        if e.get("ph") != "X": continue
        cat[c] += 1
        if c == "kernel": kern[n] += 1
        if c in ("cuda_runtime", "cuda_driver") and n in LAUNCH_NAMES: launch[(c, n)] += 1
    nk = cat["kernel"]; nm = cat["gpu_memcpy"] + cat["gpu_memset"]
    nb = sum(launch.values())
    top = kern.most_common(12)
    return {"nunits": nunits, "kernels_per_unit": nk / nunits, "memops_per_unit": nm / nunits, "host_launch_calls_per_unit": nb / nunits,
            "host_launch_breakdown_per_unit": {f"{c}:{n}": v / nunits for (c, n), v in sorted(launch.items())},
            "top_kernels_per_unit": [(n[:90], v / nunits) for n, v in top], "distinct_kernel_names": len(kern)}
