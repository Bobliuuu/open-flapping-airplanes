# b300-run: these are B300 measurements

Everything in this folder (`B300_REPORT.md`, `B300_ROUND2.md`, `r2_*`, `r3_*`, `r4_*`, `mk_*`, `bl_*`, `gemm_*`) was measured on a Modal
**B300** (SXM6, compute capability 10.3, 148 SMs), not on a B200. The name `b200` in the path comes from the Modal request
`gpu="B200+"`, which can be placed on either chip; one early round-2 call landed on a real B200, see section 0 of
[`B300_ROUND2.md`](B300_ROUND2.md).

The real **B200** (`gpu="B200"`, compute capability 10.0, 148 SMs) results are in [`../../../b200-run/results/`](../../../b200-run/results/),
summarised in [`../../../b200-run/README.md`](../../../b200-run/README.md). Do not mix the two when comparing with Flapping Airplanes' B200 numbers.

The folder keeps its path because scripts and documents refer to it.
