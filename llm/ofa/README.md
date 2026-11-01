# llm/ofa: training + inference + real Doom in ONE kernel (A100)

`ofa.cu` = the performance-tuned LLM megakernel (`llm/megakernel/`, fetch-add task claim, AdamW overlap, config F) **plus** real Doom on a
6-SM partition and the live dashboard (`dashboard.cuh`, `index.html`). One persistent launch: 216 blocks, SMs 0-16 serve tokens, 17-22 run
Doom (35 Hz), the other 85 train. Doom runs its own `doom_loop`, so the new claim path never touches it.

```
MODEL=S ./run_ofa.sh ubuntu@<ip> 8080        # 162M params, 12 layers, d=768 (default)
MODEL=F ./run_ofa.sh ubuntu@<ip> 8080        # ~408M params, 4 layers, d=2048 (-DMODEL_F), needs ~/data/init_flat_F.f32
```
Flags: `--no-doom` (leave the Doom SMs idle), `--exit` (with `--port`, quit when done instead of serving the final screen forever; use it
under `flock`/`timeout` or the process holds the GPU lock). `shot.js` takes a headless-Chrome DevTools screenshot (`--screenshot` hangs on `/stream`).

## Results (A100-SXM4-40GB, 85 train / 17 infer / 6 Doom SMs, 4,096 tokens/step)

| check | result |
|---|---|
| loss vs `llm/megakernel` fast build, S, steps 0..100 | **bit-identical** (`cmp` clean; 11.710649 -> 4.416876) |
| S, 4,000 steps, Doom ON + dashboard | 224 s, **56.0 ms/step**, 73.2k tok/s, final loss **2.1490**, Doom **35.0 fps**, 7,827 frames |
| S, 600 steps Doom ON, 2 runs | 56.15 / 56.78 ms/step, decode 611 / 597 tok/s, Doom 34.9 fps during training |
| S, 600 steps Doom OFF (`--no-doom`), 2 runs | 56.45 / 56.51 ms/step, decode 600 / 601 tok/s |
| F, 250 steps Doom ON | 114.9 ms/step (35.6k tok/s), Doom 35.0 fps, decode ~254 tok/s, step-0 loss 11.8006 |

**Doom costs nothing measurable** (ON vs OFF within noise). **Known regression:** this merged build is ~18% slower than the
`llm/megakernel` build at the same 85/17/6 split (S: 56.0 vs 47.4 ms/step; F: 114.9 vs 104.7). Cause: Doom needs `-rdc=true` (its
code is a relocatable device object), and `-rdc` makes ptxas compile the training code with larger stack frames and many more spills (kernel frame 1464 B vs
552 B). Stubbing out the `doom_loop` call (`-DSTUB_DOOM`) gives the same slowdown, so it is `-rdc`, not Doom; inlining
`run_gemm` (`-DGEMM_INLINE`) is worse (58.3). Open: LTO could not link against `doom_dev.o`.

Fixed during the merge: with Doom off the default 1 KB/thread stack overflowed (illegal memory access), so the stack limit is now always set.
