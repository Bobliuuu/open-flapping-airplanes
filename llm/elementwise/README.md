# elementwise + optimizer tasks (header-only)

Device functions called by all 256 threads of one block (no blockIdx/gridDim/atomics/global barriers; `__syncthreads` inside only).
All deterministic (fixed reduction order, no float atomics). Shapes are constexpr template parameters (D, T, V); verified at D=768, T=512, V=100352, N=4096.
Include `ew_all.cuh` (or the single group files). Namespace `ew`. `smem` args = dynamic shared memory base (128 B aligned). Row-block tasks need D % 256 == 0 (LN) / D % 8 == 0.

| Task | Inputs -> outputs | Granularity / ntasks | smem | Needs complete |
|---|---|---|---|---|
| `embed_fwd_task<D,T>(task,x,wte,wpe,tok,N)` | x = bf16(wte[tok]+wpe[row%T]) | 16 rows / ceil(N/16)=256 | 0 | - |
| `ln_fwd_task<D>(task,y,mean,rstd,x,delta,h_out,gamma,N)` | y=LN(x)*gamma, mean/rstd fp32. If `delta!=nullptr`: h=bf16(x+delta) written to `h_out` (may alias x) and LN(h) is computed (fused residual add) | 16 rows (warp w: rows w, w+8) / 256 | 0 | producer of x (and delta) |
| `ln_bwd_task<D>(task,dx,dgamma_part,dy,x,mean,rstd,gamma,dres,N,smem)` | dx = LNbwd(dy) (+ dres if non-null, fp32 add then bf16 round; dx may alias dres); x = the LN input (h for fused variant); `dgamma_part[task][D]` fp32 | 16 rows / 256; partial buffer = ntasks*D floats | `ln_bwd_smem_bytes<D>()` = 24 KB | dy |
| `ln_dgamma_reduce_task<D>(task,dgamma32,part,nblk)` | `dgamma32[c] += sum_b part[b][c]` (b ascending) | 256 columns / ceil(D/256)=3 | 0 | all ln_bwd tasks of that LN |
| `residual_add_task<D>`, `cast_f32_bf16_task` | helpers | 16 rows / 64K elems | 0 | - |
| `ce_task<V,CACHE=8>(task,logits,targets,loss,N,inv_n,smem)` | in place logits -> dlogits=(softmax-onehot)*inv_n (pass inv_n=1/N_total for the mean loss); loss[r] fp32 | 4 rows / N/4=1024 | `ce_smem_bytes<8>()` = 32 KB+256 | logits GEMM |
| `ce_mean_task(mean_loss,loss,N,smem)` | mean_loss[0] = sum(loss)/N, fixed order | ONE task | 64 B | all ce_task |
| `embed_bwd_wte_task<D>(task,dwte32,dx0,tok,N,smem)` | `dwte32[tok[n]] += dx0[n]`, per row in token order | 128 vocab rows / V/128 = 784 | `embed_bwd_wte_smem_bytes(N)` = 4N+1 KB (17 KB) | dx0 (grad at embedding output) |
| `embed_bwd_wpe_task<D,T>(task,dwpe32,dx0,N)` | `dwpe32[p] += sum_b dx0[b*T+p]` (b ascending) | 4 positions / T/4 = 128 | 0 | dx0 |
| `adamw_task<ZERO_G>(task,W,G,M,V,Wbf,n,hp,nodecay,n_nodecay,clip_scale)` | AdamW update; also writes bf16 copy; `ZERO_G` also zeroes G | 64K elements / ceil(n/65536)=2472 for 162M | 0 | all gradient producers incl. dwte GEMM + embed_bwd |
| `zero_task(task,G,n)` | G=0 | 64K elems | 0 | optimizer step finished with G |
| `grad_sumsq_task` -> `grad_clip_scale_task` | per-chunk sum of squares -> ONE task computes `scale=min(1,max_norm/(norm+1e-6))`, `norm` (out[0], out[1]) | 64K / one | 64 B | grads / all sumsq |

Notes
- **dwte/dwpe `+=`**: embed_bwd tasks ADD into the fp32 buffers. The tied-head GEMM (`dwte32 += dlogits^T x`) also adds into dwte32: they are plain read-modify-writes, so they must be in different phases (never concurrently on the same buffer); either order is fine (fp32 sums differ only by order, each phase deterministic). `dgamma32` is also `+=` (zero G32 per step with `zero_task` or `ZERO_G`).
- **Weight decay rule**: PyTorch convention "decay params with ndim>=2" => wte and wpe ARE decayed; only LayerNorm gammas are not. Passed as sorted disjoint `int2{start,end}` element ranges (`nodecay`), any alignment (n and chunk edges multiples of 4; element offsets < 2^31). AdamW is exactly torch's non-fused formula (`p*=1-lr*wd; m=lerp; v=b2 v+(1-b2)g^2; p-=lr/bc1*m/(sqrt(v)/sqrt(bc2)+eps)`), bias corrections computed in double per task from the 1-based `step`.
- **Clipping** (default OFF = pass `clip_scale=nullptr`): phase `grad_sumsq_task` (per chunk) -> phase `grad_clip_scale_task` (1 task) -> AdamW with `clip_scale` pointer (gradient multiplied by it).
- Phase ordering of a step tail: embed_bwd + head dwte GEMM (separate phases) -> [sumsq -> clip scale] -> adamw (or adamw<true>) ; LN: ln_bwd tasks -> ln_dgamma_reduce.
- CE precision: fp32 online softmax in exp2 domain (`exp2f`), `logf`; the target logit is read before the in-place overwrite. Targets must be in [0,V). A 32 KB smem cache of the first vectors of each row avoids re-reading them for pass 2 (CACHE sweep: 0..12 all within 5%; 24 is slower from occupancy).
- Not built: standalone GELU backward (the GEMM epilogue fuses it).

## Test
`python3 ref_elementwise.py all /tmp/ew_data` (needs `nvcc -arch=sm_80 -O3 -std=c++17 --extended-lambda -o ew_test test_elementwise.cu`). Python generates inputs, `ew_test` runs every task as a grid and as a persistent-style kernel (216 blocks pulling a shuffled task order from an atomic counter; outputs must be bit-identical), dumps outputs; python compares with PyTorch fp32 (`F.layer_norm` autograd, `F.cross_entropy`, `torch.optim.AdamW` x3 steps, `clip_grad_norm_`). CE correctness uses 256 rows x V=100352 (full-V rows); speed uses all 4096 rows.

## Measured (A100-SXM4-40GB, shared box, so noisy)
Accuracy (err/max|ref|, tolerance per SPEC): embed fwd 2.8e-3; LN y 2.4e-3, mean 1.6e-7, rstd 1.6e-7, dx 2.3e-3, dx+dres 2.5e-3, dgamma 6e-7; CE loss 6.4e-7, mean loss 6e-7, dlogits 1.9e-3 (bf16 rounding); dwte/dwpe exact to fp32 (0 error); AdamW W32 after 3 steps 6e-8, bf16 copy exact, with clipping 6.4e-8; clip norm exact. All PASS; grid vs persistent bit-identical: PASS.

Speed (GB/s of actual minimum traffic):
| op | time | GB/s |
|---|---|---|
| embed_fwd (N=4096) | 9.8 us | ~1900 (partly L2) |
| embed_bwd_wte | 171 us | latency bound (784 tasks, each scans 4096 tokens); runs once per step |
| embed_bwd_wpe | 12 us | ~780 |
| ln_fwd (rotating 8 buffer sets, HBM) | 14.5 us | ~870 |
| ln_bwd | 22 us | ~895 |
| add+ln_fwd | 24 us | ~1040 |
| ln_dgamma_reduce | 11 us | - |
| CE N=4096 V=100352 | 1.86 ms | ~880 (1.64 GB min traffic) |
| AdamW 162M | 3.63 ms | ~1330 (30 B/param) |
| AdamW + fused zero G | 4.17 ms | ~1320 (34 B/param) |
| zero_task / grad_sumsq (162M) | 0.44 ms | ~1480 |
(A100 HBM peak 1555 GB/s; another agent's jobs were running intermittently, run-to-run spread ~5-10%.) LN is 256 blocks of 256 threads, latency-limited at ~13-22 us; CE is ALU/latency limited at ~57% of HBM (two exp2 per element); an attempt with `ex2.approx.ftz` and cheaper one-hot handling measured slower and was reverted.

## Limits / open issues
- CE at ~0.88 TB/s is the weakest op; the per-element exp is the cost (2 exps/element: could be halved by caching probabilities, which does not fit smem). Overlap with the GEMM in the megakernel will hide some of it.
- embed_bwd_wte uses 17 KB smem and N is runtime; tokens > 4096 need more smem (4N bytes).
- Element offsets in AdamW ranges are int32 (< 2.1G params).

## B200 notes
Same code runs; to chase 8 TB/s HBM: raise per-SM bytes in flight (TMA/bulk-async loads of CE rows into the 200+ KB smem so a whole 200 KB bf16 row stays resident and is read once), use 2-4 rows per block concurrently, and fuse zero-grad into AdamW (already `ZERO_G`). FP8/bf16 packed math is not needed here.
