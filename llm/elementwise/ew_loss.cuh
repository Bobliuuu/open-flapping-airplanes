// Softmax cross-entropy (tied-head logits) + embedding backward.
#pragma once
#include "ew_common.cuh"

namespace ew {
// ---------------------------------------------------------------- softmax cross-entropy
// In place: logits[r,:] (bf16, V) -> dlogits[r,:] = (softmax - onehot(target[r])) * inv_n   (inv_n = 1/N_total
// for the gradient of the MEAN loss). loss[r] = logsumexp - logit[target] (fp32).
// Pass 1: per-thread online max/sum-exp (fp32, exp2 domain), fixed-order block combine. Pass 2: recompute
// probabilities and overwrite. The first CACHE*256 16-byte vectors of each row (32 KB for CACHE=8, default) are kept
// in smem so only ~1/3 of pass-2 reads go back to L2/HBM.
// Task = CE_ROWS (4) rows. ntasks = ceil(N/4). Needs the logits GEMM complete. smem: ce_smem_bytes<CACHE>().
// Targets must be in [0,V). Deterministic.
constexpr int CE_ROWS = 4;
__device__ __forceinline__ float fast_exp2(float x) { return exp2f(x); }  // (ex2.approx.ftz variant measured slower)
template <int CACHE = 8>
__host__ __device__ constexpr size_t ce_smem_bytes() { return 256 + (size_t)CACHE * NT * 16; }

template <int V, int CACHE = 8>
__device__ __forceinline__ void ce_task(int task, bf16* logits, const int* targets, float* loss, int N,
                                        float inv_n, char* smem) {
  static_assert(V % 8 == 0, "V%8");
  constexpr int NVEC = V / 8, NI = (NVEC + NT - 1) / NT, CI = CACHE < NI ? CACHE : NI;
  constexpr float L2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
  float* red = reinterpret_cast<float*>(smem);
  uint4* cache = reinterpret_cast<uint4*>(smem + 256);
  const int t = threadIdx.x;
  for (int rr = 0; rr < CE_ROWS; rr++) {
    const int r = task * CE_ROWS + rr;
    if (r >= N) break;
    uint4* row = reinterpret_cast<uint4*>(logits + (size_t)r * V);
    const int tgt = targets[r];
    float m = -INFINITY, s = 0.f;
#pragma unroll 7
    for (int i = 0; i < NI; i++) {
      const int v = i * NT + t;
      if (v < NVEC) {
        uint4 u = row[v];
        if (i < CI) cache[i * NT + t] = u;
        float f[8]; unpack8(u, f);
        float cm = f[0];
#pragma unroll
        for (int k = 1; k < 8; k++) cm = fmaxf(cm, f[k]);
        cm *= L2E;
        if (cm > m) { s *= fast_exp2(m - cm); m = cm; }
#pragma unroll
        for (int k = 0; k < 8; k++) s += fast_exp2(fmaf(f[k], L2E, -m));
      }
    }
    // target logit read BEFORE any thread can start overwriting (the syncs below order it)
    float lt = 0.f;
    if (t == 0) lt = __bfloat162float(logits[(size_t)r * V + tgt]);
    const float M = block_max(m, red);
    const float S = block_sum(s * fast_exp2(m - M), red);
    if (t == 0) loss[r] = M * LN2 + logf(S) - lt;
    const float invS = 1.f / S;
#pragma unroll 7
    for (int i = 0; i < NI; i++) {
      const int v = i * NT + t;
      if (v < NVEC) {
        uint4 u = (i < CI) ? cache[i * NT + t] : row[v];
        float f[8]; unpack8(u, f);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          f[k] = fast_exp2(fmaf(f[k], L2E, -M)) * invS;
        }
#pragma unroll
        for (int k = 0; k < 8; k++) if (v * 8 + k == tgt) f[k] -= 1.f;
#pragma unroll
        for (int k = 0; k < 8; k++) f[k] *= inv_n;
        row[v] = pack8(f);
      }
    }
  }
}

// mean_loss[0] = (sum_r loss[r]) / N. ONE task, fixed order (thread-strided partial sums, fixed block tree).
// Needs all ce_tasks done. smem: 64 B.
__device__ __forceinline__ void ce_mean_task(float* mean_loss, const float* loss, int N, char* smem) {
  float s = 0.f;
  for (int i = threadIdx.x; i < N; i += NT) s += loss[i];
  s = block_sum(s, reinterpret_cast<float*>(smem));
  if (threadIdx.x == 0) mean_loss[0] = s / (float)N;
}

// ---------------------------------------------------------------- embedding backward
// dwte32[tok[n],:] += dx0[n,:]  for all n, per-row contributions added in increasing n  (ADDS to the buffer).
// Deterministic, atomic-free: task = a vocab chunk of 128 rows; it scans all N tokens in order, compacts the
// matches (block scan), and the 256 threads (column-parallel) accumulate them in token order. ntasks = V/128
// (ceil). smem: embed_bwd_wte_smem_bytes(N) = 5N+1024. Needs dx0 (grad at embedding output) complete.
// The tied-head GEMM (dwte32 += dlogits^T . x) ALSO adds into dwte32. Both are plain read-modify-write, so they
// must NOT run concurrently on the same buffer: put them in different phases (either order is fine; the final
// fp32 value differs only by summation order, and each phase is deterministic).
constexpr int EMB_VCHUNK = 128;
__host__ __device__ constexpr size_t embed_bwd_wte_smem_bytes(int N) { return 5 * (size_t)N + 1024; }

// The chunk's matches are compacted in token order (list[], with the row of each in rowof[]); then warp w accumulates rows
// w, w+8, ... in registers: for each row, the dx0 rows of its tokens are added in increasing n (the same order, and so the same
// fp32 result, as one read-modify-write per token), and the row is written once. A row with hundreds of tokens (" the", ".") no longer
// serializes the whole chunk through global memory.
template <int D>
__device__ __noinline__ void embed_bwd_wte_task(int task, float* dwte32, const bf16* dx0, const int* tok,
                                                int N, char* smem) {   // own function: its registers stay out of the scheduler loop
  static_assert(D % 256 == 0, "lane owns D/256 groups of 8 columns");
  constexpr int G = D / 256;
  int* cnt = reinterpret_cast<int*>(smem);         // 256
  int* list = reinterpret_cast<int*>(smem + 1024);  // N
  unsigned char* rowof = reinterpret_cast<unsigned char*>(smem + 1024 + 4 * (size_t)N);  // N
  const int t = threadIdx.x, lo = task * EMB_VCHUNK;
  const int per = (N + NT - 1) / NT, a = t * per, b = min(N, a + per);
  int c = 0;
  for (int n = a; n < b; n++) { int v = tok[n] - lo; c += (unsigned)v < (unsigned)EMB_VCHUNK; }
  cnt[t] = c;
  __syncthreads();
  int off = 0, total = 0;
  for (int i = 0; i < NT; i++) { int x = cnt[i]; if (i < t) off += x; total += x; }
  for (int n = a; n < b; n++) {
    int v = tok[n] - lo;
    if ((unsigned)v < (unsigned)EMB_VCHUNK) { list[off] = n; rowof[off] = (unsigned char)v; off++; }
  }
  __syncthreads();
  const int warp = t >> 5, lane = t & 31;
  for (int r = warp; r < EMB_VCHUNK; r += NT / 32) {
    float acc[G][8];
    bool any = false;
    float* dst = dwte32 + (size_t)(lo + r) * D;
    for (int k0 = 0; k0 < total; k0 += 32) {
      const int k = k0 + lane;
      unsigned m = __ballot_sync(0xffffffffu, k < total && rowof[k] == r);
      while (m) {
        int ns[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          if (m) { ns[q] = list[k0 + __ffs(m) - 1]; m &= m - 1; } else ns[q] = -1;
        }
        if (!any) {
#pragma unroll
          for (int g = 0; g < G; g++) {
            const float4 x0 = *reinterpret_cast<const float4*>(dst + g * 256 + lane * 8);
            const float4 x1 = *reinterpret_cast<const float4*>(dst + g * 256 + lane * 8 + 4);
            acc[g][0] = x0.x; acc[g][1] = x0.y; acc[g][2] = x0.z; acc[g][3] = x0.w;
            acc[g][4] = x1.x; acc[g][5] = x1.y; acc[g][6] = x1.z; acc[g][7] = x1.w;
          }
          any = true;
        }
        uint4 v[4][G];
#pragma unroll
        for (int q = 0; q < 4; q++)
          if (ns[q] >= 0) {
#pragma unroll
            for (int g = 0; g < G; g++) v[q][g] = *reinterpret_cast<const uint4*>(dx0 + (size_t)ns[q] * D + g * 256 + lane * 8);
          }
#pragma unroll
        for (int q = 0; q < 4; q++)
          if (ns[q] >= 0) {
#pragma unroll
            for (int g = 0; g < G; g++) {
              const uint32_t w[4] = {v[q][g].x, v[q][g].y, v[q][g].z, v[q][g].w};
#pragma unroll
              for (int j = 0; j < 4; j++) {
                acc[g][2 * j] += __uint_as_float(w[j] << 16);
                acc[g][2 * j + 1] += __uint_as_float(w[j] & 0xffff0000u);
              }
            }
          }
      }
    }
    if (any) {
#pragma unroll
      for (int g = 0; g < G; g++) {
        *reinterpret_cast<float4*>(dst + g * 256 + lane * 8) = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        *reinterpret_cast<float4*>(dst + g * 256 + lane * 8 + 4) = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
      }
    }
  }
  __syncthreads();
}

// dwpe32[p,:] += sum_b dx0[b*T+p,:] (sum over the sequences in increasing b, then one add). Task = 4 positions;
// ntasks = T/4. smem 0. Deterministic.
template <int D, int T>
__device__ __forceinline__ void embed_bwd_wpe_task(int task, float* dwpe32, const bf16* dx0, int N) {
  for (int pp = 0; pp < 4; pp++) {
    const int p = task * 4 + pp;
    if (p >= T) break;
    for (int c = threadIdx.x; c < D; c += NT) {
      float s = 0.f;
      for (int n = p; n < N; n += T) s += __bfloat162float(dx0[(size_t)n * D + c]);
      dwpe32[(size_t)p * D + c] += s;
    }
  }
}
}  // namespace ew
