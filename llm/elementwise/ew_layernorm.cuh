// Embedding forward, LayerNorm fwd/bwd (+dgamma partial/reduce), residual add, casts.
#pragma once
#include "ew_common.cuh"

namespace ew {
constexpr int ROWS_PER_TASK = 16;  // row-block granularity for embed / LN / add

// ---------------------------------------------------------------- embedding forward
// x[r,:] = wte[tok[r],:] + wpe[r % T,:]   (bf16 add in fp32, rounded once)
// Task = 16 consecutive token rows. ntasks = ceil(N/16). No prerequisites. smem: 0.
template <int D, int T>
__device__ __forceinline__ void embed_fwd_task(int task, bf16* x, const bf16* wte, const bf16* wpe,
                                               const int* tok, int N) {
  static_assert(D % 8 == 0, "D%8");
  constexpr int V8 = D / 8;
  const int r0 = task * ROWS_PER_TASK;
  for (int i = threadIdx.x; i < ROWS_PER_TASK * V8; i += NT) {
    int r = r0 + i / V8;
    if (r >= N) break;
    int c = (i % V8) * 8;
    int t = tok[r];
    uint4 a = *reinterpret_cast<const uint4*>(wte + (size_t)t * D + c);
    uint4 b = *reinterpret_cast<const uint4*>(wpe + (size_t)(r % T) * D + c);
    float fa[8], fb[8];
    unpack8(a, fa); unpack8(b, fb);
#pragma unroll
    for (int k = 0; k < 8; k++) fa[k] += fb[k];
    *reinterpret_cast<uint4*>(x + (size_t)r * D + c) = pack8(fa);
  }
}

// ---------------------------------------------------------------- LayerNorm forward
// y = (h-mean)*rstd*gamma, h = x (+ delta if delta != nullptr: h = bf16(x+delta) is ALSO written to h_out,
// h_out may alias x -> fused residual add + LN, saves one pass over the residual stream).
// Saves mean[N], rstd[N] (fp32). Task = 16 rows (warp w handles rows w and w+8). ntasks=ceil(N/16).
// Requires D % 256 == 0. smem: 0. gamma is the bf16 working copy.
template <int D>
__device__ __forceinline__ void ln_fwd_task(int task, bf16* y, float* mean_o, float* rstd_o, const bf16* x,
                                            const bf16* delta, bf16* h_out, const bf16* gamma, int N,
                                            float eps = 1e-5f) {
  static_assert(D % 256 == 0, "D%256");
  constexpr int NV = D / 256;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int j = 0; j < 2; j++) {
    const int r = task * ROWS_PER_TASK + warp + j * 8;
    if (r >= N) continue;
    float f[NV][8];
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * D);
#pragma unroll
    for (int v = 0; v < NV; v++) unpack8(xr[v * 32 + lane], f[v]);
    if (delta) {
      const uint4* dr = reinterpret_cast<const uint4*>(delta + (size_t)r * D);
      uint4* hr = reinterpret_cast<uint4*>(h_out + (size_t)r * D);
#pragma unroll
      for (int v = 0; v < NV; v++) {
        float g[8]; unpack8(dr[v * 32 + lane], g);
#pragma unroll
        for (int k = 0; k < 8; k++) f[v][k] += g[k];
        uint4 p = pack8(f[v]);
        hr[v * 32 + lane] = p;
        unpack8(p, f[v]);  // continue with the bf16-rounded residual stream value
      }
    }
    float s = 0.f;
#pragma unroll
    for (int v = 0; v < NV; v++)
#pragma unroll
      for (int k = 0; k < 8; k++) s += f[v][k];
    const float mean = warp_sum(s) * (1.f / D);
    float q = 0.f;
#pragma unroll
    for (int v = 0; v < NV; v++)
#pragma unroll
      for (int k = 0; k < 8; k++) { float d = f[v][k] - mean; q += d * d; }
    const float rstd = rsqrtf(warp_sum(q) * (1.f / D) + eps);
    uint4* yr = reinterpret_cast<uint4*>(y + (size_t)r * D);
    const uint4* gr = reinterpret_cast<const uint4*>(gamma);
#pragma unroll
    for (int v = 0; v < NV; v++) {
      float g[8]; unpack8(gr[v * 32 + lane], g);
      float o[8];
#pragma unroll
      for (int k = 0; k < 8; k++) o[k] = (f[v][k] - mean) * rstd * g[k];
      yr[v * 32 + lane] = pack8(o);
    }
    if (lane == 0) { mean_o[r] = mean; rstd_o[r] = rstd; }
  }
}

// ---------------------------------------------------------------- LayerNorm backward
// dx = rstd*(g - mean(g) - xhat*mean(g*xhat)) [+ dres], g = dy*gamma, xhat=(x-mean)*rstd.
// x is the LN INPUT (post residual add if the fused variant was used: pass h). dres may be nullptr.
// dx may alias dres. Also writes the per-task partial dgamma[task, D] (fp32) = sum over the task's rows of
// dy*xhat, summed over the 8 warps in fixed order -> deterministic. Task = 16 rows. ntasks = ceil(N/16);
// dgamma_part must hold ntasks*D floats. smem: ln_bwd_smem_bytes<D>() (24 KB for D=768).
template <int D>
__host__ __device__ constexpr size_t ln_bwd_smem_bytes() { return 8 * (size_t)D * sizeof(float); }

template <int D>
__device__ __forceinline__ void ln_bwd_task(int task, bf16* dx, float* dgamma_part, const bf16* dy, const bf16* x,
                                            const float* mean, const float* rstd, const bf16* gamma,
                                            const bf16* dres, int N, char* smem) {
  static_assert(D % 256 == 0, "D%256");
  constexpr int NV = D / 256;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* sdg = reinterpret_cast<float*>(smem);  // [8][D]
  float gam[NV][8], dg[NV][8];
  {
    const uint4* gr = reinterpret_cast<const uint4*>(gamma);
#pragma unroll
    for (int v = 0; v < NV; v++) { unpack8(gr[v * 32 + lane], gam[v]);
#pragma unroll
      for (int k = 0; k < 8; k++) dg[v][k] = 0.f; }
  }
  for (int j = 0; j < 2; j++) {
    const int r = task * ROWS_PER_TASK + warp + j * 8;
    if (r >= N) continue;
    const float mu = mean[r], rs = rstd[r];
    float xh[NV][8], g[NV][8], dyv[NV][8];
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)r * D);
    const uint4* dr = reinterpret_cast<const uint4*>(dy + (size_t)r * D);
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < NV; v++) {
      unpack8(xr[v * 32 + lane], xh[v]); unpack8(dr[v * 32 + lane], dyv[v]);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        xh[v][k] = (xh[v][k] - mu) * rs;
        g[v][k] = dyv[v][k] * gam[v][k];
        s1 += g[v][k]; s2 += g[v][k] * xh[v][k];
        dg[v][k] += dyv[v][k] * xh[v][k];
      }
    }
    s1 = warp_sum(s1) * (1.f / D); s2 = warp_sum(s2) * (1.f / D);
    uint4* dxr = reinterpret_cast<uint4*>(dx + (size_t)r * D);
    const uint4* rr = dres ? reinterpret_cast<const uint4*>(dres + (size_t)r * D) : nullptr;
#pragma unroll
    for (int v = 0; v < NV; v++) {
      float o[8], e[8];
      if (rr) unpack8(rr[v * 32 + lane], e);
#pragma unroll
      for (int k = 0; k < 8; k++) { o[k] = rs * (g[v][k] - s1 - xh[v][k] * s2); if (rr) o[k] += e[k]; }
      dxr[v * 32 + lane] = pack8(o);
    }
  }
#pragma unroll
  for (int v = 0; v < NV; v++) {
    float4* p = reinterpret_cast<float4*>(sdg + warp * D + (v * 32 + lane) * 8);
    p[0] = make_float4(dg[v][0], dg[v][1], dg[v][2], dg[v][3]);
    p[1] = make_float4(dg[v][4], dg[v][5], dg[v][6], dg[v][7]);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += NT) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) s += sdg[w * D + c];
    dgamma_part[(size_t)task * D + c] = s;
  }
  __syncthreads();  // smem reusable by the caller's next task
}

// Sums the nblk partials in index order: dgamma32[c] += sum_b part[b,c]  (ADDS; zero dgamma32 first via
// zero_task if a fresh gradient is wanted). Task = 256 columns; ntasks = ceil(D/256). Needs all ln_bwd tasks
// of this LayerNorm done. smem: 0.
template <int D>
__device__ __forceinline__ void ln_dgamma_reduce_task(int task, float* dgamma32, const float* part, int nblk) {
  const int c = task * NT + threadIdx.x;
  if (c < D) {
    float s = 0.f;
    for (int b = 0; b < nblk; b++) s += part[(size_t)b * D + c];
    dgamma32[c] += s;
  }
}

// ---------------------------------------------------------------- helpers
// out = bf16(a + b), task = 16 rows of D (elementwise; out may alias a or b).
template <int D>
__device__ __forceinline__ void residual_add_task(int task, bf16* out, const bf16* a, const bf16* b, int N) {
  constexpr int V8 = D / 8;
  const int r0 = task * ROWS_PER_TASK;
  for (int i = threadIdx.x; i < ROWS_PER_TASK * V8; i += NT) {
    int r = r0 + i / V8;
    if (r >= N) break;
    size_t o = (size_t)r * D + (i % V8) * 8;
    float fa[8], fb[8];
    unpack8(*reinterpret_cast<const uint4*>(a + o), fa); unpack8(*reinterpret_cast<const uint4*>(b + o), fb);
#pragma unroll
    for (int k = 0; k < 8; k++) fa[k] += fb[k];
    *reinterpret_cast<uint4*>(out + o) = pack8(fa);
  }
}

// dst[bf16] = src[fp32] for chunk `task` of 64K elements (n % 8 == 0).
__device__ __forceinline__ void cast_f32_bf16_task(int task, bf16* dst, const float* src, size_t n) {
  size_t s = (size_t)task * 65536, e = s + 65536 < n ? s + 65536 : n;
  for (size_t i = s + (size_t)threadIdx.x * 8; i < e; i += NT * 8) {
    float4 a = *reinterpret_cast<const float4*>(src + i), b = *reinterpret_cast<const float4*>(src + i + 4);
    float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    *reinterpret_cast<uint4*>(dst + i) = pack8(f);
  }
}
}  // namespace ew
