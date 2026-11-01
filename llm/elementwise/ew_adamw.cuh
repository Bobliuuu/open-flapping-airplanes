// AdamW over flat fp32 buffers + zero-grad + optional global-norm clipping.
#pragma once
#include "ew_common.cuh"

namespace ew {
constexpr int OPT_CHUNK = 65536;  // elements per task (n and chunk edges must be multiples of 4)

struct AdamHP { float lr, beta1, beta2, eps, weight_decay; int step; };  // step is 1-based

// Decoupled-decay AdamW exactly like torch.optim.AdamW (non-fused, non-amsgrad):
//   p *= 1 - lr*wd ; m = lerp(m,g,1-b1) ; v = b2 v + (1-b2) g^2 ;
//   p -= lr/(1-b1^t) * m / (sqrt(v)/sqrt(1-b2^t) + eps)
// W32/G32/M/Vv flat fp32 [n]; Wbf = bf16 working copy written from the new W32.
// nodecay: sorted, disjoint [start,end) element ranges (int2{x=start,y=end}) that get NO weight decay
// (LayerNorm gammas). Convention: decay all params with ndim>=2 INCLUDING wte and wpe; gammas excluded.
// clip_scale: nullptr, or pointer to the float produced by grad_clip_scale_task (gradient multiplier).
// ZERO_G: also zero G32 after use (fuses the zero-grad pass; saves 4 B/param read... costs 4 B write).
// Task = 64K-element chunk; ntasks = ceil(n/65536). Needs all grad producers done (incl. the final dwte GEMM
// and embed_bwd). smem 0. Deterministic (pure elementwise).
template <bool ZERO_G = false>
__device__ __forceinline__ void adamw_task(int task, float* W, float* G, float* M, float* Vv, bf16* Wbf, size_t n,
                                           AdamHP hp, const int2* nodecay, int n_nodecay,
                                           const float* clip_scale) {
  const size_t cs = (size_t)task * OPT_CHUNK;
  if (cs >= n) return;
  const size_t ce = cs + OPT_CHUNK < n ? cs + OPT_CHUNK : n;
  const double bc1 = 1.0 - pow((double)hp.beta1, (double)hp.step);
  const double bc2 = 1.0 - pow((double)hp.beta2, (double)hp.step);
  const float step_size = (float)((double)hp.lr / bc1);
  const float inv_sqrt_bc2 = (float)(1.0 / sqrt(bc2));
  const float decay_mul = 1.f - hp.lr * hp.weight_decay;
  const float gscale = clip_scale ? *clip_scale : 1.f;
  const float omb1 = 1.f - hp.beta1, omb2 = 1.f - hp.beta2;
  int ri = 0;
  if (n_nodecay > 0) {  // first range with end > chunk start
    int lo = 0, hi = n_nodecay;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (nodecay[mid].y <= (int)cs) lo = mid + 1; else hi = mid; }
    ri = lo;
  }
  for (size_t i = cs + (size_t)threadIdx.x * 4; i < ce; i += NT * 4) {
    float4 w = *reinterpret_cast<const float4*>(W + i), g = *reinterpret_cast<const float4*>(G + i);
    float4 m = *reinterpret_cast<const float4*>(M + i), v = *reinterpret_cast<const float4*>(Vv + i);
    float* pw = &w.x; float* pg = &g.x; float* pm = &m.x; float* pv = &v.x;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      bool dec = true;
      if (ri < n_nodecay) {
        const int idx = (int)(i + k);
        while (ri < n_nodecay && nodecay[ri].y <= idx) ri++;
        if (ri < n_nodecay && idx >= nodecay[ri].x) dec = false;
      }
      const float gg = pg[k] * gscale;
      float p = dec ? pw[k] * decay_mul : pw[k];
      pm[k] = pm[k] + (gg - pm[k]) * omb1;
      pv[k] = fmaf(pv[k], hp.beta2, omb2 * gg * gg);
      const float denom = sqrtf(pv[k]) * inv_sqrt_bc2 + hp.eps;
      pw[k] = p - step_size * (pm[k] / denom);
    }
    *reinterpret_cast<float4*>(W + i) = w;
    *reinterpret_cast<float4*>(M + i) = m;
    *reinterpret_cast<float4*>(Vv + i) = v;
    if (ZERO_G) *reinterpret_cast<float4*>(G + i) = make_float4(0.f, 0.f, 0.f, 0.f);
    __nv_bfloat162 b0 = __floats2bfloat162_rn(w.x, w.y), b1 = __floats2bfloat162_rn(w.z, w.w);
    uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&b0); pk.y = *reinterpret_cast<uint32_t*>(&b1);
    *reinterpret_cast<uint2*>(Wbf + i) = pk;
  }
}

// G32[chunk] = 0. Task = 64K elements, n % 4 == 0. smem 0.
__device__ __forceinline__ void zero_task(int task, float* G, size_t n) {
  size_t s = (size_t)task * OPT_CHUNK, e = s + OPT_CHUNK < n ? s + OPT_CHUNK : n;
  for (size_t i = s + (size_t)threadIdx.x * 4; i < e; i += NT * 4)
    *reinterpret_cast<float4*>(G + i) = make_float4(0.f, 0.f, 0.f, 0.f);
}

// Global-norm clipping, phase A: part[task] = sum of squares of the chunk (fixed order). Task = 64K chunk.
// smem 64 B.
__device__ __forceinline__ void grad_sumsq_task(int task, const float* G, size_t n, float* part, char* smem) {
  size_t s = (size_t)task * OPT_CHUNK, e = s + OPT_CHUNK < n ? s + OPT_CHUNK : n;
  float a = 0.f;
  for (size_t i = s + (size_t)threadIdx.x * 4; i < e; i += NT * 4) {
    float4 g = *reinterpret_cast<const float4*>(G + i);
    a += g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w;
  }
  a = block_sum(a, reinterpret_cast<float*>(smem));
  if (threadIdx.x == 0) part[task] = a;
}
// Phase B (ONE task, after all phase A): out[0] = min(1, max_norm/(norm+1e-6)) (torch clip_grad_norm_), out[1]=norm.
__device__ __forceinline__ void grad_clip_scale_task(const float* part, int nparts, float max_norm, float* out,
                                                     char* smem) {
  float a = 0.f;
  for (int i = threadIdx.x; i < nparts; i += NT) a += part[i];
  a = block_sum(a, reinterpret_cast<float*>(smem));
  if (threadIdx.x == 0) {
    float norm = sqrtf(a);
    out[0] = fminf(1.f, max_norm / (norm + 1e-6f));
    out[1] = norm;
  }
}
}  // namespace ew
