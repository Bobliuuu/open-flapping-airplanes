// Shared helpers for the elementwise/optimizer tasks. Header-only.
// Every task: __device__ function called by ALL 256 threads of one block with a task index.
// No blockIdx/gridDim use. Deterministic (fixed reduction order, no float atomics).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace ew {
typedef __nv_bfloat16 bf16;
constexpr int NT = 256;  // threads per block (contract)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// Block reductions; `red` = 8 floats of shared scratch. Result valid in all threads.
// Safe to call back-to-back (leading __syncthreads protects scratch reuse). Fixed order => deterministic.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int i = 1; i < NT / 32; i++) r += red[i];
  return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
  v = warp_max(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int i = 1; i < NT / 32; i++) r = fmaxf(r, red[i]);
  return r;
}

__device__ __forceinline__ void unpack8(uint4 u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 t = __bfloat1622float2(p[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 u;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
}  // namespace ew
