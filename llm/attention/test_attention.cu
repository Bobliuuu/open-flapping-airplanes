// test_attention.cu -- standalone test/bench for attention.cuh.
// Usage: ./attn_test [outdir]   writes q,k,v,do,o,lse,dqkv .bin files; then run ref_attention.py outdir
// Compile-time: -DATT_T=512 -DATT_H=12 -DATT_B=8
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include "attention.cuh"
using namespace attn;
#ifndef ATT_T
#define ATT_T 512
#endif
#ifndef ATT_H
#define ATT_H 12
#endif
#ifndef ATT_B
#define ATT_B 8
#endif
constexpr int T = ATT_T, H = ATT_H, B = ATT_B, D = H * 64, N = B * T, NQ = T / 64;
constexpr int NTASK = B * H * NQ;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

// ---- grid kernels (heavy tiles first)
__global__ void __launch_bounds__(256, 2) k_fwd(const bf16* qkv, bf16* o, float* lse) {
  extern __shared__ __align__(128) char sm[];
  int t = blockIdx.x; int qt = NQ - 1 - t % NQ; int bh = t / NQ;
  attn_fwd_tile<T, H>(qkv, o, lse, bh / H, bh % H, qt, sm);
}
__global__ void __launch_bounds__(256, 2) k_prep(const bf16* o, const bf16* dout, float* Dr) {
  int t = blockIdx.x; int rt = t % NQ; int bh = t / NQ;
  attn_bwd_prep_tile<T, H>(o, dout, Dr, bh / H, bh % H, rt);
}
__global__ void __launch_bounds__(256, 2) k_dkdv(const bf16* qkv, const bf16* dout, const float* lse, const float* Dr, bf16* dqkv) {
  extern __shared__ __align__(128) char sm[];
  int t = blockIdx.x; int kt = t % NQ; int bh = t / NQ;
  attn_bwd_dkdv_tile<T, H>(qkv, dout, lse, Dr, dqkv, bh / H, bh % H, kt, sm);
}
__global__ void __launch_bounds__(256, 2) k_dq(const bf16* qkv, const bf16* dout, const float* lse, const float* Dr, bf16* dqkv) {
  extern __shared__ __align__(128) char sm[];
  int t = blockIdx.x; int qt = NQ - 1 - t % NQ; int bh = t / NQ;
  attn_bwd_dq_tile<T, H>(qkv, dout, lse, Dr, dqkv, bh / H, bh % H, qt, sm);
}
// ---- persistent-style kernels: blocks pull tile indices from an atomic counter, in a scrambled order
__device__ __forceinline__ int scramble(int i, int n) { return (int)(((long long)i * 7919 + 13) % n); }  // n coprime to 7919
__global__ void __launch_bounds__(256, 2) p_fwd(const bf16* qkv, bf16* o, float* lse, int* ctr) {
  extern __shared__ __align__(128) char sm[]; __shared__ int s_t;
  for (;;) {
    if (threadIdx.x == 0) s_t = atomicAdd(ctr, 1);
    __syncthreads(); int i = s_t; __syncthreads();
    if (i >= NTASK) return;
    int t = scramble(i, NTASK); int qt = t % NQ; int bh = t / NQ;
    attn_fwd_tile<T, H>(qkv, o, lse, bh / H, bh % H, qt, sm);
  }
}
__global__ void __launch_bounds__(256, 2) p_prep(const bf16* o, const bf16* dout, float* Dr, int* ctr) {
  __shared__ int s_t;
  for (;;) {
    if (threadIdx.x == 0) s_t = atomicAdd(ctr, 1);
    __syncthreads(); int i = s_t; __syncthreads();
    if (i >= NTASK) return;
    int t = scramble(i, NTASK);
    attn_bwd_prep_tile<T, H>(o, dout, Dr, t / NQ / H, t / NQ % H, t % NQ);
  }
}
__global__ void __launch_bounds__(256, 2) p_bwd(const bf16* qkv, const bf16* dout, const float* lse, const float* Dr, bf16* dqkv, int* ctr) {
  extern __shared__ __align__(128) char sm[]; __shared__ int s_t;
  for (;;) {  // mixes dkdv and dq tasks in one queue (they are independent)
    if (threadIdx.x == 0) s_t = atomicAdd(ctr, 1);
    __syncthreads(); int i = s_t; __syncthreads();
    if (i >= 2 * NTASK) return;
    int t = scramble(i % NTASK, NTASK); int tile = t % NQ; int bh = t / NQ;
    if (i < NTASK) attn_bwd_dkdv_tile<T, H>(qkv, dout, lse, Dr, dqkv, bh / H, bh % H, tile, sm);
    else attn_bwd_dq_tile<T, H>(qkv, dout, lse, Dr, dqkv, bh / H, bh % H, tile, sm);
  }
}

static uint64_t rng = 88172645463325252ull;
static float urand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static float nrand() { float u = urand() + 1e-12f, v = urand(); return sqrtf(-2.f * logf(u)) * cosf(6.2831853f * v); }
static void dump(const std::string& dir, const char* name, const void* p, size_t bytes) {
  std::string f = dir + "/" + name; FILE* fp = fopen(f.c_str(), "wb"); fwrite(p, 1, bytes, fp); fclose(fp);
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "attn_data";
  system(("mkdir -p " + dir).c_str());
  size_t nqkv = (size_t)N * 3 * D, nd = (size_t)N * D;
  std::vector<bf16> hqkv(nqkv), hdo(nd);
  for (auto& x : hqkv) x = __float2bfloat16(nrand());
  for (auto& x : hdo) x = __float2bfloat16(nrand());
  bf16 *qkv, *dout, *o, *dqkv, *dqkv2, *o2; float *lse, *Dr, *lse2, *Dr2; int* ctr;
  CK(cudaMalloc(&qkv, nqkv * 2)); CK(cudaMalloc(&dout, nd * 2)); CK(cudaMalloc(&o, nd * 2)); CK(cudaMalloc(&o2, nd * 2));
  CK(cudaMalloc(&dqkv, nqkv * 2)); CK(cudaMalloc(&dqkv2, nqkv * 2));
  CK(cudaMalloc(&lse, (size_t)B * H * T * 4)); CK(cudaMalloc(&Dr, (size_t)B * H * T * 4));
  CK(cudaMalloc(&lse2, (size_t)B * H * T * 4)); CK(cudaMalloc(&Dr2, (size_t)B * H * T * 4)); CK(cudaMalloc(&ctr, 16));
  CK(cudaMemcpy(qkv, hqkv.data(), nqkv * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dout, hdo.data(), nd * 2, cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(k_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FWD));
  CK(cudaFuncSetAttribute(k_dkdv, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DKDV));
  CK(cudaFuncSetAttribute(k_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DQ));
  CK(cudaFuncSetAttribute(p_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FWD));
  CK(cudaFuncSetAttribute(p_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DKDV > SMEM_DQ ? SMEM_DKDV : SMEM_DQ));
  CK(cudaMemset(dqkv, 0, nqkv * 2)); CK(cudaMemset(dqkv2, 0, nqkv * 2));

  auto fwd = [&]() { k_fwd<<<NTASK, 256, SMEM_FWD>>>(qkv, o, lse); };
  auto bwd = [&]() {
    k_prep<<<NTASK, 256>>>(o, dout, Dr);
    k_dkdv<<<NTASK, 256, SMEM_DKDV>>>(qkv, dout, lse, Dr, dqkv);
    k_dq<<<NTASK, 256, SMEM_DQ>>>(qkv, dout, lse, Dr, dqkv);
  };
  fwd(); bwd(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  // persistent-style run, compare bit-for-bit
  int nblk = 216;
  CK(cudaMemset(ctr, 0, 16)); p_fwd<<<nblk, 256, SMEM_FWD>>>(qkv, o2, lse2, ctr);
  CK(cudaMemset(ctr + 1, 0, 4)); p_prep<<<nblk, 256>>>(o2, dout, Dr2, ctr + 1);
  CK(cudaMemset(ctr + 2, 0, 4)); p_bwd<<<nblk, 256, SMEM_DKDV>>>(qkv, dout, lse2, Dr2, dqkv2, ctr + 2);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<bf16> ho(nd), ho2(nd), hd(nqkv), hd2(nqkv); std::vector<float> hl((size_t)B * H * T), hl2(hl.size());
  CK(cudaMemcpy(ho.data(), o, nd * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ho2.data(), o2, nd * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hd.data(), dqkv, nqkv * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hd2.data(), dqkv2, nqkv * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hl.data(), lse, hl.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hl2.data(), lse2, hl.size() * 4, cudaMemcpyDeviceToHost));
  bool same = !memcmp(ho.data(), ho2.data(), nd * 2) && !memcmp(hd.data(), hd2.data(), nqkv * 2) && !memcmp(hl.data(), hl2.data(), hl.size() * 4);
  printf("persistent-vs-grid bit-identical: %s\n", same ? "PASS" : "FAIL");

  dump(dir, "qkv.bin", hqkv.data(), nqkv * 2); dump(dir, "do.bin", hdo.data(), nd * 2);
  dump(dir, "o.bin", ho.data(), nd * 2); dump(dir, "lse.bin", hl.data(), hl.size() * 4); dump(dir, "dqkv.bin", hd.data(), nqkv * 2);
  FILE* fp = fopen((dir + "/shape.txt").c_str(), "w"); fprintf(fp, "%d %d %d\n", B, T, H); fclose(fp);

  // timing
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int IT = 20; float msf, msb;
  for (int i = 0; i < 3; i++) { fwd(); bwd(); }
  cudaEventRecord(e0); for (int i = 0; i < IT; i++) fwd(); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&msf, e0, e1); msf /= IT;
  cudaEventRecord(e0); for (int i = 0; i < IT; i++) bwd(); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&msb, e0, e1); msb /= IT;
  double ff = 4.0 * B * H * (double)T * T * 64 / 2;
  printf("TIMING B=%d T=%d H=%d fwd_ms %.4f fwd_TFLOPs %.1f bwd_ms %.4f bwd_TFLOPs(2.5x) %.1f\n", B, T, H, msf, ff / (msf * 1e-3) / 1e12, msb, 2.5 * ff / (msb * 1e-3) / 1e12);
  return 0;
}
