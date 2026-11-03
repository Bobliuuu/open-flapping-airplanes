// occ_probe.cu : does a kernel that uses tcgen05.alloc/TMEM co-reside 2 CTAs per SM on this GPU?  (round 2)
// build: nvcc -gencode arch=compute_100f,code=sm_100f -O2 -o occ_probe occ_probe.cu ; run: ./occ_probe
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
template <int COLS>
__global__ void __launch_bounds__(256, 2) k(unsigned long long* t0, unsigned long long* t1, unsigned* sm, unsigned long long spin_ns) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ uint32_t slot;
  if (COLS > 0 && threadIdx.x < 32) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(&slot);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s), "r"(COLS) : "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); sm[blockIdx.x] = id;
    unsigned long long a, b; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a)); t0[blockIdx.x] = a;
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b)); } while (b - a < spin_ns);
    t1[blockIdx.x] = b;
  }
  __syncthreads();
  if (COLS > 0 && threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(slot), "r"(COLS) : "memory");
  }
}
template <int COLS> void run(const char* name, int smem_kb) {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  int nb = 2 * nsm;
  CK(cudaFuncSetAttribute(k<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_kb * 1024));
  int occ = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k<COLS>, 256, smem_kb * 1024);
  unsigned long long *t0, *t1; unsigned* sm; CK(cudaMallocManaged(&t0, nb * 8)); CK(cudaMallocManaged(&t1, nb * 8)); CK(cudaMallocManaged(&sm, nb * 4));
  k<COLS><<<nb, 256, smem_kb * 1024>>>(t0, t1, sm, 200000ull); CK(cudaDeviceSynchronize());
  // max overlap per SM
  int maxc = 0;
  for (int s = 0; s < nsm; s++) {
    int c = 0; for (int i = 0; i < nb; i++) if (sm[i] == (unsigned)s) { int cc = 0; for (int j = 0; j < nb; j++) if (sm[j] == (unsigned)s && t0[j] < t1[i] && t1[j] > t0[i]) cc++; if (cc > c) c = cc; }
    if (c > maxc) maxc = c;
  }
  printf("%-28s smem %3d KB  occupancyAPI=%d  max concurrent CTAs per SM observed=%d\n", name, smem_kb, occ, maxc);
  cudaFree(t0); cudaFree(t1); cudaFree(sm);
}
int main() {
  run<0>("no tcgen05 (control)", 100);
  run<128>("tcgen05.alloc 128 cols", 100);
  run<256>("tcgen05.alloc 256 cols", 100);
  run<512>("tcgen05.alloc 512 cols", 100);
  run<128>("tcgen05.alloc 128 cols", 64);
  return 0;
}
