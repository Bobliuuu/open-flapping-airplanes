// mbar_probe.cu : calibrate mbarrier state words / try_wait.parity polarity / tcgen05.commit arrival on this GPU (round 2).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ uint32_t tw(uint32_t a, uint32_t par) { uint32_t ok; asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory"); return ok; }
__device__ unsigned long long rd(uint32_t a) { unsigned long long w; asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(w) : "r"(a)); return w; }
__global__ void k() {
  __shared__ __align__(8) unsigned long long bar[4];
  __shared__ uint32_t slot;
  uint32_t b0 = (uint32_t)__cvta_generic_to_shared(&bar[0]), b1 = b0 + 8, b2 = b0 + 16;
  if (threadIdx.x < 32) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(&slot);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" ::"r"(s) : "memory");
  }
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 6;" ::"r"(b2));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    printf("fresh   b0=%016llx b1=%016llx b2(count6)=%016llx | tw(b0,0)=%u tw(b0,1)=%u\n", rd(b0), rd(b1), rd(b2), tw(b0, 0), tw(b0, 1));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b0) : "memory");
    printf("arrived b0=%016llx | tw(b0,0)=%u tw(b0,1)=%u\n", rd(b0), tw(b0, 0), tw(b0, 1));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(b1) : "memory");
    for (int i = 0; i < 1000; i++) asm volatile("nanosleep.u32 100;");
    printf("commit  b1=%016llx | tw(b1,0)=%u tw(b1,1)=%u\n", rd(b1), tw(b1, 0), tw(b1, 1));
  }
  __syncthreads();
  if (threadIdx.x < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(slot) : "memory");
}
int main() { k<<<1, 128>>>(); cudaError_t e = cudaDeviceSynchronize(); printf("%s\n", cudaGetErrorString(e)); return 0; }
