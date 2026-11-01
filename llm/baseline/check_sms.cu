// Counts distinct SMs that blocks of one grid actually run on (verifies MPS SM restriction).
#include <cstdio>
__global__ void k(unsigned* m, long long spin) {
  if (threadIdx.x == 0) {
    unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    atomicOr(&m[id >> 5], 1u << (id & 31));
    long long t0 = clock64(); while (clock64() - t0 < spin) {}
  }
}
int main() {
  unsigned* m; cudaMallocManaged(&m, 64 * 4); for (int i = 0; i < 64; i++) m[i] = 0;
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  k<<<sms * 16, 128>>>(m, 100000); cudaDeviceSynchronize();
  int n = 0; for (int i = 0; i < 64; i++) n += __builtin_popcount(m[i]);
  printf("device_reported_sms=%d distinct_smids_used=%d\n", sms, n);
}
