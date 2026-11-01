// Standalone test: boot Doom inside ONE GPU thread, run N frames on a simulated clock, dump the screen.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
extern "C" __device__ void gpu_doom_setup(const unsigned char*, long, volatile char*, long);
extern "C" __device__ void gpu_doom_create(unsigned long long);
extern "C" __device__ void gpu_doom_frame(unsigned long long);
extern "C" __device__ void gpu_doom_worker(int);
extern "C" __device__ void gpu_doom_enable_par(int);
extern "C" __device__ unsigned char* I_VideoBuffer;
extern "C" __device__ unsigned int colors[256];
extern "C" __device__ unsigned int* DG_ScreenBuffer;

__global__ void k_setup(const unsigned char* wad, long n, volatile char* dbg, long dbglen) { if (threadIdx.x == 0) gpu_doom_setup(wad, n, dbg, dbglen); }
__global__ void k_create() { if (threadIdx.x == 0) { gpu_doom_enable_par(1); gpu_doom_create(0ull); } }
__global__ void k_frames(int n, int first) { for (int i = 0; i < n; i++) { if (threadIdx.x == 0) gpu_doom_frame((unsigned long long)(first + i) * 28571429ull); else gpu_doom_worker(threadIdx.x); } }
__global__ void k_grab(unsigned char* rgb) { for (int i = threadIdx.x; i < 320 * 200; i += blockDim.x) { const unsigned p = colors[I_VideoBuffer[i]]; rgb[i * 3] = (p >> 16) & 255; rgb[i * 3 + 1] = (p >> 8) & 255; rgb[i * 3 + 2] = p & 255; } }

int main(int argc, char** argv) {
  const char* wadpath = argc > 1 ? argv[1] : "doom1.wad"; const int nframes = argc > 2 ? atoi(argv[2]) : 120;
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  FILE* f = fopen(wadpath, "rb"); if (!f) { fprintf(stderr, "no wad %s\n", wadpath); return 1; }
  std::vector<unsigned char> wad; { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); wad.resize(n); if (fread(wad.data(), 1, n, f) != (size_t)n) return 1; fclose(f); }
  printf("WAD %zu bytes\n", wad.size());
  unsigned char* dwad; CK(cudaMalloc(&dwad, wad.size())); CK(cudaMemcpy(dwad, wad.data(), wad.size(), cudaMemcpyHostToDevice));
  char *hdbg, *ddbg; CK(cudaHostAlloc((void**)&hdbg, 1 << 20, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&ddbg, hdbg, 0)); memset(hdbg, 0, 1 << 20);
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 24 * 1024));
  auto now = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  k_setup<<<1, 1>>>(dwad, (long)wad.size(), ddbg, 1 << 20); CK(cudaDeviceSynchronize());
  double t = now(); k_create<<<1, 256>>>();
  { size_t shown = 0; double tl = now();                                  // stream Doom's own log while the kernel runs
    while (cudaStreamQuery(0) == cudaErrorNotReady && now() - t < 60) { usleep(300000); size_t n = strlen((const char*)hdbg); if (n > shown) { fwrite(hdbg + shown, 1, n - shown, stdout); fflush(stdout); shown = n; tl = now(); } }
    if (cudaStreamQuery(0) == cudaErrorNotReady) { printf("\n[still running after %.0fs, last log %.1fs ago]\n", now() - t, now() - tl); fflush(stdout); _exit(3); } }
  cudaError_t e = cudaDeviceSynchronize();
  printf("create: %s (%.2fs)\n--- doom output ---\n%s\n-------------------\n", cudaGetErrorString(e), now() - t, hdbg);
  if (e != cudaSuccess) return 1;
  t = now(); k_frames<<<1, 256>>>(nframes, 1);
  { size_t shown = strlen((const char*)hdbg); while (cudaStreamQuery(0) == cudaErrorNotReady && now() - t < 20) { usleep(200000); size_t n = strlen((const char*)hdbg); if (n > shown) { fwrite(hdbg + shown, 1, n - shown, stdout); fflush(stdout); shown = n; } }
    if (cudaStreamQuery(0) == cudaErrorNotReady) { printf("\n[frames kernel still running after 20s]\n"); fflush(stdout); _exit(3); } }
  e = cudaDeviceSynchronize();
  printf("%d frames: %s (%.3fs = %.1f ms/frame)\n%s\n", nframes, cudaGetErrorString(e), now() - t, (now() - t) * 1e3 / nframes, hdbg + 0);
  if (e != cudaSuccess) return 1;
  unsigned char* drgb; CK(cudaMalloc(&drgb, 320 * 200 * 3)); k_grab<<<1, 256>>>(drgb); CK(cudaDeviceSynchronize());
  std::vector<unsigned char> rgb(320 * 200 * 3); CK(cudaMemcpy(rgb.data(), drgb, rgb.size(), cudaMemcpyDeviceToHost));
  FILE* o = fopen("doom_frame.ppm", "wb"); fprintf(o, "P6\n320 200\n255\n"); fwrite(rgb.data(), 1, rgb.size(), o); fclose(o);
  long sum = 0; for (unsigned char c : rgb) sum += c; printf("frame dumped (pixel sum %ld)\n", sum);
  return 0;
}
