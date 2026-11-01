// M0: persistent-kernel VM on sm_80 (A100).
//
// Workload: N layers of a Llama-1B-shaped MLP decode step at batch 1
//   x -> rmsnorm -> (gate, up) -> silu(gate)*up -> down -> x += down
// H=2048, I=8192, bf16 weights, fp32 residual stream.
//
// Variants timed:
//   base       : 2 kernels per layer, plain stream launches
//   base+graph : same kernels captured in a CUDA graph
//   mega-s     : one persistent kernel, static round-robin task lists
//   mega-d     : one persistent kernel, dynamic atomic claim queue
//
// Mechanisms under test (see ../README.md section 2):
//   * one resident block per SM
//   * dependency counters in global memory, fence before every signal
//   * weight loads issued BEFORE the dependency wait (weights do not depend on it)
//   * claim queue for balance
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 mk.cu -o mk
// Run:   ./mk [--layers N] [--iters N] [--check]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

constexpr int H  = 2048;
constexpr int I  = 8192;
constexpr int GR = 8;            // intermediate indices per gate/up task (one per warp)
constexpr int DR = 4;            // output rows per down task (two warps per row, each owns half of K)
constexpr int NG = I / GR;       // 1024 gate/up tasks per layer
constexpr int ND = H / DR;       // 256 down tasks per layer
constexpr int NT = 256;          // threads per block (8 warps)

struct Layer { const bf16 *wg, *wu, *wd, *gamma; };
struct Instr { int op, idx, wait_ctr, wait_target, sig_ctr, layer, p0, p1; };

// ---------------------------------------------------------------- primitives
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(int* p) {
  asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p) : "memory");
}
// Block-wide wait: thread 0 spins with acquire, then barrier.
__device__ __forceinline__ void wait_ge(const int* c, int target) {
  if (threadIdx.x == 0) {
    while (ld_acquire(c) < target) __nanosleep(16);
  }
  __syncthreads();
  __threadfence();
}
// Block-wide signal: every thread fences its own writes, barrier, then one release-add.
__device__ __forceinline__ void signal_ctr(int* c, int dbg = 0) {
  if (!(dbg & 4)) __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && !(dbg & 8)) red_release(c);
}
__device__ __forceinline__ void fma8(const uint4 w, const float* xs, float& acc) {
  const float4 x0 = *reinterpret_cast<const float4*>(xs);
  const float4 x1 = *reinterpret_cast<const float4*>(xs + 4);
  acc = fmaf(__uint_as_float(w.x << 16),          x0.x, acc);
  acc = fmaf(__uint_as_float(w.x & 0xffff0000u),  x0.y, acc);
  acc = fmaf(__uint_as_float(w.y << 16),          x0.z, acc);
  acc = fmaf(__uint_as_float(w.y & 0xffff0000u),  x0.w, acc);
  acc = fmaf(__uint_as_float(w.z << 16),          x1.x, acc);
  acc = fmaf(__uint_as_float(w.z & 0xffff0000u),  x1.y, acc);
  acc = fmaf(__uint_as_float(w.w << 16),          x1.z, acc);
  acc = fmaf(__uint_as_float(w.w & 0xffff0000u),  x1.w, acc);
}

// ---------------------------------------------------------------- tasks
// Gate/up task: rmsnorm(x) then rows idx of Wg and Wu, writes act[idx] (bf16).
// Weight loads are issued before the dependency wait.
__device__ __forceinline__ void gateup_task(const Layer& w, int t, const float* x, bf16* act,
                                            float* xs, float* red,
                                            const int* wait_c, int wait_target) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int idx = t * GR + warp;
  const uint4* pg = reinterpret_cast<const uint4*>(w.wg + (size_t)idx * H);
  const uint4* pu = reinterpret_cast<const uint4*>(w.wu + (size_t)idx * H);
  uint4 g[8], u[8];
#pragma unroll
  for (int j = 0; j < 8; j++) { g[j] = ld_stream(pg + lane + 32 * j); u[j] = ld_stream(pu + lane + 32 * j); }

  if (wait_c) wait_ge(wait_c, wait_target);

  float ss = 0.f;
  for (int i = tid; i < H; i += NT) { float v = __ldcg(x + i); xs[i] = v; ss += v * v; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(~0u, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) tot += red[i];
  const float inv = rsqrtf(tot * (1.f / H) + 1e-5f);
  for (int i = tid; i < H; i += NT) xs[i] = xs[i] * inv * __bfloat162float(w.gamma[i]);
  __syncthreads();

  float ag = 0.f, au = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    fma8(g[j], xs + 8 * (lane + 32 * j), ag);
    fma8(u[j], xs + 8 * (lane + 32 * j), au);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { ag += __shfl_xor_sync(~0u, ag, o); au += __shfl_xor_sync(~0u, au, o); }
  if (lane == 0) act[idx] = __float2bfloat16(ag / (1.f + expf(-ag)) * au);
}

// Down task: DR rows of Wd over act (K=I), x[row] += result. Two warps per row, each owns half of K,
// so the WHOLE task's weights (16 x 16B per lane) are in registers before the dependency wait.
__device__ __forceinline__ void down_task(const Layer& w, int t, float* x, const bf16* act,
                                          float* xs, float* red, const int* wait_c, int wait_target) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = t * DR + (warp >> 1);
  const int half = warp & 1;
  const uint4* pw = reinterpret_cast<const uint4*>(w.wd + (size_t)row * I);
  uint4 d[16];
#pragma unroll
  for (int j = 0; j < 16; j++) d[j] = ld_stream(pw + lane + 32 * (16 * half + j));

  if (wait_c) wait_ge(wait_c, wait_target);

  for (int v = tid; v < I / 8; v += NT) {
    const uint4 a = __ldcg(reinterpret_cast<const uint4*>(act) + v);
    float* o = xs + 8 * v;
    o[0] = __uint_as_float(a.x << 16); o[1] = __uint_as_float(a.x & 0xffff0000u);
    o[2] = __uint_as_float(a.y << 16); o[3] = __uint_as_float(a.y & 0xffff0000u);
    o[4] = __uint_as_float(a.z << 16); o[5] = __uint_as_float(a.z & 0xffff0000u);
    o[6] = __uint_as_float(a.w << 16); o[7] = __uint_as_float(a.w & 0xffff0000u);
  }
  __syncthreads();

  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) fma8(d[j], xs + 8 * (lane + 32 * (16 * half + j)), acc);
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  // fixed order: half 0 + half 1
  if (half == 0 && lane == 0) x[row] = __ldcg(x + row) + (red[warp] + red[warp + 1]);
}

// Prefetch a task's weight rows into L2 (called one task ahead).
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
__device__ __forceinline__ void prefetch_task(const Layer* Ls, const Instr& in) {
  const Layer& w = Ls[in.layer];
  const int tid = threadIdx.x;
  if (in.op == 0) {            // 2 matrices x 8 rows x 32 lines of 128B
    for (int L = tid; L < 512; L += NT) {
      const int m = L >> 8, r = (L >> 5) & 7, c = L & 31;
      const bf16* base = m ? w.wu : w.wg;
      pf_l2(reinterpret_cast<const char*>(base + (size_t)(in.idx * GR + r) * H) + c * 128);
    }
  } else {                     // DR rows x 128 lines of 128B
    for (int L = tid; L < DR * 128; L += NT) {
      const int r = L >> 7, c = L & 127;
      pf_l2(reinterpret_cast<const char*>(w.wd + (size_t)(in.idx * DR + r) * I) + c * 128);
    }
  }
}

// ---------------------------------------------------------------- baseline kernels
__global__ void __launch_bounds__(NT) gateup_k(const Layer* Ls, int l, float* x, bf16* act) {
  __shared__ __align__(16) float xs[H];
  __shared__ float red[NT / 32];
  gateup_task(Ls[l], blockIdx.x, x, act, xs, red, nullptr, 0);
}
__global__ void __launch_bounds__(NT) down_k(const Layer* Ls, int l, float* x, bf16* act) {
  __shared__ __align__(16) float xs[I];
  __shared__ float red[NT / 32];
  down_task(Ls[l], blockIdx.x, x, act, xs, red, nullptr, 0);
}

// ---------------------------------------------------------------- megakernel
// mode 0: static round-robin (block b runs instrs b, b+G, ...)
// mode 1: dynamic claim queue (atomic ticket in global order)
#ifndef MK_MINB
#define MK_MINB 2
#endif
__global__ void __launch_bounds__(NT, MK_MINB) mega(const Layer* Ls, const Instr* prog, int ntasks,
                                           int* ctr, int* ticket, float* x, bf16* act, int mode, int pf, int dbg) {
  __shared__ __align__(16) float xs[I];
  __shared__ float red[NT / 32];
  __shared__ int tk;
  auto claim = [&](int n) -> int {
    if (mode == 0) return blockIdx.x + n * gridDim.x;
    if (threadIdx.x == 0) tk = atomicAdd(ticket, 1);
    __syncthreads();
    const int v = tk;
    __syncthreads();
    return v;
  };
  int cur = claim(0);
  for (int n = 1; cur < ntasks; n++) {
    const int nxt = claim(n);                 // claim one ahead; deadlock-free (deps only point backwards)
    if (pf && nxt < ntasks) prefetch_task(Ls, prog[nxt]);
    const Instr in = prog[cur];
    const int* wc = (in.wait_ctr >= 0 && !(dbg & 2)) ? ctr + in.wait_ctr : nullptr;
    if (in.op == 0) gateup_task(Ls[in.layer], in.idx, x, act, xs, red, wc, in.wait_target);
    else            down_task(Ls[in.layer], in.idx, x, act, xs, red, wc, in.wait_target);
    signal_ctr(ctr + in.sig_ctr, dbg);
    cur = nxt;
  }
}

// ---------------------------------------------------------------- host
__global__ void init_k(bf16* p, size_t n, uint32_t seed, float scale, float offset) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    float f = ((h & 0xffffff) / 16777216.f) * 2.f - 1.f;
    p[i] = __float2bfloat16(offset + f * scale);
  }
}

static double time_ms(const std::function<void()>& fn, int iters, cudaStream_t s) {
  for (int i = 0; i < 20; i++) fn();
  CK(cudaStreamSynchronize(s));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, s));
  for (int i = 0; i < iters; i++) fn();
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  return ms / iters;
}

int main(int argc, char** argv) {
  int NL = 16, iters = 200, bps = 1; bool check = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--layers") && i + 1 < argc) NL = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bps") && i + 1 < argc) bps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--check")) check = true;
  }
  if (check && NL > 4) { NL = 2; }

  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  int nsm, memclk, buswidth;
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev));
  CK(cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, dev));
  const double peak_gbs = 2.0 * memclk * 1e3 * (buswidth / 8.0) / 1e9;
  printf("device: %s  sm_%d%d  SMs=%d  peak HBM ~%.0f GB/s\n", prop.name, prop.major, prop.minor, nsm, peak_gbs);
  if (prop.major < 8) { fprintf(stderr, "needs sm_80+\n"); return 1; }

  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mega, NT, 0));
  if (occ < 1) { fprintf(stderr, "megakernel does not fit on an SM\n"); return 1; }
  printf("layers=%d  iters=%d  mega blocks/SM possible=%d, launching %d blocks\n", NL, iters, occ, nsm * bps);
  if (bps > occ) { fprintf(stderr, "bps > occupancy: would deadlock\n"); return 1; }

  // weights
  std::vector<Layer> hl(NL);
  const size_t nGU = (size_t)I * H, nD = (size_t)H * I;
  for (int l = 0; l < NL; l++) {
    bf16 *wg, *wu, *wd, *ga;
    CK(cudaMalloc(&wg, nGU * 2)); CK(cudaMalloc(&wu, nGU * 2));
    CK(cudaMalloc(&wd, nD * 2));  CK(cudaMalloc(&ga, H * 2));
    init_k<<<1024, 256>>>(wg, nGU, 1000 + l * 4 + 0, 1.7f / sqrtf((float)H), 0.f);
    init_k<<<1024, 256>>>(wu, nGU, 1000 + l * 4 + 1, 1.7f / sqrtf((float)H), 0.f);
    init_k<<<1024, 256>>>(wd, nD,  1000 + l * 4 + 2, 1.7f / sqrtf((float)I), 0.f);
    init_k<<<8, 256>>>(ga, H, 1000 + l * 4 + 3, 0.2f, 1.f);
    hl[l] = {wg, wu, wd, ga};
  }
  CK(cudaDeviceSynchronize());
  Layer* dL; CK(cudaMalloc(&dL, NL * sizeof(Layer)));
  CK(cudaMemcpy(dL, hl.data(), NL * sizeof(Layer), cudaMemcpyHostToDevice));

  // activations
  std::vector<float> x0(H);
  { uint32_t s = 12345; for (int i = 0; i < H; i++) { s = s * 1664525u + 1013904223u; x0[i] = ((s >> 8) / 16777216.f) * 2.f - 1.f; } }
  float *dx, *dx0; bf16* dact;
  CK(cudaMalloc(&dx, H * 4)); CK(cudaMalloc(&dx0, H * 4)); CK(cudaMalloc(&dact, I * 2));
  CK(cudaMemcpy(dx0, x0.data(), H * 4, cudaMemcpyHostToDevice));

  // program: layer-major, global topological order
  // counters: ctrG[l] = 2l, ctrD[l] = 2l+1; ticket = index 2*NL
  std::vector<Instr> prog;
  for (int l = 0; l < NL; l++) {
    for (int t = 0; t < NG; t++)
      prog.push_back({0, t, l > 0 ? 2 * (l - 1) + 1 : -1, l > 0 ? ND : 0, 2 * l, l, 0, 0});
    for (int t = 0; t < ND; t++)
      prog.push_back({1, t, 2 * l, NG, 2 * l + 1, l, 0, 0});
  }
  const int ntasks = (int)prog.size();
  Instr* dprog; CK(cudaMalloc(&dprog, ntasks * sizeof(Instr)));
  CK(cudaMemcpy(dprog, prog.data(), ntasks * sizeof(Instr), cudaMemcpyHostToDevice));
  int* dctr; CK(cudaMalloc(&dctr, (2 * NL + 1) * sizeof(int)));
  const size_t ctr_bytes = (2 * NL + 1) * sizeof(int);

  cudaStream_t s; CK(cudaStreamCreate(&s));

  auto reset = [&]() {
    CK(cudaMemcpyAsync(dx, dx0, H * 4, cudaMemcpyDeviceToDevice, s));
    CK(cudaMemsetAsync(dctr, 0, ctr_bytes, s));
  };
  auto run_base = [&]() {
    CK(cudaMemcpyAsync(dx, dx0, H * 4, cudaMemcpyDeviceToDevice, s));
    for (int l = 0; l < NL; l++) {
      gateup_k<<<NG, NT, 0, s>>>(dL, l, dx, dact);
      down_k<<<ND, NT, 0, s>>>(dL, l, dx, dact);
    }
  };
  auto run_mega = [&](int mode, int pf = 0, int dbg = 0) {
    reset();
    mega<<<nsm * bps, NT, 0, s>>>(dL, dprog, ntasks, dctr, dctr + 2 * NL, dx, dact, mode, pf, dbg);
  };
  auto fetch_x = [&](std::vector<float>& out) {
    out.resize(H);
    CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(out.data(), dx, H * 4, cudaMemcpyDeviceToHost));
  };

  // ---- correctness
  std::vector<float> xb, xs_, xd, xg;
  run_base(); CK(cudaGetLastError()); fetch_x(xb);
  run_mega(0); CK(cudaGetLastError()); fetch_x(xs_);
  run_mega(1); CK(cudaGetLastError()); fetch_x(xd);
  std::vector<float> xsp, xdp;
  run_mega(0, 1); CK(cudaGetLastError()); fetch_x(xsp);
  run_mega(1, 1); CK(cudaGetLastError()); fetch_x(xdp);
  bool eq_p = !memcmp(xb.data(), xsp.data(), H * 4) && !memcmp(xb.data(), xdp.data(), H * 4);
  printf("bitwise vs base: prefetch variants %s\n", eq_p ? "EQUAL" : "DIFFERENT");
  bool eq_s = !memcmp(xb.data(), xs_.data(), H * 4);
  bool eq_d = !memcmp(xb.data(), xd.data(), H * 4);
  printf("bitwise vs base: mega-s %s, mega-d %s\n", eq_s ? "EQUAL" : "DIFFERENT", eq_d ? "EQUAL" : "DIFFERENT");

  // graph
  cudaGraph_t graph; cudaGraphExec_t gexec;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  run_base();
  CK(cudaStreamEndCapture(s, &graph));
  CK(cudaGraphInstantiate(&gexec, graph, 0));
  CK(cudaGraphLaunch(gexec, s)); CK(cudaGetLastError()); fetch_x(xg);
  printf("bitwise base vs graph: %s\n", !memcmp(xb.data(), xg.data(), H * 4) ? "EQUAL" : "DIFFERENT");

  if (check) {
    // CPU reference on the (small) layer count
    auto d2h = [&](const bf16* p, size_t n) { std::vector<bf16> v(n); CK(cudaMemcpy(v.data(), p, n * 2, cudaMemcpyDeviceToHost)); return v; };
    std::vector<float> xr = x0;
    for (int l = 0; l < NL; l++) {
      auto wg = d2h(hl[l].wg, nGU), wu = d2h(hl[l].wu, nGU), wd = d2h(hl[l].wd, nD), ga = d2h(hl[l].gamma, H);
      double ss = 0; for (int i = 0; i < H; i++) ss += (double)xr[i] * xr[i];
      float inv = 1.f / sqrtf((float)(ss / H) + 1e-5f);
      std::vector<float> xn(H), actr(I);
      for (int i = 0; i < H; i++) xn[i] = xr[i] * inv * __bfloat162float(ga[i]);
      for (int k = 0; k < I; k++) {
        double g = 0, u = 0;
        for (int i = 0; i < H; i++) { float a = xn[i]; g += (double)__bfloat162float(wg[(size_t)k * H + i]) * a; u += (double)__bfloat162float(wu[(size_t)k * H + i]) * a; }
        float gf = (float)g, uf = (float)u;
        actr[k] = __bfloat162float(__float2bfloat16(gf / (1.f + expf(-gf)) * uf));
      }
      for (int r = 0; r < H; r++) {
        double a = 0; for (int k = 0; k < I; k++) a += (double)__bfloat162float(wd[(size_t)r * I + k]) * actr[k];
        xr[r] += (float)a;
      }
    }
    double maxe = 0, maxv = 0;
    for (int i = 0; i < H; i++) { maxe = fmax(maxe, fabs(xr[i] - xb[i])); maxv = fmax(maxv, fabs(xr[i])); }
    printf("CPU ref: max abs err %.3e, max |x| %.3e, rel %.3e  -> %s\n", maxe, maxv, maxe / maxv, (maxe / maxv < 2e-2) ? "PASS" : "FAIL");
    return (eq_s && eq_d && eq_p && maxe / maxv < 2e-2) ? 0 : 2;
  }

  // ---- timing
  const double bytes = (double)NL * (2.0 * nGU + nD) * 2.0;
  auto report = [&](const char* name, double ms) {
    printf("%-12s %8.1f us/forward   %7.1f GB/s   %5.1f%% of peak\n", name, ms * 1e3, bytes / (ms * 1e-3) / 1e9, 100.0 * bytes / (ms * 1e-3) / 1e9 / peak_gbs);
  };
  printf("weight bytes per forward: %.1f MB\n", bytes / 1e6);
  double t_base  = time_ms(run_base, iters, s);                                   report("base", t_base);
  double t_graph = time_ms([&]() { CK(cudaGraphLaunch(gexec, s)); }, iters, s);   report("base+graph", t_graph);
  double t_ms    = time_ms([&]() { run_mega(0); }, iters, s);                     report("mega-static", t_ms);
  double t_md    = time_ms([&]() { run_mega(1); }, iters, s);                     report("mega-dynamic", t_md);
  double t_msp   = time_ms([&]() { run_mega(0, 1); }, iters, s);                  report("mega-s+pf", t_msp);
  double t_mdp   = time_ms([&]() { run_mega(1, 1); }, iters, s);                  report("mega-d+pf", t_mdp);
  if (getenv("MK_DBG")) {
    report("d  nowait",   time_ms([&]() { run_mega(1, 0, 2); }, iters, s));
    report("d  nofence",  time_ms([&]() { run_mega(1, 0, 4); }, iters, s));
    report("d  nowait+nofence", time_ms([&]() { run_mega(1, 0, 6); }, iters, s));
    report("s  nowait",   time_ms([&]() { run_mega(0, 0, 2); }, iters, s));
    report("s  nowait+nofence", time_ms([&]() { run_mega(0, 0, 6); }, iters, s));
  }
  CK(cudaGetLastError());
  printf("speedup vs base+graph: static %.2fx, dynamic %.2fx, static+pf %.2fx, dynamic+pf %.2fx\n", t_graph / t_ms, t_graph / t_md, t_graph / t_msp, t_graph / t_mdp);
  // final determinism re-check after timing churn
  run_mega(1); fetch_x(xd);
  printf("post-timing bitwise mega-d vs base: %s\n", !memcmp(xb.data(), xd.data(), H * 4) ? "EQUAL" : "DIFFERENT");
  return 0;
}
