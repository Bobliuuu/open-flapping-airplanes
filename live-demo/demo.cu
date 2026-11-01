// Live megakernel demo (A100): ONE persistent kernel runs, at the same time,
//   * TRAINING   a character-level language model (embedding -> tanh MLP -> softmax), forward+backward+SGD in-kernel
//   * INFERENCE  streaming text generation from the LIVE weights while they train (the text improves as the loss falls)
//   * GAME       real Doom (doomgeneric, shareware E1M1) running its own game loop in a GPU thread, frames copied to pinned host memory
// A browser dashboard (served by this binary, reached through an SSH tunnel) shows all three, then compares against the
// same work as separate per-op kernels on CUDA streams (kernel launches, training and inference throughput).
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 -lpthread demo.cu -o demo        Run: ./demo --port 8080

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>

typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

// ------------------------------------------------------------------ model / job dimensions
constexpr int V = 96, D = 16, C = 8, DIN = C * D, HID = 512, B = 256, TU = 8, NT = 256;
constexpr int NP1 = HID / TU, NP2 = B / 16, NP3 = HID / TU, TPT = NP1 + NP2 + NP3;     // training tasks per step
constexpr int IA = HID / 16, TPI = IA + 1;                                              // inference tasks per token
constexpr int GW = 320, GH = 200, RT = 5, TPG = 1 + RT, FBSZ = GW * GH + 1024;     // frame = 64000 palette indices + 256 x RGBA palette
                                 // Doom: one tick task + RT copy tasks per frame
constexpr float LR = 0.012f;
constexpr int SMF = 11000;                                                              // floats of per-block scratch smem
constexpr int RING = 1 << 16;

struct GState { float px, py, pa; int turn_dir; };
struct Live { volatile unsigned frames, chars, steps; volatile float loss; };

struct Params {
  u64* t0; int* ctr; int* ticket; int* limit;
  long long dur_ns[3]; long long pace_ns[3];
  const unsigned char* text; int ntext; const float* E;
  float *W1, *b1, *W2T, *b2, *Hb, *dz, *dzT, *lossb, *lossv, *hinf; int* ctxi;
  GState* gs; unsigned char* fb; unsigned char* ring; Live* live; u64* busy;
  u64* frame_done; u64* tok_done; u64* step_done; unsigned seed; u64* tprof; int* tq;
  const unsigned char* wad; long wad_len; volatile char* dbg; long dbg_len; u64* game_epoch;
};

// Doom engine: compiled separately by clang for NVPTX (see ../doom-gpu) and linked in as a device object
extern "C" __device__ void gpu_doom_setup(const unsigned char*, long, volatile char*, long);
extern "C" __device__ void gpu_doom_create(unsigned long long);
extern "C" __device__ void gpu_doom_frame(unsigned long long);
extern "C" __device__ void gpu_doom_worker_always(int);
extern "C" __device__ void gpu_doom_enable_par(int);
extern "C" __device__ unsigned char* I_VideoBuffer;
extern "C" __device__ unsigned int colors[256];

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_rlx(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ int ld_acquire(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int signal_ret(int* c) {
  __threadfence(); __syncthreads();
  int old = 0; if (threadIdx.x == 0) old = atomicAdd(c, 1);
  return old;
}
__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

__device__ unsigned long long g_sec[16];
#ifdef SECT
#define SEC(i) do { if (threadIdx.x == 0 && blockIdx.x == 1) g_sec[i] = clock64(); } while (0)
#else
#define SEC(i) do {} while (0)
#endif
// ------------------------------------------------------------------ TRAINING tasks
__device__ __forceinline__ int sample_pos(const Params& P, int s, int b) { return (int)(hash32((unsigned)(s * B + b) + P.seed) % (unsigned)(P.ntext - C - 1)); }

// P1: hidden activations for TU units, thread = batch element
__device__ void train_p1(const Params& P, int jb, int s, float* sm) {
  const int tid = threadIdx.x;
  float* wt = sm;                                           // [TU][DIN]
  const int pos = sample_pos(P, s, tid);
  unsigned char ch[C];
#pragma unroll
  for (int k = 0; k < C; k++) ch[k] = __ldg(P.text + pos + k);
  float acc[TU];
#pragma unroll
  for (int u = 0; u < TU; u++) acc[u] = __ldcg(P.b1 + jb * TU + u);
  for (int p = tid; p < TU * DIN; p += NT) wt[p] = __ldcg(P.W1 + (size_t)jb * TU * DIN + p);
  __syncthreads();
#pragma unroll
  for (int k = 0; k < C; k++) {
    float e[D];
#pragma unroll
    for (int t = 0; t < D; t += 4) { const float4 v = __ldg(reinterpret_cast<const float4*>(P.E + ch[k] * D + t)); e[t] = v.x; e[t + 1] = v.y; e[t + 2] = v.z; e[t + 3] = v.w; }
#pragma unroll
    for (int u = 0; u < TU; u++)
#pragma unroll
      for (int t = 0; t < D; t += 4) {
        const float4 w = *reinterpret_cast<const float4*>(wt + u * DIN + k * D + t);
        acc[u] = fmaf(w.x, e[t], acc[u]); acc[u] = fmaf(w.y, e[t + 1], acc[u]); acc[u] = fmaf(w.z, e[t + 2], acc[u]); acc[u] = fmaf(w.w, e[t + 3], acc[u]);
      }
  }
#pragma unroll
  for (int u = 0; u < TU; u++) P.Hb[(size_t)(jb * TU + u) * B + tid] = tanhf(acc[u]);
}

// P2: logits + softmax + loss + dlogits for 16 batch elements.
// Register blocking (4 samples x 6 logits per thread) keeps the shared-memory wavefront count low; the 512-wide reduction is split over 4 thread groups
// (group g owns rows g*16..g*16+15 of every 64-row tile) and combined in a fixed order, so results are deterministic.
__device__ void train_p2(const Params& P, int pb, int s, float* sm) {
  const int tid = threadIdx.x, g = tid >> 6, lt = tid & 63, bq = lt & 3, vg6 = lt >> 2;     // group, 4-sample quad, 6-logit group
  float* ws = sm;                                           // [64][V]; reused for the group partials [4][16][V] after the loop
  float* hs = sm + 64 * V;                                  // [64][16]
  float* red1 = sm + 64 * V + 64 * 16;                      // [16 vg][16 b]
  float* red2 = red1 + 256;
  float acc[4][6];
#pragma unroll
  for (int bi = 0; bi < 4; bi++)
#pragma unroll
    for (int i = 0; i < 6; i++) acc[bi][i] = 0.f;
  float wr[24], hr[4];
  SEC(9);
#pragma unroll
  for (int i = 0; i < 24; i++) wr[i] = __ldcg(P.W2T + (size_t)(tid + NT * i));
#pragma unroll
  for (int i = 0; i < 4; i++) { const int p = tid + NT * i; hr[i] = __ldcg(P.Hb + (size_t)(p >> 4) * B + pb * 16 + (p & 15)); }
  SEC(6);
  for (int jt = 0; jt < HID / 64; jt++) {
#pragma unroll
    for (int i = 0; i < 24; i++) ws[tid + NT * i] = wr[i];
#pragma unroll
    for (int i = 0; i < 4; i++) hs[tid + NT * i] = hr[i];
    __syncthreads();
    if (jt + 1 < HID / 64) {
      const int tn = jt + 1;
#pragma unroll
      for (int i = 0; i < 24; i++) wr[i] = __ldcg(P.W2T + (size_t)tn * 64 * V + tid + NT * i);
#pragma unroll
      for (int i = 0; i < 4; i++) { const int p = tid + NT * i; hr[i] = __ldcg(P.Hb + (size_t)(tn * 64 + (p >> 4)) * B + pb * 16 + (p & 15)); }
    }
#pragma unroll 4
    for (int jj = 0; jj < 16; jj++) {
      const int row = g * 16 + jj;
      const float4 h = *reinterpret_cast<const float4*>(hs + row * 16 + bq * 4);
      const float hv[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
      for (int m = 0; m < 3; m++) {
        const float2 w = *reinterpret_cast<const float2*>(ws + row * V + vg6 * 6 + 2 * m);
#pragma unroll
        for (int bi = 0; bi < 4; bi++) { acc[bi][2 * m] = fmaf(w.x, hv[bi], acc[bi][2 * m]); acc[bi][2 * m + 1] = fmaf(w.y, hv[bi], acc[bi][2 * m + 1]); }
      }
    }
    __syncthreads();
  }
  SEC(7);
  float* part = sm;                                         // [4 groups][16 b][V]
#pragma unroll
  for (int bi = 0; bi < 4; bi++)
#pragma unroll
    for (int i = 0; i < 6; i++) part[(g * 16 + bq * 4 + bi) * V + vg6 * 6 + i] = acc[bi][i];
  __syncthreads();
  // combine the groups in fixed order and add the bias; thread (bl, vg) owns 6 logits of sample bl
  const int bl = tid & 15, vg = tid >> 4, b = pb * 16 + bl, sidx = s & 1;
  float z[6];
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int v = vg * 6 + i;
    z[i] = __ldcg(P.b2 + sidx * V + v) + (((part[bl * V + v] + part[(16 + bl) * V + v]) + part[(32 + bl) * V + v]) + part[(48 + bl) * V + v]);
  }
  float lm = z[0];
#pragma unroll
  for (int i = 1; i < 6; i++) lm = fmaxf(lm, z[i]);
  red1[vg * 16 + bl] = lm;
  __syncthreads();
  float m = red1[bl];
#pragma unroll
  for (int gg = 1; gg < 16; gg++) m = fmaxf(m, red1[gg * 16 + bl]);
  float e[6], ls = 0.f;
#pragma unroll
  for (int i = 0; i < 6; i++) { e[i] = expf(z[i] - m); ls += e[i]; }
  red2[vg * 16 + bl] = ls;
  __syncthreads();
  float sum = 0.f;
#pragma unroll
  for (int gg = 0; gg < 16; gg++) sum += red2[gg * 16 + bl];
  const float inv = 1.f / sum;
  const int tgt = __ldg(P.text + sample_pos(P, s, b) + C);
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int v = vg * 6 + i;
    const float p = e[i] * inv, d = (p - (v == tgt ? 1.f : 0.f)) * (1.f / B);
    P.dz[(size_t)b * V + v] = d; P.dzT[(size_t)v * B + b] = d;
    if (v == tgt) P.lossb[b] = -logf(p + 1e-20f);
  }
  SEC(8);
}

// P3: backward + SGD update for TU hidden units
__device__ void train_p3(const Params& P, int jb, int s, float* sm) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  SEC(0);
  float* dh = sm;                                           // [TU][B]
  float* hs = sm + TU * B;                                  // [TU][B]
  float* w2s = sm + 2 * TU * B;                             // [TU][V]
  float* Es = sm + 2 * TU * B + TU * V;                     // [128][D] embedding table
  unsigned char* chT = reinterpret_cast<unsigned char*>(Es + 128 * D);   // [C][B] context characters, batch-contiguous
  {
    const int pos = sample_pos(P, s, tid);
#pragma unroll
    for (int k = 0; k < C; k++) chT[k * B + tid] = __ldg(P.text + pos + k);
  }
  for (int p = tid; p < 128 * D; p += NT) Es[p] = __ldg(P.E + p);
#pragma unroll
  for (int u = 0; u < TU; u++) hs[u * B + tid] = __ldcg(P.Hb + (size_t)(jb * TU + u) * B + tid);
  for (int p = tid; p < TU * V; p += NT) w2s[p] = __ldcg(P.W2T + (size_t)jb * TU * V + p);
  __syncthreads();
  SEC(1);
  {                                                         // dh[u][b] = (1-h^2) * sum_v W2[u][v] dz[b][v]
    float a[TU];
#pragma unroll
    for (int u = 0; u < TU; u++) a[u] = 0.f;
#pragma unroll 16
    for (int v = 0; v < V; v++) {
      const float d = __ldcg(P.dzT + (size_t)v * B + tid);
#pragma unroll
      for (int u = 0; u < TU; u++) a[u] = fmaf(w2s[u * V + v], d, a[u]);
    }
#pragma unroll
    for (int u = 0; u < TU; u++) { const float h = hs[u * B + tid]; dh[u * B + tid] = a[u] * (1.f - h * h); }
  }
  __syncthreads();
  SEC(2);
  if (tid < 2 * V) {                                        // dW2[u][v] = sum_b h[u][b] dz[b][v]; one thread per (v, half of the units)
    constexpr int UH = TU / 2;
    const int v = tid % V, uq = tid / V;
    float g[UH], sdz = 0.f;
#pragma unroll
    for (int u = 0; u < UH; u++) g[u] = 0.f;
#pragma unroll 4
    for (int b = 0; b < B; b += 4) {
      float d[4];
#pragma unroll
#ifdef NODZ
      for (int k = 0; k < 4; k++) d[k] = hs[(b + k) & 255] * 0.5f;
#else
      for (int k = 0; k < 4; k++) d[k] = __ldcg(P.dz + (size_t)(b + k) * V + v);
#endif
      sdz += d[0]; sdz += d[1]; sdz += d[2]; sdz += d[3];   // output-bias gradient comes for free (used by task 0 only)
#pragma unroll
      for (int u = 0; u < UH; u++) {
        const float4 h4 = *reinterpret_cast<const float4*>(hs + (uq * UH + u) * B + b);
        g[u] = fmaf(h4.x, d[0], g[u]); g[u] = fmaf(h4.y, d[1], g[u]); g[u] = fmaf(h4.z, d[2], g[u]); g[u] = fmaf(h4.w, d[3], g[u]);
      }
    }
#pragma unroll
    for (int u = 0; u < UH; u++) P.W2T[(size_t)(jb * TU + uq * UH + u) * V + v] -= LR * g[u];
    if (jb == 0 && uq == 0) P.b2[((s + 1) & 1) * V + v] = P.b2[(s & 1) * V + v] - LR * sdz;
  }
  SEC(3); __syncthreads(); SEC(4);
  for (int u = warp; u < TU; u += NT / 32) {                // db1
    float s2 = 0.f;
#pragma unroll
    for (int k = 0; k < B / 32; k++) s2 += dh[u * B + lane + 32 * k];
#pragma unroll
    for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(~0u, s2, o);
    if (lane == 0) P.b1[jb * TU + u] -= LR * s2;
  }
  {                                                         // dW1[u][i] = sum_b dh[u][b] x[b][i]
    constexpr int UG = TU / 2;
    const int i = tid & (DIN - 1), ug = tid / DIN, e0 = i & 15;
    const unsigned char* cp = chT + (i >> 4) * B;
    float g[UG];
#pragma unroll
    for (int u = 0; u < UG; u++) g[u] = 0.f;
#pragma unroll 2
    for (int b = 0; b < B; b += 4) {
      const unsigned c4 = *reinterpret_cast<const unsigned*>(cp + b);
      const float x0 = Es[(c4 & 255u) * D + e0], x1 = Es[((c4 >> 8) & 255u) * D + e0], x2 = Es[((c4 >> 16) & 255u) * D + e0], x3 = Es[(c4 >> 24) * D + e0];
#pragma unroll
      for (int u = 0; u < UG; u++) {
        const float4 d4 = *reinterpret_cast<const float4*>(dh + (ug * UG + u) * B + b);
        g[u] = fmaf(d4.x, x0, g[u]); g[u] = fmaf(d4.y, x1, g[u]); g[u] = fmaf(d4.z, x2, g[u]); g[u] = fmaf(d4.w, x3, g[u]);
      }
    }
#pragma unroll
    for (int u = 0; u < UG; u++) P.W1[(size_t)(jb * TU + ug * UG + u) * DIN + i] -= LR * g[u];
  }
  __syncthreads(); SEC(5);
  if (jb == 0 && tid < 32) {                                // step loss: fixed-order warp reduction
    float l = 0.f;
#pragma unroll
    for (int k = 0; k < B / 32; k++) l += __ldcg(P.lossb + lane + 32 * k);
#pragma unroll
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(~0u, l, o);
    if (lane == 0) P.lossv[s & 1] = l / B;
  }
}

// ------------------------------------------------------------------ INFERENCE tasks (read the live training weights)
__device__ void infer_a(const Params& P, int r) {           // 16 hidden units of the generation context
  const int t = threadIdx.x;
  if (t < 16) {
    const int j = r * 16 + t;
    float acc = __ldcg(P.b1 + j);
    for (int k = 0; k < C; k++) {
      const int ch = __ldcg(P.ctxi + k);
      for (int d = 0; d < D; d += 4) {
        const float4 w = __ldcg(reinterpret_cast<const float4*>(P.W1 + (size_t)j * DIN + k * D + d));
        const float4 e = __ldg(reinterpret_cast<const float4*>(P.E + ch * D + d));
        acc = fmaf(w.x, e.x, acc); acc = fmaf(w.y, e.y, acc); acc = fmaf(w.z, e.z, acc); acc = fmaf(w.w, e.w, acc);
      }
    }
    P.hinf[j] = tanhf(acc);
  }
}
__device__ void infer_b(const Params& P, int n, float* sm) {  // logits -> sample -> append to the output ring and the context
  const int t = threadIdx.x;
  float* hs = sm;                                           // [HID]
  float* part = sm + HID;                                   // [2][V]
  for (int j = t; j < HID; j += NT) hs[j] = __ldcg(P.hinf + j);
  __syncthreads();
  if (t < 2 * V) {
    const int v = t % V, half = t / V;
    float acc = 0.f;
#pragma unroll 16
    for (int j = half * (HID / 2); j < (half + 1) * (HID / 2); j++) acc = fmaf(__ldcg(P.W2T + (size_t)j * V + v), hs[j], acc);
    part[half * V + v] = acc;
  }
  __syncthreads();
  if (t < 32) {                                             // warp-parallel softmax and sampling; lane l owns logits 3l..3l+2
    float z[3]; float m = -1e30f;
#pragma unroll
    for (int i = 0; i < 3; i++) { const int v = t * 3 + i; z[i] = (part[v] + part[V + v] + __ldcg(P.b2 + v)) * 1.25f; m = fmaxf(m, z[i]); }
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    float ls = 0.f;
#pragma unroll
    for (int i = 0; i < 3; i++) { z[i] = expf(z[i] - m); ls += z[i]; }
    float inc = ls;                                         // inclusive scan of the per-lane sums, in lane order
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const float up = __shfl_up_sync(~0u, inc, o); if (t >= o) inc += up; }
    const float total = __shfl_sync(~0u, inc, 31);
    const float r = (hash32((unsigned)n * 2654435761u + 77u) & 0xffffff) / 16777216.f * total;
    const float before = inc - ls;
    int pick = -1;
    if (before < r && r <= inc) { float c = before; pick = t * 3 + 2; for (int i = 0; i < 3; i++) { c += z[i]; if (c >= r) { pick = t * 3 + i; break; } } }
    const unsigned mk = __ballot_sync(~0u, pick >= 0);
    int win = mk ? (__ffs(mk) - 1) : 31;
    int ch = __shfl_sync(~0u, pick, win);
    if (ch < 0) ch = V - 1;
    if (t == 0) {
      P.ring[n & (RING - 1)] = (unsigned char)(ch == 95 ? '\n' : 32 + ch);
      for (int k = 0; k + 1 < C; k++) P.ctxi[k] = __ldcg(P.ctxi + k + 1);
      P.ctxi[C - 1] = ch;
    }
  }
}

// ------------------------------------------------------------------ GAME tasks (real Doom)
// Task 0 of each frame runs one Doom tic (the whole game: AI, physics, BSP renderer) on a single GPU thread; frame 0 also boots the engine.
// Doom's clock is virtual (35 tics per second of game time), so the game is deterministic and independent of scheduling.
__device__ void game_tick(const Params& P, int f) {
  // All 256 threads of the block take part: thread 0 runs the game, the others execute Doom's queued column/span draws in parallel.
  if (threadIdx.x != 0) { gpu_doom_worker_always(threadIdx.x); return; }
  if (f == 0) { gpu_doom_setup(P.wad, P.wad_len, P.dbg, P.dbg_len); gpu_doom_enable_par(1); gpu_doom_create(0ull); *reinterpret_cast<volatile u64*>(P.game_epoch) = gtimer(); }
  gpu_doom_frame((unsigned long long)(f + 1) * 28571429ull);
}
// Tasks 1..RT: copy 200/RT rows of Doom's 8-bit screen into the pinned frame buffer (task 1 also copies the palette).
__device__ void game_copy(const Params& P, int rt) {
  const int rows = GH / RT, p0 = rt * rows * GW;
  const unsigned char* vb = I_VideoBuffer;
  for (int p = threadIdx.x; p < rows * GW; p += NT) P.fb[p0 + p] = __ldcg(vb + p0 + p);
  if (rt == 0) for (int i = threadIdx.x; i < 256; i += NT) reinterpret_cast<unsigned int*>(P.fb + GW * GH)[i] = __ldcg(colors + i);
}

// ------------------------------------------------------------------ scheduler: sharded tickets, dependency counters
// counters: 0 game physics, 1 game render, 2 infer A, 3 infer B, 4 train P1, 5 train P2, 6 train P3
__device__ __forceinline__ int tasks_per(int j) { return j == 0 ? TPG : j == 1 ? TPI : TPT; }
__device__ __forceinline__ bool admit_check(const Params& P, int j, int step, u64 T0, u64 now) {   // true = blocked by deadline
  if (P.dur_ns[j] > 0 && now >= T0 + (u64)P.dur_ns[j]) { atomicMin(P.limit + j, step); return true; }
  return false;
}
__device__ __forceinline__ void cand_info(const Params& P, int j, int g, u64 T0, u64 now, int& cidx, int& tg, bool& blocked) {
  blocked = false;
  if (j == 0) {
    const int f = g / TPG, r = g % TPG;
    if (r == 0) {
      cidx = 1; tg = RT * f; blocked = admit_check(P, 0, f, T0, now);
      if (!blocked && f > 0) { const u64 ep = *reinterpret_cast<volatile u64*>(P.game_epoch); blocked = (ep == 0) || now < ep + (u64)f * P.pace_ns[0]; }
    }
    else { cidx = 0; tg = f + 1; }
  } else if (j == 1) {
    const int n = g / TPI, r = g % TPI;
    if (r < IA) { cidx = 3; tg = n; if (r == 0 && admit_check(P, 1, n, T0, now)) blocked = true; else if (now < T0 + (u64)n * P.pace_ns[1]) blocked = true; }
    else { cidx = 2; tg = IA * (n + 1); }
  } else {
    const int s = g / TPT, r = g % TPT;
    if (r < NP1) { cidx = 6; tg = NP3 * s; if (r == 0) blocked = admit_check(P, 2, s, T0, now); }
    else if (r < NP1 + NP2) { cidx = 4; tg = NP1 * (s + 1); }
    else { cidx = 5; tg = NP2 * (s + 1); }
  }
}
__device__ void try_claim(const Params& P, u64 T0, int home, int lane, int& sel, int& gg, bool& alldone) {
  sel = -1; gg = 0;
  const u64 now = gtimer();
  int tk[3], g[3], cv[3], cid[3], tgt[3]; bool valid[3], blk[3];
#pragma unroll
  for (int j = 0; j < 3; j++) tk[j] = (j == 0 && lane != 0) ? 0 : *reinterpret_cast<volatile int*>(P.ticket + j * 32 + lane);
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const int nsh = (j == 0) ? 1 : 32, total = *reinterpret_cast<volatile int*>(P.limit + j) * tasks_per(j);
    g[j] = tk[j] * nsh + lane; valid[j] = ((j == 0) ? (lane == 0) : (j == 1)) && g[j] < total;   // training uses the phase queue below
    cv[j] = 0; cid[j] = 0; tgt[j] = 0; blk[j] = false;
    if (valid[j]) { cand_info(P, j, g[j], T0, now, cid[j], tgt[j], blk[j]); cv[j] = ld_rlx(P.ctr + cid[j]); }
  }
  alldone = !__any_sync(~0u, valid[0] | valid[1]);
  const int rot = home & 31;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    if (sel >= 0) continue;
    unsigned m = __ballot_sync(~0u, valid[j] && !blk[j] && cv[j] >= tgt[j]);
    const int r = j == 0 ? 0 : rot;
    while (m && sel < 0) {
      const unsigned mr = r ? ((m >> r) | (m << (32 - r))) : m;
      const int k = (__ffs(mr) - 1 + r) & 31;
      int ok = 0;
      if (lane == k) ok = atomicCAS(P.ticket + j * 32 + k, tk[j], tk[j] + 1) == tk[j];
      ok = __shfl_sync(~0u, ok, k);
      if (ok) { sel = j; gg = __shfl_sync(~0u, g[j], k); } else m &= ~(1u << k);
    }
  }
  // ---- training: the ready tasks of a phase are interchangeable, so claim with one fetch-add per block (every contender succeeds in a single
  // round trip, unlike a CAS ticket). Counter slot (step&7, phase) is recycled by the last finisher of the step 4 steps earlier.
  int tsel = -1, tgg = 0; bool tpend = false;
  if (lane == 0) {
    const int c6 = ld_rlx(P.ctr + 6), c4 = ld_rlx(P.ctr + 4), c5 = ld_rlx(P.ctr + 5);
    const int st = c6 / NP3, lim2 = *reinterpret_cast<volatile int*>(P.limit + 2);
    tpend = st < lim2;
    if (tpend && sel < 0 && blockIdx.x < (gridDim.x >> 1)) {   // first resident block of each SM only: a phase has < 108 tasks, so each gets an SM to itself
      const int ph = (c5 >= NP2 * (st + 1)) ? 2 : (c4 >= NP1 * (st + 1)) ? 1 : 0;
      const int N = ph == 0 ? NP1 : ph == 1 ? NP2 : NP3;
      int* cnt = P.tq + ((st & 7) * 3 + ph);
      const int peek = ld_rlx(cnt);
      if (peek >= 0 && peek < N) {
        if (ph == 0 && peek == 0 && P.dur_ns[2] > 0 && now >= T0 + (u64)P.dur_ns[2]) {      // deadline: only a step nobody has started may be cancelled
          if (atomicCAS(cnt, 0, -1000000) == 0) atomicMin(P.limit + 2, st);
        } else {
          const int i = atomicAdd(cnt, 1);
          if (i >= 0 && i < N) { tsel = 2; tgg = st * TPT + (ph == 0 ? 0 : ph == 1 ? NP1 : NP1 + NP2) + i; }
        }
      }
    }
  }
  tsel = __shfl_sync(~0u, tsel, 0); tgg = __shfl_sync(~0u, tgg, 0); tpend = __shfl_sync(~0u, (int)tpend, 0);
  if (sel < 0 && tsel >= 0) { sel = 2; gg = tgg; }
  if (tpend) alldone = false;
}

__global__ void __launch_bounds__(NT, 2) vm(const Params P) {
  __shared__ __align__(16) float sm[SMF];
  __shared__ int sel_s, g_s;
  const int tid = threadIdx.x, lane = tid & 31;
  if (blockIdx.x == 0 && tid == 0) atomicExch(P.t0, gtimer() + 5000000ull);
  while (*reinterpret_cast<volatile u64*>(P.t0) == 0) __nanosleep(200);
  const u64 T0 = *reinterpret_cast<volatile u64*>(P.t0);
  __syncthreads();
  const int home = blockIdx.x;
  u64 busy[3] = {0, 0, 0}, lastflush = 0;
  for (;;) {
    if (tid < 32) {
      int sel, gg; bool alldone;
      try_claim(P, T0, home, lane, sel, gg, alldone);
      if (lane == 0) { sel_s = (sel < 0 && alldone) ? -2 : sel; g_s = gg; }
      if (sel < 0 && !alldone) __nanosleep(40);
      __syncwarp();
    }
    __syncthreads();
    const int sel = sel_s, g = g_s;
    __syncthreads();
    if (sel == -2) break;
    if (sel < 0) continue;
    __threadfence();
    const u64 ts = gtimer();
    if (sel == 0) {
      const int f = g / TPG, r = g % TPG;
      if (r == 0) { game_tick(P, f); signal_ret(P.ctr + 0); }
      else {
        game_copy(P, r - 1);
        const int old = signal_ret(P.ctr + 1);
        if (tid == 0 && old == RT * (f + 1) - 1) { P.frame_done[f] = gtimer(); P.live->frames = f + 1; __threadfence_system(); }
      }
    } else if (sel == 1) {
      const int n = g / TPI, r = g % TPI;
      if (r < IA) { infer_a(P, r); signal_ret(P.ctr + 2); }
      else {
        infer_b(P, n, sm);
        const int old = signal_ret(P.ctr + 3);
        if (tid == 0) { P.tok_done[n] = gtimer(); __threadfence(); P.live->chars = n + 1; __threadfence_system(); }
        (void)old;
      }
    } else {
      const int s = g / TPT, r = g % TPT;
      if (P.tprof && tid == 0 && s < 64) P.tprof[((size_t)s * TPT + r) * 2] = gtimer();
      if (r < NP1) { train_p1(P, r, s, sm); signal_ret(P.ctr + 4); }
      else if (r < NP1 + NP2) { train_p2(P, r - NP1, s, sm); signal_ret(P.ctr + 5); }
      else {
        train_p3(P, r - NP1 - NP2, s, sm);
        const int old = signal_ret(P.ctr + 6);
        if (tid == 0 && old == NP3 * (s + 1) - 1) {
          P.tq[((s + 4) & 7) * 3] = 0; P.tq[((s + 4) & 7) * 3 + 1] = 0; P.tq[((s + 4) & 7) * 3 + 2] = 0;
          P.step_done[s & 4095] = gtimer(); P.live->loss = P.lossv[s & 1]; P.live->steps = s + 1; __threadfence_system();
        }
      }
      if (P.tprof && tid == 0 && s < 64) P.tprof[((size_t)s * TPT + r) * 2 + 1] = gtimer();
    }
    if (tid == 0) {
      const u64 te = gtimer(); busy[sel] += te - ts;
      if (te - lastflush > 2000000ull) { lastflush = te; for (int j = 0; j < 3; j++) *reinterpret_cast<volatile u64*>(P.busy + blockIdx.x * 3 + j) = busy[j]; }
    }
  }
  if (tid == 0) for (int j = 0; j < 3; j++) *reinterpret_cast<volatile u64*>(P.busy + blockIdx.x * 3 + j) = busy[j];
}

// ------------------------------------------------------------------ baseline: same task code as ordinary per-op kernels
__global__ void __launch_bounds__(NT, 2) b_p1(const Params P, const int* sd) { __shared__ __align__(16) float sm[SMF]; train_p1(P, blockIdx.x, *sd, sm); }
__global__ void __launch_bounds__(NT, 2) b_p2(const Params P, const int* sd) { __shared__ __align__(16) float sm[SMF]; train_p2(P, blockIdx.x, *sd, sm); }
__global__ void __launch_bounds__(NT, 2) b_p3(const Params P, const int* sd) { __shared__ __align__(16) float sm[SMF]; train_p3(P, blockIdx.x, *sd, sm); }
__global__ void b_step_inc(int* sd) { if (threadIdx.x == 0) *sd += 1; }
__global__ void __launch_bounds__(NT, 2) b_ia(const Params P) { infer_a(P, blockIdx.x); }
__global__ void __launch_bounds__(NT, 2) b_ib(const Params P, int* nd) { __shared__ __align__(16) float sm[SMF]; infer_b(P, *nd, sm); }
__global__ void b_tok_inc(int* nd) { if (threadIdx.x == 0) *nd += 1; }

__global__ void init_f(float* p, size_t n, unsigned seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (((hash32((unsigned)i * 2654435761u ^ seed) & 0xffffff) / 16777216.f) * 2.f - 1.f) * scale;
}

// ------------------------------------------------------------------ host: corpus, shared state, http server
static const char* QA[] = {
 "Q: What is a megakernel?\nA: A megakernel is a single GPU kernel that stays resident and runs a whole program, so there are no kernel launches between the steps.\n",
 "Q: Why use one kernel?\nA: Every kernel launch has a cost, and on a fast GPU the gaps between small kernels add up. One resident kernel pays that cost once.\n",
 "Q: What runs in this kernel?\nA: Three programs at the same time: a language model is trained, text is generated from its weights, and a game is rendered.\n",
 "Q: Who schedules the work?\nA: A tiny virtual machine inside the kernel. Each block claims the next ready task, runs it, and tells the tasks that wait for it.\n",
 "Q: How do tasks wait for each other?\nA: Every task adds one to a counter in global memory when it finishes. A task that needs it waits until the counter reaches a target.\n",
 "Q: Why is the output the same every time?\nA: The math uses a fixed order of additions, so the answer does not depend on which block ran which task.\n",
 "Q: What is a thread block?\nA: A thread block is a group of threads that run on one streaming multiprocessor and share a fast scratchpad memory.\n",
 "Q: What is a streaming multiprocessor?\nA: It is one of the many small processors inside a GPU. An A100 has one hundred and eight of them.\n",
 "Q: Why is training hard to schedule?\nA: Each step is a chain of small dependent pieces, so the GPU spends a lot of time waiting between kernels.\n",
 "Q: What does the language model learn?\nA: It reads a few letters and learns to guess the next one. After enough steps the guesses start to look like words.\n",
 "Q: Can the game and the model share a GPU?\nA: Yes. The game needs a new frame every sixteen milliseconds, and the scheduler always serves it first.\n",
 "Q: What happens to the idle time?\nA: The gaps while one job waits are filled by work from the other jobs, so the machine stays busy.\n",
 "Q: Is this faster than separate kernels?\nA: It keeps the GPU busier when several jobs run together, and it launches one kernel instead of hundreds of thousands.\n",
 "Q: What is a launch?\nA: A launch is the call that asks the GPU to start a kernel. Small kernels spend more time in launches than in math.\n",
 "Q: What is a fence?\nA: A fence makes sure earlier writes are visible to other blocks before the counter changes, so a waiting task reads fresh data.\n",
 "Q: What is gradient descent?\nA: It nudges every weight a little in the direction that lowers the loss, again and again.\n",
 "Q: What is the loss?\nA: The loss measures how surprised the model is by the next letter. Lower is better.\n",
 "Q: What does the player do in the game?\nA: The player walks through a maze of walls and turns away when a wall is close.\n",
 "Q: How big is the model?\nA: It is tiny: five hundred hidden units and a vocabulary of ninety six characters, small enough to live in the cache.\n",
 "Q: Why show a comparison?\nA: To count how many kernels the same work needs when every operation is launched on its own.\n"};

static std::mutex g_mu;
static std::string g_text;                    // generated text so far
static std::vector<float> g_curve_t, g_curve_l;
static std::atomic<int> g_phase{0};           // 0 start, 1 live, 2 comparing, 3 done
static double g_kernel_t0 = 0, g_kernel_t = 0;
static double g_steps_s = 0, g_chars_s = 0, g_fps = 0, g_sm_g = 0, g_sm_i = 0, g_sm_t = 0;
static std::string g_results = "null";
static std::string g_index;
static unsigned char* g_fb_host = nullptr;
static Live* g_live = nullptr;
static unsigned char* g_ring_host = nullptr;
static u64* g_busy_host = nullptr;
static int g_nblocks = 0, g_nsm = 0;
static std::string g_note;
static char* g_doom_log = nullptr;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::string jesc(const std::string& s) { std::string o; for (unsigned char c : s) { if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c >= 32 && c < 127) o += (char)c; } return o; }

static void stream_frames(int c) {
  const char* h = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
  if (send(c, h, strlen(h), MSG_NOSIGNAL) < 0) { close(c); return; }
  unsigned last = 0xffffffffu; std::vector<unsigned char> buf(FBSZ);
  for (;;) {
    const unsigned f = g_live->frames;
    if (f == last) { usleep(1500); continue; }
    last = f; memcpy(buf.data(), g_fb_host, FBSZ);
    size_t off = 0; while (off < buf.size()) { ssize_t w = send(c, buf.data() + off, buf.size() - off, MSG_NOSIGNAL); if (w <= 0) { close(c); return; } off += w; }
  }
}
static void serve_client(int c) {
  char req[2048]; const int n = (int)recv(c, req, sizeof req - 1, 0);
  if (n <= 0) { close(c); return; }
  req[n] = 0; std::string path = "/"; { char* sp = strchr(req, ' '); if (sp) { char* e = strchr(sp + 1, ' '); if (e) path.assign(sp + 1, e); } }
  if (path == "/stream") { stream_frames(c); return; }
  std::string body, ctype = "text/html; charset=utf-8";
  if (path == "/frame") { body.assign((const char*)g_fb_host, FBSZ); ctype = "application/octet-stream"; }
  else if (path == "/state") {
    ctype = "application/json";
    std::lock_guard<std::mutex> lk(g_mu);
    std::ostringstream o;
    o << "{\"phase\":" << g_phase.load() << ",\"t\":" << g_kernel_t << ",\"steps\":" << g_live->steps << ",\"loss\":" << g_live->loss << ",\"chars\":" << g_live->chars
      << ",\"frames\":" << g_live->frames << ",\"steps_s\":" << g_steps_s << ",\"chars_s\":" << g_chars_s << ",\"fps\":" << g_fps
      << ",\"sm\":[" << g_sm_t << "," << g_sm_i << "," << g_sm_g << "],\"nsm\":" << g_nsm << ",\"batch\":" << B << ",\"note\":\"" << jesc(g_note) << "\"";
    const size_t tl = g_text.size(); o << ",\"text\":\"" << jesc(g_text.substr(tl > 900 ? tl - 900 : 0)) << "\",\"curve\":[";
    for (size_t i = 0; i < g_curve_t.size(); i++) o << (i ? "," : "") << "[" << g_curve_t[i] << "," << g_curve_l[i] << "]";
    o << "],\"results\":" << g_results << "}";
    body = o.str();
  } else if (path == "/" || path == "/index.html") { std::ifstream f("index.html"); std::stringstream ss; ss << f.rdbuf(); body = ss.str(); if (body.empty()) body = g_index; }
  else { std::string h = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"; send(c, h.data(), h.size(), 0); close(c); return; }
  std::ostringstream h; h << "HTTP/1.1 200 OK\r\nContent-Type: " << ctype << "\r\nContent-Length: " << body.size() << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
  const std::string hs = h.str(); send(c, hs.data(), hs.size(), MSG_NOSIGNAL);
  size_t off = 0; while (off < body.size()) { ssize_t w = send(c, body.data() + off, body.size() - off, MSG_NOSIGNAL); if (w <= 0) break; off += w; }
  close(c);
}
static void server_thread(int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
  if (bind(s, (sockaddr*)&a, sizeof a) < 0 || listen(s, 64) < 0) { perror("bind/listen"); exit(1); }
  for (;;) { int c = accept(s, nullptr, nullptr); if (c >= 0) std::thread(serve_client, c).detach(); }
}

// samples live counters ~10x/s: text, loss curve, rates, SM shares
static std::atomic<bool> g_sampling{false};
static void sampler_thread() {
  unsigned last_chars = 0, last_steps = 0, last_frames = 0; double lt = now_s(); std::vector<u64> lb(3, 0);
  size_t ring_pos = 0;
  while (true) {
    usleep(100000);
    if (!g_sampling) { lt = now_s(); continue; }
    const double t = now_s(), dt = t - lt; lt = t;
    const unsigned ch = g_live->chars, st = g_live->steps, fr = g_live->frames; const float loss = g_live->loss;
    u64 bs[3] = {0, 0, 0}; for (int b = 0; b < g_nblocks; b++) for (int j = 0; j < 3; j++) bs[j] += *(volatile u64*)(g_busy_host + b * 3 + j);
    std::lock_guard<std::mutex> lk(g_mu);
    g_kernel_t = t - g_kernel_t0;
    while (ring_pos < ch) { g_text.push_back((char)g_ring_host[ring_pos & (RING - 1)]); ring_pos++; if (ring_pos + RING / 2 < ch) ring_pos = ch - RING / 2; }
    g_steps_s = (st - last_steps) / dt; g_chars_s = (ch - last_chars) / dt; g_fps = (fr - last_frames) / dt;
    g_sm_g = (bs[0] - lb[0]) / 1e9 / dt / 2.0; g_sm_i = (bs[1] - lb[1]) / 1e9 / dt / 2.0; g_sm_t = (bs[2] - lb[2]) / 1e9 / dt / 2.0;
    last_chars = ch; last_steps = st; last_frames = fr; lb[0] = bs[0]; lb[1] = bs[1]; lb[2] = bs[2];
    if (st > 0 && (g_curve_t.empty() || g_kernel_t - g_curve_t.back() > 0.12)) {
      g_curve_t.push_back((float)g_kernel_t); g_curve_l.push_back(loss);
      if (g_curve_t.size() > 700) { std::vector<float> a, b; for (size_t i = 0; i < g_curve_t.size(); i += 2) { a.push_back(g_curve_t[i]); b.push_back(g_curve_l[i]); } g_curve_t = a; g_curve_l = b; }
    }
  }
}

int main(int argc, char** argv) {
  int port = 8080; double live_secs = 0, cmp_secs = 8; std::string bench_out, bench_one, child_mode = "graph"; int bench_reps = 5; double bench_secs = 6; long long pace_us = 1000; std::string wad_path = "doom1.wad"; std::string child_role; long long start_at_ns = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--secs") && i + 1 < argc) live_secs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--cmp-secs") && i + 1 < argc) cmp_secs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--pace-us") && i + 1 < argc) pace_us = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--wad") && i + 1 < argc) wad_path = argv[++i];
    else if (!strcmp(argv[i], "--child") && i + 1 < argc) child_role = argv[++i];
    else if (!strcmp(argv[i], "--child-mode") && i + 1 < argc) child_mode = argv[++i];
    else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench_out = argv[++i];
    else if (!strcmp(argv[i], "--bench-one") && i + 1 < argc) bench_one = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) bench_reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bench-secs") && i + 1 < argc) bench_secs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--start-at") && i + 1 < argc) start_at_ns = atoll(argv[++i]);
  }
  { std::ifstream f("index.html"); std::stringstream ss; ss << f.rdbuf(); g_index = ss.str(); if (g_index.empty()) g_index = "index.html not found next to the binary"; }
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 24 * 1024));      // Doom recurses (BSP walk); the limit is reserved per resident thread
  int dev = 0; CK(cudaGetDevice(&dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  CK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev));
  g_nblocks = g_nsm * 2;
  printf("GPU: %s (%d SMs), %d resident blocks. Serving http://localhost:%d (tunnel it to your Mac)\n", prop.name, g_nsm, g_nblocks, port);

  // corpus: a few fixed Q/A pairs plus many generated ones (distinct sentences, so the model cannot just memorise them)
  std::vector<unsigned char> text;
  { static const char* SUBJ[] = {"a megakernel","the scheduler","each thread block","the language model","the game","the GPU","a counter in memory","the training loop","the virtual machine","a task"};
    static const char* PRED[] = {"keeps the GPU busy","waits for the tasks it depends on","runs without a kernel launch","claims the next ready task","learns to guess the next letter","draws a new frame every sixteen milliseconds","adds one to a counter when it is done","shares the chip with the other programs","reads the weights while they change","writes its result into shared memory"};
    static const char* CONN[] = {", so ", ", and ", " because ", ", then ", ", which means "};
    static const char* TAIL[] = {"nothing sits idle","the loss goes down","the next step can start","the frame is never late","the answer stays the same","the launch cost is paid once","the other jobs keep running","the text gets better","every block stays useful"};
    auto push = [&](const std::string& str) { for (char ch : str) { if (ch == '\n') text.push_back(95); else if (ch >= 32 && ch < 127) text.push_back((unsigned char)(ch - 32)); } };
    unsigned rs = 7; auto rnd = [&](int n) { rs = rs * 1664525u + 1013904223u; return (int)((rs >> 8) % n); };
    const int NQ = (int)(sizeof QA / sizeof QA[0]);
    for (int i = 0; i < 1500; i++) {
      if (i % 5 == 0) { push(QA[rnd(NQ)]); continue; }
      const std::string sj = SUBJ[rnd(10)];
      push("Q: What does " + sj + " do?\nA: " + std::string(1, (char)toupper(sj[0])) + sj.substr(1) + " " + PRED[rnd(10)] + CONN[rnd(5)] + TAIL[rnd(9)] + ".\n");
    } }
  const int ntext = (int)text.size();
  printf("corpus: %d characters\n", ntext);

  unsigned char* dtext; float *E, *W1, *b1, *W2T, *b2, *Hb, *dz, *dzT, *lossb, *lossv, *hinf; int *ctxi, *ctr, *ticket, *limit, *sdev, *ndev, *tq; u64 *t0, *frame_done, *tok_done, *step_done; GState* gs;
  CK(cudaMalloc(&dtext, ntext)); CK(cudaMemcpy(dtext, text.data(), ntext, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&E, 128 * D * 4)); CK(cudaMalloc(&W1, (size_t)HID * DIN * 4)); CK(cudaMalloc(&b1, HID * 4)); CK(cudaMalloc(&W2T, (size_t)HID * V * 4)); CK(cudaMalloc(&b2, 2 * V * 4));
  CK(cudaMalloc(&Hb, (size_t)HID * B * 4)); CK(cudaMalloc(&dz, (size_t)B * V * 4)); CK(cudaMalloc(&dzT, (size_t)V * B * 4)); CK(cudaMalloc(&lossb, B * 4)); CK(cudaMalloc(&lossv, 8)); CK(cudaMalloc(&hinf, HID * 4));
  CK(cudaMalloc(&tq, 32 * 4)); CK(cudaMalloc(&ctxi, C * 4)); CK(cudaMalloc(&ctr, 64 * 4)); CK(cudaMalloc(&ticket, 3 * 32 * 4)); CK(cudaMalloc(&limit, 16)); CK(cudaMalloc(&sdev, 4)); CK(cudaMalloc(&ndev, 4)); CK(cudaMalloc(&t0, 8));
  u64* game_epoch; CK(cudaMalloc(&game_epoch, 8));
  CK(cudaMalloc(&frame_done, (size_t)1 << 23)); CK(cudaMalloc(&tok_done, (size_t)64 << 20)); CK(cudaMalloc(&step_done, 4096 * 8)); CK(cudaMalloc(&gs, sizeof(GState)));
  unsigned char *dfb, *dring; Live* dlive; u64* dbusy;
  CK(cudaHostAlloc((void**)&g_fb_host, FBSZ, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dfb, g_fb_host, 0)); memset(g_fb_host, 0, FBSZ);
  CK(cudaHostAlloc((void**)&g_ring_host, RING, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dring, g_ring_host, 0));
  CK(cudaHostAlloc((void**)&g_live, sizeof(Live), cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dlive, g_live, 0));
  CK(cudaHostAlloc((void**)&g_busy_host, 512 * 3 * 8, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dbusy, g_busy_host, 0));
  CK(cudaFuncSetCacheConfig(vm, cudaFuncCachePreferShared));

  // prompt context for generation: last C characters of the prompt
  const char* prompt = "Q: What is a megakernel?\nA: ";
  int ctx0[C]; { const int pl = (int)strlen(prompt); for (int k = 0; k < C; k++) { const char ch = prompt[pl - C + k]; ctx0[k] = ch == '\n' ? 95 : ch - 32; } }
  g_text = ""; for (const char* p = prompt; *p; p++) {} 

  Params P = {};
  P.t0 = t0; P.ctr = ctr; P.ticket = ticket; P.limit = limit; P.text = dtext; P.ntext = ntext; P.E = E; P.W1 = W1; P.b1 = b1; P.W2T = W2T; P.b2 = b2; P.Hb = Hb; P.dz = dz; P.dzT = dzT;
  P.lossb = lossb; P.lossv = lossv; P.hinf = hinf; P.ctxi = ctxi; P.gs = gs; P.fb = dfb; P.ring = dring; P.live = dlive; P.busy = dbusy; P.frame_done = frame_done; P.tok_done = tok_done; P.step_done = step_done; P.seed = 12345u; P.tprof = nullptr; P.tq = tq;
  P.pace_ns[0] = 28571429ll;                                    // Doom runs at 35 tics per second
  { FILE* wf = fopen(wad_path.c_str(), "rb"); if (!wf) { fprintf(stderr, "cannot open WAD %s (use --wad path)\n", wad_path.c_str()); return 1; }
    std::vector<unsigned char> wad; fseek(wf, 0, SEEK_END); long wn = ftell(wf); fseek(wf, 0, SEEK_SET); wad.resize(wn); if (fread(wad.data(), 1, wn, wf) != (size_t)wn) return 1; fclose(wf);
    unsigned char* dwad; CK(cudaMalloc(&dwad, wn)); CK(cudaMemcpy(dwad, wad.data(), wn, cudaMemcpyHostToDevice));
    char *hdbg, *ddbg; CK(cudaHostAlloc((void**)&hdbg, 1 << 20, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&ddbg, hdbg, 0)); memset(hdbg, 0, 1 << 20);
    P.wad = dwad; P.wad_len = wn; P.dbg = ddbg; P.dbg_len = 1 << 20; P.game_epoch = game_epoch; g_doom_log = hdbg; printf("WAD %s: %ld bytes\n", wad_path.c_str(), wn); }

  auto reset_state = [&](bool reset_model) {
    if (reset_model) {
      init_f<<<64, 256>>>(E, 128 * D, 11, 1.0f); init_f<<<64, 256>>>(W1, (size_t)HID * DIN, 12, 0.25f); init_f<<<64, 256>>>(W2T, (size_t)HID * V, 13, 0.1f);
      CK(cudaMemset(b1, 0, HID * 4)); CK(cudaMemset(b2, 0, 2 * V * 4));
    }
    CK(cudaMemcpy(ctxi, ctx0, sizeof ctx0, cudaMemcpyHostToDevice));
    CK(cudaMemset(game_epoch, 0, 8));
    CK(cudaMemset(ctr, 0, 64 * 4)); CK(cudaMemset(tq, 0, 32 * 4)); CK(cudaMemset(ticket, 0, 3 * 32 * 4)); CK(cudaMemset(t0, 0, 8)); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(ndev, 0, 4));
    CK(cudaMemset(hinf, 0, HID * 4)); CK(cudaMemset(lossv, 0, 8));
    memset((void*)g_live, 0, sizeof(Live)); memset((void*)g_busy_host, 0, 512 * 3 * 8); memset(g_ring_host, 0, RING);
    CK(cudaDeviceSynchronize());
  };

  std::thread(server_thread, port).detach();
  std::thread(sampler_thread).detach();
  cudaStream_t s; CK(cudaStreamCreate(&s));

  if (!child_role.empty()) {                                   // a "separate program": only training, or only inference, as ordinary per-op kernels
    reset_state(true);
    cudaStream_t cs; CK(cudaStreamCreate(&cs)); cudaGraph_t cg; cudaGraphExec_t cex; const bool inf = child_role == "infer";
    const bool use_graph = child_mode != "eager";
    auto launch_once = [&]() { if (inf) { b_ia<<<IA, NT, 0, cs>>>(P); b_ib<<<1, NT, 0, cs>>>(P, ndev); b_tok_inc<<<1, 1, 0, cs>>>(ndev); } else { b_p1<<<NP1, NT, 0, cs>>>(P, sdev); b_p2<<<NP2, NT, 0, cs>>>(P, sdev); b_p3<<<NP3, NT, 0, cs>>>(P, sdev); b_step_inc<<<1, 1, 0, cs>>>(sdev); } };
    CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
    if (inf) { b_ia<<<IA, NT, 0, cs>>>(P); b_ib<<<1, NT, 0, cs>>>(P, ndev); b_tok_inc<<<1, 1, 0, cs>>>(ndev); }
    else { b_p1<<<NP1, NT, 0, cs>>>(P, sdev); b_p2<<<NP2, NT, 0, cs>>>(P, sdev); b_p3<<<NP3, NT, 0, cs>>>(P, sdev); b_step_inc<<<1, 1, 0, cs>>>(sdev); }
    CK(cudaStreamEndCapture(cs, &cg)); CK(cudaGraphInstantiate(&cex, cg, 0));
    for (int i = 0; i < 20; i++) { if (use_graph) CK(cudaGraphLaunch(cex, cs)); else launch_once(); } CK(cudaStreamSynchronize(cs)); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(ndev, 0, 4));
    auto real_ns = []() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec; };
    while (real_ns() < start_at_ns) usleep(200);                // both programs start measuring at the same wall-clock instant
    cudaEvent_t ev[2]; for (int k = 0; k < 2; k++) { CK(cudaEventCreate(&ev[k])); CK(cudaEventRecord(ev[k], cs)); }
    const double tb = now_s(); int it = 0;
    while (now_s() - tb < cmp_secs) { for (int k = 0; k < 8; k++) { if (use_graph) CK(cudaGraphLaunch(cex, cs)); else launch_once(); } CK(cudaEventSynchronize(ev[it & 1])); CK(cudaEventRecord(ev[it & 1], cs)); it++; }
    CK(cudaStreamSynchronize(cs)); const double el = now_s() - tb; int cnt = 0; CK(cudaMemcpy(&cnt, inf ? ndev : sdev, 4, cudaMemcpyDeviceToHost));
    printf("RESULT %s %d %.4f\n", child_role.c_str(), cnt, el); fflush(stdout);
    return 0;
  }
  if (getenv("DEMO_TBENCH")) {
    reset_state(true); cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); float t;
    auto tm = [&](const char* nm, auto fn, int tasks) { for (int i = 0; i < 20; i++) fn(); CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(a, s)); for (int i = 0; i < 500; i++) fn(); CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&t, a, b)); printf("  %-10s %6.1f us each (%d tasks)\n", nm, t * 1e3 / 500, tasks); };
    tm("train P1", [&]() { b_p1<<<NP1, NT, 0, s>>>(P, sdev); }, NP1); tm("train P2", [&]() { b_p2<<<NP2, NT, 0, s>>>(P, sdev); }, NP2); tm("train P3", [&]() { b_p3<<<NP3, NT, 0, s>>>(P, sdev); }, NP3);
#ifdef SECT
    { b_p3<<<NP3, NT, 0, s>>>(P, sdev); CK(cudaStreamSynchronize(s)); unsigned long long sec[16]; CK(cudaMemcpyFromSymbol(sec, g_sec, sizeof sec)); printf("  P3 sections (cycles, block thread0): load %llu | dh %llu | dW2 %llu | sync %llu | db1+dW1 %llu\n", sec[1]-sec[0], sec[2]-sec[1], sec[3]-sec[2], sec[4]-sec[3], sec[5]-sec[4]); b_p2<<<NP2, NT, 0, s>>>(P, sdev); CK(cudaStreamSynchronize(s)); CK(cudaMemcpyFromSymbol(sec, g_sec, sizeof sec)); printf("  P2 sections: first-tile load %llu | tile loop %llu | softmax tail %llu\n", sec[6]-sec[9], sec[7]-sec[6], sec[8]-sec[7]); }
#endif
    tm("infer A", [&]() { b_ia<<<IA, NT, 0, s>>>(P); }, IA); tm("infer B", [&]() { b_ib<<<1, NT, 0, s>>>(P, ndev); }, 1);
    return 0;
  }
  if (getenv("DEMO_SOLO")) {                                   // training alone: one megakernel launch vs per-op graph, no other jobs
    reset_state(true); { int lim[3] = {0, 0, 1 << 20}; CK(cudaMemcpy(limit, lim, 12, cudaMemcpyHostToDevice)); }
    P.dur_ns[0] = P.dur_ns[1] = 0; P.dur_ns[2] = (long long)(cmp_secs * 1e9); P.pace_ns[1] = 0;
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); float t;
    CK(cudaEventRecord(a, s)); vm<<<g_nblocks, NT, 0, s>>>(P); CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError()); CK(cudaEventElapsedTime(&t, a, b));
    printf("SOLO training, megakernel: %.0f steps/s\n", g_live->steps / (t / 1e3));
    { u64* tp; CK(cudaMalloc(&tp, (size_t)64 * TPT * 16)); CK(cudaMemset(tp, 0, (size_t)64 * TPT * 16)); P.tprof = tp; reset_state(true);
      int lim[3] = {0, 0, 1 << 20}; CK(cudaMemcpy(limit, lim, 12, cudaMemcpyHostToDevice)); P.dur_ns[2] = (long long)(0.5e9);
      vm<<<g_nblocks, NT, 0, s>>>(P); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
      std::vector<u64> h((size_t)64 * TPT * 2); CK(cudaMemcpy(h.data(), tp, h.size() * 8, cudaMemcpyDeviceToHost));
      double sp[3] = {0, 0, 0}, mt[3] = {0, 0, 0}, gp[3] = {0, 0, 0}, n = 0; const int lo[3] = {0, NP1, NP1 + NP2}, hi[3] = {NP1, NP1 + NP2, TPT};
      for (int st = 20; st < 60; st++) {
        double mn[3], mx[3];
        for (int p = 0; p < 3; p++) { mn[p] = 1e30; mx[p] = 0; for (int r = lo[p]; r < hi[p]; r++) { const double a = (double)h[((size_t)st * TPT + r) * 2], e = (double)h[((size_t)st * TPT + r) * 2 + 1]; mn[p] = std::min(mn[p], a); mx[p] = std::max(mx[p], e); mt[p] += (e - a) / (hi[p] - lo[p]); } sp[p] += mx[p] - mn[p]; }
        double nx = 1e30; for (int r = 0; r < NP1; r++) nx = std::min(nx, (double)h[((size_t)(st + 1) * TPT + r) * 2]);
        gp[0] += mn[1] - mx[0]; gp[1] += mn[2] - mx[1]; gp[2] += nx - mx[2]; n++;
      }
      printf("  timeline us: P1 span %.1f (mean task %.1f) gap %.1f | P2 span %.1f (mean task %.1f) gap %.1f | P3 span %.1f (mean task %.1f) gap %.1f\n", sp[0] / n / 1e3, mt[0] / n / 1e3, gp[0] / n / 1e3, sp[1] / n / 1e3, mt[1] / n / 1e3, gp[1] / n / 1e3, sp[2] / n / 1e3, mt[2] / n / 1e3, gp[2] / n / 1e3);
      for (int ph = 0; ph < 3; ph++) { const int st = 30; std::vector<double> o; double mn = 1e30; for (int r = lo[ph]; r < hi[ph]; r++) { const double a = (double)h[((size_t)st * TPT + r) * 2]; o.push_back(a); mn = std::min(mn, a); }
        std::sort(o.begin(), o.end()); printf("  step %d phase %d start offsets (us):", st, ph + 1); for (size_t i = 0; i < o.size(); i += std::max<size_t>(1, o.size() / 16)) printf(" %.1f", (o[i] - mn) / 1e3); printf("\n"); }
      P.tprof = nullptr; }
    reset_state(true); cudaGraph_t gt; cudaGraphExec_t et;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); b_p1<<<NP1, NT, 0, s>>>(P, sdev); b_p2<<<NP2, NT, 0, s>>>(P, sdev); b_p3<<<NP3, NT, 0, s>>>(P, sdev); b_step_inc<<<1, 1, 0, s>>>(sdev); CK(cudaStreamEndCapture(s, &gt)); CK(cudaGraphInstantiate(&et, gt, 0));
    for (int i = 0; i < 50; i++) CK(cudaGraphLaunch(et, s)); CK(cudaStreamSynchronize(s)); CK(cudaMemset(sdev, 0, 4));
    const double tb = now_s(); int n = 0; while (now_s() - tb < cmp_secs) { for (int k = 0; k < 16; k++) CK(cudaGraphLaunch(et, s)); n += 16; CK(cudaStreamSynchronize(s)); }
    printf("SOLO training, per-op graph: %.0f steps/s\n", n / (now_s() - tb));
    return 0;
  }
  // ---------------------------------------------------------------- benchmark mode: repeated runs, mean/sd, written as JSON
  if (!bench_out.empty() || !bench_one.empty()) {
    std::vector<std::pair<std::string, std::vector<double>>> M;
    auto add = [&](const std::string& k, double v) { if (!bench_one.empty()) { printf("METRIC %s %.6f\n", k.c_str(), v); fflush(stdout); return; } for (auto& m : M) if (m.first == k) { m.second.push_back(v); return; } M.push_back({k, {v}}); };
    cudaEvent_t b0, b1; CK(cudaEventCreate(&b0)); CK(cudaEventCreate(&b1));
    // one megakernel launch with the chosen jobs; Doom is paced at 35 Hz when enabled unless unpaced_game
    auto mega = [&](const std::string& tag, bool game, bool infer, bool train, bool paced_inf, bool unpaced_game, double secs) {
      reset_state(true);
      int lim[3] = {game ? 12 << 20 : 0, infer ? 12 << 20 : 0, train ? 12 << 20 : 0}; CK(cudaMemcpy(limit, lim, 12, cudaMemcpyHostToDevice));
      P.dur_ns[0] = P.dur_ns[1] = P.dur_ns[2] = (long long)(secs * 1e9); P.pace_ns[0] = unpaced_game ? 0 : 28571429ll; P.pace_ns[1] = paced_inf ? 1000000ll : 0;
      CK(cudaEventRecord(b0, s)); vm<<<g_nblocks, NT, 0, s>>>(P); CK(cudaEventRecord(b1, s)); CK(cudaEventSynchronize(b1)); CK(cudaGetLastError());
      float ms; CK(cudaEventElapsedTime(&ms, b0, b1)); const double el = ms / 1e3;
      const double steps = g_live->steps, chars = g_live->chars; const unsigned frames = g_live->frames;
      double bsum[3] = {0, 0, 0}; for (int b = 0; b < g_nblocks; b++) for (int j = 0; j < 3; j++) bsum[j] += (double)*(volatile u64*)(g_busy_host + b * 3 + j);
      if (train) { add(tag + ".train_steps_s", steps / el); add(tag + ".train_tok_s", steps * B / el); add(tag + ".sm_train", bsum[2] / 1e9 / el / 2.0); }
      if (infer) { add(tag + ".infer_tok_s", chars / el); add(tag + ".sm_infer", bsum[1] / 1e9 / el / 2.0); }
      if (game && frames > 2) { std::vector<u64> d(frames); CK(cudaMemcpy(d.data(), frame_done, frames * 8, cudaMemcpyDeviceToHost)); add(tag + ".doom_fps", (frames - 1) / ((double)(d[frames - 1] - d[0]) * 1e-9)); add(tag + ".sm_doom", bsum[0] / 1e9 / el / 2.0); }
      printf("  %-22s steps/s %8.0f  tok/s %8.0f  doom fps %5.1f\n", tag.c_str(), steps / el, chars / el, (game && frames > 2) ? 0.0 : 0.0); fflush(stdout);
    };
    // one per-op program alone on the GPU: eager launches (like a PyTorch/vLLM loop) or one CUDA graph per iteration
    auto base_single = [&](const std::string& tag, bool infer, bool graph, double secs) {
      reset_state(true); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(ndev, 0, 4));
      cudaStream_t bs; CK(cudaStreamCreate(&bs)); cudaGraph_t g; cudaGraphExec_t ex;
      auto once = [&]() { if (infer) { b_ia<<<IA, NT, 0, bs>>>(P); b_ib<<<1, NT, 0, bs>>>(P, ndev); b_tok_inc<<<1, 1, 0, bs>>>(ndev); } else { b_p1<<<NP1, NT, 0, bs>>>(P, sdev); b_p2<<<NP2, NT, 0, bs>>>(P, sdev); b_p3<<<NP3, NT, 0, bs>>>(P, sdev); b_step_inc<<<1, 1, 0, bs>>>(sdev); } };
      if (graph) { CK(cudaStreamBeginCapture(bs, cudaStreamCaptureModeGlobal)); once(); CK(cudaStreamEndCapture(bs, &g)); CK(cudaGraphInstantiate(&ex, g, 0)); }
      for (int i = 0; i < 20; i++) { if (graph) CK(cudaGraphLaunch(ex, bs)); else once(); } CK(cudaStreamSynchronize(bs)); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(ndev, 0, 4));
      cudaEvent_t ev[2]; for (int k = 0; k < 2; k++) { CK(cudaEventCreate(&ev[k])); CK(cudaEventRecord(ev[k], bs)); }
      const double tb = now_s(); int it = 0;
      while (now_s() - tb < secs) { for (int k = 0; k < 16; k++) { if (graph) CK(cudaGraphLaunch(ex, bs)); else once(); } CK(cudaEventSynchronize(ev[it & 1])); CK(cudaEventRecord(ev[it & 1], bs)); it++; }
      CK(cudaStreamSynchronize(bs)); const double el = now_s() - tb; int cnt = 0; CK(cudaMemcpy(&cnt, infer ? ndev : sdev, 4, cudaMemcpyDeviceToHost));
      add(tag + (infer ? ".infer_tok_s" : ".train_steps_s"), cnt / el);
      printf("  %-22s %8.0f /s\n", tag.c_str(), cnt / el); fflush(stdout); CK(cudaStreamDestroy(bs));
    };
    // training and inference at the same time in one process on two CUDA-graph streams
    auto base_streams = [&](const std::string& tag, double secs) {
      reset_state(true); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(ndev, 0, 4));
      cudaStream_t si, st_; int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); CK(cudaStreamCreateWithPriority(&si, cudaStreamNonBlocking, hi)); CK(cudaStreamCreateWithPriority(&st_, cudaStreamNonBlocking, lo));
      cudaGraph_t gi, gt; cudaGraphExec_t ei, et;
      CK(cudaStreamBeginCapture(si, cudaStreamCaptureModeGlobal)); b_ia<<<IA, NT, 0, si>>>(P); b_ib<<<1, NT, 0, si>>>(P, ndev); b_tok_inc<<<1, 1, 0, si>>>(ndev); CK(cudaStreamEndCapture(si, &gi)); CK(cudaGraphInstantiate(&ei, gi, 0));
      CK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeGlobal)); b_p1<<<NP1, NT, 0, st_>>>(P, sdev); b_p2<<<NP2, NT, 0, st_>>>(P, sdev); b_p3<<<NP3, NT, 0, st_>>>(P, sdev); b_step_inc<<<1, 1, 0, st_>>>(sdev); CK(cudaStreamEndCapture(st_, &gt)); CK(cudaGraphInstantiate(&et, gt, 0));
      std::atomic<bool> stop_b{false}; const double tb = now_s();
      auto worker = [&](cudaStream_t st, cudaGraphExec_t ex) { cudaEvent_t ev[2]; for (int k = 0; k < 2; k++) { CK(cudaEventCreate(&ev[k])); CK(cudaEventRecord(ev[k], st)); } int it = 0;
        while (!stop_b) { for (int k = 0; k < 8; k++) CK(cudaGraphLaunch(ex, st)); CK(cudaEventSynchronize(ev[it & 1])); CK(cudaEventRecord(ev[it & 1], st)); it++; } CK(cudaStreamSynchronize(st)); };
      std::thread wi(worker, si, ei), wt(worker, st_, et); while (now_s() - tb < secs) usleep(20000); stop_b = true; wi.join(); wt.join();
      const double el = now_s() - tb; int sd = 0, nd = 0; CK(cudaMemcpy(&sd, sdev, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nd, ndev, 4, cudaMemcpyDeviceToHost));
      add(tag + ".train_steps_s", sd / el); add(tag + ".infer_tok_s", nd / el); printf("  %-22s steps/s %8.0f  tok/s %8.0f\n", tag.c_str(), sd / el, nd / el); fflush(stdout);
    };
    // the same two programs as two separate OS processes time-sharing the GPU
    auto base_procs = [&](const std::string& tag, double secs) {
      char exe[1024]; { ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1); exe[n > 0 ? n : 0] = 0; }
      struct timespec rts; clock_gettime(CLOCK_REALTIME, &rts); const long long start = (long long)rts.tv_sec * 1000000000ll + rts.tv_nsec + 14000000000ll;
      FILE* pf[2]; const char* roles[2] = {"train", "infer"};
      for (int k = 0; k < 2; k++) { char cmd[2048]; snprintf(cmd, sizeof cmd, "%s --child %s --child-mode graph --cmp-secs %g --start-at %lld --wad %s --port 0 2>&1", exe, roles[k], secs, start, wad_path.c_str()); pf[k] = popen(cmd, "r"); }
      double rate[2] = {0, 0};
      for (int k = 0; k < 2; k++) { char line[512]; while (fgets(line, sizeof line, pf[k])) { char role[16]; long long cnt; double el; if (sscanf(line, "RESULT %15s %lld %lf", role, &cnt, &el) == 3) rate[k] = cnt / el; } pclose(pf[k]); }
      add(tag + ".train_steps_s", rate[0]); add(tag + ".infer_tok_s", rate[1]); printf("  %-22s steps/s %8.0f  tok/s %8.0f\n", tag.c_str(), rate[0], rate[1]); fflush(stdout);
    };
    struct Cfg { const char* tag; bool game, infer, train, pinf, ug; double f; };
    const Cfg cfgs[] = {{"mega_all", true, true, true, false, false, 1.0}, {"mega_all_live", true, true, true, true, false, 1.0}, {"mega_train", false, false, true, false, false, 1.0},
                        {"mega_infer", false, true, false, false, false, 1.0}, {"mega_doom", true, false, false, false, true, 0.6}};
    if (!bench_one.empty()) { for (const Cfg& c : cfgs) if (bench_one == c.tag) mega(c.tag, c.game, c.infer, c.train, c.pinf, c.ug, bench_secs * c.f); return 0; }
    // Doom cannot be re-created inside one process (the engine keeps static state), so every Doom measurement runs in a fresh process
    auto mega_isolated = [&](const Cfg& c) {
      char exe[1024]; { ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1); exe[n > 0 ? n : 0] = 0; }
      char cmd[2048]; snprintf(cmd, sizeof cmd, "%s --bench-one %s --bench-secs %g --wad %s --port 0 2>&1", exe, c.tag, bench_secs, wad_path.c_str());
      FILE* pf = popen(cmd, "r"); char line[512]; while (fgets(line, sizeof line, pf)) { char key[128]; double v; if (sscanf(line, "METRIC %127s %lf", key, &v) == 2) add(key, v); } pclose(pf);
      printf("  %-22s (own process) done\n", c.tag); fflush(stdout);
    };
    printf("benchmark: %d reps, %.0f s per measurement\n", bench_reps, bench_secs);
    for (int rep = 0; rep < bench_reps; rep++) {
      printf("rep %d/%d\n", rep + 1, bench_reps);
      for (const Cfg& c : cfgs) { if (c.game) mega_isolated(c); else mega(c.tag, c.game, c.infer, c.train, c.pinf, c.ug, bench_secs * c.f); }
      base_single("eager_train", false, false, bench_secs); base_single("eager_infer", true, false, bench_secs);
      base_single("graph_train", false, true, bench_secs);  base_single("graph_infer", true, true, bench_secs);
      base_streams("streams_both", bench_secs);
      base_procs("procs_both", bench_secs);
    }
    // write JSON
    FILE* jf = fopen(bench_out.c_str(), "w"); if (!jf) { fprintf(stderr, "cannot write %s\n", bench_out.c_str()); return 1; }
    time_t tt = time(nullptr); char ds[64]; strftime(ds, sizeof ds, "%Y-%m-%dT%H:%M:%SZ", gmtime(&tt));
    fprintf(jf, "{\n  \"meta\": {\"gpu\": \"%s\", \"sms\": %d, \"reps\": %d, \"secs_per_measurement\": %.1f, \"date\": \"%s\", \"batch\": %d, \"model\": \"char-level MLP LM, %d hidden units, %d-way softmax, %d-char context\", \"launches_per_step_perop\": 4, \"launches_per_token_perop\": 3},\n  \"metrics\": {\n", prop.name, g_nsm, bench_reps, bench_secs, ds, B, HID, V, C);
    for (size_t i = 0; i < M.size(); i++) {
      const auto& v = M[i].second; double mean = 0, mn = 1e300, mx = -1e300; for (double x : v) { mean += x; mn = std::min(mn, x); mx = std::max(mx, x); } mean /= v.size();
      double var = 0; for (double x : v) var += (x - mean) * (x - mean); const double sd = v.size() > 1 ? sqrt(var / (v.size() - 1)) : 0;
      fprintf(jf, "    \"%s\": {\"mean\": %.4f, \"sd\": %.4f, \"min\": %.4f, \"max\": %.4f, \"n\": %zu, \"values\": [", M[i].first.c_str(), mean, sd, mn, mx, v.size());
      for (size_t k = 0; k < v.size(); k++) fprintf(jf, "%s%.4f", k ? ", " : "", v[k]); fprintf(jf, "]}%s\n", i + 1 < M.size() ? "," : "");
    }
    fprintf(jf, "  }\n}\n"); fclose(jf); printf("wrote %s\n", bench_out.c_str());
    return 0;
  }
  // ---------------------------------------------------------------- the demo: the three jobs, running together in one kernel launch
  g_note = "starting"; reset_state(true);
  { const int cap_inf = 1 << 20 / TPI;
    int lim[3] = {12 << 20, 12 << 20, 12 << 20}; CK(cudaMemcpy(limit, lim, 12, cudaMemcpyHostToDevice));
    P.dur_ns[0] = P.dur_ns[1] = P.dur_ns[2] = (long long)(live_secs * 1e9); P.pace_ns[1] = pace_us * 1000ll; (void)cap_inf; }
  g_kernel_t0 = now_s(); g_phase = 1; g_note = "one kernel launch: training + inference + game"; g_sampling = true;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, s)); vm<<<g_nblocks, NT, 0, s>>>(P); CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  g_sampling = false; usleep(250000);
  int lim1[3]; CK(cudaMemcpy(lim1, limit, 12, cudaMemcpyDeviceToHost));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  const float live_ms = ms;
  const int steps_live = (int)g_live->steps, chars_live = (int)g_live->chars, frames_live = (int)g_live->frames; const float loss_live = g_live->loss;
  printf("live run: %.1fs  train steps %d (loss %.3f)  generated chars %d  frames %d  (1 kernel launch)\n", ms / 1e3, steps_live, loss_live, chars_live, frames_live);
  { std::lock_guard<std::mutex> lk(g_mu); g_kernel_t = ms / 1e3; }

  g_phase = 3; g_note = "done";
  printf("done (the demo ends when its step limit or --secs is reached). Ctrl-C to quit.\n");
  for (;;) sleep(60);
}
