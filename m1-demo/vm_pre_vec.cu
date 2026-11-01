// M1 demo: ONE persistent kernel on an A100 running three jobs at once.
//   game      : Pong physics + rasteriser, paced at 60 FPS (frame deadline), frames land in pinned host memory
//   inference : 8-layer Llama-1B-shaped MLP decode chain, paced at a token rate (token deadline)
//   training  : 2-layer MLP trained with SGD, forward+backward+update entirely inside the kernel (throughput filler)
//
// Scheduler: every block loops { try game, then inference, then training; claim the head task of a job
// only if its dependencies are already satisfied (CAS on a per-job ticket); else sleep }.
// Preemption happens at task boundaries. Dependencies are monotonic global counters (never reset),
// release-fenced by the producer, acquire-read by the claimer.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 vm.cu -o vm      Run: ./vm [--secs 5] [--live] [--tok-us 3000]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
typedef unsigned long long u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

// ---------------- inference dims (same as m0-a100/mk.cu v2)
constexpr int H = 2048, I = 8192, GR = 8, NG = I / GR, KC = 4, KCH = I / KC, NRB = H / 8, NDT = NRB * KC, NT = 256, CS = KC + 1;
// ---------------- training dims
#ifndef TUNITS
#define TUNITS 16
#endif
constexpr int TB = 256, TD = 64, TH = 1024, TU = TUNITS, T1 = TH / TU, T3 = TH / TU, TPT = T1 + T3;
constexpr float LR = 0.03f;
constexpr int NS = 128;
constexpr int NST = 32;                          // training ticket shards (all visible in one window)
constexpr int NLI_C = 8;                         // compile-time inference layers; host --layers must match                          // ticket shards per job
constexpr int NBATCH = 16;                      // dataset = NBATCH*TB samples
// ---------------- game dims
constexpr int GW = 128, GH = 64, GRB = GH / 8, TPG = 1 + GRB;

struct Layer { const bf16 *wg, *wu, *wd, *gamma; };
struct GameState { float bx, by, vx, vy, pl, pr; int sl, sr, serves; };
struct Stats { volatile unsigned frames, tokens, train_steps; volatile float loss; volatile int sl, sr; };

struct Params {
  u64* t0; int* ctr; int* ticket;
  // game
  int F; long long frame_ns; GameState* gs; unsigned char* fb; u64* frame_done;
  // inference
  int T, NLI; long long token_ns; const Layer* Ls; float* xbuf; float* part; bf16* act; u64* tok_done;
  // training
  int K; long long train_dur_ns; int* train_limit;
  int* stop;
  const float *X, *XT, *Tg; float *W1, *b1, *W2, *b2s, *Hb, *Eb, *Lb, *loss_hist; u64* train_done;
  Stats* stats;
  u64* tprof;                                        // optional: [64 steps][TPT][2] start/end timestamps
};

// counters: 0 game-phys, 1 game-render, 2 T1, 4 T3, 8.. inference
__device__ __forceinline__ int cG(int l, int c) { return 8 + CS * l + c; }
__device__ __forceinline__ int cD(int l) { return 8 + CS * l + KC; }

// ---------------- primitives
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ int ld_rlx(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
// producer side: every thread fences its writes, barrier, one thread bumps the counter. Returns old value (thread 0 only).
__device__ __forceinline__ int signal_ret(int* c) {
  __threadfence(); __syncthreads();
  int old = 0; if (threadIdx.x == 0) old = atomicAdd(c, 1);
  return old;
}
__device__ __forceinline__ void fma8(const uint4 w, const float* xs, float& acc) {
  const float4 x0 = *reinterpret_cast<const float4*>(xs), x1 = *reinterpret_cast<const float4*>(xs + 4);
  acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc); acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
  acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc); acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
  acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc); acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
  acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc); acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
}

// ---------------- inference tasks (identical math to m0 v2)
__device__ void gateup_task(const Layer& w, int t, int l, float* xbuf, const float* part, bf16* act, float* xs, float* red, bool reuse = false) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, idx = t * GR + warp;
  const uint4* pg = reinterpret_cast<const uint4*>(w.wg + (size_t)idx * H);
  const uint4* pu = reinterpret_cast<const uint4*>(w.wu + (size_t)idx * H);
  uint4 g[8], u[8];
#pragma unroll
  for (int j = 0; j < 8; j++) { g[j] = ld_stream(pg + lane + 32 * j); u[j] = ld_stream(pu + lane + 32 * j); }
  if (!reuse) {
  float ss = 0.f;
  for (int i = tid; i < H; i += NT) {
    float v;
    if (l == 0) v = __ldcg(xbuf + i);
    else {
      v = __ldcg(xbuf + (size_t)(l - 1) * H + i);
#pragma unroll
      for (int c = 0; c < KC; c++) v += __ldcg(part + ((size_t)(l - 1) * KC + c) * H + i);
      if (t == 0) xbuf[(size_t)l * H + i] = v;
    }
    xs[i] = v; ss += v * v;
  }
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
  }
  float ag = 0.f, au = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) { fma8(g[j], xs + 8 * (lane + 32 * j), ag); fma8(u[j], xs + 8 * (lane + 32 * j), au); }
#pragma unroll
  for (int o = 16; o; o >>= 1) { ag += __shfl_xor_sync(~0u, ag, o); au += __shfl_xor_sync(~0u, au, o); }
  if (lane == 0) act[idx] = __float2bfloat16(ag / (1.f + expf(-ag)) * au);
}
__device__ void down_task(const Layer& w, int idx, int l, float* part, const bf16* act, float* xs, bool reuse = false) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = idx / NRB, rb = idx % NRB, row = rb * 8 + warp;
  const uint4* pw = reinterpret_cast<const uint4*>(w.wd + (size_t)row * I + (size_t)c * KCH);
  uint4 d[8];
#pragma unroll
  for (int j = 0; j < 8; j++) d[j] = ld_stream(pw + lane + 32 * j);
  if (!reuse) {
  {
    const uint4 a = __ldcg(reinterpret_cast<const uint4*>(act + (size_t)c * KCH) + tid);
    float* o = xs + 8 * tid;
    o[0] = __uint_as_float(a.x << 16); o[1] = __uint_as_float(a.x & 0xffff0000u);
    o[2] = __uint_as_float(a.y << 16); o[3] = __uint_as_float(a.y & 0xffff0000u);
    o[4] = __uint_as_float(a.z << 16); o[5] = __uint_as_float(a.z & 0xffff0000u);
    o[6] = __uint_as_float(a.w << 16); o[7] = __uint_as_float(a.w & 0xffff0000u);
  }
  __syncthreads();
  }
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) fma8(d[j], xs + 8 * (lane + 32 * j), acc);
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
  if (lane == 0) part[((size_t)l * KC + c) * H + row] = acc;
}

// ---------------- training tasks (fp32 SIMT, fixed reduction orders => deterministic)
__device__ void train_p1(const Params& P, int jb, int s, float* sm) {
  const int tid = threadIdx.x;                 // thread = batch element
  float* wt = sm;                              // [TU][TD]
  for (int p = tid; p < TU * TD; p += NT) wt[p] = __ldcg(P.W1 + (size_t)jb * TU * TD + p);
  __syncthreads();
  const float* xc = P.XT + (size_t)(s % NBATCH) * TD * TB + tid;     // feature-major: coalesced across the batch
  float xv[TD];
#pragma unroll
  for (int i = 0; i < TD; i++) xv[i] = __ldg(xc + (size_t)i * TB);
  float yp = 0.f;
  for (int u = 0; u < TU; u++) {
    const int j = jb * TU + u;
    float acc = __ldcg(P.b1 + j);
#pragma unroll
    for (int i = 0; i < TD; i++) acc = fmaf(wt[u * TD + i], xv[i], acc);
    const float hv = tanhf(acc);
    P.Hb[(size_t)j * TB + tid] = hv;
    yp = fmaf(__ldcg(P.W2 + j), hv, yp);
  }
  P.Eb[(size_t)jb * TB + tid] = yp;                    // partial output of this unit block, fixed order over u
}
__device__ void train_p3(const Params& P, int jb, int s, float* sm) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float* dh = sm;                              // [TU][TB]
  float* pr = sm + TU * TB;                    // [TU][TB]
  float y = __ldcg(P.b2s + (s & 1));
#pragma unroll 8
  for (int q = 0; q < T1; q++) y += __ldcg(P.Eb + (size_t)q * TB + tid);       // fixed order over unit blocks
  const float e = y - P.Tg[(s % NBATCH) * TB + tid];
  for (int u = 0; u < TU; u++) {
    const int j = jb * TU + u;
    const float hv = __ldcg(P.Hb + (size_t)j * TB + tid), w2 = __ldcg(P.W2 + j);
    dh[u * TB + tid] = e * w2 * (1.f - hv * hv);
    pr[u * TB + tid] = e * hv;
  }
  __syncthreads();
  if (jb == 0) {                                       // step loss + b2 update (double-buffered so other P3 tasks never see a half update)
    float ge = e, gl = 0.5f * e * e;
#pragma unroll
    for (int o = 16; o; o >>= 1) { ge += __shfl_xor_sync(~0u, ge, o); gl += __shfl_xor_sync(~0u, gl, o); }
    if (lane == 0) { sm[2 * TU * TB + warp] = ge; sm[2 * TU * TB + 8 + warp] = gl; }
    __syncthreads();
    if (tid == 0) {
      float gb = 0.f, l = 0.f;
      for (int w2 = 0; w2 < NT / 32; w2++) { gb += sm[2 * TU * TB + w2]; l += sm[2 * TU * TB + 8 + w2]; }
      P.loss_hist[s] = l / TB;
      P.b2s[(s + 1) & 1] = P.b2s[s & 1] - LR * gb / TB;
    }
  }
  for (int u = warp; u < TU; u += NT / 32) {
    const int j = jb * TU + u;
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int k = 0; k < TB / 32; k++) { s1 += pr[u * TB + lane + 32 * k]; s2 += dh[u * TB + lane + 32 * k]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    if (lane == 0) { P.W2[j] -= LR * s1 / TB; P.b1[j] -= LR * s2 / TB; }
  }
  {                                                    // gW1: each X load feeds TU/4 units; per-(u,i) sum order over b unchanged
    constexpr int UPG = TU / 4;
    const int i = tid & (TD - 1), ug = tid / TD;
    float gacc[UPG];
#pragma unroll
    for (int u = 0; u < UPG; u++) gacc[u] = 0.f;
    const float* xb = P.X + (size_t)((s % NBATCH) * TB) * TD + i;
#pragma unroll 4
    for (int b = 0; b < TB; b++) {
      const float xv = __ldg(xb + (size_t)b * TD);
#pragma unroll
      for (int u = 0; u < UPG; u++) gacc[u] = fmaf(dh[(ug * UPG + u) * TB + b], xv, gacc[u]);
    }
#pragma unroll
    for (int u = 0; u < UPG; u++) P.W1[(size_t)(jb * TU + ug * UPG + u) * TD + i] -= LR * gacc[u] / TB;
  }
}

// ---------------- game tasks
__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
__device__ void game_reset_ball(GameState* g) {
  const unsigned h = hash32(g->serves++ * 2654435761u + 17u);
  g->bx = GW / 2; g->by = GH / 2 + (float)(int)(h % 21) - 10.f;
  g->vx = (h & 0x100) ? 1.3f : -1.3f; g->vy = ((h >> 9) % 2 ? 1.f : -1.f) * (0.4f + (float)((h >> 10) % 8) * 0.1f);
}
__device__ void game_physics(GameState* g) {
  g->bx += g->vx; g->by += g->vy;
  if (g->by < 1.f) { g->by = 1.f; g->vy = -g->vy; }
  if (g->by > GH - 2.f) { g->by = GH - 2.f; g->vy = -g->vy; }
  g->pl += fminf(fmaxf(g->by - g->pl, -0.95f), 0.95f);
  g->pr += fminf(fmaxf(g->by - g->pr, -0.70f), 0.70f);
  g->pl = fminf(fmaxf(g->pl, 6.f), GH - 7.f); g->pr = fminf(fmaxf(g->pr, 6.f), GH - 7.f);
  if (g->vx < 0.f && g->bx <= 5.f) {
    if (fabsf(g->by - g->pl) <= 7.f) { g->bx = 5.f; g->vx = fminf(-g->vx * 1.04f, 3.f); g->vy += (g->by - g->pl) * 0.07f; }
    else if (g->bx < 0.f) { g->sr++; game_reset_ball(g); }
  } else if (g->vx > 0.f && g->bx >= GW - 6.f) {
    if (fabsf(g->by - g->pr) <= 7.f) { g->bx = GW - 6.f; g->vx = fmaxf(-g->vx * 1.04f, -3.f); g->vy += (g->by - g->pr) * 0.07f; }
    else if (g->bx > GW - 1.f) { g->sl++; game_reset_ball(g); }
  }
}
__device__ void game_render(const Params& P, int band) {
  struct { float bx, by, pl, pr; } g = { __ldcg(&P.gs->bx), __ldcg(&P.gs->by), __ldcg(&P.gs->pl), __ldcg(&P.gs->pr) };
  for (int p = threadIdx.x; p < 8 * GW; p += NT) {
    const int y = band * 8 + p / GW, x = p % GW;
    unsigned char v = 0;
    if (x == GW / 2 && (y & 3) < 2) v = 1;
    if ((x == 3 || x == 4) && fabsf(y - g.pl) <= 6.f) v = 2;
    if ((x == GW - 5 || x == GW - 4) && fabsf(y - g.pr) <= 6.f) v = 2;
    if (fabsf(x - g.bx) <= 1.2f && fabsf(y - g.by) <= 1.2f) v = 3;
    P.fb[y * GW + x] = v;
  }
}

// ---------------- scheduler
__device__ __forceinline__ int job_total(const Params& P, int j) {
  if (j == 0) return P.F * TPG;
  if (j == 1) return P.T * NLI_C * (NG + NDT);
  return *reinterpret_cast<volatile int*>(P.train_limit) * TPT;
}
__device__ bool job_ready(const Params& P, int j, int g, u64 T0, u64 now) {
  if (j == 0) {
    const int f = g / TPG, t = g % TPG;
    if (t == 0) return now >= T0 + (u64)f * P.frame_ns && ld_acquire(P.ctr + 1) >= GRB * f;
    return ld_acquire(P.ctr + 0) >= f + 1;
  }
  if (j == 1) {
    const int per = NG + NDT, TI = NLI_C * per, s = g / TI, tid = g % TI, l = tid / per, r = tid % per;
    if (r < NG) {
      if (l == 0 && P.token_ns > 0 && now < T0 + (u64)s * P.token_ns) return false;
      return ld_acquire(P.ctr + (l > 0 ? cD(l - 1) : cD(NLI_C - 1))) >= (l > 0 ? NDT * (s + 1) : NDT * s);
    }
    return ld_acquire(P.ctr + cG(l, (r - NG) / NRB)) >= (NG / KC) * (s + 1);
  }
  const int s = g / TPT, r = g % TPT;
  if (r < T1) {
    if (r == 0 && ((P.train_dur_ns > 0 && now >= T0 + (u64)P.train_dur_ns) || *reinterpret_cast<volatile int*>(P.stop))) { atomicMin(P.train_limit, s); return false; }
    return ld_acquire(P.ctr + 4) >= T3 * s;
  }
  return ld_acquire(P.ctr + 2) >= T1 * (s + 1);
}

// dependency description of a candidate task: which counter, what target, and whether it is blocked by time/deadline
__device__ __forceinline__ void cand_info(const Params& P, int j, int g, u64 T0, u64 now, int& cidx, int& tg, bool& blocked) {
  blocked = false;
  if (j == 0) {
    const int f = g / TPG, t = g % TPG;
    if (t == 0) { cidx = 1; tg = GRB * f; blocked = now < T0 + (u64)f * P.frame_ns; } else { cidx = 0; tg = f + 1; }
  } else if (j == 1) {
    const int per = NG + NDT, TI = NLI_C * per, s = g / TI, tid = g % TI, l = tid / per, r = tid % per;
    if (r < NG) {
      blocked = (l == 0 && P.token_ns > 0 && now < T0 + (u64)s * P.token_ns);
      cidx = l > 0 ? cD(l - 1) : cD(NLI_C - 1); tg = l > 0 ? NDT * (s + 1) : NDT * s;
    } else { cidx = cG(l, (r - NG) / NRB); tg = (NG / KC) * (s + 1); }
  } else {
    const int s = g / TPT, r = g % TPT;
    if (r < T1) {
      cidx = 4; tg = T3 * s;
      if (r == 0 && ((P.train_dur_ns > 0 && now >= T0 + (u64)P.train_dur_ns) || *reinterpret_cast<volatile int*>(P.stop))) { atomicMin(P.train_limit, s); blocked = true; }
    } else { cidx = 2; tg = T1 * (s + 1); }
  }
}

// Hot path: each lane looks at ONE shard per job, the 32-shard window starting at the block's home shard (lane 0 = home).
// Idle path (nothing ready): scan all shards for termination and for claim-ahead eligibility.
// Claim-ahead takes an inference task whose dependency is pending only if every task of an earlier 256-task group is
// already claimed, so a waiting claim can never block the work it depends on.
__device__ __forceinline__ void try_claim(const Params& P, u64 T0, int home, int lane, int& sel, int& gg, int& ahead, bool& alldone) {
  sel = -1; gg = 0; ahead = 0; alldone = false;
  const u64 now = gtimer();
  const int tlim = *reinterpret_cast<volatile int*>(P.train_limit);
  const int shw = (home + lane) & (NS - 1);
  int tk[3], g[3], cv[3], cid[3], tgt[3]; bool valid[3], blk[3];
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const bool has = (j == 0) ? (lane == 0) : true;
    tk[j] = has ? *reinterpret_cast<volatile int*>(P.ticket + j * NS + (j == 0 ? 0 : j == 1 ? shw : lane)) : 0;
  }
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const int nsh = (j == 0) ? 1 : (j == 1 ? NS : NST), shard = (j == 0) ? 0 : (j == 1 ? shw : lane);
    const int total = (j == 0) ? P.F * TPG : (j == 1) ? P.T * NLI_C * (NG + NDT) : tlim * TPT;
    g[j] = tk[j] * nsh + shard; valid[j] = ((j == 0) ? (lane == 0) : true) && g[j] < total;
    cv[j] = 0; cid[j] = 0; tgt[j] = 0; blk[j] = false;
    if (valid[j]) { cand_info(P, j, g[j], T0, now, cid[j], tgt[j], blk[j]); cv[j] = ld_rlx(P.ctr + cid[j]); }
  }
#pragma unroll
  for (int j = 0; j < 3; j++) {
    if (sel >= 0) continue;
    unsigned m = __ballot_sync(~0u, valid[j] && !blk[j] && cv[j] >= tgt[j]);
    const int rot = (j == 2) ? (home & 31) : 0;              // spread blocks over different training shards
    while (m && sel < 0) {
      const unsigned mr = rot ? ((m >> rot) | (m << (32 - rot))) : m;
      const int k = (__ffs(mr) - 1 + rot) & 31;
      int ok = 0;
      if (lane == k) ok = atomicCAS(P.ticket + j * NS + (j == 0 ? 0 : j == 1 ? shw : lane), tk[j], tk[j] + 1) == tk[j];
      ok = __shfl_sync(~0u, ok, k);
      if (ok) { sel = j; ahead = 0; gg = __shfl_sync(~0u, g[j], k); } else m &= ~(1u << k);
    }
  }
  if (sel >= 0) return;
  // ---- idle path: global scan (termination check only; no claim-ahead in this variant)
  bool nd = false; unsigned lo1 = 0xffffffffu;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const int nsh = (j == 0) ? 1 : (j == 1 ? NS : NST);
    const int total = (j == 0) ? P.F * TPG : (j == 1) ? P.T * NLI_C * (NG + NDT) : tlim * TPT;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int shard = lane + 32 * i;
      if (shard < nsh) {
        const int t = *reinterpret_cast<volatile int*>(P.ticket + j * NS + shard), gx = t * nsh + shard;
        if (gx < total) { nd = true; if (j == 1) lo1 = min(lo1, (unsigned)gx); }
      }
    }
  }
  alldone = !__any_sync(~0u, nd);
  (void)lo1;
}

__global__ void __launch_bounds__(NT, 2) vm(const Params P) {
  __shared__ __align__(16) float sm[2 * TU * TB + 16];
  __shared__ float red[NT / 32];
  __shared__ int sel_s, g_s;
  const int tid = threadIdx.x;
  if (blockIdx.x == 0 && tid == 0) atomicExch(P.t0, gtimer() + 3000000ull);
  while (*reinterpret_cast<volatile u64*>(P.t0) == 0) __nanosleep(200);
  const u64 T0 = *reinterpret_cast<volatile u64*>(P.t0);
  __syncthreads();
  const int home = blockIdx.x % NS, lane = tid & 31;
  long long tag = 0;                                    // what the scratch smem currently holds (0 = nothing reusable)
  for (;;) {
    if (tid < 32) {                                     // warp 0 schedules
      int sel, gg, ah; bool alldone;
      try_claim(P, T0, home, lane, sel, gg, ah, alldone);
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
    if (sel == 0) {
      tag = 0;
      const int f = g / TPG, t = g % TPG;
      if (t == 0) {
        if (tid == 0) game_physics(P.gs);
        signal_ret(P.ctr + 0);
      } else {
        game_render(P, t - 1);
        const int old = signal_ret(P.ctr + 1);
        if (tid == 0 && old == GRB * (f + 1) - 1) {
          P.frame_done[f] = gtimer();
          P.stats->sl = P.gs->sl; P.stats->sr = P.gs->sr; P.stats->frames = f + 1; __threadfence_system();
        }
      }
    } else if (sel == 1) {
      const int per = NG + NDT, TI = NLI_C * per, s = g / TI, tt = g % TI, l = tt / per, r = tt % per;
      const long long ntag = (long long)(s * NLI_C + l) * 8 + (r < NG ? 1 : 2 + (r - NG) / NRB);
      const bool reuse = (ntag == tag) && !(r == 0);       // same token/layer/chunk as the previous task in this block
      tag = ntag;
      if (r < NG) {
        gateup_task(P.Ls[l], r, l, P.xbuf, P.part, P.act, sm, red, reuse);
        signal_ret(P.ctr + cG(l, r / (NG / KC)));
      } else {
        down_task(P.Ls[l], r - NG, l, P.part, P.act, sm, reuse);
        const int old = signal_ret(P.ctr + cD(l));
        if (tid == 0 && l == NLI_C - 1 && old == NDT * (s + 1) - 1) {
          P.tok_done[s] = gtimer(); P.stats->tokens = s + 1; if (P.token_ns == 0 && s == P.T - 1) *reinterpret_cast<volatile int*>(P.stop) = 1; __threadfence_system();
        }
      }
    } else {
      tag = 0;
      const int s = g / TPT, r = g % TPT;
      if (P.tprof && tid == 0 && s < 64) P.tprof[((size_t)s * TPT + r) * 2] = gtimer();
      if (r < T1)            { train_p1(P, r, s, sm);            signal_ret(P.ctr + 2); }
      else {
        train_p3(P, r - T1, s, sm);
        const int old = signal_ret(P.ctr + 4);
        if (tid == 0 && old == T3 * (s + 1) - 1) {
          P.train_done[s] = gtimer(); P.stats->loss = P.loss_hist[s]; P.stats->train_steps = s + 1;
        }
      }
      if (P.tprof && tid == 0 && s < 64) P.tprof[((size_t)s * TPT + r) * 2 + 1] = gtimer();
    }
  }
}

// ---------------- baseline: the same task code as ordinary per-op kernels (for the 2-stream comparison)
__global__ void __launch_bounds__(NT, 2) b_gateup(const Layer* Ls, int l, float* xbuf, const float* part, bf16* act) {
  __shared__ __align__(16) float xs[H]; __shared__ float red[NT / 32];
  gateup_task(Ls[l], blockIdx.x, l, xbuf, part, act, xs, red);
}
__global__ void __launch_bounds__(NT, 2) b_down(const Layer* Ls, int l, float* part, bf16* act) {
  __shared__ __align__(16) float xs[KCH];
  down_task(Ls[l], blockIdx.x, l, part, act, xs);
}
__global__ void __launch_bounds__(NT, 2) b_p1(const Params P, const int* sd) { __shared__ __align__(16) float sm[2 * TU * TB + 16]; train_p1(P, blockIdx.x, *sd, sm); }
__global__ void __launch_bounds__(NT, 2) b_p3(const Params P, const int* sd) { __shared__ __align__(16) float sm[2 * TU * TB + 16]; train_p3(P, blockIdx.x, *sd, sm); }
__global__ void b_step_inc(int* sd) { if (threadIdx.x == 0) *sd += 1; }

// ---------------- host
__global__ void init_bf(bf16* p, size_t n, unsigned seed, float scale, float off) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = hash32((unsigned)i * 2654435761u ^ seed);
    p[i] = __float2bfloat16(off + (((h & 0xffffff) / 16777216.f) * 2.f - 1.f) * scale);
  }
}
__global__ void init_f(float* p, size_t n, unsigned seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = hash32((unsigned)i * 2654435761u ^ seed);
    p[i] = (((h & 0xffffff) / 16777216.f) * 2.f - 1.f) * scale;
  }
}
__global__ void init_xt(const float* X, float* XT) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NBATCH * TB * TD; i += gridDim.x * blockDim.x) {
    const int bt = i / (TB * TD), r = i % (TB * TD), b = r / TD, f = r % TD;
    XT[(size_t)(bt * TD + f) * TB + b] = X[i];
  }
}
__global__ void init_targets(const float* X, float* Tg, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { const float* x = X + (size_t)i * TD; Tg[i] = sinf(2.f * x[0]) + 0.5f * x[1] * x[2] - 0.3f * cosf(3.f * x[3]); }
}

static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))]; }
static uint64_t fnv(const void* d, size_t n, uint64_t h = 1469598103934665603ull) { const unsigned char* p = (const unsigned char*)d; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h; }

int main(int argc, char** argv) {
  double secs = 5.0; bool live = false; long long tok_us = 3000; int NLI = 8, detK = 400; bool skip_solo = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--live")) live = true;
    else if (!strcmp(argv[i], "--tok-us") && i + 1 < argc) tok_us = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--layers") && i + 1 < argc) NLI = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--det-steps") && i + 1 < argc) detK = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--all-only")) skip_solo = true;
  }
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  int dev = 0; CK(cudaGetDevice(&dev)); cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm, NT, 0));
  if (occ < 1) { fprintf(stderr, "vm does not fit\n"); return 1; }
  const int nblocks = nsm * std::min(occ, 2);
  if (NLI != NLI_C) { fprintf(stderr, "--layers must equal NLI_C=%d (recompile)\n", NLI_C); return 1; }
  printf("device: %s  SMs=%d  blocks/SM=%d  grid=%d  (1 persistent kernel, 3 job types)\n", prop.name, nsm, std::min(occ, 2), nblocks);

  const int FMAX = 60 * 40, TMAX = 20000, KMAX = 400000;
  // --- inference state
  std::vector<Layer> hl(NLI);
  for (int l = 0; l < NLI; l++) {
    bf16 *wg, *wu, *wd, *ga; const size_t n1 = (size_t)I * H;
    CK(cudaMalloc(&wg, n1 * 2)); CK(cudaMalloc(&wu, n1 * 2)); CK(cudaMalloc(&wd, n1 * 2)); CK(cudaMalloc(&ga, H * 2));
    init_bf<<<1024, 256>>>(wg, n1, 1000 + l * 4, 1.7f / sqrtf((float)H), 0.f);
    init_bf<<<1024, 256>>>(wu, n1, 1001 + l * 4, 1.7f / sqrtf((float)H), 0.f);
    init_bf<<<1024, 256>>>(wd, n1, 1002 + l * 4, 1.7f / sqrtf((float)I), 0.f);
    init_bf<<<8, 256>>>(ga, H, 1003 + l * 4, 0.2f, 1.f);
    hl[l] = {wg, wu, wd, ga};
  }
  Layer* dL; CK(cudaMalloc(&dL, NLI * sizeof(Layer))); CK(cudaMemcpy(dL, hl.data(), NLI * sizeof(Layer), cudaMemcpyHostToDevice));
  float *xbuf, *part; bf16* act;
  CK(cudaMalloc(&xbuf, (size_t)NLI * H * 4)); CK(cudaMalloc(&part, (size_t)NLI * KC * H * 4)); CK(cudaMalloc(&act, I * 2));
  { std::vector<float> x0(H); unsigned s0 = 12345; for (int i = 0; i < H; i++) { s0 = s0 * 1664525u + 1013904223u; x0[i] = ((s0 >> 8) / 16777216.f) * 2.f - 1.f; }
    CK(cudaMemcpy(xbuf, x0.data(), H * 4, cudaMemcpyHostToDevice)); }
  // --- training state
  float *X, *XT, *Tg, *W1, *b1, *W2, *b2s, *Hb, *Eb, *Lb, *loss_hist; u64* train_done;
  CK(cudaMalloc(&X, (size_t)NBATCH * TB * TD * 4)); CK(cudaMalloc(&XT, (size_t)NBATCH * TB * TD * 4)); CK(cudaMalloc(&Tg, NBATCH * TB * 4));
  CK(cudaMalloc(&W1, (size_t)TH * TD * 4)); CK(cudaMalloc(&b1, TH * 4)); CK(cudaMalloc(&W2, TH * 4)); CK(cudaMalloc(&b2s, 8));
  CK(cudaMalloc(&Hb, (size_t)TH * TB * 4)); CK(cudaMalloc(&Eb, (size_t)T1 * TB * 4)); CK(cudaMalloc(&Lb, 64));
  CK(cudaMalloc(&loss_hist, KMAX * 4)); CK(cudaMalloc(&train_done, KMAX * 8));
  init_f<<<64, 256>>>(X, (size_t)NBATCH * TB * TD, 777, 1.f);
  init_xt<<<64, 256>>>(X, XT);
  init_targets<<<(NBATCH * TB + 255) / 256, 256>>>(X, Tg, NBATCH * TB);
  // --- game / scheduler state
  GameState* gs; CK(cudaMalloc(&gs, sizeof(GameState)));
  u64 *frame_done, *tok_done, *t0; int *ctr, *ticket, *tlimit, *stopf, *sdev;
  CK(cudaMalloc(&frame_done, FMAX * 8)); CK(cudaMalloc(&tok_done, TMAX * 8)); CK(cudaMalloc(&t0, 8));
  const int nctr = 8 + CS * NLI;
  CK(cudaMalloc(&ctr, nctr * 4)); CK(cudaMalloc(&ticket, 3 * NS * 4)); CK(cudaMalloc(&tlimit, 4)); CK(cudaMalloc(&stopf, 4)); CK(cudaMalloc(&sdev, 4));
  unsigned char *hfb, *dfb; Stats *hst, *dst;
  CK(cudaHostAlloc((void**)&hfb, GW * GH, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dfb, hfb, 0));
  CK(cudaHostAlloc((void**)&hst, sizeof(Stats), cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dst, hst, 0));
  u64* tprof; CK(cudaMalloc(&tprof, (size_t)64 * TPT * 2 * 8)); CK(cudaMemset(tprof, 0, (size_t)64 * TPT * 2 * 8));
  CK(cudaDeviceSynchronize());

  cudaStream_t s; CK(cudaStreamCreate(&s));
  const long long frame_ns = 16666667ll, token_ns = tok_us * 1000ll;

  struct Result { double tok_per_s, tr_steps_window; double lat_tok_p50, lat_tok_p99, lat_tok_max, lat_fr_p50, lat_fr_p99, lat_fr_max, steps_per_s; int tokens, frames, steps, fr_missed; double loss0, loss1; uint64_t whash; std::vector<float> xfinal; double wall_s; };
  int Tunp = 1500;
  auto run = [&](const char* name, int jobs, double D, bool det, bool show_live, bool unpaced = false) -> Result {
    // reset
    GameState g0 = {GW / 2.f, GH / 2.f, 1.3f, 0.6f, GH / 2.f, GH / 2.f, 0, 0, 1};
    CK(cudaMemcpy(gs, &g0, sizeof(g0), cudaMemcpyHostToDevice));
    CK(cudaMemset(ctr, 0, nctr * 4)); CK(cudaMemset(ticket, 0, 3 * NS * 4)); CK(cudaMemset(t0, 0, 8));
    CK(cudaMemset(frame_done, 0, FMAX * 8)); CK(cudaMemset(tok_done, 0, TMAX * 8)); CK(cudaMemset(train_done, 0, KMAX * 8));
    CK(cudaMemset(loss_hist, 0, KMAX * 4)); CK(cudaMemset(part, 0, (size_t)NLI * KC * H * 4)); CK(cudaMemset(b2s, 0, 8));
    CK(cudaMemset(b1, 0, TH * 4)); CK(cudaMemset(stopf, 0, 4));
    init_f<<<16, 256>>>(W1, (size_t)TH * TD, 31, 1.5f / 8.f);
    init_f<<<4, 256>>>(W2, TH, 32, 1.7f / 32.f);
    memset(hfb, 0, GW * GH); memset((void*)hst, 0, sizeof(Stats));
    const int K = (jobs & 4) ? (det ? detK : KMAX - 1) : 0;
    CK(cudaMemcpy(tlimit, &K, 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    Params P = {};
    P.t0 = t0; P.ctr = ctr; P.ticket = ticket;
    P.F = (jobs & 1) ? std::min(FMAX - 1, (int)(D * 60.0)) : 0; P.frame_ns = frame_ns; P.gs = gs; P.fb = dfb; P.frame_done = frame_done;
    P.T = (jobs & 2) ? (unpaced ? Tunp : std::min(TMAX - 1, (int)(D * 1e9 / token_ns))) : 0; P.NLI = NLI; P.token_ns = unpaced ? 0 : token_ns; P.Ls = dL;
    P.xbuf = xbuf; P.part = part; P.act = act; P.tok_done = tok_done;
    P.K = K; P.train_dur_ns = (det || unpaced) ? 0 : (long long)(D * 1e9); P.train_limit = tlimit;
    P.X = X; P.XT = XT; P.Tg = Tg; P.W1 = W1; P.b1 = b1; P.W2 = W2; P.b2s = b2s; P.Hb = Hb; P.Eb = Eb; P.Lb = Lb; P.loss_hist = loss_hist; P.train_done = train_done;
    P.stats = dst; P.stop = stopf; P.tprof = (getenv("MK_TPROF") && jobs == 4) ? tprof : nullptr;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    vm<<<nblocks, NT, 0, s>>>(P);
    CK(cudaEventRecord(e1, s));
    if (show_live) {
      const char* ch[4] = {" ", ":", "|", "O"};
      while (cudaStreamQuery(s) == cudaErrorNotReady) {
        std::string out = "\033[H";
        char line[256];
        for (int y = 0; y < GH; y += 2) {
          for (int x = 0; x < GW; x += 2) {
            unsigned char a = std::max(std::max(hfb[y * GW + x], hfb[y * GW + x + 1]), std::max(hfb[(y + 1) * GW + x], hfb[(y + 1) * GW + x + 1]));
            out += (a == 3 ? "O" : a == 2 ? "#" : a == 1 ? ":" : " ");
          }
          out += "\n";
        }
        snprintf(line, sizeof line, "[%s] frame %u  score %d-%d | tokens %u | train steps %u  loss %.4f      \n", name, hst->frames, hst->sl, hst->sr, hst->tokens, hst->train_steps, hst->loss);
        out += line;
        fputs(out.c_str(), stdout); fflush(stdout);
        usleep(40000);
      }
    }
    CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    Result R = {}; R.wall_s = ms / 1e3;
    u64 h_t0; CK(cudaMemcpy(&h_t0, t0, 8, cudaMemcpyDeviceToHost));
    if (jobs & 1) {
      std::vector<u64> d(P.F); CK(cudaMemcpy(d.data(), frame_done, P.F * 8, cudaMemcpyDeviceToHost));
      std::vector<double> lat; for (int f = 0; f < P.F; f++) { double l = ((double)d[f] - (double)(h_t0 + (u64)f * frame_ns)) / 1e3; lat.push_back(l); if (l * 1e3 > frame_ns) R.fr_missed++; }
      R.frames = P.F; R.lat_fr_p50 = pct(lat, .5); R.lat_fr_p99 = pct(lat, .99); R.lat_fr_max = pct(lat, 1.0);
    }
    if (jobs & 2) {
      std::vector<u64> d(P.T); CK(cudaMemcpy(d.data(), tok_done, P.T * 8, cudaMemcpyDeviceToHost));
      std::vector<double> lat; for (int t = 0; t < P.T; t++) lat.push_back(((double)d[t] - (double)(h_t0 + (u64)t * token_ns)) / 1e3);
      if (unpaced) { double win = ((double)d[P.T - 1] - (double)h_t0) / 1e9; R.tok_per_s = P.T / win; R.wall_s = win;
        if (jobs & 4) { std::vector<u64> td(KMAX); CK(cudaMemcpy(td.data(), train_done, KMAX * 8, cudaMemcpyDeviceToHost)); long long c = 0; for (int k = 0; k < KMAX; k++) if (td[k] && td[k] <= d[P.T - 1]) c++; R.tr_steps_window = c / win; } }
      R.tokens = P.T; R.lat_tok_p50 = pct(lat, .5); R.lat_tok_p99 = pct(lat, .99); R.lat_tok_max = pct(lat, 1.0);
      std::vector<float> xl(H), pt((size_t)KC * H);
      CK(cudaMemcpy(xl.data(), xbuf + (size_t)(NLI - 1) * H, H * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(pt.data(), part + (size_t)(NLI - 1) * KC * H, (size_t)KC * H * 4, cudaMemcpyDeviceToHost));
      R.xfinal.resize(H); for (int i = 0; i < H; i++) { float v = xl[i]; for (int c = 0; c < KC; c++) v += pt[(size_t)c * H + i]; R.xfinal[i] = v; }
    }
    if (jobs & 4) {
      int lim; CK(cudaMemcpy(&lim, tlimit, 4, cudaMemcpyDeviceToHost));
      R.steps = lim; R.steps_per_s = lim / D;
      std::vector<float> lh(lim); if (lim) CK(cudaMemcpy(lh.data(), loss_hist, lim * 4, cudaMemcpyDeviceToHost));
      auto avg = [&](int a, int b) { double t = 0; for (int i = a; i < b; i++) t += lh[i]; return b > a ? t / (b - a) : 0.0; };
      R.loss0 = avg(0, std::min(lim, 50)); R.loss1 = avg(std::max(0, lim - 50), lim);
      std::vector<float> w1(TH * TD), bb1(TH), w2(TH), bb2(1); float b2v[2];
      CK(cudaMemcpy(w1.data(), W1, TH * TD * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bb1.data(), b1, TH * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(w2.data(), W2, TH * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2v, b2s, 8, cudaMemcpyDeviceToHost)); bb2[0] = b2v[lim & 1];
      uint64_t h = fnv(w1.data(), w1.size() * 4); h = fnv(bb1.data(), bb1.size() * 4, h); h = fnv(w2.data(), w2.size() * 4, h); h = fnv(bb2.data(), 4, h);
      R.whash = h;
    }
    return R;
  };
  auto print = [&](const char* name, int jobs, const Result& R) {
    printf("\n[%s]  wall %.2fs\n", name, R.wall_s);
    if (jobs & 1) printf("  game      %4d frames @60Hz   latency us p50 %.0f  p99 %.0f  max %.0f   missed deadlines (>16.7ms): %d\n", R.frames, R.lat_fr_p50, R.lat_fr_p99, R.lat_fr_max, R.fr_missed);
    if (jobs & 2) printf("  inference %4d tokens @%.0f/s latency us p50 %.0f  p99 %.0f  max %.0f\n", R.tokens, 1e6 / tok_us, R.lat_tok_p50, R.lat_tok_p99, R.lat_tok_max);
    if (jobs & 4) printf("  training  %6d steps  = %.0f steps/s   loss %.4f -> %.4f\n", R.steps, R.steps_per_s, R.loss0, R.loss1);
  };

  printf("config: %.1fs per run, %d-layer decode chain, token period %lld us, 60 FPS game, SGD MLP %dx%dx1 batch %d\n", secs, NLI, tok_us, TD, TH, TB);
  if (getenv("MK_TBENCH")) {
    Params B = {}; B.X = X; B.XT = XT; B.Tg = Tg; B.W1 = W1; B.b1 = b1; B.W2 = W2; B.b2s = b2s; B.Hb = Hb; B.Eb = Eb; B.Lb = Lb; B.loss_hist = loss_hist;
    CK(cudaMemset(sdev, 0, 4)); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); float ms;
    for (int ph = 0; ph < 2; ph++) {
      for (int rep = 0; rep < 20; rep++) { if (ph == 0) b_p1<<<T1, NT, 0, s>>>(B, sdev); else b_p3<<<T3, NT, 0, s>>>(B, sdev); }
      CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(a, s));
      for (int rep = 0; rep < 500; rep++) { if (ph == 0) b_p1<<<T1, NT, 0, s>>>(B, sdev); else b_p3<<<T3, NT, 0, s>>>(B, sdev); }
      CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
      printf("  train phase %s kernel (back-to-back launches): %.1f us each (%d tasks)\n", ph == 0 ? "P1" : "P3", ms * 1e3 / 500, ph == 0 ? T1 : T3);
    }
    return 0;
  }
  Result Ri, Rg, Rt;
  if (!skip_solo) {
    Ri = run("solo inference", 2, secs, false, false); print("solo inference", 2, Ri);
    Rg = run("solo game", 1, secs, false, false);      print("solo game", 1, Rg);
    Rt = run("solo training", 4, secs, false, false);  print("solo training", 4, Rt);
    if (getenv("MK_TPROF")) {
      std::vector<u64> tp((size_t)64 * TPT * 2); CK(cudaMemcpy(tp.data(), tprof, tp.size() * 8, cudaMemcpyDeviceToHost));
      double sp1 = 0, sp3 = 0, g13 = 0, g31 = 0, e1 = 0, e3 = 0, n = 0;
      for (int st = 20; st < 60; st++) {
        auto A = [&](int lo, int hi, int k, bool mn) { u64 v = mn ? ~0ull : 0; for (int r = lo; r < hi; r++) { u64 x = tp[((size_t)st * TPT + r) * 2 + k]; v = mn ? std::min(v, x) : std::max(v, x); } return (double)v; };
        const double p1s = A(0, T1, 0, true), p1e = A(0, T1, 1, false), p3s = A(T1, TPT, 0, true), p3e = A(T1, TPT, 1, false);
        double nxt = 0; { u64 v = ~0ull; for (int r = 0; r < T1; r++) v = std::min(v, tp[((size_t)(st + 1) * TPT + r) * 2]); nxt = (double)v; }
        double tmin1 = 1e18, tmax1 = 0; for (int r = 0; r < T1; r++) { double d = (double)tp[((size_t)st * TPT + r) * 2 + 1] - (double)tp[((size_t)st * TPT + r) * 2]; tmin1 = std::min(tmin1, d); tmax1 = std::max(tmax1, d); e1 += d; }
        double e3x = 0; for (int r = T1; r < TPT; r++) e3x += (double)tp[((size_t)st * TPT + r) * 2 + 1] - (double)tp[((size_t)st * TPT + r) * 2]; e3 += e3x;
        sp1 += p1e - p1s; sp3 += p3e - p3s; g13 += p3s - p1e; g31 += nxt - p3e; n++;
      }
      printf("  [train timeline, us] P1 span %.1f (mean task %.1f) | gap P1end->P3start %.1f | P3 span %.1f (mean task %.1f) | gap P3end->nextP1start %.1f | step %.1f\n",
             sp1 / n / 1e3, e1 / n / T1 / 1e3, g13 / n / 1e3, sp3 / n / 1e3, e3 / n / T3 / 1e3, g31 / n / 1e3, (sp1 + sp3 + g13 + g31) / n / 1e3);
    }
  }
  if (live) { printf("\033[2J"); }
  Result Ra = run("ALL THREE", 7, secs, false, live);
  if (live) printf("\033[2J\033[H");
  print("ALL THREE concurrently, one kernel", 7, Ra);
  if (!live) {
    // one snapshot frame
    printf("\nlast game frame:\n");
    for (int y = 0; y < GH; y += 2) { for (int x = 0; x < GW; x += 2) { unsigned char a = std::max(std::max(hfb[y * GW + x], hfb[y * GW + x + 1]), std::max(hfb[(y + 1) * GW + x], hfb[(y + 1) * GW + x + 1])); putchar(a == 3 ? 'O' : a == 2 ? '#' : a == 1 ? ':' : ' '); } putchar('\n'); }
  }
  if (!skip_solo) {
    printf("\n== summary: solo vs concurrent ==\n");
    printf("  token latency p99   : %.0f us solo  ->  %.0f us concurrent\n", Ri.lat_tok_p99, Ra.lat_tok_p99);
    printf("  frame latency p99   : %.0f us solo  ->  %.0f us concurrent  (deadline 16667 us)\n", Rg.lat_fr_p99, Ra.lat_fr_p99);
    printf("  training throughput : %.0f steps/s solo  ->  %.0f steps/s while serving + gaming (%.0f%% retained)\n", Rt.steps_per_s, Ra.steps_per_s, 100.0 * Ra.steps_per_s / Rt.steps_per_s);
    printf("  inference output bitwise identical solo vs concurrent: %s\n", (Ri.xfinal.size() && !memcmp(Ri.xfinal.data(), Ra.xfinal.data(), H * 4)) ? "YES" : "NO");
    Result D1 = run("det solo", 4, secs, true, false);
    Result D2 = run("det all", 7, secs, true, false);
    printf("  training weights after exactly %d steps, bitwise identical solo vs concurrent: %s  (hash %016llx vs %016llx)\n", detK,
           D1.whash == D2.whash ? "YES" : "NO", (unsigned long long)D1.whash, (unsigned long long)D2.whash);
  }

  // ================= throughput interleave: megakernel vs 2-stream CUDA graphs (unpaced) =================
  if (!skip_solo) {
    printf("\n== interleave gain: unpaced inference (%d tokens) + training, megakernel vs two CUDA-graph streams ==\n", Tunp);
    Result Mi = run("mega solo inf", 2, secs, false, false, true);
    Result Mb = run("mega both", 6, secs, false, false, true);
    double m_inf = Mi.tok_per_s, m_trn = Rt.steps_per_s, m_b_inf = Mb.tok_per_s, m_b_trn = Mb.tr_steps_window;
    // baseline state
    Params B = {}; B.Ls = dL; B.xbuf = xbuf; B.part = part; B.act = act; B.X = X; B.XT = XT; B.Tg = Tg; B.W1 = W1; B.b1 = b1; B.W2 = W2; B.b2s = b2s;
    B.Hb = Hb; B.Eb = Eb; B.Lb = Lb; B.loss_hist = loss_hist; B.NLI = NLI;
    auto base_reset = [&]() {
      init_f<<<16, 256>>>(W1, (size_t)TH * TD, 31, 1.5f / 8.f); init_f<<<4, 256>>>(W2, TH, 32, 1.7f / 32.f);
      CK(cudaMemset(b1, 0, TH * 4)); CK(cudaMemset(b2s, 0, 8)); CK(cudaMemset(sdev, 0, 4)); CK(cudaMemset(loss_hist, 0, KMAX * 4));
      CK(cudaMemset(part, 0, (size_t)NLI * KC * H * 4)); CK(cudaDeviceSynchronize());
    };
    int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    cudaStream_t s1, s2; CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, hi)); CK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, lo));
    cudaGraph_t gtok, gstep; cudaGraphExec_t etok, estep;
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < NLI; l++) { b_gateup<<<NG, NT, 0, s1>>>(dL, l, xbuf, part, act); b_down<<<NDT, NT, 0, s1>>>(dL, l, part, act); }
    CK(cudaStreamEndCapture(s1, &gtok)); CK(cudaGraphInstantiate(&etok, gtok, 0));
    CK(cudaStreamBeginCapture(s2, cudaStreamCaptureModeGlobal));
    b_p1<<<T1, NT, 0, s2>>>(B, sdev); b_p3<<<T3, NT, 0, s2>>>(B, sdev); b_step_inc<<<1, 1, 0, s2>>>(sdev);
    CK(cudaStreamEndCapture(s2, &gstep)); CK(cudaGraphInstantiate(&estep, gstep, 0));
    auto now_s = []() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; };
    // baseline solo inference
    base_reset(); for (int i = 0; i < 20; i++) CK(cudaGraphLaunch(etok, s1)); CK(cudaStreamSynchronize(s1));
    double t = now_s(); for (int i = 0; i < Tunp; i++) CK(cudaGraphLaunch(etok, s1)); CK(cudaStreamSynchronize(s1));
    double b_inf = Tunp / (now_s() - t);
    // baseline solo training
    base_reset(); t = now_s(); int launched = 0;
    while (now_s() - t < secs) { for (int k = 0; k < 16; k++) CK(cudaGraphLaunch(estep, s2)); launched += 16; CK(cudaStreamSynchronize(s2)); }
    double b_trn = launched / (now_s() - t);
    // baseline both: inference on high-priority stream, training on low-priority stream until inference completes
    base_reset();
    cudaEvent_t es, ee; CK(cudaEventCreate(&es)); CK(cudaEventCreate(&ee));
    std::vector<cudaEvent_t> evs; std::vector<int> evn;
    CK(cudaEventRecord(es, 0)); CK(cudaStreamSynchronize(0));
    CK(cudaEventRecord(es, s1)); CK(cudaEventRecord(es, s2));
    t = now_s();
    for (int i = 0; i < Tunp; i++) CK(cudaGraphLaunch(etok, s1));
    CK(cudaEventRecord(ee, s1));
    launched = 0;
    while (cudaEventQuery(ee) == cudaErrorNotReady) {
      for (int k = 0; k < 8; k++) CK(cudaGraphLaunch(estep, s2));
      launched += 8; cudaEvent_t e; CK(cudaEventCreate(&e)); CK(cudaEventRecord(e, s2)); evs.push_back(e); evn.push_back(launched);
      while (launched - (evn.size() > 4 ? evn[evn.size() - 4] : 0) > 64 && cudaEventQuery(ee) == cudaErrorNotReady) { usleep(20); if (cudaEventQuery(evs[evs.size() - 4 > 0 ? evs.size() - 4 : 0]) == cudaSuccess) break; }
    }
    CK(cudaEventSynchronize(ee)); CK(cudaStreamSynchronize(s2));
    float win_ms; CK(cudaEventElapsedTime(&win_ms, es, ee));
    long long done_in_window = 0;
    for (size_t k = 0; k < evs.size(); k++) { float tm; CK(cudaEventElapsedTime(&tm, es, evs[k])); if (tm <= win_ms) done_in_window = evn[k]; }
    double b_b_inf = Tunp / (win_ms / 1e3), b_b_trn = done_in_window / (win_ms / 1e3);
    // determinism of baseline training vs megakernel after exactly detK steps
    base_reset(); for (int i = 0; i < detK; i++) CK(cudaGraphLaunch(estep, s2)); CK(cudaStreamSynchronize(s2));
    std::vector<float> w1(TH * TD), bb1(TH), w2(TH), bb2(1); float b2v[2];
    CK(cudaMemcpy(w1.data(), W1, TH * TD * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bb1.data(), b1, TH * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w2.data(), W2, TH * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2v, b2s, 8, cudaMemcpyDeviceToHost)); bb2[0] = b2v[detK & 1];
    uint64_t bh = fnv(w1.data(), w1.size() * 4); bh = fnv(bb1.data(), bb1.size() * 4, bh); bh = fnv(w2.data(), w2.size() * 4, bh); bh = fnv(bb2.data(), 4, bh);

    printf("  %-34s %12s %14s %24s\n", "", "tokens/s", "train steps/s", "fraction of solo (sum)");
    printf("  %-34s %12.0f %14s\n", "2-stream graphs: inference alone", b_inf, "-");
    printf("  %-34s %12s %14.0f\n", "2-stream graphs: training alone", "-", b_trn);
    printf("  %-34s %12.0f %14.0f %14.2f + %.2f = %.2f\n", "2-stream graphs: both", b_b_inf, b_b_trn, b_b_inf / b_inf, b_b_trn / b_trn, b_b_inf / b_inf + b_b_trn / b_trn);
    printf("  %-34s %12.0f %14s\n", "megakernel: inference alone", m_inf, "-");
    printf("  %-34s %12s %14.0f\n", "megakernel: training alone", "-", m_trn);
    printf("  %-34s %12.0f %14.0f %14.2f + %.2f = %.2f\n", "megakernel: both", m_b_inf, m_b_trn, m_b_inf / m_inf, m_b_trn / m_trn, m_b_inf / m_inf + m_b_trn / m_trn);
    printf("  megakernel vs 2-stream, both running:  tokens/s %.2fx   train steps/s %.2fx\n", m_b_inf / b_b_inf, m_b_trn / b_b_trn);
    Result D1 = run("det solo", 4, secs, true, false);
    printf("  training weights after %d steps, baseline graphs vs megakernel bitwise identical: %s\n", detK, bh == D1.whash ? "YES" : "NO");
  }
  return 0;
}
