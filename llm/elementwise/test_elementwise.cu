// Standalone test + benchmark for the elementwise/optimizer tasks. Driven by ref_elementwise.py:
//   python3 ref_elementwise.py gen DIR ; ./ew_test DIR ; python3 ref_elementwise.py check DIR
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "ew_all.cuh"
using namespace ew;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

constexpr int D = 768, T = 512, V = 100352, N = 4096, NCE = 256;
static std::string DIR;

// ---- launchers: grid (one block per task) vs persistent (blocks pull tasks in a shuffled order) ----
template <class F> __global__ void k_grid(F f) { extern __shared__ __align__(128) char sm[]; f((int)blockIdx.x, sm); }
template <class F> __global__ void k_pers(F f, int nt, int* ctr, const int* perm) {
  extern __shared__ __align__(128) char sm[];
  __shared__ int s_t;
  while (true) {
    if (threadIdx.x == 0) s_t = atomicAdd(ctr, 1);
    __syncthreads();
    int i = s_t;
    __syncthreads();
    if (i >= nt) break;
    f(perm[i], sm);
  }
}
static int *g_ctr, *g_perm; static int g_permN = 0;
template <class F> void launch_grid(F f, int nt, size_t sm) {
  CK(cudaFuncSetAttribute(k_grid<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  k_grid<F><<<nt, NT, sm>>>(f);
}
template <class F> void launch_pers(F f, int nt, size_t sm) {
  CK(cudaFuncSetAttribute(k_pers<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  std::vector<int> p(nt); for (int i = 0; i < nt; i++) p[i] = i;
  std::mt19937 rng(123); std::shuffle(p.begin(), p.end(), rng);
  if (!g_ctr) CK(cudaMalloc(&g_ctr, 4));
  if (g_permN < nt) { if (g_perm) cudaFree(g_perm); CK(cudaMalloc(&g_perm, 4 * nt)); g_permN = nt; }
  CK(cudaMemcpy(g_perm, p.data(), 4 * nt, cudaMemcpyHostToDevice));
  CK(cudaMemset(g_ctr, 0, 4));
  k_pers<F><<<216, NT, sm>>>(f, nt, g_ctr, g_perm);
}
template <class F> void launch(int mode, F f, int nt, size_t sm) { if (mode == 0) launch_grid(f, nt, sm); else launch_pers(f, nt, sm); }

// ---- io ----
static std::vector<char> rd(const char* name) {
  std::string p = DIR + "/" + name; FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("missing %s\n", p.c_str()); exit(1);} 
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<char> b(n); if (fread(b.data(), 1, n, f) != n) exit(1); fclose(f); return b;
}
static void* up(const char* name) { auto b = rd(name); void* d; CK(cudaMalloc(&d, b.size())); CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice)); return d; }
static std::vector<char> down(const void* d, size_t n) { std::vector<char> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h; }
static void dump(const char* name, const void* d, size_t n) { auto h = down(d, n); std::string p = DIR + "/out_" + name; FILE* f = fopen(p.c_str(), "wb"); fwrite(h.data(), 1, n, f); fclose(f); }
template <class A> static void* dalloc(size_t n) { void* d; CK(cudaMalloc(&d, n)); return d; }
static void* dal(size_t n) { void* d; CK(cudaMalloc(&d, n)); return d; }
static bool same(const std::vector<char>& a, const std::vector<char>& b) { return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size()); }

__global__ void k_fill(float* p, size_t n, float v) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] = v; }
__global__ void k_rand_bf16(bf16* p, size_t n, float scale, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
  uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  uint32_t h2 = h * 747796405u + 2891336453u; h2 ^= h2 >> 16;
  float u = (h & 0xffff) / 65536.f + (h2 & 0xffff) / 65536.f - 1.f;  // triangular
  p[i] = __float2bfloat16(u * scale);
}
__global__ void k_rand_f32(float* p, size_t n, float scale, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
  uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  p[i] = ((h & 0xffffff) / 16777216.f - 0.5f) * scale;
}
static void fillf(float* p, size_t n, float v) { k_fill<<<(n + 255) / 256, 256>>>(p, n, v); }

static bool g_all_ok = true;
static void verdict(const char* what, bool ok) { printf("[bitident] %-28s %s\n", what, ok ? "PASS" : "FAIL"); if (!ok) g_all_ok = false; }

struct Ev { cudaEvent_t a, b; Ev() { cudaEventCreate(&a); cudaEventCreate(&b); } };
template <class L> static float time_ms(L l, int iters) {
  Ev e; l(); CK(cudaDeviceSynchronize());
  cudaEventRecord(e.a); for (int i = 0; i < iters; i++) l(); cudaEventRecord(e.b); cudaEventSynchronize(e.b);
  float ms; cudaEventElapsedTime(&ms, e.a, e.b); return ms / iters;
}

// ------------------------------------------------------------------ correctness
static void test_embed() {
  int* tok = (int*)up("tok.bin"); bf16* wte = (bf16*)up("wte.bin"); bf16* wpe = (bf16*)up("wpe.bin");
  bf16* x = (bf16*)dal((size_t)N * D * 2); bf16* dx0 = (bf16*)up("dx0.bin");
  float* dwte = (float*)dal((size_t)V * D * 4); float* dwpe = (float*)dal((size_t)T * D * 4);
  std::vector<char> ref[4];
  for (int mode = 0; mode < 2; mode++) {
    CK(cudaMemset(x, 0, (size_t)N * D * 2));
    launch(mode, [=] __device__(int t, char*) { embed_fwd_task<D, T>(t, x, wte, wpe, tok, N); }, N / 16, 0);
    fillf(dwte, (size_t)V * D, 1.f); fillf(dwpe, (size_t)T * D, 1.f);
    launch(mode, [=] __device__(int t, char* sm) { embed_bwd_wte_task<D>(t, dwte, dx0, tok, N, sm); }, V / EMB_VCHUNK, embed_bwd_wte_smem_bytes(N));
    launch(mode, [=] __device__(int t, char*) { embed_bwd_wpe_task<D, T>(t, dwpe, dx0, N); }, T / 4, 0);
    CK(cudaDeviceSynchronize());
    auto a = down(x, (size_t)N * D * 2), b = down(dwte, (size_t)V * D * 4), c = down(dwpe, (size_t)T * D * 4);
    if (mode == 0) { ref[0] = a; ref[1] = b; ref[2] = c; dump("x_emb", x, (size_t)N * D * 2); dump("dwte", dwte, (size_t)V * D * 4); dump("dwpe", dwpe, (size_t)T * D * 4); }
    else verdict("embed fwd/bwd", same(a, ref[0]) && same(b, ref[1]) && same(c, ref[2]));
  }
  // timing
  float ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { embed_fwd_task<D, T>(t, x, wte, wpe, tok, N); }, N / 16, 0); }, 50);
  printf("[speed] embed_fwd      %8.1f us  %7.1f GB/s (rd 2x%.1fMB wr %.1fMB)\n", ms * 1e3, 3.0 * N * D * 2 / ms / 1e6, N * D * 2 / 1e6, N * D * 2 / 1e6);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char* sm) { embed_bwd_wte_task<D>(t, dwte, dx0, tok, N, sm); }, V / EMB_VCHUNK, embed_bwd_wte_smem_bytes(N)); }, 20);
  printf("[speed] embed_bwd_wte  %8.1f us  (%.1f GB/s on touched traffic: rd dx0 + rmw fp32 rows)\n", ms * 1e3, (N * D * 2.0 + N * D * 8.0) / ms / 1e6);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { embed_bwd_wpe_task<D, T>(t, dwpe, dx0, N); }, T / 4, 0); }, 50);
  printf("[speed] embed_bwd_wpe  %8.1f us  %7.1f GB/s\n", ms * 1e3, (N * D * 2.0 + T * D * 8.0) / ms / 1e6);
}

static void test_ln() {
  bf16* x = (bf16*)up("ln_x.bin"); bf16* delta = (bf16*)up("ln_delta.bin"); bf16* gamma = (bf16*)up("ln_gamma.bin");
  bf16* dy = (bf16*)up("ln_dy.bin"); bf16* dres = (bf16*)up("ln_dres.bin");
  size_t RB = (size_t)N * D * 2; int nblk = N / 16;
  bf16 *y = (bf16*)dal(RB), *h = (bf16*)dal(RB), *y2 = (bf16*)dal(RB), *dx = (bf16*)dal(RB), *dx2 = (bf16*)dal(RB);
  float *mean = (float*)dal(N * 4), *rstd = (float*)dal(N * 4), *mean2 = (float*)dal(N * 4), *rstd2 = (float*)dal(N * 4);
  float *part = (float*)dal((size_t)nblk * D * 4), *dg = (float*)dal(D * 4);
  std::vector<char> ref[8];
  for (int mode = 0; mode < 2; mode++) {
    launch(mode, [=] __device__(int t, char*) { ln_fwd_task<D>(t, y, mean, rstd, x, nullptr, nullptr, gamma, N); }, nblk, 0);
    CK(cudaMemcpy(h, x, RB, cudaMemcpyDeviceToDevice));
    launch(mode, [=] __device__(int t, char*) { ln_fwd_task<D>(t, y2, mean2, rstd2, h, delta, h, gamma, N); }, nblk, 0);  // in-place residual
    launch(mode, [=] __device__(int t, char* sm) { ln_bwd_task<D>(t, dx, part, dy, x, mean, rstd, gamma, nullptr, N, sm); }, nblk, ln_bwd_smem_bytes<D>());
    fillf(dg, D, 1.f);
    launch(mode, [=] __device__(int t, char*) { ln_dgamma_reduce_task<D>(t, dg, part, nblk); }, (D + 255) / 256, 0);
    launch(mode, [=] __device__(int t, char* sm) { ln_bwd_task<D>(t, dx2, part, dy, x, mean, rstd, gamma, dres, N, sm); }, nblk, ln_bwd_smem_bytes<D>());
    CK(cudaDeviceSynchronize());
    std::vector<char> o[8] = {down(y, RB), down(mean, N * 4), down(rstd, N * 4), down(h, RB), down(y2, RB), down(dx, RB), down(dg, D * 4), down(dx2, RB)};
    if (mode == 0) {
      for (int i = 0; i < 8; i++) ref[i] = o[i];
      dump("ln_y", y, RB); dump("ln_mean", mean, N * 4); dump("ln_rstd", rstd, N * 4); dump("ln_h", h, RB); dump("ln_y2", y2, RB);
      dump("ln_dx", dx, RB); dump("ln_dgamma", dg, D * 4); dump("ln_dx_res", dx2, RB);
    } else { bool ok = true; for (int i = 0; i < 8; i++) ok &= same(o[i], ref[i]); verdict("layernorm fwd/bwd", ok); }
  }
  // timing: rotate through K buffer sets (> L2) so the traffic is HBM
  const int K = 8;
  bf16 *bx = (bf16*)dal(RB * K), *by = (bf16*)dal(RB * K), *bdy = (bf16*)dal(RB * K), *bdx = (bf16*)dal(RB * K);
  float *bm = (float*)dal(N * 4 * K), *br = (float*)dal(N * 4 * K);
  k_rand_bf16<<<(N * D * K + 255) / 256, 256>>>(bx, (size_t)N * D * K, 2.f, 1); k_rand_bf16<<<(N * D * K + 255) / 256, 256>>>(bdy, (size_t)N * D * K, 1.f, 2);
  fillf(br, N * K, 1.f); fillf(bm, N * K, 0.f);
  int it = 0;
  float ms = time_ms([&] { int k = (it++) % K; launch_grid([=] __device__(int t, char*) { ln_fwd_task<D>(t, by + k * (size_t)N * D, bm + k * N, br + k * N, bx + k * (size_t)N * D, nullptr, nullptr, gamma, N); }, nblk, 0); }, 64);
  printf("[speed] ln_fwd         %8.1f us  %7.1f GB/s (rd+wr %.1f MB, rotating %d sets)\n", ms * 1e3, (2.0 * RB + 8.0 * N) / ms / 1e6, 2.0 * RB / 1e6, K);
  it = 0;
  ms = time_ms([&] { int k = (it++) % K; launch_grid([=] __device__(int t, char* sm) { ln_bwd_task<D>(t, bdx + k * (size_t)N * D, part, bdy + k * (size_t)N * D, bx + k * (size_t)N * D, bm + k * N, br + k * N, gamma, nullptr, N, sm); }, nblk, ln_bwd_smem_bytes<D>()); }, 64);
  printf("[speed] ln_bwd         %8.1f us  %7.1f GB/s (rd 2 + wr 1 act tensors)\n", ms * 1e3, (3.0 * RB + 8.0 * N + 4.0 * nblk * D) / ms / 1e6);
  it = 0;
  ms = time_ms([&] { int k = (it++) % K; launch_grid([=] __device__(int t, char*) { ln_fwd_task<D>(t, by + k * (size_t)N * D, bm + k * N, br + k * N, bx + k * (size_t)N * D, bdy + k * (size_t)N * D, bdx + k * (size_t)N * D, gamma, N); }, nblk, 0); }, 64);
  printf("[speed] add+ln_fwd     %8.1f us  %7.1f GB/s (rd 2 + wr 2 act tensors)\n", ms * 1e3, 4.0 * RB / ms / 1e6);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { ln_dgamma_reduce_task<D>(t, dg, part, nblk); }, (D + 255) / 256, 0); }, 50);
  printf("[speed] ln_dgamma_red  %8.1f us\n", ms * 1e3);
  cudaFree(bx); cudaFree(by); cudaFree(bdy); cudaFree(bdx);
}

static void test_ce() {
  size_t LB = (size_t)NCE * V * 2;
  bf16* lg = (bf16*)up("ce_logits.bin"); bf16* lg0 = (bf16*)dal(LB); CK(cudaMemcpy(lg0, lg, LB, cudaMemcpyDeviceToDevice));
  int* tgt = (int*)up("ce_tgt.bin"); float *loss = (float*)dal(N * 4), *ml = (float*)dal(4);
  float inv = 1.f / NCE; std::vector<char> ref[3];
  for (int mode = 0; mode < 2; mode++) {
    CK(cudaMemcpy(lg, lg0, LB, cudaMemcpyDeviceToDevice));
    launch(mode, [=] __device__(int t, char* sm) { ce_task<V>(t, lg, tgt, loss, NCE, inv, sm); }, NCE / CE_ROWS, ce_smem_bytes<>());
    launch(mode, [=] __device__(int t, char* sm) { ce_mean_task(ml, loss, NCE, sm); }, 1, 64);
    CK(cudaDeviceSynchronize());
    std::vector<char> o[3] = {down(lg, LB), down(loss, NCE * 4), down(ml, 4)};
    if (mode == 0) { for (int i = 0; i < 3; i++) ref[i] = o[i]; dump("ce_dlogits", lg, LB); dump("ce_loss", loss, NCE * 4); dump("ce_mean", ml, 4); }
    else verdict("cross-entropy", same(o[0], ref[0]) && same(o[1], ref[1]) && same(o[2], ref[2]));
  }
  // full-size speed: N=4096 rows
  size_t FB = (size_t)N * V * 2; bf16 *big = (bf16*)dal(FB), *big0 = (bf16*)dal(FB);
  k_rand_bf16<<<(N * (size_t)V + 255) / 256, 256>>>(big0, (size_t)N * V, 6.f, 7);
  int* tg = (int*)dal(N * 4); std::vector<int> th(N); for (int i = 0; i < N; i++) th[i] = (i * 7919) % V; CK(cudaMemcpy(tg, th.data(), N * 4, cudaMemcpyHostToDevice));
  float best = 1e9, sum = 0; int reps = 5;
  for (int r = 0; r < reps; r++) {
    CK(cudaMemcpy(big, big0, FB, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
    Ev e; cudaEventRecord(e.a);
    launch_grid([=] __device__(int t, char* sm) { ce_task<V>(t, big, tg, loss, N, 1.f / N, sm); }, N / CE_ROWS, ce_smem_bytes<>());
    cudaEventRecord(e.b); cudaEventSynchronize(e.b); float ms; cudaEventElapsedTime(&ms, e.a, e.b); best = std::min(best, ms); sum += ms;
  }
  CK(cudaGetLastError());
  printf("[speed] ce (N=4096,V=100352) best %.3f ms avg %.3f ms -> %.0f GB/s best (min traffic rd+wr = %.2f GB; %.0f GB/s avg)\n", best, sum / reps, 2.0 * FB / best / 1e6, 2.0 * FB / 1e9, 2.0 * FB / (sum / reps) / 1e6);
  if (getenv("EW_SWEEP")) {
#define SW(C) { float b2 = 1e9; for (int r = 0; r < 4; r++) { CK(cudaMemcpy(big, big0, FB, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize()); Ev e; cudaEventRecord(e.a); \
    launch_grid([=] __device__(int t, char* sm) { ce_task<V, C>(t, big, tg, loss, N, 1.f / N, sm); }, N / CE_ROWS, ce_smem_bytes<C>()); \
    cudaEventRecord(e.b); cudaEventSynchronize(e.b); float ms; cudaEventElapsedTime(&ms, e.a, e.b); b2 = std::min(b2, ms);} printf("sweep CACHE=%d: %.3f ms %.0f GB/s\n", C, b2, 2.0 * FB / b2 / 1e6); }
    SW(0) SW(4) SW(8) SW(12) SW(16) SW(24)
  }
  cudaFree(big); cudaFree(big0);
}

static void test_adam() {
  const size_t n = 3 * 65536 + 1000; const int nnd = 3;
  int2 nd[nnd] = {{1000, 1768}, {70000, 70768}, {197000, 197608}};
  int2* dnd = (int2*)dal(sizeof(nd)); CK(cudaMemcpy(dnd, nd, sizeof(nd), cudaMemcpyHostToDevice));
  float* W0 = (float*)up("ad_w0.bin"); float* G = (float*)dal(n * 4);
  float *W = (float*)dal(n * 4), *M = (float*)dal(n * 4), *Vv = (float*)dal(n * 4); bf16* Wb = (bf16*)dal(n * 2);
  float *part = (float*)dal(64 * 4), *cs = (float*)dal(8);
  int nt = (n + OPT_CHUNK - 1) / OPT_CHUNK;
  std::vector<char> ref[4], cref[2];
  for (int mode = 0; mode < 2; mode++) {
    CK(cudaMemcpy(W, W0, n * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemset(M, 0, n * 4)); CK(cudaMemset(Vv, 0, n * 4));
    for (int s = 1; s <= 3; s++) {
      char nm[32]; snprintf(nm, 32, "ad_g%d.bin", s - 1); static std::vector<char> hg; hg = rd(nm);
      CK(cudaMemcpy(G, hg.data(), n * 4, cudaMemcpyHostToDevice));
      AdamHP hp{1e-3f, 0.9f, 0.95f, 1e-8f, 0.1f, s};
      launch(mode, [=] __device__(int t, char*) { adamw_task<false>(t, W, G, M, Vv, Wb, n, hp, dnd, nnd, nullptr); }, nt, 0);
    }
    // zero-grad (fused variant) check: run zero_task and verify
    launch(mode, [=] __device__(int t, char*) { zero_task(t, G, n); }, nt, 0);
    CK(cudaDeviceSynchronize());
    std::vector<char> o[4] = {down(W, n * 4), down(M, n * 4), down(Vv, n * 4), down(Wb, n * 2)};
    auto g = down(G, n * 4); bool z = true; for (size_t i = 0; i < n * 4; i++) z &= g[i] == 0;
    if (mode == 0) { for (int i = 0; i < 4; i++) ref[i] = o[i]; dump("ad_w", W, n * 4); dump("ad_wbf", Wb, n * 2); dump("ad_m", M, n * 4); dump("ad_v", Vv, n * 4); printf("[check] zero_task %s\n", z ? "PASS" : "FAIL"); if (!z) g_all_ok = false; }
    else verdict("adamw 3 steps", same(o[0], ref[0]) && same(o[1], ref[1]) && same(o[2], ref[2]) && same(o[3], ref[3]));
    // clipping: 1 step from W0, G0, max_norm 1.0
    {
      std::vector<char> hg = rd("ad_g0.bin"); CK(cudaMemcpy(G, hg.data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(W, W0, n * 4, cudaMemcpyDeviceToDevice)); CK(cudaMemset(M, 0, n * 4)); CK(cudaMemset(Vv, 0, n * 4));
      launch(mode, [=] __device__(int t, char* sm) { grad_sumsq_task(t, G, n, part, sm); }, nt, 64);
      launch(mode, [=] __device__(int t, char* sm) { grad_clip_scale_task(part, nt, 1.0f, cs, sm); }, 1, 64);
      AdamHP hp{1e-3f, 0.9f, 0.95f, 1e-8f, 0.1f, 1};
      launch(mode, [=] __device__(int t, char*) { adamw_task<true>(t, W, G, M, Vv, Wb, n, hp, dnd, nnd, cs); }, nt, 0);
      CK(cudaDeviceSynchronize());
      auto w = down(W, n * 4), c = down(cs, 8), gz = down(G, n * 4);
      bool z2 = true; for (size_t i = 0; i < n * 4; i++) z2 &= gz[i] == 0;
      if (mode == 0) { cref[0] = w; cref[1] = c; dump("ad_wclip", W, n * 4); dump("ad_clipscale", cs, 8); printf("[check] fused ZERO_G %s\n", z2 ? "PASS" : "FAIL"); if (!z2) g_all_ok = false; }
      else verdict("adamw clip + fused zero", same(w, cref[0]) && same(c, cref[1]));
    }
  }
  // speed: 162M params
  size_t P = 162000000; P = (P / OPT_CHUNK) * OPT_CHUNK + OPT_CHUNK; int ntt = P / OPT_CHUNK;
  float *bW = (float*)dal(P * 4), *bG = (float*)dal(P * 4), *bM = (float*)dal(P * 4), *bV = (float*)dal(P * 4); bf16* bB = (bf16*)dal(P * 2);
  k_rand_f32<<<(P + 255) / 256, 256>>>(bW, P, 0.04f, 1); k_rand_f32<<<(P + 255) / 256, 256>>>(bG, P, 0.01f, 2); fillf(bM, P, 0.f); fillf(bV, P, 1e-6f);
  AdamHP hp{1e-4f, 0.9f, 0.95f, 1e-8f, 0.1f, 5};
  float ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { adamw_task<false>(t, bW, bG, bM, bV, bB, P, hp, dnd, nnd, nullptr); }, ntt, 0); }, 10);
  double by = (double)P * 30;
  printf("[speed] adamw          %8.3f ms  %7.1f GB/s (30 B/param: rd 16 wr 14; %zu params, %d tasks)\n", ms, by / ms / 1e6, P, ntt);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { adamw_task<true>(t, bW, bG, bM, bV, bB, P, hp, dnd, nnd, nullptr); }, ntt, 0); }, 10);
  printf("[speed] adamw+zeroG    %8.3f ms  %7.1f GB/s (34 B/param)\n", ms, P * 34.0 / ms / 1e6);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char*) { zero_task(t, bG, P); }, ntt, 0); }, 10);
  printf("[speed] zero_task      %8.3f ms  %7.1f GB/s\n", ms, P * 4.0 / ms / 1e6);
  float* bp = (float*)dal(4096 * 4);
  ms = time_ms([&] { launch_grid([=] __device__(int t, char* sm) { grad_sumsq_task(t, bG, P, bp, sm); }, ntt, 64); }, 10);
  printf("[speed] grad_sumsq     %8.3f ms  %7.1f GB/s\n", ms, P * 4.0 / ms / 1e6);
  cudaFree(bW); cudaFree(bG); cudaFree(bM); cudaFree(bV); cudaFree(bB);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: ew_test DIR\n"); return 1; }
  DIR = argv[1];
  size_t fr, tot; cudaMemGetInfo(&fr, &tot); printf("GPU free %.1f GB / %.1f GB\n", fr / 1e9, tot / 1e9);
  test_embed(); test_ln(); test_ce(); test_adam();
  CK(cudaGetLastError());
  printf("%s bit-identical grid-vs-persistent: %s\n", "OVERALL", g_all_ok ? "PASS" : "FAIL");
  return g_all_ok ? 0 : 2;
}
