// test_decode.cu : correctness (logits dump for ref_decode.py), grid-vs-persistent determinism, performance.
// usage: ./decode_test <dir> [perf_only]    (dir contains weights.bin, tokens.bin from `ref_decode.py gen`)
#include "../decode_wide.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unistd.h>
using namespace decode;
typedef ModelF C;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2);} } while (0)

static std::vector<char> readfile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) { printf("cannot open %s\n", p.c_str()); exit(2); }
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> b(n); if (fread(b.data(), 1, n, f) != n) exit(2); fclose(f); return b;
}

struct Run {
  Params<C> P; State<C>* st;
  std::vector<int> toks;
};
static int g_nsm = 108;

static void set_forced(Run& r, int nforced, int* d_forced) {
  r.P.forced = d_forced; r.P.nforced = nforced;
}
template <class T> static std::vector<T> d2h(const T* d, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static std::vector<int> get_tokens(Run& r, int n) {
  int* ot; CK(cudaMemcpy(&ot, &r.st->out_tokens, sizeof(ot), cudaMemcpyDeviceToHost));
  return d2h(ot, n);
}
static std::vector<float> get_logits(Run& r) {
  float* lg; CK(cudaMemcpy(&lg, &r.st->logits, sizeof(lg), cudaMemcpyDeviceToHost));
  return d2h(lg, C::V);
}
static std::vector<bf16> get_kv(Run& r) {
  bf16* kv; CK(cudaMemcpy(&kv, &r.st->kv, sizeof(kv), cudaMemcpyDeviceToHost));
  return d2h(kv, (size_t)C::L * 2 * C::H * C::T * 64);
}
static void begin(Run& r, int first_tok, int start_pos = 0) {
  begin_context<C>(r.P, first_tok, start_pos);
}

static float run_grid(Run& r, int nsteps, bool graph) {
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float ms = 0;
  if (graph) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    launch_step_grid<C>(r.P, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0, 0, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));  // warm (this consumes step 0; callers account for it via begin())
    // re-begin is the caller's job; here we just time nsteps replays
    cudaEventRecord(a, s);
    for (int i = 0; i < nsteps; i++) CK(cudaGraphLaunch(ge, s));
    cudaEventRecord(b, s); CK(cudaStreamSynchronize(s));
    cudaEventElapsedTime(&ms, a, b);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  } else {
    cudaEventRecord(a, s);
    for (int i = 0; i < nsteps; i++) launch_step_grid<C>(r.P, s);
    cudaEventRecord(b, s); CK(cudaStreamSynchronize(s));
    cudaEventElapsedTime(&ms, a, b);
  }
  CK(cudaGetLastError());
  cudaStreamDestroy(s);
  return ms;
}

static unsigned *d_next, *d_done, *d_tk; static unsigned long long* d_ts = nullptr;
static float run_persistent(Run& r, int nsteps, int sm_limit, int per_sm, int scramble, int prefetch) {
  size_t nc = (size_t)nsteps * C::PHASES;
  CK(cudaMemset(d_next, 0, nc * 4)); CK(cudaMemset(d_done, 0, nc * 4)); CK(cudaMemset(d_tk, 0, 4 * 1100));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int nactive = sm_limit * per_sm;
  cudaEventRecord(a);
  k_persistent<C><<<2 * g_nsm, NTHREADS, smem_bytes<C>()>>>(r.P, nsteps, d_next, d_done, sm_limit, per_sm, d_tk, scramble, prefetch, nactive, d_ts);
  cudaEventRecord(b); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
  float ms; cudaEventElapsedTime(&ms, a, b); return ms;
}

static bool nvidia_idle() {
  FILE* p = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", "r");
  int u = 0; if (p) { if (fscanf(p, "%d", &u) != 1) u = 0; pclose(p); } return u < 5;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  bool perf_only = argc > 2;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); g_nsm = prop.multiProcessorCount;
  printf("device %s, %d SMs\n", prop.name, g_nsm);
  int occ = 0; CK(cudaFuncSetAttribute(k_persistent<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<C>()));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_persistent<C>, NTHREADS, smem_bytes<C>()));
  printf("smem/task %zu B, blocks/SM %d, phases/step %d\n", smem_bytes<C>(), occ, C::PHASES);
  if (occ < 2) { printf("need 2 blocks/SM\n"); return 2; }
  // ---- load weights
  auto wb = readfile(dir + "/weights.bin"); auto tb = readfile(dir + "/tokens.bin");
  std::vector<int> toks(tb.size() / 4); memcpy(toks.data(), tb.data(), tb.size());
  int ntok = (int)toks.size();
  char* dw; CK(cudaMalloc(&dw, wb.size())); CK(cudaMemcpy(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice));
  Weights w; size_t o = 0; auto take = [&](size_t n) { const bf16* p = (const bf16*)(dw + o); o += n * 2; return p; };
  w.wte = take((size_t)C::V * C::D); w.wpe = take((size_t)C::T * C::D); w.gf = take(C::D);
  for (int l = 0; l < C::L; l++) {
    w.g1[l] = take(C::D); w.g2[l] = take(C::D); w.wqkv[l] = take((size_t)C::D * 3 * C::D); w.wo[l] = take((size_t)C::D * C::D);
    w.wup[l] = take((size_t)C::D * C::FF); w.wdn[l] = take((size_t)C::FF * C::D);
  }
  if (o != wb.size()) { printf("weights.bin size mismatch\n"); return 2; }
  const int OUTCAP = 1024;
  Run r; r.P.w = w; r.st = alloc_state<C>(OUTCAP); r.P.st = r.st; r.P.temperature = 0.f; r.P.seed = 0; r.P.valid_vocab = 100277;
  r.P.auto_reset = 0; r.P.nforced = 0; r.P.forced = nullptr;
  int* d_forced; CK(cudaMalloc(&d_forced, 4 * 1024)); CK(cudaMemcpy(d_forced, toks.data(), 4 * ntok, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_next, 4 * 200000)); CK(cudaMalloc(&d_done, 4 * 200000)); CK(cudaMalloc(&d_tk, 4 * 1100));
  CK(cudaFuncSetAttribute(k_phase<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<C>()));
  bool all_ok = true;

  if (!perf_only) {
    // ---- 1. teacher-forced prefill-by-decode, per-step logits dump (plain grid)
    set_forced(r, ntok, d_forced); begin(r, toks[0]);
    std::vector<float> all((size_t)ntok * C::V);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int i = 0; i < ntok; i++) {
      if (getenv("DBG")) { for (int ph = 0; ph < C::PHASES; ph++) { k_phase<C><<<phase_tasks<C>(ph), NTHREADS, smem_bytes<C>(), s>>>(r.P, ph); cudaError_t e = cudaStreamSynchronize(s); printf("step %d ph %d: %s\n", i, ph, cudaGetErrorString(e)); fflush(stdout);} } else launch_step_grid<C>(r.P, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
      auto lg = get_logits(r); memcpy(&all[(size_t)i * C::V], lg.data(), C::V * 4);
    }
    FILE* f = fopen((dir + "/logits_cu.bin").c_str(), "wb"); fwrite(all.data(), 4, all.size(), f); fclose(f);
    printf("wrote %d steps of logits\n", ntok);

    // ---- 2. determinism: grid vs persistent (ordered / scrambled / different block counts), sampling with temperature
    const int NS = 96, NP = 8;  // 8 prompt tokens then free-running sampling
    set_forced(r, NP, d_forced); r.P.temperature = 0.9f; r.P.seed = 0xC0FFEEull;
    begin(r, toks[0]);
    run_grid(r, NS - 1, false);  // NS-1 steps: launch NS-1 steps
    launch_step_grid<C>(r.P, 0); CK(cudaDeviceSynchronize());
    auto tg = get_tokens(r, NS); auto lgg = get_logits(r); auto kvg = get_kv(r);
    printf("grid tokens (first 24):"); for (int i = 0; i < 24; i++) printf(" %d", tg[i]); printf("\n");
    struct V { const char* name; int sm, per, scr, pf; } vars[] = {
      {"persistent 108 SMs x2 ordered", g_nsm, 2, 0, 0}, {"persistent 108x2 scrambled+delays", g_nsm, 2, 1, 0},
      {"persistent 17 SMs x2 scrambled", 17, 2, 1, 0}, {"persistent 17 SMs x1 ordered+L2 prefetch", 17, 1, 0, 3},
      {"persistent 5 SMs x1 scrambled", 5, 1, 1, 0}, {"persistent 1 block", 1, 1, 1, 0}};
    for (auto& v : vars) {
      begin(r, toks[0]);
      run_persistent(r, NS, v.sm, v.per, v.scr, v.pf);
      auto tp = get_tokens(r, NS); auto lgp = get_logits(r); auto kvp = get_kv(r);
      bool ok = tp == tg && memcmp(lgp.data(), lgg.data(), C::V * 4) == 0 && memcmp(kvp.data(), kvg.data(), kvp.size() * 2) == 0;
      printf("  %-44s tokens+logits+KV bitwise identical to grid: %s\n", v.name, ok ? "yes" : "NO");
      all_ok &= ok;
    }
    // sampling sanity: distinct seed gives a different stream; greedy runs
    r.P.seed = 1; begin(r, toks[0]); run_persistent(r, NS, g_nsm, 2, 0, 0);
    auto t2 = get_tokens(r, NS); printf("  different seed -> different stream: %s\n", t2 != tg ? "yes" : "NO"); all_ok &= (t2 != tg);
    printf("DETERMINISM %s\n", all_ok ? "PASS" : "FAIL");
  }

  // ---- 3. performance
  int tries = 0; while (!nvidia_idle() && tries++ < 5) { printf("GPU busy, retrying...\n"); usleep(2000000); }
  printf("GPU idle at start: %s\n", nvidia_idle() ? "yes" : "no (timings noisy)");
  const double BYTES = (double)((size_t)C::L * ((size_t)C::D * 3 * C::D + (size_t)C::D * C::D + 2ull * C::D * C::FF) + (size_t)C::V * C::D) * 2.0;
  printf("weight bytes per token: %.1f MB\n", BYTES / 1e6);
  r.P.temperature = 0.f; r.P.nforced = 0; r.P.forced = nullptr;
  auto report = [&](const char* name, float ms, int steps) {
    double us = ms * 1e3 / steps;
    printf("  %-52s %7.1f us/token  %8.0f tok/s  %6.0f GB/s\n", name, us, 1e6 / us, BYTES / us / 1e3);
  };
  const bool quick = getenv("QUICK") != nullptr;
  for (int startpos : {0, 384}) {
    if (quick && startpos != (getenv("QPOS") ? atoi(getenv("QPOS")) : 0)) continue;
    int NSTEP = 128;
    printf("-- context positions %d..%d (KV read grows with pos)\n", startpos, startpos + NSTEP - 1);
    begin(r, toks[0], startpos); run_grid(r, 4, false);   // warm
    float ms = 0;
    if (!quick) { begin(r, toks[0], startpos); ms = run_grid(r, NSTEP, false); report("grid, 38 launches/step (stream)", ms, NSTEP); }
    begin(r, toks[0], startpos); ms = run_grid(r, NSTEP - 1, true); report("grid, CUDA graph per step", ms, NSTEP - 1);
    for (int pf : {0, 1, 3}) {
      if (quick && pf == 3) continue;
      struct { int sm, per; } cfgs[] = {{g_nsm, 2}, {g_nsm, 1}, {34, 2}, {17, 2}, {17, 1}, {24, 2}};
      for (auto& c : cfgs) {
        char nm[100]; snprintf(nm, 100, "persistent %3d SMs x %d blocks%s", c.sm, c.per, pf == 0 ? "" : (pf == 1 ? " + prefetch(next task)" : " + prefetch(task+phase)"));
        begin(r, toks[0], startpos); run_persistent(r, 4, c.sm, c.per, 0, pf);  // warm
        float best = 1e30f;
        for (int rep = 0; rep < 3; rep++) { begin(r, toks[0], startpos); best = std::min(best, run_persistent(r, NSTEP, c.sm, c.per, 0, pf)); }
        report(nm, best, NSTEP);
      }
    }
  }
  {  // per-phase timeline (block rank 0 timestamps at each phase completion), 108x2 blocks, pos 0..63
    int NSTEP = 64; CK(cudaMalloc(&d_ts, 8 * NSTEP * C::PHASES)); CK(cudaMemset(d_ts, 0, 8 * NSTEP * C::PHASES));
    int tsm = getenv("TL_SM") ? atoi(getenv("TL_SM")) : g_nsm; begin(r, toks[0], 256); run_persistent(r, NSTEP, tsm, getenv("TL_PER") ? atoi(getenv("TL_PER")) : 2, 0, 0);
    auto ts = d2h(d_ts, (size_t)NSTEP * C::PHASES);
    double acc[5] = {0}; int cnt[5] = {0};
    for (int st = 4; st < NSTEP; st++) for (int ph = 0; ph < C::PHASES; ph++) {
      size_t gp = (size_t)st * C::PHASES + ph; double dt = (double)(ts[gp] - ts[gp - 1]) / 1e3;
      int k = ph < 3 * C::L ? ph % 3 : (ph == 3 * C::L ? 3 : 4); acc[k] += dt; cnt[k]++;
    }
    const char* nm[] = {"QKV+LN1", "ATT+WO", "UP+DOWN", "HEAD(all)", "SAMPLE"};
    printf("-- phase timeline (us per phase instance, %d SMs x2 blocks, pos~256+); weight bytes / time\n", tsm);
    double by[] = {C::D * 3.0 * C::D * 2, (double)C::D * C::D * 2, 2.0 * C::D * C::FF * 2, (double)C::V * C::D * 2, 0};
    for (int k = 0; k < 5; k++) printf("  %-12s %7.2f us  (x%d per step)  %.0f GB/s\n", nm[k], acc[k] / cnt[k], k < 3 ? C::L : 1, by[k] > 0 ? by[k] / (acc[k] / cnt[k]) / 1e3 : 0.0);
    double tot = 0; for (int k = 0; k < 3; k++) tot += acc[k] / cnt[k] * C::L; tot += acc[3] / cnt[3] + acc[4] / cnt[4];
    printf("  sum %.1f us\n", tot);
  }
  printf("%s\n", all_ok ? "ALL-OK" : "SOME-FAIL");
  return all_ok ? 0 : 1;
}
