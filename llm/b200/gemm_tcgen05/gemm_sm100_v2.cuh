// gemm_sm100_v2.cuh -- round-2 Blackwell GEMM backend for the open-flapping-airplanes megakernel (TMA + tcgen05.mma + TMEM), B300-verified descriptors from v1.
//
// Same task contract as gemm_sm100.cuh / gemm.cuh: one task = one output tile, executed by ALL 256 threads of a block via tile<LAYOUT,EPI>(p, tile_m, tile_n, smem),
// __syncthreads() at the end, no blockIdx/gridDim, deterministic reduction order over K.  What changed vs v1:
//   * TASK TILE = 128 x TILE_N with TILE_N = G2_SUBN * G2_NSUB (compile time; tile_n indexes TILE_N-wide column blocks).  G2_SUBN = N of one tcgen05.mma (128 or 256),
//     G2_NSUB = number of sequential sub-tiles inside one task.  Every dimension of the S/F/M models is a multiple of 256, so TILE_N<=256 divides all N.
//   * TMEM accumulator is double buffered across sub-tiles (G2_NSUB>=2): the epilogue of sub-tile j (warps 2..7) overlaps the MMAs of sub-tile j+1 (warp 1);
//     the last sub-tile's epilogue uses all 8 warps.  Producer (warp 0 lane 0), MMA issuer (warp 1 lane 0) and epilogue warps run concurrently inside the task:
//     warp specialisation IS compatible with the megakernel contract, because roles are assigned by warp id inside one task and all 256 threads still enter and leave together.
//   * Epilogue stores 16 B per thread (8 bf16 / 4 fp32) instead of 4 B (8x fewer store transactions); two tcgen05.ld in flight before the wait.
//   * Feature guard: G2_FORCE_TC (set by the build line) or the FEAT/family macros; otherwise falls back to mma.sync (gemm::gemm_tile, requires TILE_N==128).
// Mirrored CUTLASS sources (structure, not copied): include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp (producer/MMA/epilogue warp roles, acc full/empty
// pipeline), include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp (TMEM alloc, role dispatch), include/cute/arch/mma_sm100_umma.hpp, mma_sm100_desc.hpp.
// CUTLASS's 2-CTA (cta_group::2) tiles need a cluster launch; the megakernel is a plain persistent grid with a ticket scheduler, so 2-CTA is NOT used (see B300_ROUND2.md).
#pragma once
#include "../../gemm/gemm.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Two compile-time configs, chosen PER GEMM PHASE at run time by ArgsTC::wide (the host picks "wide" when the GEMM has enough tiles to keep all SM slots busy):
//   narrow (wide=0): G2_SUBN x G2_NSUB x G2_STAGES, default 128 x 1 x 3  -> task tile 128x128 (more, smaller tasks: small-N / weight-gradient GEMMs)
//   wide   (wide=1): G2_WSUBN x G2_WNSUB x G2_WSTAGES, default 256 x 1 x 2 -> task tile 128x256 (higher arithmetic intensity: L2->SM bandwidth is the roofline)
#ifndef G2_SUBN
#define G2_SUBN 128
#endif
#ifndef G2_NSUB
#define G2_NSUB 4
#endif
#ifndef G2_STAGES
#define G2_STAGES 6
#endif
#ifndef G2_WSUBN
#define G2_WSUBN 256
#endif
#ifndef G2_WNSUB
#define G2_WNSUB 4
#endif
#ifndef G2_WSTAGES
#define G2_WSTAGES 4
#endif
#ifndef GEMM100_LAYOUT_MASK
#define GEMM100_LAYOUT_MASK 7
#endif
#if defined(G2_FORCE_TC) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC)
#define G2_TC 1
#endif

namespace gemm {
namespace sm100v2 {

constexpr int BK2 = 64;
template <int SUBN_, int NSUB_, int STAGES_> struct Cfg {
  static constexpr int SUBN = SUBN_, NSUB = NSUB_, STAGES = STAGES_;
  static constexpr int TILE_N = SUBN * NSUB;
  static constexpr int A_BYTES = 128 * BK2 * 2, B_BYTES = SUBN * BK2 * 2, STAGE_BYTES = A_BYTES + B_BYTES;
  static constexpr int TMEM_COLS = (NSUB > 1 ? 2 : 1) * SUBN;
  static constexpr int META_BYTES = 512, EPI_BYTES = 8 * 4096, ALIGN_SLACK = 1024;   // barriers/slot | epilogue staging (8 warps x 32 rows x 128 B) | slack for the 1024-B alignment
  static constexpr int SMEM = STAGES * STAGE_BYTES + META_BYTES + EPI_BYTES + ALIGN_SLACK;
  static_assert(SUBN == 128 || SUBN == 256, "SUBN");
  static_assert(TMEM_COLS == 128 || TMEM_COLS == 256 || TMEM_COLS == 512, "TMEM cols must be a power of two");
};
using CfgN = Cfg<G2_SUBN, G2_NSUB, G2_STAGES>;
using CfgW = Cfg<G2_WSUBN, G2_WNSUB, G2_WSTAGES>;
constexpr int TILE_N = CfgN::TILE_N;                                   // narrow task width (also the width of backends 0/1)
__host__ __device__ constexpr int smem_bytes() { return CfgN::SMEM > CfgW::SMEM ? CfgN::SMEM : CfgW::SMEM; }

struct ArgsTC : Args {
  const CUtensorMap* tmA;
  const CUtensorMap* tmB;
  int wide;                 // 0: narrow config, 1: wide config (set by prepare)
  int nsub;                 // tiles per task (1..NSUB of the chosen config): a task = nsub consecutive 128 x SUBN tiles (flattened, tile_m fastest), pipelined with a double-buffered TMEM accumulator
};
__host__ __device__ constexpr int cfg_subn(int wide) { return wide ? CfgW::SUBN : CfgN::SUBN; }
__host__ __device__ constexpr int cfg_nsub(int wide) { return wide ? CfgW::NSUB : CfgN::NSUB; }
__host__ inline long ntiles(int M, int N, int wide) { return (long)(M / 128) * (N / cfg_subn(wide)); }
__host__ inline int ntasks(const ArgsTC& p) { const long t = ntiles(p.M, p.N, p.wide); return (int)((t + p.nsub - 1) / p.nsub); }
// host policy: wide config whenever N % 256 == 0 (else narrow); nsub = ceil(tiles/slots) clamped to the config's NSUB (one wave of ~slots tasks, each a pipelined run of nsub tiles)
__host__ inline void choose(int M, int N, int slots, int* wide, int* nsub, int force_wide = -1, int force_nsub = 0) {
  int w = (N % CfgW::SUBN == 0) ? 1 : 0;      // wide (128x256 MMA) whenever N allows: end-to-end it beat the narrow config for every shape of S (13.34 vs 14.54 ms/step with narrow-only, 13.52 with the 0.6*slots threshold)
  if (force_wide >= 0) w = force_wide;
  long t = ntiles(M, N, w); int ns = (int)((t + slots - 1) / slots); if (ns < 1) ns = 1; if (ns > cfg_nsub(w)) ns = cfg_nsub(w);
  if (force_nsub > 0) ns = force_nsub < cfg_nsub(w) ? force_nsub : cfg_nsub(w);
  *wide = w; *nsub = ns;
}

__host__ inline bool args_ok(const ArgsTC& p) {
  return p.M % 128 == 0 && p.N % cfg_subn(p.wide) == 0 && p.nsub >= 1 && p.nsub <= cfg_nsub(p.wide) && p.K % BK2 == 0 && p.lda % 8 == 0 && p.ldb % 8 == 0 && p.ldc % 8 == 0 && p.ldaux % 8 == 0 &&
         ((uintptr_t)p.A % 16 == 0) && ((uintptr_t)p.B % 16 == 0) && ((uintptr_t)p.C % 16 == 0) && (p.aux == nullptr || (uintptr_t)p.aux % 16 == 0);
}

// ------------------------------------------------------------------------------------------------ host: TMA descriptors
inline const CUtensorMap* make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t ld_elems, uint32_t box_inner, uint32_t box_outer) {
  typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                               const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  { int dev = 0, major = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 10) return nullptr; }
  static EncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 13000
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
#endif
    if (!fn) { fprintf(stderr, "gemm_sm100v2: cuTensorMapEncodeTiled not found\n"); exit(1); }
    enc = (EncodeFn)fn;
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {ld_elems * 2};
  cuuint32_t box[2] = {box_inner, box_outer}, es[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "gemm_sm100v2: cuTensorMapEncodeTiled failed (%d)\n", (int)r); exit(1); }
  CUtensorMap* d; cudaMalloc(&d, sizeof(CUtensorMap) * 2);
  cudaMemcpy(d, &m, sizeof m, cudaMemcpyHostToDevice);
  return d;
}
//   A K-major (NN, NT): [M][K] -> {K, M} box {64,128}      A MN-major (TN): [K][M] -> {M, K} box {64,64} (two boxes per 128 rows)
//   B K-major (NT):     [N][K] -> {K, N} box {64,SUBN}     B MN-major (NN, TN): [K][N] -> {N, K} box {64,64} (SUBN/64 boxes per sub-tile)
inline void prepare(ArgsTC& p, int layout, int wide = 0, int nsub = 1) {
  p.wide = wide; p.nsub = nsub;
  const int SUBN = wide ? CfgW::SUBN : CfgN::SUBN;
  const bool akc = layout != TN, bkc = layout == NT;
  p.tmA = akc ? make_map(p.A, p.K, p.M, p.lda, 64, 128) : make_map(p.A, p.M, p.K, p.lda, 64, 64);
  p.tmB = bkc ? make_map(p.B, p.K, p.N, p.ldb, 64, SUBN) : make_map(p.B, p.N, p.K, p.ldb, 64, 64);
}

// ------------------------------------------------------------------------------------------------ device
#if defined(G2_TC)
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory"); }
#ifdef G2_DEBUG
__device__ int g2_tcnt[1024];
// debug build: a wait that does not complete within ~3 s (clock64) prints who is stuck and traps, so a hang becomes a message
__device__ __noinline__ void g2_hang(int tag, uint32_t addr, uint32_t par, int a, int b, int c) {
  if ((threadIdx.x & 31) != 0 && tag != 4) return;   // one line per warp
  if (tag == 4 && (threadIdx.x & 31) != 1) return;
  unsigned long long st0, st1, st2; asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(st0) : "r"(addr)); asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(st1) : "r"(addr - 8)); asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(st2) : "r"(addr - 8 * 7));
  unsigned long long w[10];
  for (int i = 0; i < 10; i++) asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(w[i]) : "r"(addr + 8 * (i - 6)));
  printf("G2 HANG tiles_started=%d blk=%d tid=%d tag=%d addr=%u parity=%u a=%#x b=%d c=%#x | bars[-6..+3]= %016llx %016llx %016llx %016llx %016llx %016llx [%016llx] %016llx %016llx %016llx\n",
         g2_tcnt[blockIdx.x & 1023], (int)blockIdx.x, (int)threadIdx.x, tag, addr, par, a, b, c, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]);
  __trap();
}
#endif
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity, int tag = 0, int x = 0, int y = 0, int z = 0) {
  uint32_t ok;
#ifdef G2_DEBUG
  const long long t0 = clock64();
#endif
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
#ifdef G2_DEBUG
    if (!ok && clock64() - t0 > 6000000000ll) g2_hang(tag, a, parity, x, y, z);
#endif
  } while (!ok);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t mbar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               ::"r"(dst), "l"(tm), "r"(mbar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc(uint32_t slot_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void umma_bf16(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void tmem_ld32_issue(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]),
                 "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
                 "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ uint64_t smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(lbo_bytes >> 4) << 16) | ((uint64_t)(sbo_bytes >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
__host__ __device__ constexpr uint32_t make_idesc(bool a_mn, bool b_mn, int SUBN) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((a_mn ? 1u : 0u) << 15) | ((b_mn ? 1u : 0u) << 16) | ((uint32_t)(SUBN >> 3) << 17) | ((128u >> 4) << 24);
}

// 8 consecutive columns of one row -> global (16-byte accesses). v = fp32 accumulators.
template <int EPI>
__device__ __forceinline__ void epi8(const ArgsTC& p, int row, int col, float (&v)[8], const uint4& l0, const uint4& l1) {   // l0/l1: prefetched aux (RES/DGELU: l0) or C (F32 accumulate: l0,l1)
#ifdef G2_NOEPI
  if (p.accumulate != 0x5eed) return;   // diagnostic build: skip every global store/load of the epilogue (results are garbage)
#endif
  if constexpr (EPI == EPI_F32) {
    float* dst = reinterpret_cast<float*>(p.C) + (size_t)row * p.ldc + col;
    float4 a = make_float4(v[0], v[1], v[2], v[3]), b = make_float4(v[4], v[5], v[6], v[7]);
    if (p.accumulate) {
      const float4 oa = *reinterpret_cast<const float4*>(&l0), ob = *reinterpret_cast<const float4*>(&l1);
      a.x += oa.x; a.y += oa.y; a.z += oa.z; a.w += oa.w; b.x += ob.x; b.y += ob.y; b.z += ob.z; b.w += ob.w;
    }
    *reinterpret_cast<float4*>(dst) = a; *reinterpret_cast<float4*>(dst + 4) = b;
  } else {
    bf16* C = reinterpret_cast<bf16*>(p.C) + (size_t)row * p.ldc + col;
    if constexpr (EPI == EPI_RES) {
      const uint4 rr = l0;
      const __nv_bfloat162* rp = reinterpret_cast<const __nv_bfloat162*>(&rr);
#pragma unroll
      for (int i = 0; i < 4; i++) { v[2 * i] += __bfloat162float(rp[i].x); v[2 * i + 1] += __bfloat162float(rp[i].y); }
    } else if constexpr (EPI == EPI_GELU) {
      uint4 pre; __nv_bfloat162* pp = reinterpret_cast<__nv_bfloat162*>(&pre);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        pp[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
        v[2 * i] = gelu_tanh(__bfloat162float(pp[i].x)); v[2 * i + 1] = gelu_tanh(__bfloat162float(pp[i].y));
      }
      *reinterpret_cast<uint4*>(p.aux + (size_t)row * p.ldaux + col) = pre;
    } else if constexpr (EPI == EPI_DGELU) {
      const uint4 pre = l0;
      const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pre);
#pragma unroll
      for (int i = 0; i < 4; i++) { v[2 * i] *= gelu_tanh_grad(__bfloat162float(pp[i].x)); v[2 * i + 1] *= gelu_tanh_grad(__bfloat162float(pp[i].y)); }
    }
    uint4 o; __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4*>(C) = o;
  }
}

__device__ __forceinline__ void st_shared_v4(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ void ld_shared_v4(uint32_t a, float (&f)[4]) {
  uint32_t x, y, z, w;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(a) : "memory");
  f[0] = __uint_as_float(x); f[1] = __uint_as_float(y); f[2] = __uint_as_float(z); f[3] = __uint_as_float(w);
}
// 32 consecutive columns of 32 rows (this warp's TMEM lane quadrant; thread = one row) -> global through a 4 KB per-warp smem staging tile (row = 128 B, 16-B chunks XOR-swizzled by row&7:
// conflict-free for both the row-per-thread write and the 4-lanes-per-row read-back) so that the global accesses are 64-128 B contiguous per row instead of 16 B per thread-row.
template <int EPI>
__device__ __forceinline__ void epi_chunk(const ArgsTC& p, uint32_t stg, int lane, int row0, int col0, const uint32_t (&r)[32]) {
  // issue all global loads of the epilogue (aux / accumulate-C) first: they are independent of the TMEM data and their latency overlaps the smem transposition
  uint4 L0[4], L1[4];
#pragma unroll
  for (int it = 0; it < 4; it++) {
    L0[it] = make_uint4(0, 0, 0, 0); L1[it] = make_uint4(0, 0, 0, 0);
    const int row = row0 + 8 * it + (lane >> 2), col = col0 + 8 * (lane & 3);
    if constexpr (EPI == EPI_RES || EPI == EPI_DGELU) {
      L0[it] = *reinterpret_cast<const uint4*>(p.aux + (size_t)row * p.ldaux + col);
    } else if constexpr (EPI == EPI_F32) {
      if (p.accumulate) {
        const float* d = reinterpret_cast<const float*>(p.C) + (size_t)row * p.ldc + col;
        L0[it] = *reinterpret_cast<const uint4*>(d); L1[it] = *reinterpret_cast<const uint4*>(d + 4);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < 8; c++) st_shared_v4(stg + lane * 128 + ((c ^ (lane & 7)) << 4), r[4 * c], r[4 * c + 1], r[4 * c + 2], r[4 * c + 3]);
  __syncwarp();
#pragma unroll
  for (int it = 0; it < 4; it++) {
    const int rr = 8 * it + (lane >> 2), c0 = 2 * (lane & 3);
    float v[8], a[4], b[4];
    ld_shared_v4(stg + rr * 128 + ((c0 ^ (rr & 7)) << 4), a);
    ld_shared_v4(stg + rr * 128 + (((c0 + 1) ^ (rr & 7)) << 4), b);
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = a[i]; v[4 + i] = b[i]; }
    epi8<EPI>(p, row0 + rr, col0 + c0 * 4, v, L0[it], L1[it]);
  }
  __syncwarp();
}

// ---- optional persistent TMEM (G2_PERSIST_TMEM): the megakernel allocates all 512 TMEM columns ONCE per block (block_begin) and frees them at the end (block_end);
//      tile_tc then neither allocates nor frees.  Needed because a tcgen05.alloc-containing kernel is limited to 1 CTA/SM anyway (see B300_ROUND2.md) and saves the
//      alloc/dealloc round trip per task.
__device__ __forceinline__ uint32_t& g2_tbase_ref() { __shared__ uint32_t v; return v; }
__device__ __forceinline__ void block_begin() {
  __shared__ uint32_t slot;
  const int warp = threadIdx.x >> 5;
  __syncthreads();
  if (warp == 0) { tmem_alloc((uint32_t)__cvta_generic_to_shared(&slot), 512); tc_fence_before(); }
  __syncthreads();
  tc_fence_after();
  if (threadIdx.x == 0) g2_tbase_ref() = *reinterpret_cast<volatile uint32_t*>(&slot);
  __syncthreads();
}
__device__ __forceinline__ void block_end() {
  __syncthreads();
  tc_fence_before();
  if ((threadIdx.x >> 5) == 0) tmem_dealloc(g2_tbase_ref(), 512);
  __syncthreads();
}

template <int LAYOUT, int EPI, class CF>
__device__ __forceinline__ void tile_tc(const ArgsTC& p, int task, char* smem) {
  constexpr bool A_MN = (LAYOUT == TN), B_MN = (LAYOUT != NT);
  constexpr int SUBN = CF::SUBN, NSUB = CF::NSUB, STAGES = CF::STAGES, TILE_N = CF::TILE_N, STAGE_BYTES = CF::STAGE_BYTES, A_BYTES = CF::A_BYTES, TMEM_COLS = CF::TMEM_COLS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (raw + 1023u) & ~1023u;
  char* gbase = smem + (sbase - raw);
  const uint32_t meta = sbase + STAGES * STAGE_BYTES;
  auto full = [&](int s) { return meta + 8u * s; };
  auto empty = [&](int s) { return meta + 8u * (STAGES + s); };
  auto accf = [&](int b) { return meta + 8u * (2 * STAGES + b); };
  auto acce = [&](int b) { return meta + 8u * (2 * STAGES + 2 + b); };
  const uint32_t slot = meta + 8u * (2 * STAGES + 4);
  const int KT = p.K / BK2;
  const int TMN = p.M / 128, id0 = task * p.nsub;                         // flattened tile id = tile_m + TMN * tile_n
  const int total = TMN * (p.N / SUBN);
  const int cnt = (total - id0 < p.nsub) ? (total - id0) : p.nsub;        // tiles in this task (>= 1)
  auto sub_m0 = [&](int j) { return ((id0 + j) % TMN) * 128; };
  auto sub_n0 = [&](int j) { return ((id0 + j) / TMN) * SUBN; };
  const int tile_m = id0 % TMN, tile_n = id0 / TMN;                       // first tile (debug prints only)
#ifdef G2_DEBUG
  __shared__ volatile int g2dbg[4];
  if (tid < 4) g2dbg[tid] = 0;
  if (tid == 0) atomicAdd(&g2_tcnt[blockIdx.x & 1023], 1);
#endif

  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(full(s), 1); mbar_init(empty(s), 1); }
    for (int b = 0; b < 2; b++) { mbar_init(accf(b), 1); mbar_init(acce(b), 6); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#ifndef G2_PERSIST_TMEM
  __syncwarp();                                                   // tcgen05.alloc is .sync.aligned: the whole warp must execute it converged (lane 0 just ran the barrier init)
  if (warp == 0) { tmem_alloc(slot, TMEM_COLS); tc_fence_before(); }
#endif
  __syncthreads();
  tc_fence_after();
#ifdef G2_PERSIST_TMEM
  const uint32_t tmem = g2_tbase_ref();
#else
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(gbase + STAGES * STAGE_BYTES + 8 * (2 * STAGES + 4));
#endif
#ifdef G2_DEBUG
#define G2DBGV (g2dbg[0] | (g2dbg[1] << 8) | (g2dbg[2] << 16) | ((int)(tmem & 0xff) << 24))
#else
#define G2DBGV 0
#endif

  if (warp == 0 && lane == 0) {                                   // ---- TMA producer
    asm volatile("fence.proxy.async;" ::: "memory");
    int it = 0;
    for (int j = 0; j < cnt; j++) {
      const int nb = sub_n0(j), m0 = sub_m0(j);
      for (int kt = 0; kt < KT; kt++, it++) {
        const int s = it % STAGES, use = it / STAGES;
        mbar_wait(empty(s), (use & 1) ^ 1, 1, it, tile_m * 1000 + tile_n, (p.wide << 24) | (LAYOUT << 20) | (EPI << 16) | (p.K >> 6));
        mbar_expect_tx(full(s), STAGE_BYTES);
        const uint32_t sA = sbase + s * STAGE_BYTES, sB = sA + A_BYTES, k0 = kt * BK2;
        if (!A_MN) tma_load_2d(sA, p.tmA, full(s), k0, m0);
        else { tma_load_2d(sA, p.tmA, full(s), m0, k0); tma_load_2d(sA + 8192, p.tmA, full(s), m0 + 64, k0); }
        if (!B_MN) tma_load_2d(sB, p.tmB, full(s), k0, nb);
        else {
#pragma unroll
          for (int c = 0; c < SUBN / 64; c++) tma_load_2d(sB + c * 8192, p.tmB, full(s), nb + 64 * c, k0);
        }
#ifdef G2_DEBUG
        g2dbg[0] = it + 1;
#endif
      }
    }
  } else if (warp == 1 && lane == 0) {                            // ---- MMA issuer (one thread)
    constexpr uint32_t IDESC = make_idesc(A_MN, B_MN, SUBN);
    int it = 0;
    for (int j = 0; j < cnt; j++) {
      const int b = j & 1;
      const uint32_t td = tmem + (uint32_t)(b * SUBN);
      if (j >= 2) { mbar_wait(acce(b), ((j >> 1) - 1) & 1, 3, j, tile_m * 1000 + tile_n, (p.wide << 24) | (LAYOUT << 20) | (EPI << 16) | (p.K >> 6)); tc_fence_after(); }
      for (int kt = 0; kt < KT; kt++, it++) {
        const int s = it % STAGES;
        mbar_wait(full(s), (it / STAGES) & 1, 2, it, tile_m * 1000 + tile_n, (p.wide << 24) | (LAYOUT << 20) | (EPI << 16) | (p.K >> 6));
        tc_fence_after();
        const uint32_t sA = sbase + s * STAGE_BYTES, sB = sA + A_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK2 / 16; kk++) {
          const uint64_t da = A_MN ? smem_desc(sA + kk * 2048, 8192, 1024) : smem_desc(sA + kk * 32, 16, 1024);
          const uint64_t db = B_MN ? smem_desc(sB + kk * 2048, 8192, 1024) : smem_desc(sB + kk * 32, 16, 1024);
          umma_bf16(td, da, db, IDESC, (kt > 0 || kk > 0) ? 1u : 0u);
        }
        umma_commit(empty(s));
#ifdef G2_DEBUG
        g2dbg[1] = it + 1;
#endif
      }
      umma_commit(accf(b));
#ifdef G2_DEBUG
      g2dbg[2] = j + 1;
#endif
    }
  }
  __syncwarp();

  // ---- epilogue. Warp w reads TMEM lane quadrant q = w&3 (hardware rule) of sub-tile j. Sub-tiles before the last one are drained by warps 2..7 only
  //      (warps 0/1 are still issuing TMA/MMA); the last sub-tile by all 8 warps.
  {
    const int q = warp & 3;
    for (int j = 0; j < cnt; j++) {
      const bool last = (j == cnt - 1);
      const int row0 = sub_m0(j) + q * 32, nbase = sub_n0(j);
      const uint32_t stg = meta + CF::META_BYTES + (uint32_t)warp * 4096u;
      if (!last && warp < 2) continue;
      const int b = j & 1;
      const int nown = (last || q >= 2) ? 2 : 1, oi = (nown == 2) ? (warp >> 2) : 0;
#ifdef G2_DEBUG
      unsigned long long wb = 0, wa = 0;
      unsigned long long bw[8];
      if (blockIdx.x == 0 && tid == 33 && tile_m == 0) { asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(wb) : "r"(accf(b))); for (int i = 0; i < 8; i++) asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(bw[i]) : "r"(meta + 8 * i)); }
#endif
      mbar_wait(accf(b), (j >> 1) & 1, 4, G2DBGV, tile_m * 1000 + tile_n, (p.wide << 24) | (LAYOUT << 20) | (EPI << 16) | (p.K >> 6));
#ifdef G2_DEBUG
      if (blockIdx.x == 0 && tid == 33 && tile_m == 0) { asm volatile("ld.volatile.shared.b64 %0, [%1];" : "=l"(wa) : "r"(accf(b))); printf("G2 OK accf before=%016llx after=%016llx j=%d  bars_before: %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n", wb, wa, j, bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], bw[6], bw[7]); }
#endif
      tc_fence_after();
      __syncwarp();
      constexpr int NCH = SUBN / 32;
      const uint32_t tb = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(b * SUBN);
      const int per = NCH / nown;                                   // chunks of 32 columns owned by this warp (2, 4 or 8)
#pragma unroll
      for (int i = 0; i < NCH; i += 2) {
        if (i >= per) break;
        const int c0 = oi + nown * i, c1 = oi + nown * (i + 1);
        uint32_t r0[32], r1[32];
        tmem_ld32_issue(tb + (uint32_t)(c0 * 32), r0);
        tmem_ld32_issue(tb + (uint32_t)(c1 * 32), r1);
        tmem_ld_wait();
        epi_chunk<EPI>(p, stg, lane, row0, nbase + c0 * 32, r0);
        epi_chunk<EPI>(p, stg, lane, row0, nbase + c1 * 32, r1);
      }
      if (j + 2 < cnt) {                                           // buffer b is free for sub-tile j+2
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(acce(b));
      }
    }
  }
  tc_fence_before();
  if (tid == 0) {   // drain: consume the LAST arrival of every empty[] barrier (tcgen05.commit arrivals are asynchronous and may land after accf) so no stale arrival hits the next task's re-initialised barrier
    const int IT = cnt * KT;
    for (int s = 0; s < STAGES && s < IT; s++) { const int last = s + ((IT - 1 - s) / STAGES) * STAGES; mbar_wait(empty(s), (last / STAGES) & 1, 5, s, tile_m * 1000 + tile_n, 0); }
    for (int s = 0; s < STAGES; s++) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(full(s)) : "memory"); asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(empty(s)) : "memory"); }
    for (int b = 0; b < 2; b++) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(accf(b)) : "memory"); asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(acce(b)) : "memory"); }
  }
  __syncthreads();
#ifndef G2_PERSIST_TMEM
  __syncwarp();
  if (warp == 0) tmem_dealloc(tmem, TMEM_COLS);
#endif
  __syncthreads();
}
#endif  // G2_TC

__device__ __forceinline__ void blk_begin() {
#if defined(G2_TC) && defined(G2_PERSIST_TMEM)
  block_begin();
#endif
}
__device__ __forceinline__ void blk_end() {
#if defined(G2_TC) && defined(G2_PERSIST_TMEM)
  block_end();
#endif
}

// task = index of a group of p.nsub consecutive tiles (see ArgsTC::nsub); the second index argument is unused (kept for the tg:: interface).
template <int LAYOUT, int EPI>
__device__ __forceinline__ void tile(const ArgsTC& p, int task, int /*unused*/, char* smem) {
#if defined(G2_TC)
  if constexpr (((GEMM100_LAYOUT_MASK >> LAYOUT) & 1) != 0) {
    if (p.wide) tile_tc<LAYOUT, EPI, CfgW>(p, task, smem); else tile_tc<LAYOUT, EPI, CfgN>(p, task, smem);
    return;
  }
#endif
}

}  // namespace sm100v2
}  // namespace gemm
