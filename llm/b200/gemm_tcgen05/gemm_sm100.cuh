// gemm_sm100.cuh -- Blackwell (sm_100a) GEMM backend for the Triad megakernel: TMA loads + tcgen05.mma into TMEM, single-CTA (cta_group::1).
//
// *** UNTESTED. Written WITHOUT B200 hardware, from the PTX ISA 8.7 tcgen05 chapter and CUTLASS 3.8+/4.x sm100 code. It may not even be correct
// *** on the first run; see README.md (test plan). Compile-checked only (nvcc 12.8 -arch=sm_100a, ptxas) where the README says so. ***
//
// Same task contract as llm/gemm/gemm.cuh: one task = one 128x128 output tile of C[M,N], executed by ALL 256 threads of a block that call
// tile<LAYOUT,EPI>(p, tile_m, tile_n, smem). Same Layout (NN/NT/TN) and Epi enums, same epilogues (they are the epilogue helpers of gemm.cuh,
// reused verbatim), same deterministic reduction order. Differences visible to the caller:
//   * Args is ArgsTC = gemm::Args + two CUtensorMap pointers. Call prepare(args, LAYOUT) ONCE on the host after filling A,B,lda,ldb,M,N,K
//     (it encodes the TMA descriptors into device memory). Copies of the struct that only change C/aux (as the tests do) stay valid.
//   * K must be a multiple of 64 (BK=64 bf16 = one 128-B swizzle row), not 32.
//   * smem_bytes() = 100,352 B (3 stages x 32 KB + 1 KB barriers + 1 KB alignment slack). 2 blocks/SM => ~197 KB of the 227 KB/SM.
//   * TMEM: each block allocates TMEM_COLS of the SM's 512 columns ONCE (block_init(), all 256 threads, before the first tile) and frees
//     them in block_exit() before it exits. block_init() relinquishes the allocation permit right after allocating: while a CTA holds
//     the permit, another CTA's tcgen05.alloc on the same SM waits until the holder exits, so a per-tile allocation would serialize (or,
//     in a persistent kernel, deadlock) two co-resident blocks. The occupancy API reports 1 block/SM for any kernel with tcgen05
//     instructions; the hardware runs 2 (measured on a B200), so size the grid from registers and shared memory instead.
//   * Needs sm_100a code (macro __CUDA_ARCH_FEAT_SM100_ALL). For any other target (sm_80, sm_90, plain sm_100) tile<> compiles to the
//     mma.sync path of gemm.cuh, so the same source builds everywhere. -DGEMM100_LAYOUT_MASK=<bits> chooses per layout (bit0 NN, bit1 NT, bit2 TN)
//     which layouts use tcgen05; the rest use mma.sync (debug switch: start with 2 = NT only, K-major on both operands, the best-understood case).
//
// Mirrored sources (from memory, unverified against the files):
//   cutlass/include/cute/arch/mma_sm100_umma.hpp     SM100_MMA_F16BF16_SS: "tcgen05.mma.cta_group::1.kind::f16 [d], a_desc, b_desc, i_desc, p;"
//   cutlass/include/cute/arch/mma_sm100_desc.hpp     UMMA::SmemDescriptor (start>>4 @0, LBO>>4 @16, SBO>>4 @32, version=1 @46, layout_type @61),
//                                                    UMMA::InstrDescriptor (c_fmt@4, a_fmt@7, b_fmt@10, a_major@15, b_major@16, N>>3 @17, M>>4 @24),
//                                                    LayoutType: SWIZZLE_NONE=0, 128B_BASE32B=1, 128B=2, 64B=4, 32B=6
//   cutlass/include/cute/arch/copy_sm100.hpp         SM100_TMEM_LOAD_32dp32b{1..32}x ("tcgen05.ld.sync.aligned.32x32b.xN.b32"), tcgen05.wait::ld
//   cutlass/include/cute/arch/tmem_allocator_sm100.hpp  TMEM::Allocator1Sm: tcgen05.alloc/dealloc/relinquish_alloc_permit
//   cutlass/include/cutlass/arch/barrier.h           ClusterTransactionBarrier (init, arrive_and_expect_tx, try_wait.parity), umma_arrive = tcgen05.commit
//   cutlass/include/cute/arch/copy_sm90_tma.hpp      SM90_TMA_LOAD_2D: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
//   cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp   mainloop structure (TMA warp, MMA warp, epilogue warps; acc full/empty)
//   cutlass/include/cute/atom/mma_traits_sm100.hpp   make_umma_desc (the K-major / MN-major SW128 LBO/SBO rules used below)
//   DeepGEMM sm100 kernels (deep_gemm/include/deep_gemm/...) for the "advance start address by 32 B per UMMA_K=16 inside a SW128 atom" idiom.
#pragma once
#include "../../gemm/gemm.cuh"
#include <cuda.h>            // CUtensorMap
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef GEMM100_STAGES
#define GEMM100_STAGES 3
#endif
#ifndef GEMM100_LAYOUT_MASK
#define GEMM100_LAYOUT_MASK 7
#endif

namespace gemm {
namespace sm100 {

constexpr int BK100 = 64;                       // K per stage: 64 bf16 = 128 B = one SWIZZLE_128B row
constexpr int TILE_BYTES = 128 * BK100 * 2;     // 16 KB per operand per stage
constexpr int STAGE_BYTES = 2 * TILE_BYTES;     // A + B
constexpr int STAGES100 = GEMM100_STAGES;
constexpr int META_BYTES = 1024, ALIGN_SLACK = 1024;
constexpr int TMEM_COLS = 128;                  // 128x128 fp32 accumulator = 128 lanes x 128 columns
__host__ __device__ constexpr int smem_bytes() { return STAGES100 * STAGE_BYTES + META_BYTES + ALIGN_SLACK; }

struct ArgsTC : Args {
  const CUtensorMap* tmA;   // device pointers to 128-B-aligned tensor maps, filled by prepare()
  const CUtensorMap* tmB;
};

__host__ inline bool args_ok(const ArgsTC& p) {
  return gemm::args_ok(p) && p.K % BK100 == 0;
}

// ------------------------------------------------------------------------------------------------ host: TMA descriptors
inline const CUtensorMap* make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t ld_elems, uint32_t box_inner, uint32_t box_outer) {
  typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                               const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  { int dev = 0, major = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 10) return nullptr; }   // pre-Blackwell: tile() runs the mma.sync fallback, no TMA descriptors needed (encode returns 801 on A100)
  static EncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 13000
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
#endif
    if (!fn) { fprintf(stderr, "gemm_sm100: cuTensorMapEncodeTiled not found\n"); exit(1); }
    enc = (EncodeFn)fn;
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer}, strides[1] = {ld_elems * 2};
  cuuint32_t box[2] = {box_inner, box_outer}, es[2] = {1, 1};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "gemm_sm100: cuTensorMapEncodeTiled failed (%d)\n", (int)r); exit(1); }
  CUtensorMap* d; cudaMalloc(&d, sizeof(CUtensorMap) * 2);   // 256-B aligned by cudaMalloc
  cudaMemcpy(d, &m, sizeof m, cudaMemcpyHostToDevice);
  return d;
}
// Call once per GEMM description (after A,B,lda,ldb,M,N,K are set). Operand shapes (row-major, ld in elements):
//   A K-major (NN, NT): [M][K] -> map dims {K, M}, box {64, 128}     A MN-major (TN): [K][M] -> dims {M, K}, box {64, 64} (two boxes per tile)
//   B K-major (NT):     [N][K] -> map dims {K, N}, box {64, 128}     B MN-major (NN, TN): [K][N] -> dims {N, K}, box {64, 64}
inline void prepare(ArgsTC& p, int layout) {
  const bool akc = layout != TN, bkc = layout == NT;
  p.tmA = akc ? make_map(p.A, p.K, p.M, p.lda, 64, 128) : make_map(p.A, p.M, p.K, p.lda, 64, 64);
  p.tmB = bkc ? make_map(p.B, p.K, p.N, p.ldb, 64, 128) : make_map(p.B, p.N, p.K, p.ldb, 64, 64);
}

// ------------------------------------------------------------------------------------------------ device: PTX wrappers (sm_100a only)
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t mbar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               ::"r"(dst), "l"(tm), "r"(mbar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc(uint32_t slot_smem, uint32_t ncols) {   // whole warp, converged
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {      // whole warp, converged. Once per block, in block_exit_tc().
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void umma_bf16(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t mbar) {   // arrive on mbar when all previously issued tcgen05.mma of this thread completed
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {   // 32 lanes x 32 columns of fp32 for this warp's lane quadrant
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]),
                 "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
                 "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Shared-memory matrix descriptor, SWIZZLE_128B (layout_type 2), version 1.
__device__ __forceinline__ uint64_t smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(lbo_bytes >> 4) << 16) | ((uint64_t)(sbo_bytes >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
// Instruction descriptor: D=f32, A=B=bf16, M=128, N=128, a_mn/b_mn = 1 for MN-major operands.
__host__ __device__ constexpr uint32_t make_idesc(bool a_mn, bool b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((a_mn ? 1u : 0u) << 15) | ((b_mn ? 1u : 0u) << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
}

__shared__ uint32_t tmem_base_s;   // this block's TMEM columns (block_init)

__device__ __forceinline__ void block_init_tc() {
  if (threadIdx.x < 32) {
    tmem_alloc((uint32_t)__cvta_generic_to_shared(&tmem_base_s), TMEM_COLS);
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
}
__device__ __forceinline__ void block_exit_tc() {
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (threadIdx.x < 32) tmem_dealloc(tmem_base_s, TMEM_COLS);
}

template <int LAYOUT, int EPI>
__device__ __forceinline__ void tile_tc(const ArgsTC& p, int tile_m, int tile_n, char* smem) {
  constexpr bool A_MN = (LAYOUT == TN), B_MN = (LAYOUT != NT);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (raw + 1023u) & ~1023u;                  // SW128 needs 1024-B aligned tiles
  char* gbase = smem + (sbase - raw);
  const uint32_t meta = sbase + STAGES100 * STAGE_BYTES;
  auto full = [&](int s) { return meta + 8u * s; };
  auto empty = [&](int s) { return meta + 8u * (STAGES100 + s); };
  const uint32_t accum = meta + 8u * (2 * STAGES100);
  const int KT = p.K / BK100, m0 = tile_m * 128, n0 = tile_n * 128;

  if (tid == 0) {
    for (int s = 0; s < STAGES100; s++) { mbar_init(full(s), 1); mbar_init(empty(s), 1); }
    mbar_init(accum, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = tmem_base_s;

  if (warp == 0 && lane == 0) {                                   // ---- TMA producer
    asm volatile("fence.proxy.async;" ::: "memory");             // make generic-proxy writes of earlier megakernel phases visible to TMA (see README, risk R4)
    for (int kt = 0; kt < KT; kt++) {
      const int s = kt % STAGES100, use = kt / STAGES100;
      mbar_wait(empty(s), (use & 1) ^ 1);
      mbar_expect_tx(full(s), STAGE_BYTES);
      const uint32_t sA = sbase + s * STAGE_BYTES, sB = sA + TILE_BYTES, k0 = kt * BK100;
      if (!A_MN) tma_load_2d(sA, p.tmA, full(s), k0, m0);
      else { tma_load_2d(sA, p.tmA, full(s), m0, k0); tma_load_2d(sA + 8192, p.tmA, full(s), m0 + 64, k0); }
      if (!B_MN) tma_load_2d(sB, p.tmB, full(s), k0, n0);
      else { tma_load_2d(sB, p.tmB, full(s), n0, k0); tma_load_2d(sB + 8192, p.tmB, full(s), n0 + 64, k0); }
    }
  } else if (warp == 1 && lane == 0) {                            // ---- MMA issuer (one thread)
    constexpr uint32_t IDESC = make_idesc(A_MN, B_MN);
    for (int kt = 0; kt < KT; kt++) {
      const int s = kt % STAGES100;
      mbar_wait(full(s), (kt / STAGES100) & 1);
      tc_fence_after();
      const uint32_t sA = sbase + s * STAGE_BYTES, sB = sA + TILE_BYTES;
#pragma unroll
      for (int kk = 0; kk < BK100 / 16; kk++) {
        // K-major SW128: tile = 128 rows x 128 B, 8-row groups 1024 B apart (SBO=1024, LBO unused=16); k16 step = +32 B inside the swizzle atom.
        // MN-major SW128: two 64-wide MN halves 8192 B apart (LBO=8192), 8-K-row groups 1024 B apart (SBO=1024); k16 step = +2 groups = +2048 B.
        const uint64_t da = A_MN ? smem_desc(sA + kk * 2048, 8192, 1024) : smem_desc(sA + kk * 32, 16, 1024);
        const uint64_t db = B_MN ? smem_desc(sB + kk * 2048, 8192, 1024) : smem_desc(sB + kk * 32, 16, 1024);
        umma_bf16(tmem, da, db, IDESC, (kt > 0 || kk > 0) ? 1u : 0u);
      }
      umma_commit(empty(s));                                      // frees the smem stage when these MMAs are done
    }
    umma_commit(accum);                                           // accumulator complete
  }
  __syncwarp();

  // ---- epilogue: all 8 warps. Warp w owns TMEM lane quadrant (w&3) and column half (w>>2): thread = one row, 64 columns.
  // TMEM -> registers -> fp32 staging tile in the (now idle) pipeline buffers, 16-B chunks XOR-swizzled by row so both the row-per-thread
  // writes and the row-per-warp reads are free of bank conflicts -> coalesced global stores, one tile row per warp instruction.
  // The per-element math is gemm::epi_store's, so results are bit-identical to the row-per-thread epilogue.
  mbar_wait(accum, 0);
  tc_fence_after();
  __syncwarp();
  static_assert(STAGES100 * STAGE_BYTES >= 128 * 128 * 4, "staging tile needs 64 KB of pipeline buffers");
  float* stg = reinterpret_cast<float*>(gbase);
  {
    const int q = warp & 3, half = warp >> 2, row = q * 32 + lane;
#pragma unroll
    for (int c = 0; c < 2; c++) {
      uint32_t r[32];
      tmem_ld32(tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(half * 64 + c * 32), r);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int chunk = half * 16 + c * 8 + j;                  // 16-B chunk of the 512-B fp32 row, 0..31
        *reinterpret_cast<float4*>(stg + row * 128 + ((chunk ^ (row & 7)) << 2)) =
            make_float4(__uint_as_float(r[4 * j]), __uint_as_float(r[4 * j + 1]), __uint_as_float(r[4 * j + 2]), __uint_as_float(r[4 * j + 3]));
      }
    }
  }
  __syncthreads();
#pragma unroll 4
  for (int rr = warp; rr < 128; rr += 8) {                        // lane = 4 consecutive columns of row rr
    const float4 a = *reinterpret_cast<const float4*>(stg + rr * 128 + ((lane ^ (rr & 7)) << 2));
    const size_t gr = (size_t)(m0 + rr);
    const int gc = n0 + lane * 4;
    if constexpr (EPI == EPI_F32) {
      float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(p.C) + gr * p.ldc + gc);
      float4 v = a;
      if (p.accumulate) { const float4 o = *dst; v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w; }
      *dst = v;
    } else {
      float v0 = a.x, v1 = a.y, v2 = a.z, v3 = a.w;
      bf16* C = reinterpret_cast<bf16*>(p.C);
      if constexpr (EPI == EPI_RES) {
        const uint2 rw = *reinterpret_cast<const uint2*>(p.aux + gr * p.ldaux + gc);
        const __nv_bfloat162 r01 = *reinterpret_cast<const __nv_bfloat162*>(&rw.x), r23 = *reinterpret_cast<const __nv_bfloat162*>(&rw.y);
        v0 += __bfloat162float(r01.x); v1 += __bfloat162float(r01.y); v2 += __bfloat162float(r23.x); v3 += __bfloat162float(r23.y);
      } else if constexpr (EPI == EPI_GELU) {
        const __nv_bfloat162 p01 = __floats2bfloat162_rn(v0, v1), p23 = __floats2bfloat162_rn(v2, v3);
        uint2 pw; pw.x = *reinterpret_cast<const uint32_t*>(&p01); pw.y = *reinterpret_cast<const uint32_t*>(&p23);
        *reinterpret_cast<uint2*>(p.aux + gr * p.ldaux + gc) = pw;
        v0 = gelu_tanh(__bfloat162float(p01.x)); v1 = gelu_tanh(__bfloat162float(p01.y));
        v2 = gelu_tanh(__bfloat162float(p23.x)); v3 = gelu_tanh(__bfloat162float(p23.y));
      } else if constexpr (EPI == EPI_DGELU) {
        const uint2 pw = *reinterpret_cast<const uint2*>(p.aux + gr * p.ldaux + gc);
        const __nv_bfloat162 p01 = *reinterpret_cast<const __nv_bfloat162*>(&pw.x), p23 = *reinterpret_cast<const __nv_bfloat162*>(&pw.y);
        v0 *= gelu_tanh_grad(__bfloat162float(p01.x)); v1 *= gelu_tanh_grad(__bfloat162float(p01.y));
        v2 *= gelu_tanh_grad(__bfloat162float(p23.x)); v3 *= gelu_tanh_grad(__bfloat162float(p23.y));
      }
      const __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
      uint2 ow; ow.x = *reinterpret_cast<const uint32_t*>(&o01); ow.y = *reinterpret_cast<const uint32_t*>(&o23);
      *reinterpret_cast<uint2*>(C + gr * p.ldc + gc) = ow;
    }
  }
  tc_fence_before();                                              // TMEM reads done before the next tile's MMAs overwrite the columns
  __syncthreads();                                                // smem (incl. barriers) is free for the caller's next task
}
#endif  // __CUDA_ARCH_FEAT_SM100_ALL

// ------------------------------------------------------------------------------------------------ public entry
// block_init() / block_exit(): call with all 256 threads of every block that runs tile<>, once before its first tile and once before
// it exits. No-ops without sm_100a features.
__device__ __forceinline__ void block_init() {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  block_init_tc();
#endif
}
__device__ __forceinline__ void block_exit() {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  block_exit_tc();
#endif
}

template <int LAYOUT, int EPI>
__device__ __forceinline__ void tile(const ArgsTC& p, int tile_m, int tile_n, char* smem) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  if constexpr (((GEMM100_LAYOUT_MASK >> LAYOUT) & 1) != 0) { tile_tc<LAYOUT, EPI>(p, tile_m, tile_n, smem); return; }
#endif
  gemm::gemm_tile<LAYOUT, EPI>(p, tile_m, tile_n, smem);          // mma.sync fallback (runs on sm_100a too); uses 64 KB of the provided smem
}

}  // namespace sm100
}  // namespace gemm
