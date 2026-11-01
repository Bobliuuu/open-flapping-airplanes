// gemm.cuh -- header-only bf16 tensor-core GEMM *tile* library for the open-flapping-airplanes LLM megakernel (sm_80).
// One task = one 128x128 output tile, executed by all 256 threads of a block (8 warps, 2x4, warp tile 64x32).
// mma.sync.m16n8k16 bf16 -> fp32, ldmatrix (swizzled, conflict-free smem), STAGES-deep cp.async pipeline, BK=32.
// No blockIdx/gridDim use, no atomics, deterministic (fixed reduction order over K).
// See README.md for the API contract.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace gemm {

typedef __nv_bfloat16 bf16;

// Layouts. Output C[M,N] (row-major) = sum_k Aop(m,k) * Bop(k,n); all matrices row-major with explicit ld.
//   NN: A stored [M,K] (lda>=K),  B stored [K,N] (ldb>=N)       forward linear  Y = X.W
//   NT: A stored [M,K],           B stored [N,K] (ldb>=K)       dX = dY.W^T (W stored [N_out=?,..]) and logits = x.wte^T
//   TN: A stored [K,M] (lda>=M),  B stored [K,N]                dW = X^T.dY  (reduce over the token dim K)
enum Layout { NN = 0, NT = 1, TN = 2 };

// Epilogues
enum Epi {
  EPI_BF16  = 0,  // C(bf16) = acc
  EPI_RES   = 1,  // C(bf16) = bf16(acc + R), R = aux (bf16, ld=ldaux), may alias C
  EPI_GELU  = 2,  // aux(bf16) = bf16(acc) (pre-activation, saved);  C(bf16) = gelu_tanh(float(aux))
  EPI_DGELU = 3,  // C(bf16) = bf16(acc * gelu'(aux)), aux = saved bf16 pre-activation (read only)
  EPI_F32   = 4   // C(fp32): accumulate ? C += acc : C = acc   (logits chunk / dW32)
};

constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;
#ifndef GEMM_STAGES
#define GEMM_STAGES 4
#endif
constexpr int kDefaultStages = GEMM_STAGES;
template <int STAGES = kDefaultStages>
__host__ __device__ constexpr int smem_bytes() { return STAGES * (BM * BK + BN * BK) * 2; }  // 64 KB for 4 stages

struct Args {
  const bf16* A;  // see Layout
  const bf16* B;
  void* C;        // bf16* (EPI_BF16/RES/GELU/DGELU) or float* (EPI_F32)
  bf16* aux;      // RES: residual R;  GELU: pre-activation output;  DGELU: pre-activation input
  int lda, ldb, ldc, ldaux;  // leading dims in ELEMENTS of the respective row-major arrays
  int M, N, K;    // GEMM dims: C is MxN, reduction length K. All must be multiples of 128 / 128 / 32.
  int accumulate; // EPI_F32 only: 1 -> C += acc, 0 -> C = acc
};

__host__ inline bool args_ok(const Args& p) {
  return p.M % BM == 0 && p.N % BN == 0 && p.K % BK == 0 && p.lda % 8 == 0 && p.ldb % 8 == 0 &&
         p.ldc % 4 == 0 && p.ldaux % 8 == 0 && ((uintptr_t)p.A % 16 == 0) && ((uintptr_t)p.B % 16 == 0);
}

// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// tanh.approx.f32: max rel. error ~2^-11, far below the bf16 rounding (2^-9) of the values it feeds.
__device__ __forceinline__ float tanh_fast(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  float t = tanh_fast(k0 * (x + k1 * x * x * x));
  return 0.5f * x * (1.f + t);
}
__device__ __forceinline__ float gelu_tanh_grad(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  float t = tanh_fast(k0 * (x + k1 * x * x * x));
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * k0 * (1.f + 3.f * k1 * x * x);
}

// Smem tile layouts (per operand, per stage, 8 KB, 128 x 32 elements)
//  K-contiguous  (tile stored [mn][k], 64 B rows): byte = mn*64 + ((k/8) ^ ((mn>>1)&3))*16 + (k%8)*2
//  MN-contiguous (tile stored [k][mn], 256 B rows): byte = k*256 + ((mn/8) ^ (k&7))*16 + (mn%8)*2
// Both are bank-conflict free for cp.async writes and for ldmatrix(.trans) reads.
template <bool KC>
__device__ __forceinline__ void load_tile(uint32_t sbase, const bf16* g, int ld, int mn0, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    int c = tid + i * THREADS;
    if (KC) {
      int row = c >> 2, kc = c & 3;
      cp_async16(sbase + row * 64 + ((kc ^ ((row >> 1) & 3)) << 4), g + (size_t)(mn0 + row) * ld + k0 + kc * 8);
    } else {
      int k = c >> 4, mc = c & 15;
      cp_async16(sbase + k * 256 + ((mc ^ (k & 7)) << 4), g + (size_t)(k0 + k) * ld + mn0 + mc * 8);
    }
  }
}

template <int EPI>
__device__ __forceinline__ void epi_store(const Args& p, int r, int c, float v0, float v1) {
  if (EPI == EPI_F32) {
    float2* dst = reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + (size_t)r * p.ldc + c);
    if (p.accumulate) { float2 o = *dst; v0 += o.x; v1 += o.y; }
    *dst = make_float2(v0, v1);
    return;
  }
  bf16* C = reinterpret_cast<bf16*>(p.C);
  if (EPI == EPI_RES) {
    __nv_bfloat162 rr = *reinterpret_cast<const __nv_bfloat162*>(p.aux + (size_t)r * p.ldaux + c);
    v0 += __bfloat162float(rr.x); v1 += __bfloat162float(rr.y);
  } else if (EPI == EPI_GELU) {
    __nv_bfloat162 pre = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<__nv_bfloat162*>(p.aux + (size_t)r * p.ldaux + c) = pre;
    v0 = gelu_tanh(__bfloat162float(pre.x)); v1 = gelu_tanh(__bfloat162float(pre.y));
  } else if (EPI == EPI_DGELU) {
    __nv_bfloat162 pre = *reinterpret_cast<const __nv_bfloat162*>(p.aux + (size_t)r * p.ldaux + c);
    v0 *= gelu_tanh_grad(__bfloat162float(pre.x)); v1 *= gelu_tanh_grad(__bfloat162float(pre.y));
  }
  *reinterpret_cast<__nv_bfloat162*>(C + (size_t)r * p.ldc + c) = __floats2bfloat162_rn(v0, v1);
}

// ---------------------------------------------------------------------------------------------
// gemm_tile: compute output tile (tile_m, tile_n) of C[M,N]. Called by all 256 threads of a block.
// smem: dynamic shared memory base, 128 B aligned, smem_bytes<STAGES>() bytes (64 KB default).
// Ends with __syncthreads(), so smem may be reused immediately after return.
template <int LAYOUT, int EPI, int STAGES = kDefaultStages>
__device__ __forceinline__ void gemm_tile(const Args& p, int tile_m, int tile_n, char* smem) {
  constexpr bool A_KC = (LAYOUT != TN);   // A stored [M,K]
  constexpr bool B_KC = (LAYOUT == NT);   // B stored [N,K]
  constexpr int A_BYTES = BM * BK * 2, STAGE_BYTES = A_BYTES + BN * BK * 2;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = tile_m * BM, n0 = tile_n * BN;
  const uint32_t sbase = smem_u32(smem);
  const int KT = p.K / BK;

  auto load_stage = [&](int kt) {
    uint32_t s = sbase + (kt % STAGES) * STAGE_BYTES;
    int k0 = kt * BK;
    if (A_KC) load_tile<true>(s, p.A, p.lda, m0, k0, tid); else load_tile<false>(s, p.A, p.lda, m0, k0, tid);
    if (B_KC) load_tile<true>(s + A_BYTES, p.B, p.ldb, n0, k0, tid); else load_tile<false>(s + A_BYTES, p.B, p.ldb, n0, k0, tid);
  };

  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load_stage(s); cp_commit(); }

  const int q = lane >> 3, r8 = lane & 7;
  auto ldA = [&](uint32_t sA, int ks, int mt, uint32_t (&r)[4]) {
    int mb = wm * 64 + mt * 16;
    if (A_KC) {
      int m = mb + r8 + (q & 1) * 8, kc = ks * 2 + (q >> 1);
      ldsm_x4(r[0], r[1], r[2], r[3], sA + m * 64 + ((kc ^ ((m >> 1) & 3)) << 4));
    } else {
      int k = ks * 16 + r8 + (q >> 1) * 8, mc = (mb >> 3) + (q & 1);
      ldsm_x4_t(r[0], r[1], r[2], r[3], sA + k * 256 + ((mc ^ (k & 7)) << 4));
    }
  };
  auto ldB = [&](uint32_t sB, int ks, uint32_t (&r)[4][2]) {
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
      int nbase = wn * 32 + nb * 16;
      if (B_KC) {
        int n = nbase + r8 + (q >> 1) * 8, kc = ks * 2 + (q & 1);
        ldsm_x4(r[nb * 2][0], r[nb * 2][1], r[nb * 2 + 1][0], r[nb * 2 + 1][1], sB + n * 64 + ((kc ^ ((n >> 1) & 3)) << 4));
      } else {
        int k = ks * 16 + r8 + (q & 1) * 8, nc = (nbase >> 3) + (q >> 1);
        ldsm_x4_t(r[nb * 2][0], r[nb * 2][1], r[nb * 2 + 1][0], r[nb * 2 + 1][1], sB + k * 256 + ((nc ^ (k & 7)) << 4));
      }
    }
  };
  auto mma_row = [&](int mt, const uint32_t (&a)[4], const uint32_t (&b)[4][2]) {
#pragma unroll
    for (int nt = 0; nt < 4; nt++) mma_bf16(acc[mt][nt], a, b[nt][0], b[nt][1]);
  };

  // Fine-grained software pipeline: A fragments are double-buffered per 16-row group, B fragments per k16 step; the
  // fragments of the next k16 step (also across stages) are fetched while the current MMAs execute.
  // Slot (kt-1)%STAGES is refilled right after the mid-iteration barrier (all warps are done reading it).
  static_assert(STAGES >= 3, "need >= 3 stages");
  uint32_t af[2][4], bf[2][4][2];
  cp_wait<STAGES - 2>();
  __syncthreads();
  ldB(sbase + A_BYTES, 0, bf[0]);
  ldA(sbase, 0, 0, af[0]);
  for (int kt = 0; kt < KT; kt++) {
    const uint32_t sA = sbase + (kt % STAGES) * STAGE_BYTES, sB = sA + A_BYTES;
    // ---- k16 step 0
    ldA(sA, 0, 1, af[1]); mma_row(0, af[0], bf[0]);
    ldA(sA, 0, 2, af[0]); mma_row(1, af[1], bf[0]);
    ldA(sA, 0, 3, af[1]); mma_row(2, af[0], bf[0]);
    ldB(sB, 1, bf[1]); ldA(sA, 1, 0, af[0]); mma_row(3, af[1], bf[0]);
    // ---- k16 step 1
    ldA(sA, 1, 1, af[1]); mma_row(0, af[0], bf[1]);
    ldA(sA, 1, 2, af[0]); mma_row(1, af[1], bf[1]);
    cp_wait<STAGES - 3>();  // tile kt+1 has landed (this thread's copies) ...
    __syncthreads();        // ... for all threads; and everyone is done with slot (kt-1)%STAGES
    { int nk = kt + STAGES - 1; if (nk < KT) load_stage(nk); cp_commit(); }
    ldA(sA, 1, 3, af[1]); mma_row(2, af[0], bf[1]);
    if (kt + 1 < KT) {
      const uint32_t nA = sbase + ((kt + 1) % STAGES) * STAGE_BYTES;
      ldB(nA + A_BYTES, 0, bf[0]); ldA(nA, 0, 0, af[0]);
    }
    mma_row(3, af[1], bf[1]);
  }
  cp_wait<0>();
  __syncthreads();  // smem free for the caller's next task

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; mt++)
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      int r = m0 + wm * 64 + mt * 16 + g, c = n0 + wn * 32 + nt * 8 + 2 * t;
      epi_store<EPI>(p, r, c, acc[mt][nt][0], acc[mt][nt][1]);
      epi_store<EPI>(p, r + 8, c, acc[mt][nt][2], acc[mt][nt][3]);
    }
}

}  // namespace gemm
