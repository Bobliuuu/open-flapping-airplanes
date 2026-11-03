// gemm2.cuh -- v2 of the bf16 tensor-core GEMM *tile* library for the open-flapping-airplanes LLM megakernel (sm_80).
// DROP-IN for llm/gemm/gemm.cuh: same namespace `gemm`, same Args / Layout / Epi, same gemm_tile<LAYOUT,EPI,STAGES>(p,tile_m,tile_n,smem)
// (= the "compat" config: 128x128 tile, 64x32 warp tiles, <=128 regs, 2 blocks/SM). Additionally exposes a configurable core
//   gemm_tile_cfg<Cfg, LAYOUT, EPI>(p, tile_m, tile_n, smem)
// with Cfg<BM,BN,WARPS_M,WARPS_N,STAGES> (8 warps = 256 threads always). Presets: CfgCompat, CfgBig (128x256, 64x64 warp tiles,
// 1 block/SM), CfgBigT (256x128), CfgSmallN (128x64) and CfgSmallM (64x128, warp tile 32x32).
// No blockIdx/gridDim, no atomics, deterministic (fixed K order), all epilogues EPI_BF16/RES/GELU/DGELU/F32.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace gemm {

typedef __nv_bfloat16 bf16;
enum Layout { NN = 0, NT = 1, TN = 2 };
enum Epi { EPI_BF16 = 0, EPI_RES = 1, EPI_GELU = 2, EPI_DGELU = 3, EPI_F32 = 4 };

struct Args {
  const bf16* A; const bf16* B; void* C; bf16* aux;
  int lda, ldb, ldc, ldaux;
  int M, N, K;
  int accumulate;
};

constexpr int THREADS = 256;

template <int BM_, int BN_, int WM_, int WN_, int ST_, int BK_ = 32>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, STAGES = ST_, BK = BK_, KS = BK_ / 16;
  static constexpr int WTM = BM / WM, WTN = BN / WN;     // warp tile
  static constexpr int MT = WTM / 16, NT8 = WTN / 8, NB16 = WTN / 16;
  static constexpr int A_BYTES = BM_ * BK_ * 2, B_BYTES = BN_ * BK_ * 2, STAGE_BYTES = A_BYTES + B_BYTES;
  static constexpr int EPI_PITCH = (WTN + 8) * 4;         // fp32 staging pitch per row (bytes), conflict free
  static constexpr int EPI_BYTES = 8 * 16 * EPI_PITCH;
  static constexpr int PIPE_BYTES = STAGES * STAGE_BYTES;
  static constexpr int SMEM = PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES;
  static_assert(WM * WN == 8 && BM % (WM * 16) == 0 && BN % (WN * 16) == 0 && MT % 2 == 0 && STAGES >= 2, "bad cfg");
};
//                          BM   BN  WM WN ST
typedef Cfg<128, 128, 2, 4, 2, 64> CfgCompat;   // DEFAULT: BK=64, double buffered, 64 KB, 2 blocks/SM, <=128 regs
typedef Cfg<128, 128, 2, 4, 4> CfgCompat32;     // v1-style: BK=32, 4 stages, 64 KB
typedef Cfg<128, 256, 2, 4, 4> CfgBig;      // 96 KB, 1 block/SM (255 regs)
typedef Cfg<128, 256, 2, 4, 3> CfgBig3;     // 72 KB
typedef Cfg<128, 256, 2, 4, 3, 64> CfgBig64s3;  // BK=64, 144 KB (exceeds the 96 KB contract; experiment)
typedef Cfg<128, 256, 2, 4, 2, 64> CfgBigDB;   // BK=64 double buffer, 96 KB, 1 block/SM
typedef Cfg<256, 128, 4, 2, 4> CfgBigT;     // 96 KB, 1 block/SM
typedef Cfg<128, 64, 4, 2, 2, 64> CfgSmallN;    // 48 KB (warp tile 32x32), BK=64 double buffered
typedef Cfg<64, 128, 2, 4, 2, 64> CfgSmallM;    // 48 KB (warp tile 32x32), BK=64 double buffered

template <class C> __host__ __device__ constexpr int smem_bytes_cfg() { return C::SMEM; }
template <class C> __host__ inline bool args_ok_cfg(const Args& p) {
  return p.M % C::BM == 0 && p.N % C::BN == 0 && p.K % C::BK == 0 && p.lda % 8 == 0 && p.ldb % 8 == 0 && p.ldc % 4 == 0 &&
         p.ldaux % 8 == 0 && ((uintptr_t)p.A % 16 == 0) && ((uintptr_t)p.B % 16 == 0);
}
// legacy (compat) names
constexpr int BM = 128, BN = 128;
#ifndef GEMM_STAGES
#define GEMM_STAGES 2   // 2 -> BK=64 double buffered (64 KB); 3..5 -> BK=32 multi-stage
#endif
constexpr int kDefaultStages = GEMM_STAGES;
template <int STAGES = kDefaultStages> __host__ __device__ constexpr int smem_bytes() { return Cfg<128, 128, 2, 4, STAGES, (STAGES == 2 ? 64 : 32)>::SMEM; }
__host__ inline bool args_ok(const Args& p) { return args_ok_cfg<CfgCompat>(p); }

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

template <int BKK> __device__ __forceinline__ int swz_kc(int row) { return BKK == 32 ? ((row >> 1) & 3) : (row & 7); }
// Smem tile layouts (per operand per stage, BMN x 32 elements)
//  K-contiguous  (stored [mn][k], 64 B rows): byte = mn*64 + ((k/8) ^ ((mn>>1)&3))*16 + (k%8)*2
//  MN-contiguous (stored [k][mn], BMN*2 B rows): byte = k*BMN*2 + ((mn/8) ^ (k&7))*16 + (mn%8)*2
template <bool KC, int BMN, int BKK>
__device__ __forceinline__ void load_tile(uint32_t sbase, const bf16* g, int ld, int mn0, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < BMN * BKK / 8 / THREADS; i++) {
    int c = tid + i * THREADS;
    if (KC) {
      constexpr int CPK = BKK / 8;
      int row = c / CPK, kc = c % CPK;
      cp_async16(sbase + row * (BKK * 2) + ((kc ^ swz_kc<BKK>(row)) << 4), g + (size_t)(mn0 + row) * ld + k0 + kc * 8);
    } else {
      constexpr int CPR = BMN / 8;
      int k = c / CPR, mc = c % CPR;
      cp_async16(sbase + k * (BMN * 2) + ((mc ^ (k & 7)) << 4), g + (size_t)(k0 + k) * ld + mn0 + mc * 8);
    }
  }
}

// ---- epilogue helpers ----
// elementwise epilogue on 4 consecutive columns (r, c..c+3), staged path
template <int EPI>
__device__ __forceinline__ void epi4(const Args& p, int r, int c, float4 v) {
  bf16* C = reinterpret_cast<bf16*>(p.C);
  __nv_bfloat162 o0, o1;
  if (EPI == EPI_RES) {
    uint2 rr = *reinterpret_cast<const uint2*>(p.aux + (size_t)r * p.ldaux + c);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&rr.x), b = *reinterpret_cast<__nv_bfloat162*>(&rr.y);
    v.x += __bfloat162float(a.x); v.y += __bfloat162float(a.y); v.z += __bfloat162float(b.x); v.w += __bfloat162float(b.y);
    o0 = __floats2bfloat162_rn(v.x, v.y); o1 = __floats2bfloat162_rn(v.z, v.w);
  } else if (EPI == EPI_GELU) {
    __nv_bfloat162 p0 = __floats2bfloat162_rn(v.x, v.y), p1 = __floats2bfloat162_rn(v.z, v.w);
    uint2 pu; pu.x = *reinterpret_cast<uint32_t*>(&p0); pu.y = *reinterpret_cast<uint32_t*>(&p1);
    *reinterpret_cast<uint2*>(p.aux + (size_t)r * p.ldaux + c) = pu;
    o0 = __floats2bfloat162_rn(gelu_tanh(__bfloat162float(p0.x)), gelu_tanh(__bfloat162float(p0.y)));
    o1 = __floats2bfloat162_rn(gelu_tanh(__bfloat162float(p1.x)), gelu_tanh(__bfloat162float(p1.y)));
  } else if (EPI == EPI_DGELU) {
    uint2 rr = *reinterpret_cast<const uint2*>(p.aux + (size_t)r * p.ldaux + c);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&rr.x), b = *reinterpret_cast<__nv_bfloat162*>(&rr.y);
    v.x *= gelu_tanh_grad(__bfloat162float(a.x)); v.y *= gelu_tanh_grad(__bfloat162float(a.y));
    v.z *= gelu_tanh_grad(__bfloat162float(b.x)); v.w *= gelu_tanh_grad(__bfloat162float(b.y));
    o0 = __floats2bfloat162_rn(v.x, v.y); o1 = __floats2bfloat162_rn(v.z, v.w);
  } else {
    o0 = __floats2bfloat162_rn(v.x, v.y); o1 = __floats2bfloat162_rn(v.z, v.w);
  }
  uint2 ou; ou.x = *reinterpret_cast<uint32_t*>(&o0); ou.y = *reinterpret_cast<uint32_t*>(&o1);
  *reinterpret_cast<uint2*>(C + (size_t)r * p.ldc + c) = ou;
}

// direct (fragment) store, used for EPI_F32 and when GEMM_EPI_DIRECT is defined
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
// gemm_tile_cfg: compute output tile (tile_m, tile_n) of C[M,N] (tile = C::BM x C::BN). All 256 threads call it.
// smem: 128 B aligned, C::SMEM bytes. Ends with __syncthreads()-free warp-local epilogue + final __syncthreads (smem reusable).
template <class CF, int LAYOUT, int EPI>
__device__ __forceinline__ void gemm_tile_cfg(const Args& p, int tile_m, int tile_n, char* smem) {
  constexpr bool A_KC = (LAYOUT != TN);
  constexpr bool B_KC = (LAYOUT == NT);
  constexpr int BKK = CF::BK, KS = CF::KS, STAGES = CF::STAGES, MT = CF::MT, NT8 = CF::NT8, NB16 = CF::NB16;
  constexpr int A_BYTES = CF::A_BYTES, STAGE_BYTES = CF::STAGE_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / CF::WN, wn = warp % CF::WN;
  const int m0 = tile_m * CF::BM, n0 = tile_n * CF::BN;
  const uint32_t sbase = smem_u32(smem);
  const int KT = p.K / BKK;

  auto load_stage = [&](int kt) {
#ifdef GEMM_DIAG_NOLOAD   // diagnostic only (wrong results): no global traffic after the prologue
    if (kt >= STAGES - 1) return;
#endif
    uint32_t s = sbase + (kt % STAGES) * STAGE_BYTES;
    int k0 = kt * BKK;
    load_tile<A_KC, CF::BM, BKK>(s, p.A, p.lda, m0, k0, tid);
    load_tile<B_KC, CF::BN, BKK>(s + A_BYTES, p.B, p.ldb, n0, k0, tid);
  };

  float acc[MT][NT8][4];
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < KT) load_stage(s); cp_commit(); }  // (DB: stage 0 only)

  const int q = lane >> 3, r8 = lane & 7;
  auto ldA = [&](uint32_t sA, int ks, int mt, uint32_t (&r)[4]) {
    int mb = wm * CF::WTM + mt * 16;
    if (A_KC) {
      int m = mb + r8 + (q & 1) * 8, kc = ks * 2 + (q >> 1);
      ldsm_x4(r[0], r[1], r[2], r[3], sA + m * (BKK * 2) + ((kc ^ swz_kc<BKK>(m)) << 4));
    } else {
      int k = ks * 16 + r8 + (q >> 1) * 8, mc = (mb >> 3) + (q & 1);
      ldsm_x4_t(r[0], r[1], r[2], r[3], sA + k * (CF::BM * 2) + ((mc ^ (k & 7)) << 4));
    }
  };
  auto ldBp = [&](uint32_t sB, int ks, uint32_t (&r)[NT8][2], int nb) {
    int nbase = wn * CF::WTN + nb * 16;
    if (B_KC) {
      int n = nbase + r8 + (q >> 1) * 8, kc = ks * 2 + (q & 1);
      ldsm_x4(r[nb * 2][0], r[nb * 2][1], r[nb * 2 + 1][0], r[nb * 2 + 1][1], sB + n * (BKK * 2) + ((kc ^ swz_kc<BKK>(n)) << 4));
    } else {
      int k = ks * 16 + r8 + (q & 1) * 8, nc = (nbase >> 3) + (q >> 1);
      ldsm_x4_t(r[nb * 2][0], r[nb * 2][1], r[nb * 2 + 1][0], r[nb * 2 + 1][1], sB + k * (CF::BN * 2) + ((nc ^ (k & 7)) << 4));
    }
  };
  auto ldB = [&](uint32_t sB, int ks, uint32_t (&r)[NT8][2]) {
#pragma unroll
    for (int nb = 0; nb < NB16; nb++) ldBp(sB, ks, r, nb);
  };
  auto mma_row = [&](int mt, const uint32_t (&a)[4], const uint32_t (&b)[NT8][2]) {
#pragma unroll
    for (int nt = 0; nt < NT8; nt++) mma_bf16(acc[mt][nt], a, b[nt][0], b[nt][1]);
  };

  uint32_t af[2][4], bf[2][NT8][2];
  constexpr bool DB = (STAGES == 2);  // double buffering: load of tile kt+1 issued right after the barrier at the top of iteration kt
  if (!DB) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    ldB(sbase + A_BYTES, 0, bf[0]);
    ldA(sbase, 0, 0, af[0]);
  }
  for (int kt = 0; kt < KT; kt++) {
    const uint32_t sA = sbase + (kt % STAGES) * STAGE_BYTES, sB = sA + A_BYTES;
    if (DB) {
      cp_wait<0>(); __syncthreads();
      if (kt + 1 < KT) load_stage(kt + 1);
      cp_commit();
      ldB(sB, 0, bf[0]); ldA(sA, 0, 0, af[0]);
    }
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
#ifdef GEMM_SPREAD_B   // B fragments of the next k16 step are fetched piecewise during this step (NB16 <= MT)
        static_assert(CF::NB16 <= CF::MT, "spread needs NB16<=MT");
        if (ks == KS - 1 && mt == 0) {
          cp_wait<STAGES - 3>(); __syncthreads();
          int nk = kt + STAGES - 1; if (nk < KT) load_stage(nk); cp_commit();
        }
        if (mt + 1 < MT) ldA(sA, ks, mt + 1, af[(mt + 1) & 1]);
        else if (ks < KS - 1) ldA(sA, ks + 1, 0, af[0]);
        else if (kt + 1 < KT) ldA(sbase + ((kt + 1) % STAGES) * STAGE_BYTES, 0, 0, af[0]);
        if (mt < NB16) {
          if (ks < KS - 1) ldBp(sB, ks + 1, bf[(ks + 1) & 1], mt);
          else if (kt + 1 < KT) ldBp(sbase + ((kt + 1) % STAGES) * STAGE_BYTES + A_BYTES, 0, bf[0], mt);
        }
#else
        if (!DB && ks == KS - 1 && mt == MT - 2) {
          cp_wait<STAGES - 3>();  // tile kt+1 landed (this thread) ...
          __syncthreads();        // ... for all; and everyone is done with slot (kt-1)%STAGES
          int nk = kt + STAGES - 1; if (nk < KT) load_stage(nk); cp_commit();
        }
        if (mt + 1 < MT) ldA(sA, ks, mt + 1, af[(mt + 1) & 1]);
        else if (ks < KS - 1) { ldB(sB, ks + 1, bf[(ks + 1) & 1]); ldA(sA, ks + 1, 0, af[0]); }
        else if (!DB && kt + 1 < KT) {
          const uint32_t nA = sbase + ((kt + 1) % STAGES) * STAGE_BYTES;
          ldB(nA + A_BYTES, 0, bf[0]); ldA(nA, 0, 0, af[0]);
        }
#endif
        mma_row(mt, af[mt & 1], bf[ks & 1]);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();  // pipeline smem free -> epilogue staging

  const int g = lane >> 2, t = lane & 3;
  if (EPI == EPI_F32
#ifdef GEMM_EPI_DIRECT
      || true
#endif
  ) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int nt = 0; nt < NT8; nt++) {
        int r = m0 + wm * CF::WTM + mt * 16 + g, c = n0 + wn * CF::WTN + nt * 8 + 2 * t;
        epi_store<EPI>(p, r, c, acc[mt][nt][0], acc[mt][nt][1]);
        epi_store<EPI>(p, r + 8, c, acc[mt][nt][2], acc[mt][nt][3]);
      }
  } else {
    // warp-private fp32 staging, 16 rows at a time, then coalesced 8-byte stores
    constexpr int P = CF::EPI_PITCH;  // bytes per staged row
    char* stg = smem + warp * (16 * P);
    constexpr int LPR = CF::WTN / 4, RPI = 32 / LPR;  // lanes per row, rows per instruction
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int nt = 0; nt < NT8; nt++) {
        *reinterpret_cast<float2*>(stg + g * P + (nt * 8 + 2 * t) * 4) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        *reinterpret_cast<float2*>(stg + (g + 8) * P + (nt * 8 + 2 * t) * 4) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 16 / RPI; i++) {
        int rr = i * RPI + lane / LPR, cc = (lane % LPR) * 4;
        float4 v = *reinterpret_cast<const float4*>(stg + rr * P + cc * 4);
        epi4<EPI>(p, m0 + wm * CF::WTM + mt * 16 + rr, n0 + wn * CF::WTN + cc, v);
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

// compat entry point (identical signature to gemm/gemm.cuh)
template <int LAYOUT, int EPI, int STAGES = kDefaultStages>
__device__ __forceinline__ void gemm_tile(const Args& p, int tile_m, int tile_n, char* smem) {
  gemm_tile_cfg<Cfg<128, 128, 2, 4, STAGES, (STAGES == 2 ? 64 : 32)>, LAYOUT, EPI>(p, tile_m, tile_n, smem);
}

}  // namespace gemm
