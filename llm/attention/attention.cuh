// attention.cuh -- causal multi-head attention (fwd + bwd) tasks for the Triad megakernel.
// bf16, head dim 64, mma.sync.m16n8k16, ldmatrix, cp.async, flash-style online softmax.
// See README.md for the contract. All functions: called by all 256 threads of one block.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace attn {

typedef __nv_bfloat16 bf16;
constexpr int DH = 64;            // head dim
constexpr int TILE = 64;          // rows per task
constexpr int NTHREADS = 256;
constexpr float SCALE = 0.125f;                       // 1/sqrt(64)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float SC2 = SCALE * LOG2E;                  // scale folded with log2(e)

// dynamic smem bytes needed by each task
constexpr int SMEM_FWD  = 8192 + 2 * 16384;                 // Q + 2 x (K,V)               = 40960
constexpr int SMEM_DKDV = 16384 + 2 * (16384 + 512);        // K,V + 2 x (Q,dO,lse,D)      = 50176
constexpr int SMEM_DQ   = 16384 + 2 * 16384;                // Q,dO + 2 x (K,V)            = 49152
constexpr int SMEM_PREP = 0;

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// 64x64 bf16 tile in smem, 128 B rows, 16 B chunks XOR-swizzled by (row & 7).
__device__ __forceinline__ uint32_t swz(int row, int chunk) { return (uint32_t)(row * 128 + ((chunk ^ (row & 7)) << 4)); }

// cooperative load of a 64x64 tile (gsrc -> row 0, ld = row stride in elements)
__device__ __forceinline__ void load_tile(uint32_t sbase, const bf16* gsrc, int ld, int tid) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    int r = (tid >> 3) + 32 * i, c = tid & 7;
    cp_async16(sbase + swz(r, c), gsrc + (size_t)r * ld + c * 8);
  }
}
// A fragment (16x16, row-major source [m][k])
__device__ __forceinline__ void ldA(uint32_t t, int r0, int k0, int lane, uint32_t* a) {
  int row = r0 + (lane & 15), ch = (k0 >> 3) + (lane >> 4);
  ldsm4(a[0], a[1], a[2], a[3], t + swz(row, ch));
}
// B fragments for two n-tiles (n0..n0+15), k0..k0+15, source stored [n][k]  (e.g. K for QK^T)
__device__ __forceinline__ void ldB_nk(uint32_t t, int n0, int k0, int lane, uint32_t* b) {
  int m = lane >> 3, row = n0 + (lane & 7) + ((m >> 1) << 3), ch = (k0 >> 3) + (m & 1);
  ldsm4(b[0], b[1], b[2], b[3], t + swz(row, ch));
}
// B fragments for two n-tiles, source stored [k][n]  (e.g. V for PV)
__device__ __forceinline__ void ldB_kn(uint32_t t, int k0, int n0, int lane, uint32_t* b) {
  int m = lane >> 3, row = k0 + (lane & 7) + ((m & 1) << 3), ch = (n0 >> 3) + (m >> 1);
  ldsm4t(b[0], b[1], b[2], b[3], t + swz(row, ch));
}

// ------------------------------------------------------------------ 1. forward
// Task (b,h,qt): O rows [qt*64,qt*64+64) of head h, lse[b,h,t].
// qkv [B*T, 3*H*64] bf16, o [B*T, H*64] bf16, lse [B,H,T] fp32.  smem >= SMEM_FWD.
template <int T, int H>
__device__ __forceinline__ void attn_fwd_tile(const bf16* __restrict__ qkv, bf16* __restrict__ o,
                                              float* __restrict__ lse, int b, int h, int qt, char* smem) {
  constexpr int D = H * DH, LD = 3 * D;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, rg = w & 3, hf = w >> 2;
  const int g = lane >> 2, q4 = lane & 3;
  const uint32_t sQ = smem_u32(smem);
  const uint32_t sKV = sQ + 8192;  // stage s: K at +s*16384, V at +s*16384+8192
  const bf16* base = qkv + (size_t)(b * T) * LD + h * DH;
  const int nb = qt + 1;

  load_tile(sQ, base + (size_t)(qt * TILE) * LD, LD, tid);
  load_tile(sKV, base + D, LD, tid);
  load_tile(sKV + 8192, base + 2 * D, LD, tid);
  cp_commit();

  uint32_t qa[4][4];
  float O[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  for (int kb = 0; kb < nb; kb++) {
    if (kb + 1 < nb) {
      uint32_t st = sKV + ((kb + 1) & 1) * 16384;
      const bf16* kp = base + D + (size_t)((kb + 1) * TILE) * LD;
      load_tile(st, kp, LD, tid);
      load_tile(st + 8192, kp + D, LD, tid);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    if (kb == 0) {
#pragma unroll
      for (int ks = 0; ks < 4; ks++) ldA(sQ, rg * 16, ks * 16, lane, qa[ks]);
    }
    const bool diag = (kb == qt);
    if (!(diag && hf * 32 > rg * 16 + 15)) {
      const uint32_t sK = sKV + (kb & 1) * 16384, sV = sK + 8192;
      float S[4][4];
#pragma unroll
      for (int j = 0; j < 4; j++) S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int jp = 0; jp < 2; jp++) {
          uint32_t bb[4];
          ldB_nk(sK, hf * 32 + jp * 16, ks * 16, lane, bb);
          mma16816(S[2 * jp], qa[ks], bb[0], bb[1]);
          mma16816(S[2 * jp + 1], qa[ks], bb[2], bb[3]);
        }
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          float s = S[j][e] * SC2;
          if (diag) {
            int col = hf * 32 + j * 8 + q4 * 2 + (e & 1);
            int row = rg * 16 + g + ((e >> 1) << 3);
            if (col > row) s = -INFINITY;
          }
          S[j][e] = s;
        }
        mx0 = fmaxf(mx0, fmaxf(S[j][0], S[j][1]));
        mx1 = fmaxf(mx1, fmaxf(S[j][2], S[j][3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
      float u0 = (n0 == -INFINITY) ? 0.f : n0, u1 = (n1 == -INFINITY) ? 0.f : n1;
      float al0 = exp2f(m0 - u0), al1 = exp2f(m1 - u1);
      m0 = n0; m1 = n1;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        S[j][0] = exp2f(S[j][0] - u0); S[j][1] = exp2f(S[j][1] - u0);
        S[j][2] = exp2f(S[j][2] - u1); S[j][3] = exp2f(S[j][3] - u1);
        rs0 += S[j][0] + S[j][1]; rs1 += S[j][2] + S[j][3];
      }
      l0 = l0 * al0 + rs0; l1 = l1 * al1 + rs1;
#pragma unroll
      for (int j = 0; j < 8; j++) { O[j][0] *= al0; O[j][1] *= al0; O[j][2] *= al1; O[j][3] *= al1; }
#pragma unroll
      for (int k2 = 0; k2 < 2; k2++) {
        uint32_t pa[4];
        pa[0] = pack2(S[2 * k2][0], S[2 * k2][1]);
        pa[1] = pack2(S[2 * k2][2], S[2 * k2][3]);
        pa[2] = pack2(S[2 * k2 + 1][0], S[2 * k2 + 1][1]);
        pa[3] = pack2(S[2 * k2 + 1][2], S[2 * k2 + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 4; dp++) {
          uint32_t bb[4];
          ldB_kn(sV, hf * 32 + k2 * 16, dp * 16, lane, bb);
          mma16816(O[2 * dp], pa, bb[0], bb[1]);
          mma16816(O[2 * dp + 1], pa, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
  }

  // merge the two key-halves (hf=1 -> smem -> hf=0), fixed order => deterministic
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  float* mb = reinterpret_cast<float*>(smem + 8192);
  const int t128 = rg * 32 + lane;
  if (hf == 1) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) mb[(j * 4 + e) * 128 + t128] = O[j][e];
    mb[32 * 128 + t128] = m0; mb[33 * 128 + t128] = m1; mb[34 * 128 + t128] = l0; mb[35 * 128 + t128] = l1;
  }
  __syncthreads();
  if (hf == 0) {
    float pm0 = mb[32 * 128 + t128], pm1 = mb[33 * 128 + t128], pl0 = mb[34 * 128 + t128], pl1 = mb[35 * 128 + t128];
    float M0 = fmaxf(m0, pm0), M1 = fmaxf(m1, pm1);
    float a0 = exp2f(m0 - M0), b0 = exp2f(pm0 - M0), a1 = exp2f(m1 - M1), b1 = exp2f(pm1 - M1);
    float L0 = l0 * a0 + pl0 * b0, L1 = l1 * a1 + pl1 * b1;
    float i0 = 1.f / L0, i1 = 1.f / L1;
    const int r0 = qt * TILE + rg * 16 + g;
    bf16* o0 = o + (size_t)(b * T + r0) * D + h * DH;
    bf16* o1 = o0 + (size_t)8 * D;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v0 = (O[j][0] * a0 + mb[(j * 4 + 0) * 128 + t128] * b0) * i0;
      float v1 = (O[j][1] * a0 + mb[(j * 4 + 1) * 128 + t128] * b0) * i0;
      float v2 = (O[j][2] * a1 + mb[(j * 4 + 2) * 128 + t128] * b1) * i1;
      float v3 = (O[j][3] * a1 + mb[(j * 4 + 3) * 128 + t128] * b1) * i1;
      *reinterpret_cast<uint32_t*>(o0 + j * 8 + q4 * 2) = pack2(v0, v1);
      *reinterpret_cast<uint32_t*>(o1 + j * 8 + q4 * 2) = pack2(v2, v3);
    }
    if (q4 == 0) {
      float* lp = lse + (size_t)(b * H + h) * T + r0;
      lp[0] = (M0 + log2f(L0)) * LN2;
      lp[8] = (M1 + log2f(L1)) * LN2;
    }
  }
  __syncthreads();  // smem may be reused by the caller afterwards
}

// ------------------------------------------------------------------ 2a. bwd prep: D = rowsum(dO * O)
// Task (b,h,rt): D[b,h,rt*64 .. +64). o, dO [B*T, H*64] bf16. No smem.
template <int T, int H>
__device__ __forceinline__ void attn_bwd_prep_tile(const bf16* __restrict__ o, const bf16* __restrict__ dout,
                                                   float* __restrict__ Dout, int b, int h, int rt) {
  constexpr int D = H * DH;
  const int tid = threadIdx.x, r = tid >> 2, part = tid & 3;
  const size_t off = (size_t)(b * T + rt * TILE + r) * D + h * DH + part * 16;
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 2; i++) {
    uint4 ov = *reinterpret_cast<const uint4*>(o + off + i * 8);
    uint4 dv = *reinterpret_cast<const uint4*>(dout + off + i * 8);
    const bf16* op = reinterpret_cast<const bf16*>(&ov);
    const bf16* dp = reinterpret_cast<const bf16*>(&dv);
#pragma unroll
    for (int k = 0; k < 8; k++) s += __bfloat162float(op[k]) * __bfloat162float(dp[k]);
  }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  if (part == 0) Dout[(size_t)(b * H + h) * T + rt * TILE + r] = s;
}

// ------------------------------------------------------------------ 2b. bwd dK, dV
// Task (b,h,kt): dK,dV of key rows [kt*64,+64), iterating query tiles kt..T/64-1.
// Needs fwd (lse), prep (D). Writes dqkv[:, D + h*64..] (dK) and dqkv[:, 2D + h*64..] (dV). smem >= SMEM_DKDV.
template <int T, int H>
__device__ __forceinline__ void attn_bwd_dkdv_tile(const bf16* __restrict__ qkv, const bf16* __restrict__ dout,
                                                   const float* __restrict__ lse, const float* __restrict__ Dr,
                                                   bf16* __restrict__ dqkv, int b, int h, int kt, char* smem) {
  constexpr int D = H * DH, LD = 3 * D, NQ = T / TILE;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, rg = w & 3, qh = w >> 2;
  const int g = lane >> 2, q4 = lane & 3;
  const uint32_t sK = smem_u32(smem), sV = sK + 8192;
  const uint32_t sST = sK + 16384;  // stage s at +s*16896: Q, dO, lse[64], D[64]
  const bf16* base = qkv + (size_t)(b * T) * LD + h * DH;
  const bf16* dob = dout + (size_t)(b * T) * D + h * DH;
  const float* lseb = lse + (size_t)(b * H + h) * T;
  const float* Db = Dr + (size_t)(b * H + h) * T;
  const int ns = NQ - kt;

  auto load_stage = [&](int s, int qb) {
    uint32_t st = sST + s * 16896;
    load_tile(st, base + (size_t)(qb * TILE) * LD, LD, tid);
    load_tile(st + 8192, dob + (size_t)(qb * TILE) * D, D, tid);
    if (tid < 16) cp_async16(st + 16384 + tid * 16, lseb + qb * TILE + tid * 4);
    else if (tid < 32) cp_async16(st + 16640 + (tid - 16) * 16, Db + qb * TILE + (tid - 16) * 4);
  };
  load_tile(sK, base + D + (size_t)(kt * TILE) * LD, LD, tid);
  load_tile(sV, base + 2 * D + (size_t)(kt * TILE) * LD, LD, tid);
  load_stage(0, kt);
  cp_commit();

  float dK[8][4], dV[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) dK[j][e] = dV[j][e] = 0.f;

  for (int i = 0; i < ns; i++) {
    const int qb = kt + i;
    if (i + 1 < ns) { load_stage((i + 1) & 1, qb + 1); cp_commit(); cp_wait<1>(); }
    else cp_wait<0>();
    __syncthreads();
    const bool diag = (i == 0);
    if (!(diag && qh == 0 && rg >= 2)) {
      const uint32_t sQ = sST + (i & 1) * 16896, sdO = sQ + 8192;
      const float* sl = reinterpret_cast<const float*>(smem + 16384 + (i & 1) * 16896 + 16384);
      const float* sD = sl + 64;
      float S[4][4], P[4][4];
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) S[j][e] = P[j][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        uint32_t ka[4], va[4];
        ldA(sK, rg * 16, ks * 16, lane, ka);
        ldA(sV, rg * 16, ks * 16, lane, va);
#pragma unroll
        for (int jp = 0; jp < 2; jp++) {
          uint32_t bb[4];
          ldB_nk(sQ, qh * 32 + jp * 16, ks * 16, lane, bb);
          mma16816(S[2 * jp], ka, bb[0], bb[1]);
          mma16816(S[2 * jp + 1], ka, bb[2], bb[3]);
          ldB_nk(sdO, qh * 32 + jp * 16, ks * 16, lane, bb);
          mma16816(P[2 * jp], va, bb[0], bb[1]);       // P holds dP^T for now
          mma16816(P[2 * jp + 1], va, bb[2], bb[3]);
        }
      }
      // S -> P^T = exp(S*scale - lse), P -> dS^T = P^T * (dP^T - D)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int c0 = qh * 32 + j * 8 + q4 * 2;
        float2 lv = *reinterpret_cast<const float2*>(sl + c0);
        float2 dv = *reinterpret_cast<const float2*>(sD + c0);
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float lc = (e & 1) ? lv.y : lv.x, dc = (e & 1) ? dv.y : dv.x;
          float p = exp2f(S[j][e] * SC2 - lc * LOG2E);
          if (diag) {
            int col = c0 + (e & 1), row = rg * 16 + g + ((e >> 1) << 3);
            if (row > col) p = 0.f;
          }
          S[j][e] = p;                       // P^T
          P[j][e] = p * (P[j][e] - dc);      // dS^T
        }
      }
#pragma unroll
      for (int k2 = 0; k2 < 2; k2++) {
        uint32_t pa[4], da[4];
        pa[0] = pack2(S[2 * k2][0], S[2 * k2][1]);
        pa[1] = pack2(S[2 * k2][2], S[2 * k2][3]);
        pa[2] = pack2(S[2 * k2 + 1][0], S[2 * k2 + 1][1]);
        pa[3] = pack2(S[2 * k2 + 1][2], S[2 * k2 + 1][3]);
        da[0] = pack2(P[2 * k2][0], P[2 * k2][1]);
        da[1] = pack2(P[2 * k2][2], P[2 * k2][3]);
        da[2] = pack2(P[2 * k2 + 1][0], P[2 * k2 + 1][1]);
        da[3] = pack2(P[2 * k2 + 1][2], P[2 * k2 + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 4; dp++) {
          uint32_t bb[4];
          ldB_kn(sdO, qh * 32 + k2 * 16, dp * 16, lane, bb);
          mma16816(dV[2 * dp], pa, bb[0], bb[1]);
          mma16816(dV[2 * dp + 1], pa, bb[2], bb[3]);
          ldB_kn(sQ, qh * 32 + k2 * 16, dp * 16, lane, bb);
          mma16816(dK[2 * dp], da, bb[0], bb[1]);
          mma16816(dK[2 * dp + 1], da, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
  }

  float* mb = reinterpret_cast<float*>(smem + 16384);
  const int t128 = rg * 32 + lane;
  if (qh == 1) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        mb[(j * 4 + e) * 128 + t128] = dK[j][e];
        mb[(32 + j * 4 + e) * 128 + t128] = dV[j][e];
      }
  }
  __syncthreads();
  if (qh == 0) {
    const int r0 = kt * TILE + rg * 16 + g;
    bf16* k0p = dqkv + (size_t)(b * T + r0) * LD + D + h * DH;
    bf16* k1p = k0p + (size_t)8 * LD;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int c = j * 8 + q4 * 2;
      float a0 = (dK[j][0] + mb[(j * 4 + 0) * 128 + t128]) * SCALE;
      float a1 = (dK[j][1] + mb[(j * 4 + 1) * 128 + t128]) * SCALE;
      float a2 = (dK[j][2] + mb[(j * 4 + 2) * 128 + t128]) * SCALE;
      float a3 = (dK[j][3] + mb[(j * 4 + 3) * 128 + t128]) * SCALE;
      float v0 = dV[j][0] + mb[(32 + j * 4 + 0) * 128 + t128];
      float v1 = dV[j][1] + mb[(32 + j * 4 + 1) * 128 + t128];
      float v2 = dV[j][2] + mb[(32 + j * 4 + 2) * 128 + t128];
      float v3 = dV[j][3] + mb[(32 + j * 4 + 3) * 128 + t128];
      *reinterpret_cast<uint32_t*>(k0p + c) = pack2(a0, a1);
      *reinterpret_cast<uint32_t*>(k1p + c) = pack2(a2, a3);
      *reinterpret_cast<uint32_t*>(k0p + D + c) = pack2(v0, v1);
      *reinterpret_cast<uint32_t*>(k1p + D + c) = pack2(v2, v3);
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ 3. bwd dQ
// Task (b,h,qt): dQ of query rows [qt*64,+64), iterating key tiles 0..qt. Needs fwd (lse), prep (D).
// Writes dqkv[:, h*64..]. smem >= SMEM_DQ.
template <int T, int H>
__device__ __forceinline__ void attn_bwd_dq_tile(const bf16* __restrict__ qkv, const bf16* __restrict__ dout,
                                                 const float* __restrict__ lse, const float* __restrict__ Dr,
                                                 bf16* __restrict__ dqkv, int b, int h, int qt, char* smem) {
  constexpr int D = H * DH, LD = 3 * D;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, rg = w & 3, hf = w >> 2;
  const int g = lane >> 2, q4 = lane & 3;
  const uint32_t sQ = smem_u32(smem), sdO = sQ + 8192, sKV = sQ + 16384;
  const bf16* base = qkv + (size_t)(b * T) * LD + h * DH;
  const bf16* dob = dout + (size_t)(b * T) * D + h * DH;
  const int nb = qt + 1;

  load_tile(sQ, base + (size_t)(qt * TILE) * LD, LD, tid);
  load_tile(sdO, dob + (size_t)(qt * TILE) * D, D, tid);
  load_tile(sKV, base + D, LD, tid);
  load_tile(sKV + 8192, base + 2 * D, LD, tid);
  cp_commit();

  const int r0 = qt * TILE + rg * 16 + g;
  const float* lp = lse + (size_t)(b * H + h) * T + r0;
  const float* dp_ = Dr + (size_t)(b * H + h) * T + r0;
  const float ls0 = lp[0] * LOG2E, ls1 = lp[8] * LOG2E, Dv0 = dp_[0], Dv1 = dp_[8];

  float dQ[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++) dQ[j][0] = dQ[j][1] = dQ[j][2] = dQ[j][3] = 0.f;

  for (int kb = 0; kb < nb; kb++) {
    if (kb + 1 < nb) {
      uint32_t st = sKV + ((kb + 1) & 1) * 16384;
      const bf16* kp = base + D + (size_t)((kb + 1) * TILE) * LD;
      load_tile(st, kp, LD, tid);
      load_tile(st + 8192, kp + D, LD, tid);
      cp_commit();
      cp_wait<1>();
    } else cp_wait<0>();
    __syncthreads();
    const bool diag = (kb == qt);
    if (!(diag && hf * 32 > rg * 16 + 15)) {
      const uint32_t sK = sKV + (kb & 1) * 16384, sV = sK + 8192;
      float S[4][4], P[4][4];
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) S[j][e] = P[j][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        uint32_t qa[4], da[4];
        ldA(sQ, rg * 16, ks * 16, lane, qa);
        ldA(sdO, rg * 16, ks * 16, lane, da);
#pragma unroll
        for (int jp = 0; jp < 2; jp++) {
          uint32_t bb[4];
          ldB_nk(sK, hf * 32 + jp * 16, ks * 16, lane, bb);
          mma16816(S[2 * jp], qa, bb[0], bb[1]);
          mma16816(S[2 * jp + 1], qa, bb[2], bb[3]);
          ldB_nk(sV, hf * 32 + jp * 16, ks * 16, lane, bb);
          mma16816(P[2 * jp], da, bb[0], bb[1]);
          mma16816(P[2 * jp + 1], da, bb[2], bb[3]);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float lc = (e >> 1) ? ls1 : ls0, dc = (e >> 1) ? Dv1 : Dv0;
          float p = exp2f(S[j][e] * SC2 - lc);
          if (diag) {
            int col = hf * 32 + j * 8 + q4 * 2 + (e & 1), row = rg * 16 + g + ((e >> 1) << 3);
            if (col > row) p = 0.f;
          }
          S[j][e] = p * (P[j][e] - dc);  // dS
        }
#pragma unroll
      for (int k2 = 0; k2 < 2; k2++) {
        uint32_t da[4];
        da[0] = pack2(S[2 * k2][0], S[2 * k2][1]);
        da[1] = pack2(S[2 * k2][2], S[2 * k2][3]);
        da[2] = pack2(S[2 * k2 + 1][0], S[2 * k2 + 1][1]);
        da[3] = pack2(S[2 * k2 + 1][2], S[2 * k2 + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 4; dp++) {
          uint32_t bb[4];
          ldB_kn(sK, hf * 32 + k2 * 16, dp * 16, lane, bb);
          mma16816(dQ[2 * dp], da, bb[0], bb[1]);
          mma16816(dQ[2 * dp + 1], da, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
  }

  float* mb = reinterpret_cast<float*>(smem + 16384);
  const int t128 = rg * 32 + lane;
  if (hf == 1) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) mb[(j * 4 + e) * 128 + t128] = dQ[j][e];
  }
  __syncthreads();
  if (hf == 0) {
    bf16* q0p = dqkv + (size_t)(b * T + r0) * LD + h * DH;
    bf16* q1p = q0p + (size_t)8 * LD;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int c = j * 8 + q4 * 2;
      float a0 = (dQ[j][0] + mb[(j * 4 + 0) * 128 + t128]) * SCALE;
      float a1 = (dQ[j][1] + mb[(j * 4 + 1) * 128 + t128]) * SCALE;
      float a2 = (dQ[j][2] + mb[(j * 4 + 2) * 128 + t128]) * SCALE;
      float a3 = (dQ[j][3] + mb[(j * 4 + 3) * 128 + t128]) * SCALE;
      *reinterpret_cast<uint32_t*>(q0p + c) = pack2(a0, a1);
      *reinterpret_cast<uint32_t*>(q1p + c) = pack2(a2, a3);
    }
  }
  __syncthreads();
}

}  // namespace attn
