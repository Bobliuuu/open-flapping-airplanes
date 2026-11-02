// gemm_backend.cuh -- selects the GEMM backend for the B200 megakernel build. Both backends are tested on a B200; mma.sync also on the A100.
//   -DTRIAD_GEMM_BACKEND=0 (default) : llm/gemm/gemm.cuh (mma.sync; runs on sm_80 and, via sm_100a compile, on B200)
//   -DTRIAD_GEMM_BACKEND=1           : b200/gemm_tcgen05/gemm_sm100.cuh (TMA + tcgen05; falls back to mma.sync on non-sm_100a targets)
// Uniform interface used by triad.cu:   tg::Args, tg::args_ok(p), tg::prepare(p, layout) [host, no-op for backend 0], tg::smem_bytes(), tg::tile<L,E>(p,tm,tn,smem),
//   tg::block_init() / tg::block_exit() [device, all threads, once per block around its tiles], tg::kTcgen05
#pragma once
#ifndef TRIAD_GEMM_BACKEND
#define TRIAD_GEMM_BACKEND 0
#endif
#if TRIAD_GEMM_BACKEND == 1
#include "gemm_sm100.cuh"
namespace tg {
using Args = gemm::sm100::ArgsTC;
inline bool args_ok(const Args& p) { return gemm::sm100::args_ok(p); }
inline void prepare(Args& p, int layout) { gemm::sm100::prepare(p, layout); }
__host__ __device__ constexpr int smem_bytes() { return gemm::sm100::smem_bytes(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int tm, int tn, char* smem) { gemm::sm100::tile<L, E>(p, tm, tn, smem); }
__device__ __forceinline__ void block_init() { gemm::sm100::block_init(); }   // TMEM once per block (see gemm_sm100.cuh)
__device__ __forceinline__ void block_exit() { gemm::sm100::block_exit(); }
constexpr bool kTcgen05 = true;                                                 // the occupancy API under-reports this kernel
}  // namespace tg
#else
#include "../../gemm/gemm.cuh"
namespace tg {
using Args = gemm::Args;
inline bool args_ok(const Args& p) { return gemm::args_ok(p); }
inline void prepare(Args&, int) {}
__host__ __device__ constexpr int smem_bytes() { return gemm::smem_bytes<>(); }
template <int L, int E> __device__ __forceinline__ void tile(const Args& p, int tm, int tn, char* smem) { gemm::gemm_tile<L, E>(p, tm, tn, smem); }
__device__ __forceinline__ void block_init() {}
__device__ __forceinline__ void block_exit() {}
constexpr bool kTcgen05 = false;
}  // namespace tg
#endif
