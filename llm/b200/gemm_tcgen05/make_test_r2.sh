#!/usr/bin/env bash
# make_test_r2.sh : derive test_gemm_r2.cu from llm/gemm/test_gemm.cu for the round-2 backend (task tile 128 x tg::TILE_N).
# Adds: --nocheck (skip fp32 reference/cuBLAS-compare/bit-exact; timing vs cuBLAS only), --iters N, tile count uses tg::TILE_N.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import re
s = open("../../gemm/test_gemm.cu").read()
s = s.replace('#include "gemm.cuh"', '#include "gemm_backend_r2.cuh"')
s = re.sub(r'\bArgs\b', 'tg::Args', s)
s = s.replace('gemm_tile<L, E>(p,', 'tg::tile<L, E>(p,')
s = s.replace('constexpr int SM = smem_bytes();', 'constexpr int SM = tg::smem_bytes();')
s = s.replace('nsm, smem_bytes());', 'nsm, tg::smem_bytes());')
s = s.replace('p.A = A; p.B = B; p.C = C; p.aux = aux;', 'p.A = A; p.B = B; p.C = C; p.aux = aux; tg::prepare(p, L, wide, nsub);')
s = s.replace('int tm = p.M / BM, tn = p.N / BN;', 'int tm = tg::ntasks(p), tn = 1;')
s = s.replace('int tm = M / BM, tn = N / BN, nt = tm * tn;', 'int tm = tg::ntasks(p), tn = 1, nt = tm * tn;')
s = s.replace('tg::tile<L, E>(p, t % tilesM, t / tilesM, smem)', 'tg::tile<L, E>(p, t, 0, smem)')
assert s.count('tg::ntasks') == 2 and s.count('tg::prepare') == 1 and 'tg::tile<L, E>' in s and s.count('tg::smem_bytes') == 2
s = s.replace('int M = cs.M, N = cs.N, K = cs.K, L = cs.L, E = cs.E;', 'int M = cs.M, N = cs.N, K = cs.K, L = cs.L, E = cs.E;\n    int wide = 0, nsub = 1; tg::choose(M, N, slotsOv, &wide, &nsub, wideOv, nsubOv);')
s = s.replace('bool cfgM = false,', 'int wideOv = -1, nsubOv = 0, slotsOv = 148; bool cfgM = false,')
s = s.replace('else if (!strcmp(argv[i], "--nopersist")) persist = false;', 'else if (!strcmp(argv[i], "--nopersist")) persist = false;\n    else if (!strcmp(argv[i], "--wide") && i + 1 < argc) wideOv = atoi(argv[++i]);\n    else if (!strcmp(argv[i], "--nsub") && i + 1 < argc) nsubOv = atoi(argv[++i]);\n    else if (!strcmp(argv[i], "--slots") && i + 1 < argc) slotsOv = atoi(argv[++i]);',1)
# --nocheck: skip reference + comparisons (they cost minutes for the big shapes); still runs the kernel once and times ours vs cuBLAS
s = s.replace('bool cfgM = false, bench = true, persist = true;', 'bool cfgM = false, bench = true, persist = true, nocheck = false; int itersOv = 0;')
s = s.replace('else if (!strcmp(argv[i], "--nopersist")) persist = false;', 'else if (!strcmp(argv[i], "--nopersist")) persist = false;\n    else if (!strcmp(argv[i], "--nocheck")) { nocheck = true; persist = false; }\n    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) itersOv = atoi(argv[++i]);')
s = s.replace('ref_gemm<<<dim3(N / 64, M / 64), 256>>>(A, B, ref, M, N, K, p.lda, p.ldb, akc, bkc);', 'if (!nocheck) ref_gemm<<<dim3(N / 64, M / 64), 256>>>(A, B, ref, M, N, K, p.lda, p.ldb, akc, bkc);')
s = s.replace('Stat s = compare(C, cbf, N, nullptr, 0, ref, E, aux, N, C0, M, N);\n    bool ok = s.rmax <= tol;', 'Stat s{}; if (!nocheck) s = compare(C, cbf, N, nullptr, 0, ref, E, aux, N, C0, M, N);\n    bool ok = s.rmax <= tol;')
s = s.replace('if (E == EPI_GELU) {  // check saved pre-activation too', 'if (E == EPI_GELU && !nocheck) {  // check saved pre-activation too')
s = s.replace('Stat sc = compare(CB_, cbf, N, nullptr, 0, ref, EXPECT_IDENT, nullptr, 0, C0, M, N);', 'Stat sc{}; if (!nocheck) sc = compare(CB_, cbf, N, nullptr, 0, ref, EXPECT_IDENT, nullptr, 0, C0, M, N);')
s = s.replace('if (E == EPI_BF16 || E == EPI_F32) {  // ours vs cuBLAS directly', 'if (!nocheck && (E == EPI_BF16 || E == EPI_F32)) {  // ours vs cuBLAS directly')
s = s.replace('if (E == EPI_RES && persist) {  // C aliases R', 'if (E == EPI_RES && persist && !nocheck) {  // C aliases R')
s = s.replace('int it = (2.0 * M * N * K > 5e11) ? 5 : 20;', 'int it = itersOv ? itersOv : ((2.0 * M * N * K > 5e11) ? 5 : 20);')
s = s.replace('%5.2f  %s\\n"', '%5.2f  %s cfg%d\\n"').replace('ok ? "PASS" : "FAIL");\n    fflush', 'ok ? "PASS" : "FAIL", wide * 10 + nsub);\n    fflush')
s = s.replace('    init = true;\n', '    { int o1 = -1, o2 = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o1, gemm_grid<L, E>, 256, SM); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2, gemm_persist<L, E>, 256, SM); printf("  occupancy (blocks/SM) grid=%d persist=%d\\n", o1, o2); }\n    init = true;\n', 1)
s = s.replace('tg::tile<L, E>(p, t % tilesM, t / tilesM, smem);', 'tg::tile<L, E>(p, t % tilesM, t / tilesM, smem + SMEM_SHIFT);')
s = s.replace('constexpr int SM = tg::smem_bytes();', 'constexpr int SM = tg::smem_bytes() + SMEM_SHIFT;')
s = s.replace('using namespace gemm;', 'using namespace gemm;\n#ifndef SMEM_SHIFT\n#define SMEM_SHIFT 0\n#endif', 1)
s = s.replace('    tg::tile<L, E>(p, t % tilesM, t / tilesM, smem + SMEM_SHIFT);\n  }', '''    tg::tile<L, E>(p, t % tilesM, t / tilesM, smem + SMEM_SHIFT);
#ifdef JOBLOOP_EMU
    { __shared__ volatile int s_sel[4]; __threadfence(); __syncthreads();
      if (threadIdx.x == 0) { atomicAdd(&g_emu[0], 1); s_sel[0] = (int)g_emu[1]; __nanosleep(100); }
      __syncthreads(); int rr = s_sel[0]; __syncthreads(); (void)rr; }
#endif
  }''', 1)
s = s.replace('using namespace gemm;', 'using namespace gemm;\n__device__ unsigned g_emu[4];', 1)
s = "// GENERATED by make_test_r2.sh from llm/gemm/test_gemm.cu (round 2, B300). Build: see modal/app2.py.\n" + s
open("test_gemm_r2.cu", "w").write(s)
print("wrote test_gemm_r2.cu")
PY
