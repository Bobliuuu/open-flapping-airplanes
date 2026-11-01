# Builds flat init weights (triad_llm layout) from baseline golden files, and the exact batch-start file.
import numpy as np, os, sys
M = os.environ.get('MODEL', 'S')
g = os.path.expanduser('~/llm/baseline/golden' + ('' if M == 'S' else '/' + M)); out = os.path.expanduser('~/data')
L = {'S': 12, 'F': 4}[M]
def r(n): return np.fromfile(f'{g}/{n}.f32', dtype=np.float32)
parts = [r('w_wte'), r('w_wpe')]
for l in range(L):
    p = f'w_L{l:02d}_'
    parts += [r(p+'gamma1'), r(p+'Wqkv'), r(p+'Wo'), r(p+'gamma2'), r(p+'Wup'), r(p+'Wdown')]
parts.append(r('w_gamma_f'))
flat = np.concatenate([x.ravel() for x in parts]); print('params', flat.size)
flat.tofile(f'{out}/init_flat_{M}.f32')
ntok = os.path.getsize(f'{out}/tinystories_cl100k_train.u32') // 4
st = np.random.default_rng(1234).integers(0, ntok - 512 - 1, size=(4000, 8)).astype(np.uint32)
st.tofile(f'{out}/starts_4000.u32'); print('starts', st.shape, ntok)
