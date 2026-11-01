# Compare triad_llm --dump-grads output with baseline golden step0.json (per-tensor L2 and sampled values).
import json, sys, numpy as np, os
sys.path.insert(0, os.path.dirname(__file__))
M=os.environ.get('MODEL','S'); L,D={'S':(12,768),'F':(4,2048)}[M]; V=100352; T=512; FF=4*D
g = np.fromfile(sys.argv[1], dtype=np.float32)
d = json.load(open(os.path.expanduser('~/llm/baseline/golden/' + ('' if M=='S' else M+'/') + 'step0.json')))
off = {}; o = 0
def put(n, sz): global o; off[n] = (o, sz); o += sz
put('wte', V*D); put('wpe', T*D)
for l in range(L):
    p = f'L{l:02d}_'
    put(p+'gamma1', D); put(p+'Wqkv', D*3*D); put(p+'Wo', D*D); put(p+'gamma2', D); put(p+'Wup', D*FF); put(p+'Wdown', FF*D)
put('gamma_f', D)
tot = 0; worst = 0
for k, v in d['grads'].items():
    a, n = off[k]; x = g[a:a+n].astype(np.float64); l2 = np.linalg.norm(x); tot += l2**2
    rel = abs(l2 - v['l2']) / (v['l2'] + 1e-30); worst = max(worst, rel)
    sv = np.array(v['sample_vals']); mine = x[v['sample_flat_idx']]
    srel = np.abs(mine - sv).max() / (np.abs(sv).max() + 1e-30)
    if rel > 0.02 or k in ('wte', 'wpe', 'L00_Wqkv', f'L{L-1:02d}_Wdown', 'gamma_f'): print(f'{k:14s} l2 {l2:.5g} vs {v["l2"]:.5g} rel {rel:.3%}  sample max-rel {srel:.2%}')
print('global grad norm', tot**0.5, 'vs', d['global_grad_norm'], ' worst per-tensor l2 rel err', f'{worst:.3%}')
