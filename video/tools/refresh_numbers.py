#!/usr/bin/env python3
"""One-command refresh: python3 tools/refresh_numbers.py [--dry]
Re-reads repo result files and updates ONLY value/status in data.json (scenes, captions, sources untouched; curves replaced). Prints a diff.
Sources: llm/baseline/results/baseline_v2_a100_*.json (+ v1 for the old no-TE number), llm/baseline/launch_count_notes.md (numbers live in the v2 JSON),
llm/ofa/README.md, llm/b200/results/{mk_split_100a.txt,B300_REPORT.md,bl_gpu1.log}, benchmarks/fa_published.json.
Only keys already present in data.json are touched. A pending value that gets filled becomes 'measured' (A100) or 'interim' (B300)."""
import json, re, os, sys, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, '..', '..'))
DJ = os.path.join(H, '..', 'data.json'); d = json.load(open(DJ))
rd = lambda p: open(os.path.join(R, p)).read()
new = {}
def put(p, v):
    if v is not None: new[p] = v
def guard(name, fn):
    try: fn()
    except Exception as e: print(f'[{name}] skipped: {e!r}')
def fa():
    f = json.load(open(f'{R}/benchmarks/fa_published.json')); t = f['training']; i = f['inference']
    put('fa.launches_kvm', f['kernel_launches']['kvm']); put('fa.launches_base', f['kernel_launches']['baseline'])
    put('fa.train_s_kvm', t['seconds_kvm']); put('fa.train_s_base', t['seconds_baseline']); put('fa.train_toks_base', t['tok_s_baseline_at_frame'])
    put('fa.loss_start', t['loss_initial']); put('fa.loss_end', t['loss_final']); put('fa.infer_toks_kvm', i['tok_s_kvm']); put('fa.infer_toks_base', i['tok_s_baseline'])
    put('fa.slower_pct', t['baseline_slower_pct'])
def v2():
    p = sorted(glob.glob(f'{R}/llm/baseline/results/baseline_v2_a100_*.json'))[-1]; b = json.load(open(p)); T = b['training']; I = b['inference']['vllm']; L = b['launch_counts']
    te = T['megatron_te']; m = lambda x: x['mean']
    put('a100.train_s_base', round(m(te['S']['sm_restricted']['seconds']), 1)); put('a100.train_toks_base', round(m(te['S']['sm_restricted']['tok_s'])))
    put('a100.loss_start', round(m(te['S']['sm_restricted']['loss_initial']), 2)); put('a100.loss_end_base', round(m(te['S']['sm_restricted']['loss_final']), 3))
    put('a100.infer_toks_base', round(m(I['S']['sm_restricted']['tok_s']), 1))
    pe = T['pytorch_eager']['S']['sm_restricted']; put('a100.train_s_eager', round(m(pe['seconds']), 1)); put('a100.train_toks_eager', round(m(pe['tok_s'])))
    lt = L['training']; put('a100.kernels_per_step_base', lt['megatron_te']['S']['a_kernels_per_step']); put('a100.kernels_per_step_base_F', lt['megatron_te']['F']['a_kernels_per_step'])
    put('a100.kernels_per_step_eager', lt['pytorch_eager']['S']['a_kernels_per_step']); put('a100.kernels_per_step_eager_F', lt['pytorch_eager']['F']['a_kernels_per_step'])
    put('a100.host_calls_graphed_per_step_te_S', lt['megatron_te']['S']['c_host_launch_calls_per_step_one_graph']); put('a100.host_calls_graphed_per_step_te_F', lt['megatron_te']['F']['c_host_launch_calls_per_step_one_graph'])
    put('a100.host_calls_graphed_per_step_eager', lt['pytorch_eager']['S']['c_host_launch_calls_per_step_one_graph'])
    row = [r for r in L['demo_equivalent']['rows'] if r['system'] == 'megatron_te' and r['model'] == 'S'][0]['fixed_fa_tokens_49604']
    put('a100.launch_def_a', round(row['a_gpu_kernel_executions'])); put('a100.launch_def_b', round(row['b_host_launch_calls']))
    put('a100.launch_def_c', round(row['c_host_launch_calls_graphed'])); put('a100.launch_def_c_ideal', round(row['c_with_ideal_1_launch_per_token'])); put('a100.launch_fa_tokens', round(row['served_tokens']))
    put('a100.F_te_s_full', round(m(te['F']['full_gpu']['seconds']), 1)); put('a100.F_te_s_restricted', round(m(te['F']['sm_restricted']['seconds']), 1)); put('a100.F_te_loss_end', round(m(te['F']['full_gpu']['loss_final']), 3))
    pf = T['pytorch_eager']['F']; put('a100.F_eager_s_full', round(m(pf['full_gpu']['seconds']), 1)); put('a100.F_eager_s_restricted', round(m(pf['sm_restricted']['seconds']), 1))
    put('a100.F_vllm_toks_restricted', round(m(I['F']['sm_restricted']['tok_s']), 1)); put('a100.F_vllm_toks_full', round(m(I['F']['full_gpu']['tok_s']), 1))
    c = te['S']['sm_restricted']['loss_curve_every25_rep1']; d['curves']['megatron_a100']['points'] = [[int(k), round(x, 4)] for k, x in sorted(c.items(), key=lambda kv: int(kv[0]))]
    d['curves']['megatron_a100']['source'] = os.path.basename(p) + ' megatron_te.S.sm_restricted loss_curve_every25_rep1'
def v1():
    b = json.load(open(f'{R}/llm/baseline/results/baseline_a100_20261101.json'))
    put('a100.train_s_notE', round(b['training']['megatron_core']['sm_restricted']['seconds']['mean'], 1))
def ofa():
    tr = rd('llm/ofa/README.md')
    mm = re.search(r'S, 4,000 steps, Doom ON.*?\| (\d+) s, \*\*[\d.]+ ms/step\*\*, ([\d.]+)k tok/s, final loss \*\*([\d.]+)\*\*, Doom \*\*([\d.]+) fps', tr)
    if mm: put('a100.train_s_ofa', float(mm[1])); put('a100.train_toks_ofa', round(float(mm[2]) * 1000)); put('a100.loss_end_ofa', float(mm[3])); put('a100.doom_fps', float(mm[4]))
    dm = re.search(r'S, 600 steps Doom ON, 2 runs \| [\d./ ]+ ms/step, decode (\d+) / (\d+) tok/s', tr)
    if dm: put('a100.infer_toks_ofa', round((int(dm[1]) + int(dm[2])) / 2))
def mk():
    t = rd('llm/b200/results/mk_split_100a.txt')
    mm = re.search(r'steady state: [\d.]+ ms/step .*?-> (\d+) tok/s;\s+4,000 steps = ([\d.]+) s', t)
    if mm: put('b300.train_toks_ofa', int(mm[1])); put('b300.train_s_ofa', float(mm[2]))
    lm = re.search(r'loss: step0 ([\d.]+).*s3999 ([\d.]+)', t)
    if lm: put('b300.loss_start', round(float(lm[1]), 2)); put('b300.loss_end_ofa', float(lm[2]))
    dm = re.search(r'-> (\d+) tok/s\n', t[t.index('decode:'):]) if 'decode:' in t else None
    if dm: put('b300.infer_toks_ofa', int(dm[1]))
    pm = re.search(r'loss: (.*)', t)
    cp = [[int(a), float(b)] for a, b in re.findall(r'(?:step| s)(\d+) ([\d.]+)', pm[1])] if pm else []
    if cp: d['curves']['ofa_b300']['points'] = cp
def rep():
    t = rd('llm/b200/results/B300_REPORT.md')
    mg = re.search(r'\| Megatron-core local spec \(no TE\), MPS-116 \| \d+ \| ([\d.]+) s \| ([\d.]+)k \| ([\d.]+) \|', t)
    if mg: put('b300.train_s_base', float(mg[1])); put('b300.train_toks_base', round(float(mg[2]) * 1000)); put('b300.loss_end_base', float(mg[3]))
    pe = re.search(r'\| PyTorch eager, MPS-116 SMs \| \d+ \| \*\*([\d.]+) s\*\* / [\d.]+ s \| ([\d.]+)k \|', t)
    if pe: put('b300.train_s_eager', float(pe[1])); put('b300.train_toks_eager', round(float(pe[2]) * 1000))
    fe = re.search(r'\| PyTorch eager, model F, MPS-116 \| \d+ \| ([\d.]+) s', t)
    if fe: put('b300.F_eager_s', float(fe[1]))
    fm = re.search(r'\| \*\*F\*\*, train\+decode \| [^|]+\| \*\*([\d.]+) s\*\* \| [\d.]+ \| [\d.]+k \| \*\*(\d+)\*\*', t)
    if fm: put('b300.F_ofa_s', float(fm[1])); put('b300.F_decode_toks', int(fm[2]))
    lg = rd('llm/b200/results/bl_gpu1.log')
    v = [float(x) for x in re.findall(r'vllm_S_mps_r\d rc=0.*?"tok_s": \[([\d.]+)\]', lg)]
    if v: put('b300.infer_toks_base', round(sum(v) / len(v), 1))
for n, f in [('fa', fa), ('baseline v2', v2), ('baseline v1', v1), ('ofa README', ofa), ('mk_split', mk), ('B300_REPORT', rep)]: guard(n, f)
changes = []
for p, v in new.items():
    g, k = p.split('.')
    if k not in d.get(g, {}): print('skip (key not in data.json):', p); continue
    o = d[g][k]
    if o['value'] != v:
        changes.append((p, o['value'], v)); o['value'] = v
        if o['status'] == 'pending': o['status'] = 'interim' if g == 'b300' else 'measured'
for p, a, b in changes: print(f'{p}: {a} -> {b}')
print(f'{len(changes)} value(s) changed' if changes else 'no changes')
print('still pending:', ', '.join(f'{g}.{k}' for g in ('a100', 'b300') for k, o in d[g].items() if isinstance(o, dict) and o.get('value', 1) is None) or 'none')
if '--dry' not in sys.argv: json.dump(d, open(DJ, 'w'), indent=1)
