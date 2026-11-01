#!/usr/bin/env python3
"""One-command refresh: python3 tools/refresh_numbers.py [--dry]
Re-reads repo result files and updates ONLY value/status in data.json (scenes, captions, sources untouched). Prints a diff.
Sources: llm/baseline/results/*.json, llm/ofa/README.md, llm/b200/results/mk_split_100a.txt + bl_gpu1.log, benchmarks/fa_published.json.
A pending value that gets filled becomes 'measured' (A100) or 'interim' (B300); existing statuses are otherwise kept."""
import json, re, os, sys
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, '..', '..'))
DJ = os.path.join(H, '..', 'data.json'); d = json.load(open(DJ))
rd = lambda p: open(os.path.join(R, p)).read()
new = {}  # dotted path -> value
def put(p, v):
    if v is not None: new[p] = v
try:
    fa = json.load(open(f'{R}/benchmarks/fa_published.json'))
    put('fa.launches_kvm', fa['kernel_launches']['kvm']); put('fa.launches_base', fa['kernel_launches']['baseline'])
    t = fa['training']; i = fa['inference']
    put('fa.train_s_kvm', t['seconds_kvm']); put('fa.train_s_base', t['seconds_baseline'])
    put('fa.train_toks_base', t['tok_s_baseline_at_frame']); put('fa.loss_start', t['loss_initial']); put('fa.loss_end', t['loss_final'])
    put('fa.infer_toks_kvm', i['tok_s_kvm']); put('fa.infer_toks_base', i['tok_s_baseline']); put('fa.slower_pct', t['baseline_slower_pct'])
except Exception as e: print('fa_published:', e)
try:
    bl = json.load(open(f'{R}/llm/baseline/results/baseline_a100_20261101.json')); m = bl['training']['megatron_core']
    r = m['sm_restricted']
    put('a100.train_s_base', round(r['seconds']['mean'], 1)); put('a100.train_toks_base', round(r['tok_s']['mean']))
    put('a100.loss_start', round(r['loss_initial']['mean'], 2)); put('a100.loss_end_base', round(r['loss_final']['mean'], 3))
    put('a100.infer_toks_base', round(bl['inference']['vllm']['sm_restricted']['tok_s']['mean'], 1))
    put('a100.kernels_per_step_base', m['full_gpu']['launches']['kernels_per_step'])
    put('a100.launches_base', m['full_gpu']['launches']['kernels_total_extrapolated'])
    c = r['loss_curve_every25_rep1']; d['curves']['megatron_a100']['points'] = [[int(k), round(v, 4)] for k, v in sorted(c.items(), key=lambda kv: int(kv[0]))]
except Exception as e: print('baseline json:', e)
try:
    tr = rd('llm/ofa/README.md')
    mm = re.search(r'S, 4,000 steps, Doom ON.*?\| (\d+) s, \*\*[\d.]+ ms/step\*\*, ([\d.]+)k tok/s, final loss \*\*([\d.]+)\*\*, Doom \*\*([\d.]+) fps', tr)
    if mm: put('a100.train_s_ofa', float(mm[1])); put('a100.train_toks_ofa', round(float(mm[2]) * 1000)); put('a100.loss_end_ofa', float(mm[3])); put('a100.doom_fps', float(mm[4]))
    dm = re.search(r'S, 600 steps Doom ON, 2 runs \| [\d./ ]+ ms/step, decode (\d+) / (\d+) tok/s', tr)
    if dm: put('a100.infer_toks_ofa', round((int(dm[1]) + int(dm[2])) / 2))
except Exception as e: print('ofa README:', e)
try:
    mk = rd('llm/b200/results/mk_split_100a.txt')
    mm = re.search(r'steady state: [\d.]+ ms/step .*?-> (\d+) tok/s;\s+4,000 steps = ([\d.]+) s', mk)
    if mm: put('b300.train_toks_ofa', int(mm[1])); put('b300.train_s_ofa', float(mm[2]))
    lm = re.search(r'loss: step0 ([\d.]+).*s3999 ([\d.]+)', mk)
    if lm: put('b300.loss_start', round(float(lm[1]), 2)); put('b300.loss_end_ofa', float(lm[2]))
    dm = re.search(r'-> (\d+) tok/s\n', mk[mk.index('decode:'):]) if 'decode:' in mk else None
    if dm: put('b300.infer_toks_ofa', int(dm[1]))
    pts = [[0, 0.0]]; pm = re.search(r'loss: (.*)', mk)
    cp = [[int(a), float(b)] for a, b in re.findall(r's?(?:tep)?(\d+) ([\d.]+)', pm[1])] if pm else []
    if cp: d['curves']['ofa_b300']['points'] = cp
except Exception as e: print('mk_split:', e)
try:
    lg = rd('llm/b200/results/bl_gpu1.log')
    v = [float(x) for x in re.findall(r'vllm_S_mps_r\d rc=0.*?"tok_s": \[([\d.]+)\]', lg)]
    if v: put('b300.infer_toks_base', round(sum(v) / len(v), 1))
except Exception as e: print('bl_gpu1.log:', e)
changes = []
for p, v in new.items():
    g, k = p.split('.'); o = d[g][k]
    if o['value'] != v:
        changes.append((p, o['value'], v)); o['value'] = v
        if o['status'] == 'pending': o['status'] = 'interim' if g == 'b300' else 'measured'
for p, a, b in changes: print(f'{p}: {a} -> {b}')
print(f'{len(changes)} value(s) changed' if changes else 'no changes')
print('still pending:', ', '.join(f'{g}.{k}' for g in ('a100', 'b300') for k, o in d[g].items() if isinstance(o, dict) and o.get('value', 1) is None) or 'none')
if '--dry' not in sys.argv: json.dump(d, open(DJ, 'w'), indent=1)
