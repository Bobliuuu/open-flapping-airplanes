#!/usr/bin/env python3
"""Merge a benchmark run (demo --bench) with Flapping Airplanes' published numbers and derive the comparison.
usage: compare.py results/a100_YYYYMMDD.json  ->  results/summary.md, ../site/data/benchmarks.json, ../site/data/benchmarks.js
       compare.py results/b200_YYYYMMDD.json --summary results/summary_b200.md --no-site   (another GPU; leaves summary.md and the site alone)"""
import json, sys, os, datetime
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
summary_path = args[args.index('--summary') + 1] if '--summary' in args else os.path.join(here, 'results', 'summary.md')
write_site = '--no-site' not in args
pos = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] != '--summary')]
src = pos[0] if pos else os.path.join(here, 'results', 'latest.json')
run = json.load(open(src)); fa = json.load(open(os.path.join(here, 'fa_published.json')))
m = {k: v for k, v in run['metrics'].items()}
g = lambda k: m[k]['mean'] if k in m else None
sd = lambda k: m[k]['sd'] if k in m else None
def slower(mega, base):               # how much slower the baseline is than the megakernel, in percent (negative = baseline faster)
    return (1 - base / mega) * 100 if mega and base else None
STEPS = 4000
mega_steps, mega_toks = g('mega_all.train_steps_s'), g('mega_all.infer_tok_s')
rows = []
for name, label in [('eager', 'per-op kernels, eager launches (the method the video uses)'), ('graph', 'per-op kernels, CUDA graphs (best case for the baseline)')]:
    rows.append({
        'baseline': name, 'label': label,
        'train_steps_s': g(f'{name}_train.train_steps_s'), 'infer_tok_s': g(f'{name}_infer.infer_tok_s'),
        'train_slower_pct': slower(mega_steps, g(f'{name}_train.train_steps_s')),
        'infer_slower_pct': slower(mega_toks, g(f'{name}_infer.infer_tok_s')),
        'seconds_for_4000_steps': STEPS / g(f'{name}_train.train_steps_s'),
        'gpu_equivalents': mega_steps / g(f'{name}_train.train_steps_s') + mega_toks / g(f'{name}_infer.infer_tok_s'),
    })
for name, label in [('streams_both', 'both jobs in one process, two CUDA-graph streams'), ('procs_both', 'two separate processes sharing the GPU (time-sliced)')]:
    rows.append({'baseline': name, 'label': label, 'train_steps_s': g(f'{name}.train_steps_s'), 'infer_tok_s': g(f'{name}.infer_tok_s'),
                 'train_slower_pct': slower(mega_steps, g(f'{name}.train_steps_s')), 'infer_slower_pct': slower(mega_toks, g(f'{name}.infer_tok_s')),
                 'seconds_for_4000_steps': STEPS / g(f'{name}.train_steps_s'), 'gpu_equivalents': None})
sm = {k: g(f'mega_all_live.sm_{k}') for k in ('train', 'infer', 'doom')}
sm['total'] = sum(v for v in sm.values() if v); sm['gpu_sms'] = run['meta']['sms']
ours = {
    'gpu': run['meta']['gpu'], 'sms': run['meta']['sms'], 'reps': run['meta']['reps'], 'secs': run['meta']['secs_per_measurement'], 'date': run['meta']['date'], 'model': run['meta']['model'],
    'mega_all': {k: {'mean': g(f'mega_all.{k}'), 'sd': sd(f'mega_all.{k}')} for k in ('train_steps_s', 'train_tok_s', 'infer_tok_s', 'doom_fps')},
    'mega_live': {k: {'mean': g(f'mega_all_live.{k}'), 'sd': sd(f'mega_all_live.{k}')} for k in ('train_steps_s', 'infer_tok_s', 'doom_fps')},
    'mega_alone': {'train_steps_s': g('mega_train.train_steps_s'), 'infer_tok_s': g('mega_infer.infer_tok_s'), 'doom_fps_capacity': g('mega_doom.doom_fps')},
    'seconds_for_4000_steps_mega': STEPS / mega_steps,
    'kernel_launches_per_step_perop': run['meta']['launches_per_step_perop'], 'kernel_launches_per_token_perop': run['meta']['launches_per_token_perop'],
    'sm_busy_equiv': sm,
}
# launches the per-op programs need for the same work as 4,000 training steps plus serving for the same time the megakernel needs
t = STEPS / mega_steps; toks = g('mega_all_live.infer_tok_s') * t
ours['same_work'] = {'steps': STEPS, 'tokens': toks, 'seconds_mega': t,
                     'launches_perop': STEPS * run['meta']['launches_per_step_perop'] + toks * run['meta']['launches_per_token_perop'], 'launches_mega': 1}
out = {'ours': ours, 'baselines': rows, 'fa': fa, 'all_metrics': m,
       'generated': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'source_file': os.path.basename(src)}
if write_site:
    os.makedirs(os.path.join(here, '..', 'site', 'data'), exist_ok=True)
    json.dump(out, open(os.path.join(here, '..', 'site', 'data', 'benchmarks.json'), 'w'), indent=1)
    open(os.path.join(here, '..', 'site', 'data', 'benchmarks.js'), 'w').write('window.BENCH = ' + json.dumps(out) + ';\n')
f = lambda v, d=0: ('%.*f' % (d, v)) if v is not None else 'n/a'
L = []
L.append(f"# Benchmark summary ({ours['gpu']}, {ours['reps']} reps x {ours['secs']:.0f} s, {ours['date']})\n")
L.append('| metric | mean | sd |\n|---|---|---|')
for k, v in ours['mega_all'].items(): L.append(f"| megakernel, all three at once: {k} | {f(v['mean'], 1)} | {f(v['sd'], 1)} |")
L.append('\n| baseline (each job alone unless stated) | train steps/s | infer tok/s | baseline slower than megakernel (train / infer) | 4,000 steps take | GPU-equivalents needed |\n|---|---|---|---|---|---|')
for r in rows: L.append(f"| {r['label']} | {f(r['train_steps_s'])} | {f(r['infer_tok_s'])} | {f(r['train_slower_pct'], 1)}% / {f(r['infer_slower_pct'], 1)}% | {f(r['seconds_for_4000_steps'], 3)} s | {f(r['gpu_equivalents'], 2)} |")
L.append(f"\nMegakernel: 4,000 steps take {ours['seconds_for_4000_steps_mega']:.3f} s. Flapping Airplanes (B200): 46.1 s vs 70.1 s, baseline 34% slower, both jobs 34% slower, 1 vs 514,562 launches.")
open(summary_path, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
