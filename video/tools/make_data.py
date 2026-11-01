#!/usr/bin/env python3
"""Bootstrap video/data.json from the repo result files. Refuses to overwrite unless --force.
After bootstrapping, edit data.json by hand: set a "pending" number's value and status to "measured"."""
import json, sys, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(os.path.dirname(__file__), '..', 'data.json')
if os.path.exists(OUT) and '--force' not in sys.argv:
    sys.exit('data.json exists; use --force to regenerate (this discards hand edits)')
bl = json.load(open(f'{R}/llm/baseline/results/baseline_a100_20261101.json'))
fa = json.load(open(f'{R}/benchmarks/fa_published.json'))
curve = bl['training']['megatron_core']['sm_restricted']['loss_curve_every25_rep1']
mega = [[int(k), round(v, 4)] for k, v in sorted(curve.items(), key=lambda kv: int(kv[0]))]

def N(value, source, unit='', status=None, note=''):
    return {'value': value, 'unit': unit, 'status': status or ('pending' if value is None else 'measured'), 'source': source, 'note': note}

BL = 'llm/baseline/results/baseline_a100_20261101.json'
TR = 'llm/ofa/README.md'
MK = 'llm/b200/results/mk_split_100a.txt'
VB = 'llm/b200/results/bl_gpu1.log'
FAJ = 'benchmarks/fa_published.json'
data = {
 'meta': {'fps': 30, 'width': 1920, 'height': 1080,
          'disclaimer': 'Independent reproduction. Not affiliated with, endorsed by, or based on code from Flapping Airplanes.',
          'edit_note': 'value=null renders as "pending" in the video. status: measured | published | pending. Derived percentages are computed in the page as 1 - slow/fast (FA convention: 70.1 s vs 46.1 s = 34%).'},
 'hw': {
  'a100': {'label': 'A100', 'sms': 108, 'split': [85, 17, 6], 'source': 'README.md, llm/ofa/README.md'},
  'b300': {'label': 'B300', 'sms': 148, 'split': [116, 24, 8], 'source': MK + ' (GPU reported: NVIDIA B300 SXM6 AC, 148 SMs; FA B200 proportions (116/24/8))'},
  'fa':   {'label': 'FA B200', 'sms': 148, 'split': [116, 24, 8], 'source': FAJ}},
 'fa': {  # published by Flapping Airplanes, read from their video
  'launches_kvm': N(1, FAJ, status='published'),
  'launches_base': N(514562, FAJ, status='published'),
  'train_s_kvm': N(46.1, FAJ, 's', 'published'), 'train_s_base': N(70.1, FAJ, 's', 'published'),
  'train_toks_kvm': N(355000, FAJ + ' (353k-361k read off video)', 'tok/s', 'published'),
  'train_toks_base': N(230000, FAJ, 'tok/s', 'published'),
  'infer_toks_kvm': N(1076, FAJ, 'tok/s', 'published'), 'infer_toks_base': N(710, FAJ, 'tok/s', 'published'),
  'loss_start': N(11.5, FAJ, '', 'published'), 'loss_end': N(2.1094, FAJ, '', 'published'),
  'slower_pct': N(34, FAJ, '%', 'published')},
 'a100': {
  'launches_ofa': N(1, 'one persistent launch (README.md)'),
  'launches_base': N(3832000, BL + ' training.megatron_core.full_gpu.launches.kernels_total_extrapolated (958 kernels/step x 4000, CUPTI, training only)', ''),
  'kernels_per_step_base': N(958, BL),
  'train_s_ofa': N(224, TR + ' (S, 4000 steps, Doom ON + dashboard, 85/17/6 split)', 's'),
  'train_s_base': N(round(bl['training']['megatron_core']['sm_restricted']['seconds']['mean'], 1), BL + ' Megatron-core, 4000 steps, SM-restricted (MPS 79% = 85 SMs), mean of 3', 's'),
  'train_toks_ofa': N(73200, TR, 'tok/s'),
  'train_toks_base': N(round(bl['training']['megatron_core']['sm_restricted']['tok_s']['mean']), BL + ' Megatron-core SM-restricted', 'tok/s'),
  'infer_toks_ofa': N(604, TR + ' (decode 611 / 597 tok/s, two 600-step runs, Doom ON; mean)', 'tok/s'),
  'infer_toks_base': N(round(bl['inference']['vllm']['sm_restricted']['tok_s']['mean'], 1), BL + ' vLLM, MPS 16% (17 SMs), batch 1', 'tok/s'),
  'loss_start': N(round(bl['training']['megatron_core']['sm_restricted']['loss_initial']['mean'], 2), BL),
  'loss_end_ofa': N(2.1490, TR + ' (4000 steps)'),
  'loss_end_base': N(round(bl['training']['megatron_core']['sm_restricted']['loss_final']['mean'], 3), BL),
  'doom_fps': N(35.0, TR + ' (Doom on 6 SMs)', 'fps'),
  'doom_frame_ms_serial': N(24, 'README.md (Doom pipeline section)', 'ms'),
  'doom_frame_ms_parallel': N(6.7, 'README.md (Doom pipeline section)', 'ms'),
  'model': 'S: 12 layers, d=768, 162M params, 4,096 tokens/step, cl100k vocab (100,352 padded)'},
 'b300': {
  'launches_ofa': N(1, MK, status='interim'),
  'launches_base': N(None, 'B300 launch count not run yet'),
  'train_s_ofa': N(87.1, MK + ' (4000 steps, 116/24/8 split)', 's'),
  'train_s_base': N(None, 'B300 Megatron restricted run not extracted yet (llm/b200/results/bl_gpu1.log is truncated)', 's'),
  'train_toks_ofa': N(188185, MK, 'tok/s'),
  'train_toks_base': N(None, 'pending, see train_s_base', 'tok/s'),
  'infer_toks_ofa': N(1762, MK + ' (decode while training)', 'tok/s'),
  'infer_toks_base': N(821.6, VB + ' vLLM, MPS 17%, mean of 2 runs (825.0, 818.1)', 'tok/s'),
  'loss_start': N(11.71, MK),
  'loss_end_ofa': N(2.1575, MK + ' (step 3999)'),
  'loss_end_base': N(None, 'pending'),
  'doom_fps': N(None, 'Doom on B300 not measured yet', 'fps')},
 'curves': {
  'megatron_a100': {'source': BL + ' loss_curve_every25_rep1 (Megatron-core, SM-restricted)', 'points': mega},
  'ofa_b300': {'source': MK, 'points': [[0, 11.7106], [1, 11.2639], [2, 10.7522], [5, 9.9998], [10, 9.6331], [20, 8.5514], [50, 5.6934], [100, 4.4137], [200, 4.0631], [500, 3.6659], [1000, 2.9972], [2000, 2.4032], [3000, 2.2013], [3999, 2.1575]]}},
 'text_sample': {'prompt': 'What is a megakernel?', 'source': BL + ' inference.text_sample (vLLM on the trained weights; open-flapping-airplanes live text is shown in the recording)'},
}
json.dump(data, open(OUT, 'w'), indent=1)
print('wrote', OUT)
