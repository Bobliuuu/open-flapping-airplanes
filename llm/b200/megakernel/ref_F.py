# PyTorch reference for model MODEL (default F): N steps with the same init/batches/optimizer as the baseline.
# usage: MODEL=F python ref_F.py <nsteps> <losses_out.json> [weights_after_step1.f32]
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/llm/baseline'))
import common as C
n = int(sys.argv[1]); outj = sys.argv[2]; wout = sys.argv[3] if len(sys.argv) > 3 else None
tok = np.fromfile(C.TRAIN_BIN, dtype=np.uint32); starts = np.fromfile(os.path.expanduser('~/data/starts_4000.u32'), dtype=np.uint32).reshape(-1, C.B)
model = C.GPT().cuda(); model.load_spec(C.init_weights()); opt = C.make_optimizer(model)
def batch(s):
    a = np.stack([tok[st:st + C.T + 1] for st in starts[s]]).astype(np.int64); a = torch.from_numpy(a).cuda()
    return a[:, :-1].contiguous(), a[:, 1:].contiguous()
losses = []
for s in range(n):
    for g in opt.param_groups: g['lr'] = C.lr_at(s)
    x, y = batch(s)
    with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); losses.append(loss.item())
    if s == 0 and wout:
        parts = [C.spec_layout(k, v) for k, v in model.spec_tensors().items()]
        # Layout order: wte, wpe, per layer [g1, Wqkv, Wo, g2, Wup, Wdn], gf
        t = dict(zip(model.spec_tensors().keys(), parts)); fl = [t['wte'], t['wpe']]
        for l in range(C.L):
            p = f'L{l:02d}_'; fl += [t[p+'gamma1'], t[p+'Wqkv'], t[p+'Wo'], t[p+'gamma2'], t[p+'Wup'], t[p+'Wdown']]
        fl.append(t['gamma_f']); np.concatenate([x.ravel() for x in fl]).astype('<f4').tofile(wout)
json.dump({'losses': losses}, open(outj, 'w')); print('done', losses[:3], losses[-3:])
