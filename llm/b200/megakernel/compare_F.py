# Compare triad_llm (MODEL_F) outputs with the PyTorch reference: step-1 updated weights and the loss curve.
# usage: python compare_F.py <w_triad_after1.f32> <ref_w_after1.f32> <init_flat.f32> <loss_triad.txt> <ref_losses.json>
import sys, json, numpy as np
wt, wr, w0, lt, lr = sys.argv[1:6]
A = np.fromfile(wt, np.float32); R = np.fromfile(wr, np.float32); I = np.fromfile(w0, np.float32)
dA, dR = A - I, R - I
print('params', A.size, ' |ref delta| rms %.3e  |triad-ref| rms %.3e' % (np.sqrt((dR**2).mean()), np.sqrt(((dA-dR)**2).mean())))
print('weights after 1 step: max |triad-ref| = %.3e (lr step = 6e-6); cosine(delta) = %.4f; frac elements with same delta sign = %.4f' %
      (np.abs(A-R).max(), (dA*dR).sum()/np.sqrt((dA**2).sum()*(dR**2).sum()), (np.sign(dA)==np.sign(dR)).mean()))
T = np.array([float(l.split()[1]) for l in open(lt)]); Rr = np.array(json.load(open(lr))['losses'])
n = min(len(T), len(Rr)); print('loss step0 triad %.4f ref %.4f' % (T[0], Rr[0]))
for s in (0, 1, 2, 5, 10, 20, 30, 50, 75, n-1): print('  step %3d  triad %.4f  ref %.4f  diff %+.4f' % (s, T[s], Rr[s], T[s]-Rr[s]))
print('max |diff| over %d steps %.4f; mean diff %+.4f; mean |diff| steps 50-%d: %.4f' % (n, np.abs(T[:n]-Rr[:n]).max(), (T[:n]-Rr[:n]).mean(), n, np.abs(T[50:n]-Rr[50:n]).mean()))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4)); plt.plot(Rr[:n], label='PyTorch (bf16 autocast, fused AdamW)'); plt.plot(T[:n], '--', label='triad_llm megakernel'); plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.title('Config F: loss, first %d steps' % n); plt.tight_layout(); plt.savefig('lossF_overlay.png', dpi=130); print('wrote lossF_overlay.png')
except Exception as e: print('no plot:', e)
