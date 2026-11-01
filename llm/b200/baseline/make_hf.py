"""Build an HF GPT2LMHeadModel checkpoint (same shape as model S) from golden init weights or a train_pytorch.py --save ckpt.
  python make_hf.py OUT_DIR [ckpt.pt]"""
import sys, os, torch
from transformers import GPT2Config, GPT2LMHeadModel
from common import *
out = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else None
if ck:
    sd = torch.load(ck, map_location="cpu"); m = GPT(); m.load_state_dict(sd)
    w = {n: spec_layout(n, t) for n, t in m.spec_tensors().items()}; w = {k: torch.from_numpy(v) for k, v in w.items()}
else:
    w = init_weights()
cfg = GPT2Config(vocab_size=V, n_positions=T, n_embd=D, n_layer=L, n_head=H, n_inner=4 * D, activation_function="gelu_new",
                 layer_norm_epsilon=1e-5, tie_word_embeddings=True, bos_token_id=100257, eos_token_id=100257,
                 resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
hf = GPT2LMHeadModel(cfg)
with torch.no_grad():
    t = hf.transformer
    t.wte.weight.copy_(w["wte"]); t.wpe.weight.copy_(w["wpe"])
    for l, b in enumerate(t.h):
        p = f"L{l:02d}_"   # HF Conv1D weights are [in,out] == SPEC layout
        b.attn.c_attn.weight.copy_(w[p + "Wqkv"]); b.attn.c_proj.weight.copy_(w[p + "Wo"])
        b.mlp.c_fc.weight.copy_(w[p + "Wup"]); b.mlp.c_proj.weight.copy_(w[p + "Wdown"])
        b.ln_1.weight.copy_(w[p + "gamma1"]); b.ln_2.weight.copy_(w[p + "gamma2"])
        for m_ in (b.attn.c_attn, b.attn.c_proj, b.mlp.c_fc, b.mlp.c_proj): m_.bias.zero_()
        b.ln_1.bias.zero_(); b.ln_2.bias.zero_()
    t.ln_f.weight.copy_(w["gamma_f"]); t.ln_f.bias.zero_()
hf.save_pretrained(out, safe_serialization=True)
print("saved", out)
