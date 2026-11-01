"""Model S with Megatron-core GPTModel (local layer spec: no Transformer Engine / apex), bf16 params + fp32 main
params in Megatron's own optimizer (get_megatron_optimizer), Megatron DDP wrapper (world size 1), no grad clipping.
Same batches / LR schedule / hyper-parameters as train_pytorch.py. Initial weights are copied from golden init.
  python megatron_train.py --out r.json [--steps 4000] [--count_launches] [--no_load_golden]"""
import argparse, json, os, time, sys
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
import te_env  # noqa: F401  (TE import shims; must precede megatron/TE imports)
import numpy as np, torch, torch.nn.functional as F
from common import *
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.distributed import DistributedDataParallel as DDP, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import Float16Module

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=STEPS); ap.add_argument("--out"); ap.add_argument("--count_launches", action="store_true")
ap.add_argument("--no_load_golden", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--impl", default="local", choices=["local", "te"])  # local = v1 baseline (no TE); te = Transformer Engine spec
ap.add_argument("--save", default=None)
ap.add_argument("--count3", action="store_true"); ap.add_argument("--cuda_graph", action="store_true")
a = ap.parse_args()
dev = "cuda"
torch.cuda.set_device(0)
torch.distributed.init_process_group("nccl", rank=0, world_size=1)
parallel_state.initialize_model_parallel(1, 1)
tensor_parallel.model_parallel_cuda_manual_seed(SEED)

TE = a.impl == "te"
kw = dict(num_layers=L, hidden_size=D, num_attention_heads=H, ffn_hidden_size=FFN, add_bias_linear=False,
          gated_linear_unit=False, normalization="LayerNorm", layernorm_epsilon=1e-5, hidden_dropout=0.0, attention_dropout=0.0,
          init_method_std=0.02, bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
          attention_softmax_in_fp32=True, apply_query_key_layer_scaling=False, bias_dropout_fusion=False, apply_rope_fusion=False)
if TE:
    # TE modules; F.gelu here is only the *selector* for TE's GELU op, which is the tanh approximation (== F.gelu(approximate='tanh')).
    kw.update(activation_func=F.gelu, use_te_activation_func=True, bias_activation_fusion=False, masked_softmax_fusion=True,
              gradient_accumulation_fusion=False, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te", persist_layer_norm=True)
else:
    kw.update(activation_func=lambda x: F.gelu(x, approximate="tanh"), masked_softmax_fusion=False, bias_activation_fusion=False,
              gradient_accumulation_fusion=False, persist_layer_norm=False)
cfg = TransformerConfig(**kw)
spec = get_gpt_layer_with_transformer_engine_spec(use_te_activation_func=True) if TE else get_gpt_layer_local_spec()
gpt = GPTModel(cfg, spec, vocab_size=V, max_sequence_length=T, pre_process=True, post_process=True,
               fp16_lm_cross_entropy=False, parallel_output=True, share_embeddings_and_output_weights=True,
               position_embedding_type="learned_absolute")
gpt.cuda()
if not a.no_load_golden:
    w = init_weights(); sd = dict(gpt.named_parameters())
    def put(name, t):
        p = sd[name]; assert tuple(p.shape) == tuple(t.shape), (name, p.shape, t.shape)
        with torch.no_grad(): p.copy_(t.to(p.dtype))
    put("embedding.word_embeddings.weight", w["wte"]); put("embedding.position_embeddings.weight", w["wpe"])
    for l in range(L):
        p = f"L{l:02d}_"; q = f"decoder.layers.{l}."
        # Megatron QKV layout is per head [q|k|v]; SPEC is [q(all heads)|k|v]
        wq = w[p + "Wqkv"].view(D, 3, H, DH).permute(0, 2, 1, 3).reshape(D, 3 * D)
        put(q + "self_attention.linear_qkv.weight", wq.t()); put(q + "self_attention.linear_proj.weight", w[p + "Wo"].t())
        put(q + "mlp.linear_fc1.weight", w[p + "Wup"].t()); put(q + "mlp.linear_fc2.weight", w[p + "Wdown"].t())
        if TE:   # LN fused into the following linear: layer_norm_weight lives on linear_qkv / linear_fc1
            put(q + "self_attention.linear_qkv.layer_norm_weight", w[p + "gamma1"]); put(q + "mlp.linear_fc1.layer_norm_weight", w[p + "gamma2"])
        else:
            put(q + "input_layernorm.weight", w[p + "gamma1"]); put(q + "pre_mlp_layernorm.weight", w[p + "gamma2"])
    put("decoder.final_layernorm.weight", w["gamma_f"])
    for n, p in gpt.named_parameters():
        if n.endswith(("layernorm.bias", "layer_norm_bias")):
            p.data.zero_()
            if TE: p.requires_grad_(False)   # SPEC LN has gamma only: beta frozen at 0 == no beta (param count would otherwise be +d per LN)
nparams = sum(p.numel() for p in gpt.parameters())
nparams_trainable = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False, grad_reduce_in_fp32=True, overlap_grad_reduce=False)
model = DDP(cfg, ddp_cfg, Float16Module(cfg, gpt))
oc = OptimizerConfig(optimizer="adam", lr=LR_MAX, min_lr=LR_MIN, weight_decay=WD, adam_beta1=BETAS[0], adam_beta2=BETAS[1],
                     adam_eps=1e-8, clip_grad=0.0, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=False)
opt = get_megatron_optimizer(oc, [model])

tok_np = np.memmap(TRAIN_BIN, dtype=np.uint32, mode="r")
tok = torch.from_numpy(tok_np.astype(np.int32)).to(dev)
starts = torch.from_numpy(batch_starts(len(tok_np))).to(dev)
ar = torch.arange(T + 1, device=dev); pos = torch.arange(T, device=dev)[None].expand(B, T).contiguous()
def get_batch(s):
    seq = tok[(starts[s][:, None] + ar).view(-1)].view(B, T + 1).long()
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
def set_lr(s):
    for g in opt.param_groups: g["lr"] = lr_at(s)
def step(s):
    set_lr(s)
    model.zero_grad_buffer()
    x, y = get_batch(s)
    loss_tok = model(x, pos, None, labels=y)           # [B,T] fp32 per-token CE, causal mask applied internally
    loss = loss_tok.mean()
    loss.backward()
    model.finish_grad_sync()
    ok, gn, nz = opt.step()
    return loss.detach()
res = {"model": MODEL, "model_params": nparams, "model_params_trainable": nparams_trainable, "tag": a.tag,
       "impl": "megatron-core %s GPTModel %s" % (__import__("megatron.core").core.__version__, "Transformer Engine spec" if TE else "local spec"),
       "env": {"TRAIN_SMS": TRAIN_SMS, "TOTAL_SMS": TOTAL_SMS, "MPS_PCT": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "sm_count_visible": torch.cuda.get_device_properties(0).multi_processor_count, "capability": list(torch.cuda.get_device_capability(0)),
               "NVTE": {k: v for k, v in os.environ.items() if k.startswith("NVTE_")}}}
# ---- fwd+bwd (incl. batch gather + grad-buffer zeroing) as ONE CUDA graph; Megatron's optimizer (host-synchronising inf/nan check) stays eager ----
s_t = torch.zeros((), dtype=torch.long, device=dev)
def fb_static():
    model.zero_grad_buffer()
    seq = tok[(starts.index_select(0, s_t.reshape(1))[0][:, None] + ar).view(-1)].view(B, T + 1).long()
    loss = model(seq[:, :-1].contiguous(), pos, None, labels=seq[:, 1:].contiguous()).mean()
    loss.backward(); s_t.add_(1)
    return loss.detach()
def build_fb_graph():
    torch.autograd.graph.set_override_stale_capture_stream(True)   # Megatron DDP's AccumulateGrad hooks were created on the default stream
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fb_static()           # warm-up (real fwd/bwd, no optimizer step: weights unchanged)
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); s_t.zero_()
    g = torch.cuda.CUDAGraph(); out = {}
    with torch.cuda.graph(g): out["loss"] = fb_static()
    return g, out
def step_graph(s, g, out):
    set_lr(s); g.replay(); model.finish_grad_sync(); opt.step(); return out["loss"]
if a.count3:
    from torch.profiler import profile, ProfilerActivity
    from launchcount import trace_counts
    NS = 5
    for s in range(20): step(s)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for s in range(100, 100 + NS): step(s)
        torch.cuda.synchronize()
    res["launch3"] = {"eager": trace_counts(prof, NS)}
    g, out = build_fb_graph()
    for s in range(3): step_graph(s, g, out)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for s in range(NS): step_graph(s, g, out)
        torch.cuda.synchronize()
    res["launch3"]["graph_fwd_bwd_only"] = trace_counts(prof, NS)
    print(json.dumps(res["launch3"], indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    sys.exit()
losses = torch.zeros(a.steps, device=dev)
torch.cuda.synchronize(); t0 = time.time()
if a.cuda_graph:
    g, out = build_fb_graph()
    for s in range(a.steps): losses[s] = step_graph(s, g, out)
    res["cuda_graph"] = "fwd+bwd graph, optimizer eager"
else:
    for s in range(a.steps): losses[s] = step(s)
torch.cuda.synchronize(); dt = time.time() - t0
lc = losses.cpu().numpy()
res.update(steps=a.steps, seconds=dt, tok_s=a.steps * B * T / dt, loss_initial=float(lc[0]), loss_final=float(lc[-1]),
           loss_final_mean_last25=float(lc[-25:].mean()),
           loss_curve_every25={int(i): float(lc[i]) for i in range(0, a.steps, 25)} | {a.steps - 1: float(lc[-1])},
           peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
if a.save:
    torch.save({n: p.detach().cpu() for n, p in gpt.named_parameters()}, a.save)
if a.count_launches:
    from torch.profiler import profile, ProfilerActivity
    NS = 3
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for s in range(100, 100 + NS): step(s)
        torch.cuda.synchronize()
    d_ = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.name]
    kern = sum(1 for e in d_ if not e.name.startswith(("Memcpy", "Memset")))
    res["launches"] = {"method": "torch.profiler(CUPTI) over 3 steady-state steps", "kernels_per_step": kern / NS,
                       "memops_per_step": (len(d_) - kern) / NS, "kernels_total_extrapolated": kern / NS * a.steps}
print(json.dumps({k: v for k, v in res.items() if k != "loss_curve_every25"}))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
