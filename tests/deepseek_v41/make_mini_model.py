"""Build a tiny random-weight DeepSeek V4.1 Flash and dump per-stage activations as a parity oracle.

Runs the checkpoint's own inference/model.py unmodified: the six tilelang kernels are replaced by
ds41_kernels_cpu, and the two float8/float4 dtypes newer torch has are shimmed so torch 2.4 loads it.
Weights are bf16 with unquantised experts, so the GEMM path is plain F.linear and the oracle
isolates the architecture's maths rather than the quantisation.

usage: make_mini_model.py [--out DIR] [--seq N] [--no-quant]
"""
import argparse, os, sys, types, json
import numpy as np
import torch

DEFAULT_SNAP = "/Volumes/WD_BLACK/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"


class _ShimDtype:
    """Stands in for a torch dtype this build lacks; never equal to a real one."""
    def __init__(self, name): self.name = name
    def __repr__(self): return f"<shim torch.{self.name}>"


def install_reference(snapshot, quantize=True):
    for name in ("float8_e8m0fnu", "float4_e2m1fn_x2"):
        if not hasattr(torch, name):
            setattr(torch, name, _ShimDtype(name))
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import ds41_kernels_cpu as kcpu
    kcpu.QUANTIZE = quantize
    sys.modules["kernel"] = kcpu          # model.py does `from kernel import ...`
    sys.path.insert(0, os.path.join(snapshot, "inference"))
    import model as ref
    return ref


def mini_args(ref):
    """Small but structurally faithful: every mechanism the release uses is present at least once."""
    return ref.ModelArgs(
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None,
        vocab_size=512, dim=256, moe_inter_dim=128, n_layers=6, n_mtp_layers=0,
        n_heads=8, head_dim=64, rope_head_dim=16, q_lora_rank=64,
        o_groups=4, o_lora_rank=32, norm_eps=1e-20,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
        score_func="sqrtsoftplus", norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0,
        window_size=16,
        # 0 = window only, 2 = ratio-2 pooled (encoder), 1 = 1:1 latent (decoder)
        compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4), index_source_layers=(2, 4, 5),
        compress_rope_theta=160000.0, original_seq_len=32, rope_theta=10000.0,
        rope_factor=16, beta_fast=32, beta_slow=1,
        index_n_heads=4, index_head_dim=32, index_topk=8,
        candidate_source_layer=4, candidate_topk_blocks=6, candidate_block_size=2,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    )


def init_weights(model, seed=0):
    """torch.empty leaves junk; give every parameter something small and deterministic."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.endswith("norm.weight") or name.endswith("_norm.weight"):
                p.fill_(1.0)
            elif "hc_scale" in name:
                p.copy_(torch.full_like(p, 1.0))
            elif "hc_base" in name:
                p.copy_(torch.zeros_like(p))
            elif "attn_sink" in name:
                p.copy_(torch.full_like(p, -2.0))
            elif name.endswith("gate.bias") or name.endswith("gate.bias_vl"):
                p.copy_(torch.zeros_like(p))
            else:
                t = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02
                p.copy_(t.to(p.dtype))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("DS41_SNAPSHOT", DEFAULT_SNAP))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini"))
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--raw-shared-index-k", action="store_true",
                    help="keep the released stale-index_k decode behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-quant", action="store_true", help="skip the simulated fp8/fp4 KV quantisation")
    ap.add_argument("--f32", action="store_true",
                    help="build the weights and oracle in float32 so the CPU reference is scored without bf16 rounding")
    a = ap.parse_args()

    ref = install_reference(a.snapshot, quantize=not a.no_quant)
    if not a.raw_shared_index_k:
        # Indexer.forward only does `shared_attn.index_k = self.k_cache` inside
        # `if self.owns_k and latent is not None`, so at decode a compress_ratio>1 layer
        # that did not complete its group skips the publish and its consumers score
        # against whichever layer published last -- a different layer's keys.  That makes
        # the released decode disagree with its own prefill by ~15% on the same input;
        # republishing restores agreement to 4e-7.  SharedAttentionRuntime's own docstring
        # ("every source writes before its consumers read") is the intended contract, so
        # the oracle is generated with it held.  --raw-shared-index-k reproduces the
        # released behaviour instead.
        _indexer_forward = ref.Indexer.forward

        def _forward(self, x, qr, latent, start_pos, offset):
            if self.owns_k:
                ref.shared_attn.index_k = self.k_cache
            return _indexer_forward(self, x, qr, latent, start_pos, offset)

        ref.Indexer.forward = _forward
    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")

    args = mini_args(ref)
    model = init_weights(ref.Transformer(args), a.seed).eval()
    if a.f32:
        # same weight values, f32 arithmetic: lets the CPU reference be scored without
        # bf16 activation rounding standing between the two sides
        model = model.float()
        torch.set_default_dtype(torch.float32)
    os.makedirs(a.out, exist_ok=True)

    caught = {}
    def hook(tag):
        def fn(_m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                caught[tag] = t.detach().clone().float().cpu().numpy()
        return fn
    for i, layer in enumerate(model.layers):
        layer.attn.register_forward_hook(hook(f"layer{i}.attn"))
        layer.ffn.register_forward_hook(hook(f"layer{i}.ffn"))
        layer.register_forward_hook(hook(f"layer{i}.out"))
        if layer.attn.compressor is not None:
            layer.attn.compressor.register_forward_hook(hook(f"layer{i}.compressor"))
        if layer.attn.indexer is not None:
            layer.attn.indexer.register_forward_hook(hook(f"layer{i}.indexer_idxs"))

    ids = torch.randint(0, args.vocab_size, (1, a.seq))
    with torch.inference_mode():
        out_ids, logits, _ = model(ids)

    caught["input_ids"] = ids.cpu().numpy()
    caught["logits"] = logits.detach().float().cpu().numpy()
    np.savez(os.path.join(a.out, "oracle_prefill.npz"), **caught)

    # decode steps continue from the prefilled caches: the sliding-window ring, the
    # partial compressor group, and the per-step indexer all carry across
    decode = {"prefill_ids": ids.cpu().numpy()}
    step_ids = out_ids.view(1, 1)
    with torch.inference_mode():
        for step in range(a.decode):
            pos = a.seq + step
            decode[f"step{step}.id"] = step_ids.cpu().numpy()
            caught.clear()
            step_out, step_logits, _ = model(step_ids, start_pos=pos)
            for tag, arr in caught.items():
                decode[f"step{step}.{tag}"] = arr
            decode[f"step{step}.logits"] = step_logits.detach().clone().float().cpu().numpy()
            step_ids = step_out.view(1, 1)
    np.savez(os.path.join(a.out, "oracle_decode.npz"), **decode)
    print(f"decode: {a.decode} steps from start_pos {a.seq} -> {a.out}/oracle_decode.npz")

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    json.dump(cfg, open(os.path.join(a.out, "mini_config.json"), "w"), indent=1, default=str)
    torch.save(model.state_dict(), os.path.join(a.out, "mini_weights.pt"))

    fin = np.isfinite(decode["step0.logits"]).all()
    print(f"seq={a.seq} decode={a.decode} step0 logits finite={fin}")
    print(f"captured prefill -> {a.out}/oracle_prefill.npz, decode -> {a.out}/oracle_decode.npz")


if __name__ == "__main__":
    main()
