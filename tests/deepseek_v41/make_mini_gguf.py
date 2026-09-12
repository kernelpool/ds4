#!/usr/bin/env python3
"""Pack the V4.1 mini checkpoint into a DS4 GGUF so the CPU reference can run on it.

The mini model exists to score DS4's forward path against the released implementation, so
everything is written F32: no quantization error stands between the two sides. The real
checkpoint goes through gguf-tools/deepseek_v41_template.py + deepseek4-quantize instead.

  python make_mini_gguf.py --mini mini --out mini/mini.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-tools"))
from glm53_quantize import (  # noqa: E402
    GGUF_ALIGNMENT,
    GGUF_VERSION,
    QTYPE_F32,
    align,
    fail,
    kv_bool,
    kv_f32,
    kv_string,
    kv_u32,
    kv_u32_array,
    pack_string,
)
from deepseek_v41_template import kv_f32_array, kv_string_array  # noqa: E402

ARCHITECTURE = "deepseek4"


def tensor_header(name, shape, qtype, offset):
    return (pack_string(name)
            + struct.pack("<I", len(shape))
            + struct.pack(f"<{len(shape)}Q", *shape)
            + struct.pack("<IQ", qtype, offset))


def metadata(config):
    n_layer = config["n_layers"]
    hc = config["hc_mult"]
    prefix = ARCHITECTURE
    return [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "DeepSeek V4.1 Flash Mini"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", "tests/deepseek_v41/make_mini_model.py"),
        kv_string("general.source.revision", "mini"),
        kv_u32(f"{prefix}.block_count", n_layer),
        kv_u32(f"{prefix}.context_length", config["max_seq_len"]),
        kv_u32(f"{prefix}.embedding_length", config["dim"]),
        kv_u32(f"{prefix}.vocab_size", config["vocab_size"]),
        kv_u32(f"{prefix}.attention.head_count", config["n_heads"]),
        kv_u32(f"{prefix}.attention.head_count_kv", 1),
        kv_u32(f"{prefix}.attention.key_length", config["head_dim"]),
        kv_u32(f"{prefix}.attention.value_length", config["head_dim"]),
        kv_u32(f"{prefix}.attention.q_lora_rank", config["q_lora_rank"]),
        kv_u32(f"{prefix}.attention.output_lora_rank", config["o_lora_rank"]),
        kv_u32(f"{prefix}.attention.output_group_count", config["o_groups"]),
        kv_u32(f"{prefix}.attention.sliding_window", config["window_size"]),
        kv_f32(f"{prefix}.attention.layer_norm_rms_epsilon", config["norm_eps"]),
        kv_u32_array(f"{prefix}.attention.compress_ratios", config["compress_ratios"]),
        kv_f32(f"{prefix}.attention.compress_rope_freq_base", config["compress_rope_theta"]),
        kv_u32(f"{prefix}.attention.indexer.head_count", config["index_n_heads"]),
        kv_u32(f"{prefix}.attention.indexer.key_length", config["index_head_dim"]),
        kv_u32(f"{prefix}.attention.indexer.top_k", config["index_topk"]),
        kv_u32_array(f"{prefix}.attention.kv_source_layers", config["kv_source_layers"]),
        kv_u32_array(f"{prefix}.attention.index_source_layers", config["index_source_layers"]),
        kv_u32(f"{prefix}.attention.candidate_source_layer", config["candidate_source_layer"]),
        kv_u32(f"{prefix}.attention.candidate_top_blocks", config["candidate_topk_blocks"]),
        kv_u32(f"{prefix}.attention.candidate_block_size", config["candidate_block_size"]),
        kv_u32(f"{prefix}.expert_count", config["n_routed_experts"]),
        kv_u32(f"{prefix}.expert_used_count", config["n_activated_experts"]),
        kv_u32(f"{prefix}.expert_shared_count", config["n_shared_experts"]),
        kv_f32(f"{prefix}.expert_weights_scale", config["route_scale"]),
        kv_bool(f"{prefix}.expert_weights_norm", True),
        kv_u32(f"{prefix}.expert_feed_forward_length", config["moe_inter_dim"]),
        kv_u32(f"{prefix}.expert_group_count", 0),
        kv_u32(f"{prefix}.expert_group_used_count", 0),
        kv_u32(f"{prefix}.hash_layer_count", 0),
        kv_u32(f"{prefix}.hyper_connection.count", hc),
        kv_f32(f"{prefix}.hyper_connection.epsilon", config["hc_eps"]),
        kv_u32(f"{prefix}.hyper_connection.sinkhorn_iterations", config["hc_sinkhorn_iters"]),
        kv_u32(f"{prefix}.nextn_predict_layers", config["n_mtp_layers"]),
        kv_u32(f"{prefix}.engram.ngram_count", 0),
        kv_u32(f"{prefix}.engram.head_count", 0),
        kv_u32(f"{prefix}.engram.head_dim", 0),
        kv_u32(f"{prefix}.rope.dimension_count", config["rope_head_dim"]),
        kv_f32(f"{prefix}.rope.freq_base", config["rope_theta"]),
        kv_f32(f"{prefix}.rope.scaling.factor", config["rope_factor"]),
        kv_u32(f"{prefix}.rope.scaling.original_context_length", config["original_seq_len"]),
        kv_f32(f"{prefix}.rope.scaling.yarn_beta_fast", config["beta_fast"]),
        kv_f32(f"{prefix}.rope.scaling.yarn_beta_slow", config["beta_slow"]),
        kv_f32_array(f"{prefix}.swiglu_clamp_exp", [config["swiglu_limit"]] * n_layer),
        # a stand-in vocabulary: the mini model is scored on logits, never on text
        kv_string_array("tokenizer.ggml.tokens",
                        [f"<t{i}>" for i in range(config["vocab_size"])]),
        kv_string_array("tokenizer.ggml.merges", []),
        kv_u32("tokenizer.ggml.bos_token_id", 0),
        kv_u32("tokenizer.ggml.eos_token_id", 1),
    ]


def f32(tensor):
    return tensor.to(torch.float32).numpy().astype("<f4", copy=False)


def build_tensors(weights, config):
    """GGUF dim[0] is the row length, so every 2-D torch weight is stored reversed."""
    out = []

    def add(name, array):
        array = np.ascontiguousarray(array, dtype="<f4")
        out.append((name, tuple(reversed(array.shape)), array))

    add("token_embd.weight", f32(weights["embed.weight"]))
    add("output_norm.weight", f32(weights["norm.weight"]))
    add("output.weight", f32(weights["head.weight"]))

    kv_source = set(config["kv_source_layers"])
    index_source = set(config["index_source_layers"])
    for il in range(config["n_layers"]):
        src, dst = f"layers.{il}.", f"blk.{il}."
        for suffix, gguf in (
            ("hc_attn_fn", "hc_attn_fn.weight"), ("hc_attn_scale", "hc_attn_scale.weight"),
            ("hc_attn_base", "hc_attn_base.weight"), ("hc_ffn_fn", "hc_ffn_fn.weight"),
            ("hc_ffn_scale", "hc_ffn_scale.weight"), ("hc_ffn_base", "hc_ffn_base.weight"),
            ("attn_norm.weight", "attn_norm.weight"), ("ffn_norm.weight", "ffn_norm.weight"),
            ("attn.wq_a.weight", "attn_q_a.weight"), ("attn.q_norm.weight", "attn_q_a_norm.weight"),
            ("attn.wq_b.weight", "attn_q_b.weight"), ("attn.wkv.weight", "attn_kv.weight"),
            ("attn.kv_norm.weight", "attn_kv_a_norm.weight"),
            ("attn.attn_sink", "attn_sinks.weight"),
            ("attn.wo_a.weight", "attn_output_a.weight"),
            ("attn.wo_b.weight", "attn_output_b.weight"),
            ("ffn.gate.weight", "ffn_gate_inp.weight"), ("ffn.gate.bias", "exp_probs_b.bias"),
            ("ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight"),
            ("ffn.shared_experts.w3.weight", "ffn_up_shexp.weight"),
            ("ffn.shared_experts.w2.weight", "ffn_down_shexp.weight"),
        ):
            add(dst + gguf, f32(weights[src + suffix]))

        if il in kv_source:
            add(dst + "attn_compressor_kv.weight", f32(weights[src + "attn.compressor.wkv.weight"]))
            add(dst + "attn_compressor_norm.weight",
                f32(weights[src + "attn.compressor.norm.weight"]))
            gate = src + "attn.compressor.wgate.weight"
            if gate in weights:
                add(dst + "attn_compressor_gate.weight", f32(weights[gate]))
            add(dst + "indexer.attn_k.weight", f32(weights[src + "attn.indexer.wk.weight"]))
            add(dst + "indexer.k_norm.weight", f32(weights[src + "attn.indexer.k_norm.weight"]))
        if il in index_source:
            add(dst + "indexer.attn_q_b.weight", f32(weights[src + "attn.indexer.wq_b.weight"]))
            add(dst + "indexer.proj.weight",
                f32(weights[src + "attn.indexer.weights_proj.weight"]))

        # routed experts stack into one [in, out, n_expert] tensor per part
        for part, gguf in (("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")):
            stack = np.stack([f32(weights[f"{src}ffn.experts.{e}.{part}.weight"])
                              for e in range(config["n_routed_experts"])])
            out.append((dst + gguf + ".weight",
                        (stack.shape[2], stack.shape[1], stack.shape[0]),
                        np.ascontiguousarray(stack, dtype="<f4")))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", default=os.path.join(os.path.dirname(__file__), "mini"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.mini, "mini_config.json"), encoding="utf-8") as fp:
        config = json.load(fp)
    weights = torch.load(os.path.join(args.mini, "mini_weights.pt"),
                         map_location="cpu", weights_only=True)

    records = metadata(config)
    tensors = build_tensors(weights, config)

    header = 4 + 4 + 8 + 8 + sum(len(r) for r in records)
    header += sum(len(tensor_header(n, s, QTYPE_F32, 0)) for n, s, _ in tensors)
    data_offset = align(header)

    offsets, offset = [], 0
    for _, _, array in tensors:
        offsets.append(offset)
        offset += align(array.nbytes)

    with open(args.out, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(records)))
        for record in records:
            fp.write(record)
        for (name, shape, _), off in zip(tensors, offsets):
            fp.write(tensor_header(name, shape, QTYPE_F32, off))
        if fp.tell() > data_offset:
            fail("mini GGUF header exceeds its planned data offset")
        fp.write(bytes(data_offset - fp.tell()))
        for (name, _, array), off in zip(tensors, offsets):
            if fp.tell() != data_offset + off:
                fail(f"{name}: output offset mismatch")
            fp.write(array.tobytes())
            fp.write(bytes(align(array.nbytes) - array.nbytes))

    print(f"tensors: {len(tensors)}")
    print(f"metadata_records: {len(records)}")
    print(f"file_bytes: {data_offset + offset}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
