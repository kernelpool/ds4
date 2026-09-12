#!/usr/bin/env python3
"""Emit the DeepSeek V4.1 Flash template GGUF that deepseek4-quantize consumes.

deepseek4-quantize takes metadata, tensor names, shapes and ordering from a template
GGUF and never reads its payload, so this writes headers only: every `deepseek4.*` key
derived from config.json, the tokenizer tables from tokenizer.json, and the full backbone
tensor list. No V4.1 GGUF exists to copy, which is why the template is generated.

Engram lives in its own sidecar (deepseek_v41_engram.py) and vision in another
(deepseek4_vision.py), so neither appears here.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_VERSION,
    QTYPE_F16,
    QTYPE_F32,
    QTYPE_Q8_0,
    QTYPE_NAMES,
    align,
    fail,
    kv_bool,
    kv_f32,
    kv_string,
    kv_u32,
    kv_u32_array,
    pack_string,
    qtype_nbytes,
)

SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ARCHITECTURE = "deepseek4"
GGUF_STRING_TYPE = 8
GGUF_ARRAY_TYPE = 9

# Types follow what ds4.c's deepseek4 layout validation demands: F32 for norms, scales
# and biases, F16 for the mHC mixers, the router and the compressor, and a dense quant
# (Q8_0/Q4_K/Q4_0 only -- not BF16) for every projection. Routed experts are left to
# deepseek4-quantize's --experts flag.
DEFAULT_QUANT = QTYPE_Q8_0


def kv_string_array(key, values):
    payload = b"".join(pack_string(v) for v in values)
    return (pack_string(key)
            + struct.pack("<IIQ", GGUF_ARRAY_TYPE, GGUF_STRING_TYPE, len(values))
            + payload)


def kv_f32_array(key, values):
    return (pack_string(key)
            + struct.pack("<IIQ", GGUF_ARRAY_TYPE, 6, len(values))
            + struct.pack(f"<{len(values)}f", *values))


def load_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as fp:
        document = json.load(fp)
    config = document.get("text_config", document)
    expected = {
        "model_type": "deepseek_v41_text",
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_nextn_predict_layers": 3,
        "vocab_size": 129280,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2304,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "sliding_window": 128,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "rms_norm_eps": 1e-20,
        "scoring_func": "sqrtsoftplus",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"unexpected config {key}: {config.get(key)!r}")
    return config


def build_tokenizer_records(hf_dir, config):
    """tokenizer.ggml.tokens / .merges, plus the ids DS4's DeepSeek branch looks up."""
    with open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8") as fp:
        document = json.load(fp)
    if document["model"]["type"] != "BPE":
        fail(f"unexpected tokenizer model {document['model']['type']!r}")

    vocab = dict(document["model"]["vocab"])
    for added in document.get("added_tokens", []):
        vocab[added["content"]] = added["id"]
    n_vocab = config["vocab_size"]
    if len(vocab) != n_vocab:
        fail(f"tokenizer has {len(vocab)} entries, config says {n_vocab}")

    tokens = [None] * n_vocab
    for text, token_id in vocab.items():
        if not 0 <= token_id < n_vocab:
            fail(f"token id {token_id} is outside the vocabulary")
        if tokens[token_id] is not None:
            fail(f"duplicate token id {token_id}")
        tokens[token_id] = text
    missing = [i for i, t in enumerate(tokens) if t is None]
    if missing:
        fail(f"vocabulary has {len(missing)} holes, first at id {missing[0]}")

    merges = document["model"].get("merges", [])
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]

    required = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>",
                "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜"]
    for name in required:
        if name not in vocab:
            fail(f"tokenizer is missing the token DS4 requires: {name!r}")

    return [
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string_array("tokenizer.ggml.tokens", tokens),
        kv_string_array("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.bos_token_id", vocab["<｜begin▁of▁sentence｜>"]),
        kv_u32("tokenizer.ggml.eos_token_id", vocab["<｜end▁of▁sentence｜>"]),
    ]


def build_metadata(config, source_revision, tokenizer_records):
    rope = config["rope_scaling"]
    n_layer = config["num_hidden_layers"] + config["num_nextn_predict_layers"]
    prefix = ARCHITECTURE
    records = [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "DeepSeek V4.1 Flash"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", source_revision),
        kv_string(f"{prefix}.checkpoint_variant", "flash-v41"),
        kv_u32(f"{prefix}.block_count", n_layer),
        kv_u32(f"{prefix}.context_length", config["max_position_embeddings"]),
        kv_u32(f"{prefix}.embedding_length", config["hidden_size"]),
        kv_u32(f"{prefix}.vocab_size", config["vocab_size"]),
        kv_u32(f"{prefix}.attention.head_count", config["num_attention_heads"]),
        kv_u32(f"{prefix}.attention.head_count_kv", config["num_key_value_heads"]),
        kv_u32(f"{prefix}.attention.key_length", config["head_dim"]),
        kv_u32(f"{prefix}.attention.value_length", config["head_dim"]),
        kv_u32(f"{prefix}.attention.q_lora_rank", config["q_lora_rank"]),
        kv_u32(f"{prefix}.attention.output_lora_rank", config["o_lora_rank"]),
        kv_u32(f"{prefix}.attention.output_group_count", config["o_groups"]),
        kv_u32(f"{prefix}.attention.sliding_window", config["sliding_window"]),
        kv_f32(f"{prefix}.attention.layer_norm_rms_epsilon", config["rms_norm_eps"]),
        kv_u32_array(f"{prefix}.attention.compress_ratios", config["compress_ratios"]),
        kv_f32(f"{prefix}.attention.compress_rope_freq_base", config["compress_rope_theta"]),
        kv_u32(f"{prefix}.attention.indexer.head_count", config["index_n_heads"]),
        kv_u32(f"{prefix}.attention.indexer.key_length", config["index_head_dim"]),
        kv_u32(f"{prefix}.attention.indexer.top_k", config["index_topk"]),
        kv_u32_array(f"{prefix}.attention.kv_source_layers", config["kv_source_layer_ids"]),
        kv_u32_array(f"{prefix}.attention.index_source_layers", config["index_source_layer_ids"]),
        kv_u32(f"{prefix}.attention.candidate_source_layer", config["candidate_source_layer_id"]),
        kv_u32(f"{prefix}.attention.candidate_top_blocks", config["candidate_topk_blocks"]),
        kv_u32(f"{prefix}.attention.candidate_block_size", config["candidate_block_size"]),
        kv_u32(f"{prefix}.expert_count", config["n_routed_experts"]),
        kv_u32(f"{prefix}.expert_used_count", config["num_experts_per_tok"]),
        kv_u32(f"{prefix}.expert_shared_count", config["n_shared_experts"]),
        kv_u32(f"{prefix}.expert_feed_forward_length", config["moe_intermediate_size"]),
        kv_u32(f"{prefix}.expert_group_count", 0),
        kv_u32(f"{prefix}.expert_group_used_count", 0),
        kv_bool(f"{prefix}.expert_weights_norm", config["norm_topk_prob"]),
        kv_f32(f"{prefix}.expert_weights_scale", config["routed_scaling_factor"]),
        kv_u32(f"{prefix}.hash_layer_count", 0),
        kv_u32(f"{prefix}.hyper_connection.count", config["hc_mult"]),
        kv_f32(f"{prefix}.hyper_connection.epsilon", config["hc_eps"]),
        kv_u32(f"{prefix}.hyper_connection.sinkhorn_iterations", config["hc_sinkhorn_iters"]),
        kv_u32(f"{prefix}.nextn_predict_layers", config["num_nextn_predict_layers"]),
        kv_u32(f"{prefix}.engram.ngram_count", config["engram_max_ngram_size"] - 1),
        kv_u32(f"{prefix}.engram.head_count", config["engram_n_heads"]),
        kv_u32(f"{prefix}.engram.head_dim", config["engram_head_dim"]),
        kv_u32(f"{prefix}.rope.dimension_count", config["qk_rope_head_dim"]),
        kv_f32(f"{prefix}.rope.freq_base", config["rope_theta"]),
        kv_f32(f"{prefix}.rope.scaling.factor", rope["factor"]),
        kv_u32(f"{prefix}.rope.scaling.original_context_length",
               rope["original_max_position_embeddings"]),
        kv_f32(f"{prefix}.rope.scaling.yarn_beta_fast", rope["beta_fast"]),
        kv_f32(f"{prefix}.rope.scaling.yarn_beta_slow", rope["beta_slow"]),
        kv_f32_array(f"{prefix}.swiglu_clamp_exp", [config["swiglu_limit"]] * n_layer),
        kv_u32(f"{prefix}.dspark.block_size", config["dspark_block_size"]),
        kv_u32(f"{prefix}.dspark.markov_rank", config["dspark_markov_rank"]),
        kv_u32(f"{prefix}.dspark.noise_token_id", config["dspark_noise_token_id"]),
        kv_u32_array(f"{prefix}.dspark.target_layer_ids", config["dspark_target_layer_ids"]),
        kv_bool(f"{prefix}.vision.sidecar_required", True),
    ]
    return records + tokenizer_records


class Tensor:
    __slots__ = ("name", "shape", "qtype", "offset", "nbytes")

    def __init__(self, name, shape, qtype=DEFAULT_QUANT):
        self.name = name
        self.shape = tuple(shape)
        self.qtype = qtype
        self.offset = 0
        self.nbytes = 0


def backbone_tensors(config):
    """GGUF shapes are the reverse of the checkpoint's, so dim[0] is the row length."""
    embd = config["hidden_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    q_lora = config["q_lora_rank"]
    o_lora = config["o_lora_rank"]
    o_groups = config["o_groups"]
    hc = config["hc_mult"]
    mix_hc = (2 + hc) * hc
    n_expert = config["n_routed_experts"]
    ff_exp = config["moe_intermediate_size"]
    index_heads = config["index_n_heads"]
    index_dim = config["index_head_dim"]
    latent = head_dim
    ratios = config["compress_ratios"]
    kv_source = set(config["kv_source_layer_ids"])
    index_source = set(config["index_source_layer_ids"])
    n_backbone = config["num_hidden_layers"]
    n_layer = n_backbone + config["num_nextn_predict_layers"]

    out = [
        Tensor("token_embd.weight", (embd, vocab), QTYPE_F16),
        Tensor("output_norm.weight", (embd,), QTYPE_F32),
        Tensor("output.weight", (embd, vocab), QTYPE_Q8_0),
    ]
    for il in range(n_layer):
        # the DSpark draft stages carry a smaller routed pool than the backbone
        experts = n_expert if il < n_backbone else config["dspark_n_routed_experts"]
        p = f"blk.{il}."
        out += [
            Tensor(p + "hc_attn_fn.weight", (hc * embd, mix_hc), QTYPE_F16),
            Tensor(p + "hc_attn_scale.weight", (3,), QTYPE_F32),
            Tensor(p + "hc_attn_base.weight", (mix_hc,), QTYPE_F32),
            Tensor(p + "attn_norm.weight", (embd,), QTYPE_F32),
            Tensor(p + "attn_q_a.weight", (embd, q_lora), QTYPE_Q8_0),
            Tensor(p + "attn_q_a_norm.weight", (q_lora,), QTYPE_F32),
            Tensor(p + "attn_q_b.weight", (q_lora, heads * head_dim), QTYPE_Q8_0),
            Tensor(p + "attn_kv.weight", (embd, latent), QTYPE_Q8_0),
            Tensor(p + "attn_kv_a_norm.weight", (latent,), QTYPE_F32),
            Tensor(p + "attn_sinks.weight", (heads,), QTYPE_F32),
            Tensor(p + "attn_output_a.weight", (o_groups * latent, o_groups * o_lora), QTYPE_Q8_0),
            Tensor(p + "attn_output_b.weight", (o_groups * o_lora, embd), QTYPE_Q8_0),
        ]
        if il in kv_source:
            out += [
                Tensor(p + "attn_compressor_kv.weight", (embd, latent), QTYPE_F16),
                Tensor(p + "attn_compressor_norm.weight", (latent,), QTYPE_F32),
            ]
            if ratios[il] > 1:
                # ratio 1 is a plain projection, so it has no pooling gate
                out.append(Tensor(p + "attn_compressor_gate.weight", (embd, latent), QTYPE_F16))
            out += [
                Tensor(p + "indexer.attn_k.weight", (latent, index_dim), QTYPE_F16),
                Tensor(p + "indexer.k_norm.weight", (index_dim,), QTYPE_F32),
            ]
        if il in index_source:
            out += [
                Tensor(p + "indexer.attn_q_b.weight", (q_lora, index_heads * index_dim), QTYPE_Q8_0),
                Tensor(p + "indexer.proj.weight", (embd, index_heads), QTYPE_F16),
            ]
        out += [
            Tensor(p + "hc_ffn_fn.weight", (hc * embd, mix_hc), QTYPE_F16),
            Tensor(p + "hc_ffn_scale.weight", (3,), QTYPE_F32),
            Tensor(p + "hc_ffn_base.weight", (mix_hc,), QTYPE_F32),
            Tensor(p + "ffn_norm.weight", (embd,), QTYPE_F32),
            Tensor(p + "ffn_gate_inp.weight", (embd, experts), QTYPE_F16),
            Tensor(p + "exp_probs_b.bias", (experts,), QTYPE_F32),
            Tensor(p + "ffn_gate_exps.weight", (embd, ff_exp, experts)),
            Tensor(p + "ffn_up_exps.weight", (embd, ff_exp, experts)),
            Tensor(p + "ffn_down_exps.weight", (ff_exp, embd, experts)),
            Tensor(p + "ffn_gate_shexp.weight", (embd, ff_exp), QTYPE_Q8_0),
            Tensor(p + "ffn_up_shexp.weight", (embd, ff_exp), QTYPE_Q8_0),
            Tensor(p + "ffn_down_shexp.weight", (ff_exp, embd), QTYPE_Q8_0),
        ]
    return out


def tensor_header(item):
    return (pack_string(item.name)
            + struct.pack("<I", len(item.shape))
            + struct.pack(f"<{len(item.shape)}Q", *item.shape)
            + struct.pack("<IQ", item.qtype, item.offset))


def write_template(path, metadata, tensors, overwrite):
    for item in tensors:
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
    offset = 0
    for item in tensors:
        item.offset = offset
        offset += align(item.nbytes)

    header = 4 + 4 + 8 + 8 + sum(len(r) for r in metadata)
    header += sum(len(tensor_header(t)) for t in tensors)
    data_offset = align(header)

    if os.path.exists(path) and not overwrite:
        fail(f"output exists: {path}; use --overwrite")
    with open(path, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata)))
        for record in metadata:
            fp.write(record)
        for item in tensors:
            fp.write(tensor_header(item))
        if fp.tell() > data_offset:
            fail("template header exceeds its planned data offset")
        # headers only: deepseek4-quantize never reads a template payload
        fp.write(bytes(data_offset - fp.tell()))
    return data_offset, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official DeepSeek-V4.1-Flash snapshot")
    parser.add_argument("--out", required=True, help="template GGUF to write")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_revision != SOURCE_REVISION:
        parser.error(f"unsupported source revision: {args.source_revision}")

    config = load_config(args.hf)
    tokenizer_records = build_tokenizer_records(args.hf, config)
    metadata = build_metadata(config, args.source_revision, tokenizer_records)
    tensors = backbone_tensors(config)
    data_offset, payload = write_template(args.out, metadata, tensors, args.overwrite)

    print(f"tensors: {len(tensors)}")
    print(f"metadata_records: {len(metadata)}")
    print(f"header_bytes: {data_offset}")
    print(f"payload_bytes_if_bf16: {payload}")
    by_type = {}
    for item in tensors:
        by_type[item.qtype] = by_type.get(item.qtype, 0) + 1
    for qtype, count in sorted(by_type.items()):
        print(f"tensor_type: {QTYPE_NAMES[qtype]} {count}")
    print(f"deepseek4-template: wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"deepseek4-template: error: {error}", file=sys.stderr)
        raise SystemExit(1)
