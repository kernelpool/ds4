#!/usr/bin/env python3
"""Convert MiMo-V2.6 (Flash) to a DwarfStar GGUF in llama.cpp's `mimo2` layout.

The released checkpoint stores the fused qkv projection as fp8 in `tp_size`
rank chunks, each [Q_c | K_c | V_c] with its own [128,128] scale grid, and the
routed experts as MXFP4 nibbles with E8M0 scales.  The `mxfp4` recipe keeps the
experts byte-for-byte (repacked into ggml blocks) and quantizes the fp8 and
bf16 projections to Q8_0; `q8` and `f32` are parity recipes for small models.
The three MTP layers become trailing blocks with `nextn.*` extras; the DFlash
drafter is written to a separate support GGUF with `--dflash-out`.
"""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

import glm53_quantize
from glm53_quantize import (
    GGUF_ALIGNMENT, GGUF_STRING, SourceDB, TensorPlan, Quantizer, Imatrix, QTYPE_F32,
    QTYPE_F16, QTYPE_Q8_0, QTYPE_Q2_K, QTYPE_Q4_K, QTYPE_IQ2_XXS, QTYPE_BF16, align,
    conversion_signature, kv_bool, kv_f32, kv_string, kv_u32, kv_u32_array,
    load_resume_state, print_plan, qtype_nbytes, save_resume_state, tensor_header,
)
from glm53_manifest import load_index, load_safetensors_header

QTYPE_MXFP4 = 39
glm53_quantize.QTYPE_NAMES[QTYPE_MXFP4] = "MXFP4"
glm53_quantize.QTYPE_LAYOUT[QTYPE_MXFP4] = (32, 17)

ARCH = "mimo2"
FP8_BLOCK = 128
QUANTIZATION = {
    "mxfp4": "released MXFP4 gate/up/down; Q8_0 attention/dense/head",
    "q8": "Q8_0 gate/up/down; Q8_0 attention/dense/head",
    "f32": "F32 everything",
}
MATRIX_QTYPE = {"mxfp4": QTYPE_Q8_0, "q8": QTYPE_Q8_0, "f32": QTYPE_F32}
EXPERT_QTYPE = {"mxfp4": QTYPE_MXFP4, "q8": QTYPE_Q8_0, "f32": QTYPE_F32}
EMBED_QTYPE = {"mxfp4": QTYPE_BF16, "q8": QTYPE_BF16, "f32": QTYPE_F32}


def scale_inv_name(name):
    return name.removesuffix(".weight") + ".weight_scale_inv"


def mxfp4_scale_name(name):
    return name.removesuffix(".weight") + ".weight_scale"


def attn_dims(c, is_swa):
    if is_swa:
        return (c["swa_num_attention_heads"], c["swa_num_key_value_heads"], c["swa_head_dim"], c["swa_v_head_dim"])
    return (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["v_head_dim"])


def qkv_chunk_rows(dims, tp):
    n_q, n_kv, hd, vd = dims
    if n_q % tp or n_kv % tp:
        raise ValueError(f"qkv heads {n_q}/{n_kv} not divisible by tp_size {tp}")
    return n_q // tp * hd, n_kv // tp * hd, n_kv // tp * vd


def load_json_lenient(path):
    """json.load that tolerates the trailing commas in the released dflash/config.json."""
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read()
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def load_config(hf_dir):
    config = load_json_lenient(os.path.join(hf_dir, "config.json"))
    if config.get("model_type") != "mimo_v2" or config.get("attention_projection_layout") != "fused_qkv":
        raise ValueError("not a fused-qkv MiMo-V2 checkpoint")
    q = config.get("quantization_config") or {}
    if q.get("quant_method") != "fp8" or q.get("weight_block_size") != [128, 128] or q.get("store_dtype") != "mxfp4":
        raise ValueError("expected fp8 [128,128] attention and MXFP4 experts")
    if config["n_shared_experts"] not in (None, 0) or config["n_group"] != 1 or config["topk_group"] != 1:
        raise ValueError("unsupported MoE layout")
    if config["scoring_func"] != "sigmoid" or config["topk_method"] != "noaux_tc" or not config["norm_topk_prob"]:
        raise ValueError("unsupported router")
    if config.get("routed_scaling_factor") not in (None, 1, 1.0):
        raise ValueError("unsupported routed_scaling_factor")
    if config["moe_layer_freq"][0] != 0 or any(f != 1 for f in config["moe_layer_freq"][1:]):
        raise ValueError("expected one leading dense layer followed by MoE layers")
    if config["swa_num_attention_heads"] != config["num_attention_heads"] or \
            config["swa_head_dim"] != config["head_dim"] or config["swa_v_head_dim"] != config["v_head_dim"]:
        raise ValueError("global and SWA head geometry must agree")
    if int(config["head_dim"] * config["partial_rotary_factor"]) % 2:
        raise ValueError("odd rotary dimension")
    return config


def model_tp(hf_dir):
    document, _ = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
    tp = document.get("metadata", {}).get("tp_size")
    if not isinstance(tp, int) or tp < 1:
        raise ValueError("index metadata lacks tp_size (fused qkv chunk count)")
    return tp


class Layout:
    """Which source tensors exist and how each is shaped, derived from config."""

    def __init__(self, config, tp):
        self.c = config
        self.tp = tp
        c = config
        self.n_layer = c["num_hidden_layers"]
        self.n_nextn = c.get("num_nextn_predict_layers", 3)
        self.dim = c["hidden_size"]
        self.inter = c["intermediate_size"]
        self.moe_inter = c["moe_intermediate_size"]
        self.experts = c["n_routed_experts"]
        self.vocab = c["vocab_size"]

    def is_swa(self, layer):
        return layer >= self.n_layer or self.c["hybrid_layer_pattern"][layer] == 1

    def qkv_rows(self, layer):
        dims = attn_dims(self.c, self.is_swa(layer))
        return sum(qkv_chunk_rows(dims, self.tp)) * self.tp

    def qkv_scale_shape(self, layer):
        dims = attn_dims(self.c, self.is_swa(layer))
        rows_c = sum(qkv_chunk_rows(dims, self.tp))
        return [self.tp * ((rows_c + FP8_BLOCK - 1) // FP8_BLOCK), (self.dim + FP8_BLOCK - 1) // FP8_BLOCK]


def validate_index(weight_map):
    for name in weight_map:
        if not name.startswith(("model.", "lm_head.", "visual.", "audio_encoder.", "speech_embeddings.")):
            raise ValueError(f"unexpected source tensor {name}")


def make_scale_validator(layout):
    def validate(tensors):
        for name, info in tensors.items():
            dtype, shape = info["dtype"], info["shape"]
            if dtype == "F8_E4M3":
                m = re.fullmatch(r"model\.(?:mtp\.)?layers\.(\d+)\.self_attn\.qkv_proj\.weight", name)
                if m:
                    layer = int(m.group(1)) + (layout.n_layer if ".mtp." in name else 0)
                    expected = layout.qkv_scale_shape(layer)
                else:
                    expected = [(d + FP8_BLOCK - 1) // FP8_BLOCK for d in shape]
                scale = tensors.get(scale_inv_name(name))
                if not scale or scale["dtype"] != "F32" or scale["shape"] != expected:
                    raise ValueError(f"{name}: expected F32 scale_inv {expected}, found {scale and scale['shape']}")
            elif dtype == "U8" and name.endswith(".weight"):
                scale = tensors.get(mxfp4_scale_name(name))
                expected = [shape[0], shape[1] * 2 // 32]
                if not scale or scale["dtype"] != "U8" or scale["shape"] != expected:
                    raise ValueError(f"{name}: expected U8 E8M0 scales {expected}")
    return validate


def build_plan(db, layout, quant):
    if quant not in QUANTIZATION:
        raise ValueError(f"unknown quantization recipe: {quant}")
    c, tp = layout.c, layout.tp
    dim, inter, moe_inter, experts, vocab = layout.dim, layout.inter, layout.moe_inter, layout.experts, layout.vocab
    mq, eq, embq = MATRIX_QTYPE[quant], EXPERT_QTYPE[quant], EMBED_QTYPE[quant]
    plan, consumed = [], set()

    def claim(name, expected, dtype=None):
        info = db.info(name)
        if info["shape"] != list(expected) or (dtype and info["dtype"] != dtype):
            raise ValueError(f"{name}: unexpected {info['dtype']} {info['shape']}, expected {dtype} {expected}")
        consumed.add(name)
        if info["dtype"] == "F8_E4M3":
            consumed.add(scale_inv_name(name))
        elif info["dtype"] == "U8":
            consumed.add(mxfp4_scale_name(name))
        return info

    def regular(name, source, shape, qtype, role, **kw):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source, **kw))

    regular("token_embd.weight", "model.embed_tokens.weight", (vocab, dim), embq, "embedding")
    regular("output_norm.weight", "model.norm.weight", (dim,), QTYPE_F32, "norm")
    regular("output.weight", "lm_head.weight", (vocab, dim), mq, "output")

    def attention(dst, src, layer):
        n_q, n_kv, hd, vd = attn_dims(c, layout.is_swa(layer))
        regular(f"{dst}.attn_qkv.weight", f"{src}.self_attn.qkv_proj.weight",
                (layout.qkv_rows(layer), dim), mq, "attention", transform="qkv_chunks")
        regular(f"{dst}.attn_output.weight", f"{src}.self_attn.o_proj.weight", (dim, n_q * vd), mq, "attention")
        if layout.is_swa(layer):
            regular(f"{dst}.attn_sinks.weight", f"{src}.self_attn.attention_sink_bias", (n_q,), QTYPE_F32, "attention")

    def dense_ffn(dst, src):
        regular(f"{dst}.ffn_gate.weight", f"{src}.mlp.gate_proj.weight", (inter, dim), mq, "dense")
        regular(f"{dst}.ffn_up.weight", f"{src}.mlp.up_proj.weight", (inter, dim), mq, "dense")
        regular(f"{dst}.ffn_down.weight", f"{src}.mlp.down_proj.weight", (dim, inter), mq, "dense")

    for layer in range(layout.n_layer):
        src, dst = f"model.layers.{layer}", f"blk.{layer}"
        regular(f"{dst}.attn_norm.weight", f"{src}.input_layernorm.weight", (dim,), QTYPE_F32, "norm")
        regular(f"{dst}.ffn_norm.weight", f"{src}.post_attention_layernorm.weight", (dim,), QTYPE_F32, "norm")
        attention(dst, src, layer)
        if not c["moe_layer_freq"][layer]:
            dense_ffn(dst, src)
            continue
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.mlp.gate.weight", (experts, dim), QTYPE_F32, "router")
        regular(f"{dst}.exp_probs_b.bias", f"{src}.mlp.gate.e_score_correction_bias", (experts,), QTYPE_F32, "router")
        for part, shape in (("gate", (moe_inter, dim)), ("up", (moe_inter, dim)), ("down", (dim, moe_inter))):
            pattern = f"{src}.mlp.experts.{{expert}}.{part}_proj.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "U8")
            plan.append(TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts), eq,
                                   "experts", source=pattern, expert_layer=layer, expert_part=part,
                                   expert_count=experts))

    for k in range(layout.n_nextn):
        layer = layout.n_layer + k
        src, dst = f"model.mtp.layers.{k}", f"blk.{layer}"
        regular(f"{dst}.nextn.eh_proj.weight", f"{src}.eh_proj.weight", (dim, 2 * dim), mq, "nextn")
        for target, source in (("nextn.enorm", "enorm"), ("nextn.hnorm", "hnorm"), ("attn_norm", "input_layernorm"),
                               ("ffn_norm", "pre_mlp_layernorm"), ("nextn.shared_head_norm", "final_layernorm")):
            regular(f"{dst}.{target}.weight", f"{src}.{source}.weight", (dim,), QTYPE_F32, "norm")
        attention(dst, src, layer)
        dense_ffn(dst, src)

    omitted = {name for name in db.tensors if name.startswith(("visual.", "audio_encoder.", "speech_embeddings."))}
    unclaimed = set(db.tensors) - consumed - omitted
    if unclaimed:
        raise ValueError(f"unclaimed source tensors: {sorted(unclaimed)[:10]}")
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


class NativeQuantizer(Quantizer):
    def __init__(self, library_path, layout):
        super().__init__(library_path)
        self.layout = layout

    def to_f32(self, db, name, row_start=0, row_count=None):
        np = self.np
        info = db.info(name)
        if info["dtype"] == "U8":
            shape = info["shape"]
            codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(shape)
            scales = np.frombuffer(db.read(mxfp4_scale_name(name)), dtype=np.uint8)
            scales = scales.reshape(shape[0], shape[1] * 2 // 32)
            if np.any(scales == 255):
                raise ValueError(f"{name}: nonfinite scale")
            lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
            result = np.empty((shape[0], shape[1] * 2), dtype=np.float32)
            result[:, 0::2] = lut[codes & 15]
            result[:, 1::2] = lut[codes >> 4]
            result = (result.reshape(shape[0], -1, 32) * np.exp2(scales.astype(np.float32) - 127)[:, :, None])
            result = result.reshape(shape[0], -1)
        elif name.endswith("self_attn.qkv_proj.weight"):
            result = self.qkv_f32(db, name)
        else:
            result = super().to_f32(db, name)
        if row_count is not None:
            result = result[row_start:row_start + row_count]
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name}: nonfinite dequantized weight")
        return np.ascontiguousarray(result, dtype=np.float32)

    def qkv_f32(self, db, name):
        """Dequantise the rank-chunked fused qkv and reorder its rows to [Q | K | V]."""
        np = self.np
        layout = self.layout
        m = re.fullmatch(r"model\.(?:(mtp)\.)?layers\.(\d+)\.self_attn\.qkv_proj\.weight", name)
        layer = int(m.group(2)) + (layout.n_layer if m.group(1) else 0)
        dims = attn_dims(layout.c, layout.is_swa(layer))
        qr, kr, vr = qkv_chunk_rows(dims, layout.tp)
        rows_c = qr + kr + vr
        rb = (rows_c + FP8_BLOCK - 1) // FP8_BLOCK
        info = db.info(name)
        codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(info["shape"])
        if np.any((codes & 0x7F) == 0x7F):
            raise ValueError(f"{name}: nonfinite FP8 code")
        sinfo = db.info(scale_inv_name(name))
        scales = np.frombuffer(db.read(scale_inv_name(name)), dtype="<f4").reshape(sinfo["shape"])
        if not np.all(np.isfinite(scales)):
            raise ValueError(f"{name}: nonfinite scale")
        q, k, v = [], [], []
        for chunk in range(layout.tp):
            s = np.repeat(np.repeat(scales[chunk * rb:(chunk + 1) * rb], FP8_BLOCK, axis=0), FP8_BLOCK, axis=1)
            w = self.fp8_lut[codes[chunk * rows_c:(chunk + 1) * rows_c]] * s[:rows_c, :codes.shape[1]]
            q.append(w[:qr])
            k.append(w[qr:qr + kr])
            v.append(w[qr + kr:])
        return np.concatenate(q + k + v, axis=0)

    def to_mxfp4(self, db, name):
        """Repack released E2M1 pairs into GGUF MXFP4 blocks without touching the values."""
        np = self.np
        info = db.info(name)
        if info["dtype"] != "U8" or info["shape"][1] % 16:
            raise ValueError(f"{name}: not a packed MXFP4 tensor")
        rows, blocks = info["shape"][0], info["shape"][1] // 16
        codes = np.frombuffer(db.read(name), dtype=np.uint8).reshape(rows, blocks, 16)
        scales = np.frombuffer(db.read(mxfp4_scale_name(name)), dtype=np.uint8).reshape(rows, blocks)
        if np.any(scales == 255):
            raise ValueError(f"{name}: nonfinite scale")
        values = np.empty((rows, blocks, 32), dtype=np.uint8)
        values[:, :, 0::2] = codes & 15
        values[:, :, 1::2] = codes >> 4
        packed = np.empty((rows, blocks, 17), dtype=np.uint8)
        packed[:, :, 0] = scales
        packed[:, :, 1:] = values[:, :, :16] | (values[:, :, 16:] << 4)
        return packed.tobytes()

    def encode(self, array, qtype, imatrix=None):
        if qtype == QTYPE_F16 and self.np.any(self.np.abs(array) > 65504):
            raise ValueError("F16 tensor would overflow; preserve this family in BF16/F32")
        return super().encode(array, qtype, imatrix)


def array_record(key, kind, values):
    from deepseek41_metadata import array_record as record
    return record(key, kind, values)


def tokenizer_records(hf_dir, vocab_size):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    with open(os.path.join(hf_dir, "tokenizer.json"), "rb") as fp:
        tok = json.load(fp)
    if tok["model"]["type"] != "BPE" or tok["pre_tokenizer"]["type"] != "Sequence":
        raise ValueError("unsupported tokenizer")
    n_ids = tokenizer.get_vocab_size(with_added_tokens=True)
    if n_ids > vocab_size:
        raise ValueError(f"tokenizer has {n_ids} ids, embedding has {vocab_size} rows")
    tokens = [tokenizer.id_to_token(i) for i in range(n_ids)]
    if any(token is None for token in tokens):
        raise ValueError("tokenizer has missing IDs")
    added = {item["id"]: item["special"] for item in tok["added_tokens"]}
    # llama.cpp token types: 1 normal, 3 control, 4 user-defined, 5 unused
    types = [3 if added.get(i) else 4 if i in added else 1 for i in range(n_ids)]
    tokens += [f"[PAD{i}]" for i in range(n_ids, vocab_size)]
    types += [5] * (vocab_size - n_ids)
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in tok["model"]["merges"]]
    with open(os.path.join(hf_dir, "chat_template.jinja"), "r", encoding="utf-8") as fp:
        template = fp.read()
    return [
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "qwen2"),
        array_record("tokenizer.ggml.tokens", GGUF_STRING, tokens),
        array_record("tokenizer.ggml.token_type", 5, types),
        array_record("tokenizer.ggml.merges", GGUF_STRING, merges),
        kv_u32("tokenizer.ggml.eos_token_id", 151645),
        kv_u32("tokenizer.ggml.padding_token_id", 151643),
        kv_bool("tokenizer.ggml.add_bos_token", False),
        kv_bool("tokenizer.ggml.add_eos_token", False),
        kv_string("tokenizer.chat_template", template),
    ]


def model_records(hf_dir, layout, revision, name):
    c = layout.c
    n_all = layout.n_layer + layout.n_nextn
    pattern = [1 if layout.is_swa(i) else 0 for i in range(n_all)]
    kv_heads = [c["swa_num_key_value_heads"] if p else c["num_key_value_heads"] for p in pattern]
    rope_dim = int(c["head_dim"] * c["partial_rotary_factor"])
    records = [
        kv_string("general.architecture", ARCH),
        kv_string("general.name", name),
        kv_string("general.source.url", "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL"),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(f"{ARCH}.block_count", n_all),
        kv_u32(f"{ARCH}.nextn_predict_layers", layout.n_nextn),
        kv_u32(f"{ARCH}.context_length", c["max_position_embeddings"]),
        kv_u32(f"{ARCH}.embedding_length", layout.dim),
        kv_u32(f"{ARCH}.feed_forward_length", layout.inter),
        kv_u32(f"{ARCH}.expert_feed_forward_length", layout.moe_inter),
        kv_u32(f"{ARCH}.expert_count", layout.experts),
        kv_u32(f"{ARCH}.expert_used_count", c["num_experts_per_tok"]),
        kv_u32(f"{ARCH}.leading_dense_block_count", 1),
        kv_f32(f"{ARCH}.expert_weights_scale", 1.0),
        kv_bool(f"{ARCH}.expert_weights_norm", True),
        kv_u32(f"{ARCH}.expert_gating_func", 2),
        kv_u32(f"{ARCH}.attention.head_count", c["num_attention_heads"]),
        kv_u32_array(f"{ARCH}.attention.head_count_kv", kv_heads),
        kv_u32(f"{ARCH}.attention.key_length", c["head_dim"]),
        kv_u32(f"{ARCH}.attention.value_length", c["v_head_dim"]),
        kv_f32(f"{ARCH}.attention.layer_norm_rms_epsilon", c["layernorm_epsilon"]),
        kv_u32(f"{ARCH}.attention.sliding_window", c["sliding_window"]),
        kv_u32_array(f"{ARCH}.attention.sliding_window_pattern", pattern),
        kv_f32(f"{ARCH}.attention.value_scale", c["attention_value_scale"]),
        kv_u32(f"{ARCH}.rope.dimension_count", rope_dim),
        kv_f32(f"{ARCH}.rope.freq_base", c["rope_theta"]),
        kv_f32(f"{ARCH}.rope.freq_base_swa", c["swa_rope_theta"]),
        kv_u32(f"{ARCH}.vocab_size", layout.vocab),
        kv_u32(f"{ARCH}.qkv_chunks", layout.tp),
    ]
    return records + tokenizer_records(hf_dir, layout.vocab)


def dflash_plan(db, cfg, quant):
    """Plan for the drafter support GGUF (llama.cpp `dflash` names + DS4 extras)."""
    mq = MATRIX_QTYPE[quant]
    h, n_q, n_kv, hd, inter = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
    n_tgt = len(cfg["dflash_config"]["target_layer_ids"])
    plan, consumed = [], set()

    def regular(name, source, shape, qtype, role):
        info = db.info(source)
        if info["shape"] != list(shape) or info["dtype"] != "BF16":
            raise ValueError(f"{source}: unexpected {info['dtype']} {info['shape']}, expected BF16 {shape}")
        consumed.add(source)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source))

    regular("fc.weight", "fc.weight", (h, n_tgt * cfg["target_hidden_size"]), mq, "dflash")
    regular("enc.output_norm.weight", "hidden_norm.weight", (h,), QTYPE_F32, "norm")
    regular("output_norm.weight", "norm.weight", (h,), QTYPE_F32, "norm")
    for il in range(cfg["num_hidden_layers"]):
        src, dst = f"layers.{il}", f"blk.{il}"
        regular(f"{dst}.attn_norm.weight", f"{src}.input_layernorm.weight", (h,), QTYPE_F32, "norm")
        regular(f"{dst}.ffn_norm.weight", f"{src}.post_attention_layernorm.weight", (h,), QTYPE_F32, "norm")
        regular(f"{dst}.attn_q_norm.weight", f"{src}.self_attn.q_norm.weight", (hd,), QTYPE_F32, "norm")
        regular(f"{dst}.attn_k_norm.weight", f"{src}.self_attn.k_norm.weight", (hd,), QTYPE_F32, "norm")
        regular(f"{dst}.attn_sinks.weight", f"{src}.self_attn.attention_sink_bias", (n_q,), QTYPE_F32, "attention")
        regular(f"{dst}.attn_q.weight", f"{src}.self_attn.q_proj.weight", (n_q * hd, h), mq, "attention")
        regular(f"{dst}.attn_k.weight", f"{src}.self_attn.k_proj.weight", (n_kv * hd, h), mq, "attention")
        regular(f"{dst}.attn_v.weight", f"{src}.self_attn.v_proj.weight", (n_kv * hd, h), mq, "attention")
        regular(f"{dst}.attn_output.weight", f"{src}.self_attn.o_proj.weight", (h, n_q * hd), mq, "attention")
        regular(f"{dst}.ffn_gate.weight", f"{src}.mlp.gate_proj.weight", (inter, h), mq, "dense")
        regular(f"{dst}.ffn_up.weight", f"{src}.mlp.up_proj.weight", (inter, h), mq, "dense")
        regular(f"{dst}.ffn_down.weight", f"{src}.mlp.down_proj.weight", (h, inter), mq, "dense")
    if consumed != set(db.tensors):
        raise ValueError(f"unclaimed drafter tensors: {sorted(set(db.tensors) - consumed)[:10]}")
    plan.append(TensorPlan("mask_embd.weight", (h,), QTYPE_F32, "dflash", source="mask_embedding.pt"))
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


def dflash_records(cfg, layout, revision, name):
    dc = cfg["dflash_config"]
    n = cfg["num_hidden_layers"]
    if cfg["is_causal"] or cfg["layer_types"] != ["sliding_attention"] * n or cfg["block_size"] != dc["block_size"]:
        raise ValueError("unexpected drafter attention layout")
    if cfg["hidden_size"] != layout.dim or cfg["target_hidden_size"] != layout.dim or \
            cfg["num_target_layers"] != layout.n_layer or cfg["vocab_size"] != layout.vocab:
        raise ValueError("drafter does not match the target model")
    return [
        kv_string("general.architecture", "dflash"),
        kv_string("general.name", name),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32("dflash.block_count", n),
        kv_u32("dflash.context_length", cfg["max_position_embeddings"]),
        kv_u32("dflash.embedding_length", cfg["hidden_size"]),
        kv_u32("dflash.feed_forward_length", cfg["intermediate_size"]),
        kv_u32("dflash.vocab_size", cfg["vocab_size"]),
        kv_u32("dflash.attention.head_count", cfg["num_attention_heads"]),
        kv_u32("dflash.attention.head_count_kv", cfg["num_key_value_heads"]),
        kv_u32("dflash.attention.key_length", cfg["head_dim"]),
        kv_u32("dflash.attention.value_length", cfg["v_head_dim"]),
        kv_f32("dflash.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]),
        kv_u32("dflash.attention.sliding_window", cfg["sliding_window"]),
        kv_u32_array("dflash.attention.sliding_window_pattern", [1] * n),
        kv_bool("dflash.attention.causal", False),
        kv_f32("dflash.attention.value_scale", dc["attention_value_scale"]),
        kv_u32("dflash.rope.dimension_count", int(cfg["head_dim"] * cfg["partial_rotary_factor"])),
        kv_f32("dflash.rope.freq_base", cfg["rope_theta"]),
        kv_u32("dflash.block_size", dc["block_size"]),
        # llama.cpp convention: the input of layer i+1, i.e. the residual output of target layer i
        kv_u32_array("dflash.target_layers", [i + 1 for i in dc["target_layer_ids"]]),
        kv_u32("tokenizer.ggml.mask_token_id", dc["mask_token_id"]),
    ]


def read_mask_embedding(path, hidden):
    """`mask_embedding.pt` is a torch zip pickle holding {"mask_token_id", "embedding": bf16[H]}."""
    import torch

    state = torch.load(path, map_location="cpu")
    vec = state["embedding"]
    if tuple(vec.shape) != (hidden,):
        raise ValueError(f"mask embedding shape {tuple(vec.shape)}")
    return vec.float().numpy(), int(state["mask_token_id"])


def write_gguf(out, plan, records, db, quantizer, imatrix, threads, resume, extra_payloads=None):
    data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    partial, journal = out + ".partial", out + ".partial.json"
    signature = conversion_signature(plan, records, [], None)
    source_identity = [(name, db.info(name)) for name in sorted(db.tensors)]
    signature = hashlib.sha256((signature + json.dumps(source_identity, sort_keys=True)).encode()).hexdigest()
    if os.path.exists(out):
        raise ValueError(f"refusing to overwrite {out}")
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not resume or not (os.path.exists(partial) and os.path.exists(journal)):
            raise ValueError("partial file and journal require --resume")
        completed = load_resume_state(journal, signature, plan)
    end = data_start + (plan[completed - 1].offset +
                       align(plan[completed - 1].nbytes, GGUF_ALIGNMENT) if completed else 0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out)) or ".").free
    if free < data_start + data_bytes - end + (4 << 30):
        raise ValueError("insufficient disk space for remaining output plus 4 GiB reserve")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                raise ValueError("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(journal, signature, 0)
    with open(partial, "r+b") as fp, concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        fp.truncate(end)
        fp.seek(end)
        for index in range(completed, len(plan)):
            item = plan[index]
            started = time.monotonic()
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")
            if extra_payloads and item.source in extra_payloads:
                fp.write(quantizer.encode(extra_payloads[item.source], item.qtype))
            elif item.is_expert:
                def convert(expert):
                    source = item.source.format(expert=expert)
                    if item.qtype == QTYPE_MXFP4:
                        return quantizer.to_mxfp4(db, source)
                    values = quantizer.to_f32(db, source)
                    importance = imatrix.expert(item.name, expert, item.shape[0], item.expert_count)
                    return quantizer.encode(values, item.qtype, importance)
                for start in range(0, item.expert_count, threads):
                    futures = [pool.submit(convert, e) for e in range(start, min(start + threads, item.expert_count))]
                    for future in futures:
                        data = future.result()
                        if len(data) != item.nbytes // item.expert_count:
                            raise ValueError("wrong encoded expert size")
                        fp.write(data)
            else:
                fp.write(quantizer.encode(quantizer.to_f32(db, item.source), item.qtype))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(journal, signature, index + 1)
            print(f"[{index + 1}/{len(plan)}] {item.name}: {item.nbytes / (1 << 30):.3f} GiB, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    os.rename(partial, out)
    os.unlink(journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="mxfp4")
    parser.add_argument("--name", default="MiMo V2.6 Flash")
    parser.add_argument("--imatrix")
    parser.add_argument("--dflash-out", help="write the DFlash drafter support GGUF")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    config = load_config(args.hf)
    layout = Layout(config, model_tp(args.hf))
    db = SourceDB(args.hf, index_validator=validate_index, scale_validator=make_scale_validator(layout))
    try:
        plan = build_plan(db, layout, args.quant)
        records = model_records(args.hf, layout, args.source_revision, args.name)
        records.append(kv_string(f"{ARCH}.quantization", QUANTIZATION[args.quant]))
        quantizer = NativeQuantizer(args.quants_library, layout)
        imatrix = Imatrix(args.imatrix, quantizer.np)
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            write_gguf(args.out, plan, records, db, quantizer, imatrix, args.threads, args.resume)
    finally:
        db.close()
    if args.dflash_out:
        ddir = os.path.join(args.hf, "dflash")
        dcfg = load_json_lenient(os.path.join(ddir, "config.json"))
        ddb = SourceDB(ddir, index_validator=lambda _: None, scale_validator=lambda _: None)
        try:
            dplan = dflash_plan(ddb, dcfg, args.quant)
            drecords = dflash_records(dcfg, layout, args.source_revision, args.name + " DFlash")
            mask, mask_id = read_mask_embedding(os.path.join(ddir, "mask_embedding.pt"), dcfg["hidden_size"])
            if mask_id != dcfg["dflash_config"]["mask_token_id"]:
                raise ValueError("mask_embedding.pt disagrees with dflash config")
            if args.dry_run:
                print_plan(dplan, drecords, [], GGUF_ALIGNMENT)
            else:
                write_gguf(args.dflash_out, dplan, drecords, ddb, quantizer, imatrix, args.threads, args.resume,
                           extra_payloads={"mask_embedding.pt": mask})
        finally:
            ddb.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"mimo26-quantize: {error}")
