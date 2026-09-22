#!/usr/bin/env python3
"""Write the MiMo-V2.6 vision tower (`visual.*`) as a llama.cpp clip mmproj
GGUF with projector type `mimovl`: the Conv3D patch embedding split into its
two temporal taps (`v.patch_embd.weight`, `.weight.1`), the blocks as
`v.blk.N.*` with per-head sinks on the windowed blocks, the merger's
LayerNorm as `v.post_ln.weight` and its bias-free MLP as `mm.0`/`mm.2`.
Matrices are Q8_0 (`--quant q8`) or F32; vectors stay F32.
"""
import argparse
import dataclasses
import json
import os
import re
import sys

from glm53_quantize import (
    GGUF_ALIGNMENT, GGUF_STRING, SourceDB, TensorPlan, Quantizer, Imatrix, QTYPE_F32, QTYPE_Q8_0, align,
    kv_bool, kv_f32, kv_string, kv_u32, print_plan, qtype_nbytes,
)
from mimo26_quantize import array_record, load_config, load_json_lenient, validate_index, write_gguf

MATRIX_QTYPE = {"q8": QTYPE_Q8_0, "f32": QTYPE_F32}
DEFAULT_PREPROCESSOR = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}


def load_preprocessor(hf_dir):
    path = os.path.join(hf_dir, "preprocessor_config.json")
    cfg = dict(DEFAULT_PREPROCESSOR)
    if os.path.exists(path):
        cfg.update({k: v for k, v in load_json_lenient(path).items() if k in cfg})
    return cfg


def head_dim(vc):
    return int(vc.get("qk_channels", 64))   # the HF tower's default


def vision_plan(db, vc, quant, payloads, quantizer):
    """Plan in llama.cpp clip order; the split patch taps come from payloads."""
    mq = MATRIX_QTYPE[quant]
    np = quantizer.np
    h, ff, nh, nkv = vc["hidden_size"], vc["intermediate_size"], vc["num_heads"], vc["num_key_value_heads"]
    hd, patch, tp, chans, out = head_dim(vc), vc["patch_size"], vc.get("temporal_patch_size", 2), vc.get("in_chans", 3), vc["out_hidden_size"]
    merged = h * vc["spatial_merge_size"] ** 2
    types = vc["vit_window_attn_types"]
    plan, consumed = [], set()

    def regular(name, source, shape, qtype):
        info = db.info(source)
        if info["shape"] != list(shape) or info["dtype"] != "BF16":
            raise ValueError(f"{source}: unexpected {info['dtype']} {info['shape']}, expected BF16 {shape}")
        consumed.add(source)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, "vision", source=source))

    conv = "visual.patch_embed.proj.weight"
    info = db.info(conv)
    if info["shape"] != [h, chans, tp, patch, patch] or tp != 2:
        raise ValueError(f"{conv}: unexpected shape {info['shape']}")
    taps = quantizer.to_f32(db, conv).reshape(h, chans, tp, patch * patch)
    consumed.add(conv)
    for t in range(tp):
        key = f"patch.{t}"
        payloads[key] = np.ascontiguousarray(taps[:, :, t, :].reshape(h, chans * patch * patch), dtype=np.float32)
        plan.append(TensorPlan("v.patch_embd.weight" + (".1" if t else ""), (chans * patch * patch, h), mq, "vision",
                               source=key))
    for il in range(vc["depth"]):
        src, dst = f"visual.blocks.{il}", f"v.blk.{il}"
        regular(f"{dst}.ln1.weight", f"{src}.norm1.weight", (h,), QTYPE_F32)
        regular(f"{dst}.attn_qkv.weight", f"{src}.attn.qkv.weight", ((nh + 2 * nkv) * hd, h), mq)
        regular(f"{dst}.attn_qkv.bias", f"{src}.attn.qkv.bias", ((nh + 2 * nkv) * hd,), QTYPE_F32)
        regular(f"{dst}.attn_out.weight", f"{src}.attn.proj.weight", (h, nh * hd), mq)
        regular(f"{dst}.attn_out.bias", f"{src}.attn.proj.bias", (h,), QTYPE_F32)
        if types[il] != -1:
            regular(f"{dst}.attn_sinks", f"{src}.attn.sinks", (nh,), QTYPE_F32)
        regular(f"{dst}.ln2.weight", f"{src}.norm2.weight", (h,), QTYPE_F32)
        for w_name, out_dim, in_dim in (("gate", ff, h), ("up", ff, h), ("down", h, ff)):
            regular(f"{dst}.ffn_{w_name}.weight", f"{src}.mlp.{w_name}_proj.weight", (out_dim, in_dim), mq)
            regular(f"{dst}.ffn_{w_name}.bias", f"{src}.mlp.{w_name}_proj.bias", (out_dim,), QTYPE_F32)
    regular("v.post_ln.weight", "visual.merger.ln_q.weight", (h,), QTYPE_F32)
    regular("mm.0.weight", "visual.merger.mlp.0.weight", (merged, merged), mq)
    regular("mm.2.weight", "visual.merger.mlp.2.weight", (out, merged), mq)
    visual = {name for name in db.tensors if name.startswith("visual.")}
    if consumed != visual:
        raise ValueError(f"unclaimed vision tensors: {sorted(visual - consumed)[:10]}")
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


def vision_records(vc, pp, revision, name):
    types = vc["vit_window_attn_types"]
    if len(types) != vc["depth"] or any(t not in (-1, 0, 1) for t in types):
        raise ValueError("unexpected vit_window_attn_types")
    if [i for i, t in enumerate(types) if t == -1] != list(vc["fullatt_block_indexes"]):
        raise ValueError("fullatt_block_indexes disagree with vit_window_attn_types")
    if not vc["use_sink"] or vc["hidden_act"] != "silu" or vc["spatial_merge_size"] != 2:
        raise ValueError("unexpected vision tower configuration")
    return [
        kv_string("general.architecture", "clip"),
        kv_string("general.name", name),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_bool("clip.has_vision_encoder", True),
        kv_string("clip.projector_type", "mimovl"),
        kv_bool("clip.use_silu", True),
        kv_u32("clip.vision.image_size", 560),
        kv_u32("clip.vision.patch_size", vc["patch_size"]),
        kv_u32("clip.vision.embedding_length", vc["hidden_size"]),
        kv_u32("clip.vision.feed_forward_length", vc["intermediate_size"]),
        kv_u32("clip.vision.projection_dim", vc["out_hidden_size"]),
        kv_u32("clip.vision.block_count", vc["depth"]),
        kv_u32("clip.vision.attention.head_count", vc["num_heads"]),
        kv_u32("clip.vision.attention.head_count_kv", vc["num_key_value_heads"]),
        kv_u32("clip.vision.attention.head_dim", head_dim(vc)),
        kv_f32("clip.vision.attention.layer_norm_epsilon", float(vc.get("rms_norm_eps", 1e-6))),
        kv_u32("clip.vision.spatial_merge_size", vc["spatial_merge_size"]),
        kv_u32("clip.vision.window_size", vc["visual_token_window_size"]),
        array_record("clip.vision.wa_pattern_mode", 5, types),
        kv_u32("clip.vision.image_min_pixels", int(pp["min_pixels"])),
        kv_u32("clip.vision.image_max_pixels", int(pp["max_pixels"])),
        array_record("clip.vision.image_mean", 6, [float(v) for v in pp["image_mean"]]),
        array_record("clip.vision.image_std", 6, [float(v) for v in pp["image_std"]]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=MATRIX_QTYPE, default="q8")
    parser.add_argument("--name", default="MiMo V2.6 Flash Vision")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    config = load_config(args.hf)
    vc = config["vision_config"]
    if config.get("vision_model_type") != "mimovl":
        raise ValueError("not a mimovl checkpoint")
    db = SourceDB(args.hf, index_validator=validate_index, scale_validator=lambda _: None)
    try:
        quantizer = Quantizer(args.quants_library)
        payloads = {}
        plan = vision_plan(db, vc, args.quant, payloads, quantizer)
        records = vision_records(vc, load_preprocessor(args.hf), args.source_revision, args.name)
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            write_gguf(args.out, plan, records, db, quantizer, Imatrix(None, quantizer.np), args.threads, args.resume,
                       extra_payloads=payloads)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"mimo26-vision: {error}")
