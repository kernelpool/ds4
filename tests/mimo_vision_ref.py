#!/usr/bin/env python3
"""Compare DS4's MiMo-V2.6 vision tower with the Hugging Face tower.

The HF tower comes from the checkpoint's own modeling file, loaded with the
weights of the mmproj GGUF (implementation parity) and with the original
checkpoint weights (quantization quality, reported separately). It is run as
Xiaomi's serving code (SGLang, vLLM) runs it: the windowed blocks' sinks are an
extra softmax logit rather than a bias on key 0, the merger's ln_q is an
RMSNorm, and images go through the Qwen2-VL image processor with the ImageNet
mean/std (preprocessor_config.json lists CLIP's).

  python tests/mimo_vision_ref.py --snapshot /path/to/MiMo-V2.6-Flash-RL \\
      --mmproj MiMo-V2.6-Flash-Vision-F32.gguf \\
      --image tests/vision-fixtures/qwen38/orbit.png [--min-cos 0.99] [--max-abs 1e-3]
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_lenient_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", fp.read()))


def hf_tower(snapshot, mmproj=None):
    import torch
    from safetensors import safe_open
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    tower_class = get_class_from_dynamic_module("modeling_mimo_v2.MiMoVisionTransformer", snapshot)
    config = load_lenient_json(os.path.join(snapshot, "config.json"))
    tower = tower_class(SimpleNamespace(**config["vision_config"])).float().eval()
    if mmproj is not None:
        state = gguf_state(tower, mmproj)
    else:
        weight_map = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
        state = {}
        for shard in sorted({s for k, s in weight_map.items() if k.startswith("visual.")}):
            with safe_open(os.path.join(snapshot, shard), "pt") as f:
                for key in f.keys():
                    if key.startswith("visual."):
                        state[key[len("visual."):]] = f.get_tensor(key).float()
    # the checkpoint carries no merger biases; HF zeroes missing biases
    for key, param in tower.state_dict().items():
        if key.endswith(("ln_q.bias", "mlp.0.bias", "mlp.2.bias")) and key not in state:
            state[key] = torch.zeros_like(param)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        sys.exit(f"vision weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    serving_semantics(tower, sys.modules[tower_class.__module__])
    return tower, config


def serving_semantics(tower, modeling):
    """ln_q as an RMSNorm and the sinks as an extra softmax logit, as SGLang and vLLM run the tower."""
    import types
    import torch
    import torch.nn.functional as F

    tower.merger.ln_q.forward = types.MethodType(
        lambda self, x: (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight, tower.merger.ln_q)

    def forward(self, hidden_states, cu_seqlens, position_embeddings, full_attn=False):
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q_dim, kv_dim = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].view(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim:q_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim:].view(seq_len, self.num_kv_heads, self.head_dim)
        q, k = modeling._apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for q_c, k_c, v_c in zip(torch.split(q, lengths), torch.split(k, lengths), torch.split(v, lengths)):
            q_c, k_c, v_c = (t.unsqueeze(0).transpose(1, 2) for t in (q_c, k_c, v_c))
            if self.num_kv_groups > 1:
                k_c = k_c.repeat_interleave(self.num_kv_groups, dim=1)
                v_c = v_c.repeat_interleave(self.num_kv_groups, dim=1)
            scores = torch.matmul(q_c, k_c.transpose(-1, -2)) * self.scaling
            mask = None if full_attn else self._build_window_mask(q_c.shape[2], q_c.device, q_c.dtype)
            if mask is not None:
                scores = scores + mask
            if self.sinks is not None:
                sink = self.sinks.view(1, -1, 1, 1).expand(*scores.shape[:3], 1).to(scores.dtype)
                probs = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
            else:
                probs = F.softmax(scores, dim=-1)
            outputs.append(torch.matmul(probs, v_c).squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(outputs, dim=0).reshape(seq_len, -1))

    for block in tower.blocks:
        block.attn.forward = types.MethodType(forward, block.attn)


def gguf_state(tower, path):
    import torch
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    used = set()

    def tensor(name):
        used.add(name)
        t = tensors[name]
        shape = tuple(reversed(t.shape.tolist()))
        return torch.from_numpy(dequantize(t.data, t.tensor_type).reshape(shape).copy()).float()

    state = {}
    for key, param in tower.state_dict().items():
        if key == "patch_embed.proj.weight":
            taps = [tensor("v.patch_embd.weight"), tensor("v.patch_embd.weight.1")]
            out, chans = param.shape[0], param.shape[1]
            state[key] = torch.stack([t.reshape(out, chans, param.shape[3], param.shape[4]) for t in taps], dim=2)
            continue
        if key.startswith("blocks."):
            _, layer, rest = key.split(".", 2)
            for old, new in [("attn.proj.", "attn_out."), ("attn.qkv.", "attn_qkv."), ("attn.sinks", "attn_sinks"),
                             ("mlp.gate_proj.", "ffn_gate."), ("mlp.up_proj.", "ffn_up."),
                             ("mlp.down_proj.", "ffn_down."), ("norm1.", "ln1."), ("norm2.", "ln2.")]:
                if rest.startswith(old):
                    name = f"v.blk.{layer}." + rest.replace(old, new, 1)
                    break
            else:
                raise ValueError(f"unmapped HF vision tensor: {key}")
        elif key == "merger.ln_q.weight":
            name = "v.post_ln.weight"
        elif key == "merger.mlp.0.weight":
            name = "mm.0.weight"
        elif key == "merger.mlp.2.weight":
            name = "mm.2.weight"
        elif key.endswith(("ln_q.bias", "mlp.0.bias", "mlp.2.bias")):
            state[key] = torch.zeros_like(param)
            continue
        else:
            raise ValueError(f"unmapped HF vision tensor: {key}")
        state[key] = tensor(name)
    if used != set(tensors):
        raise ValueError(f"unused GGUF tensors: {sorted(set(tensors) - used)}")
    return state


def hf_embeddings(tower, config, snapshot, image_path, min_tokens, max_tokens):
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil as Qwen2VLImageProcessor

    ip = Qwen2VLImageProcessor(min_pixels=min_tokens * 32 * 32, max_pixels=max_tokens * 32 * 32,
                               patch_size=16, merge_size=2, temporal_patch_size=2,
                               image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD)
    img = Image.open(image_path).convert("RGB")
    out = ip(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = tower(out["pixel_values"].float(), grid_thw=out["image_grid_thw"])
    thw = out["image_grid_thw"][0].tolist()
    return emb.numpy(), (thw[1] // 2, thw[2] // 2)


def ds4_embeddings(binary, mmproj, image_path, out_path, min_tokens, max_tokens):
    subprocess.run([binary, mmproj, image_path, out_path, str(min_tokens), str(max_tokens), out_path + ".patches"],
                   check=True)
    with open(out_path, "rb") as f:
        n, dim, gh, gw = struct.unpack("<4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, dim)
    with open(out_path + ".patches", "rb") as f:
        n_patch, in_dim, ph, pw = struct.unpack("<4I", f.read(16))
        patches = np.frombuffer(f.read(), dtype=np.float32).reshape(n_patch, in_dim)
    return data, (gh, gw), patches, (ph, pw)


def hf_embeddings_from_patches(tower, patches, grid):
    """The tower on DS4's own patches ([c][y][x] per patch), duplicated over the two temporal taps."""
    import torch
    n = patches.shape[0]
    pv = torch.from_numpy(patches.reshape(n, 3, 1, 256).repeat(2, axis=2).reshape(n, 1536).copy())
    with torch.no_grad():
        emb = tower(pv, grid_thw=torch.tensor([[1, grid[0], grid[1]]]))
    return emb.numpy(), (grid[0] // 2, grid[1] // 2)


def compare(actual, reference, actual_grid, reference_grid, min_cos, max_abs):
    if actual.shape != reference.shape or actual_grid != reference_grid:
        return {"pass": False, "error": f"token layout mismatch {actual.shape}/{actual_grid} vs {reference.shape}/{reference_grid}"}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"pass": False, "error": "non-finite embeddings"}
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    diff = np.abs(a - b)
    ok = bool(cos.min() >= min_cos) and (max_abs is None or bool(diff.max() <= max_abs))
    return {"pass": ok, "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
            "worst_token": int(cos.argmin()), "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True, action="append")
    ap.add_argument("--binary", default=os.path.join(HERE, "test_mimo_vision"))
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--max-abs", type=float, help="also require max |diff| below this (F32 mmproj parity)")
    ap.add_argument("--json-report")
    args = ap.parse_args()
    original, config = hf_tower(args.snapshot)
    quant, _ = hf_tower(args.snapshot, args.mmproj)
    results = []
    with tempfile.TemporaryDirectory(prefix="mimo-vision-") as directory:
        for i, image_path in enumerate(args.image):
            out = os.path.join(directory, f"{i}.bin")
            actual, grid, patches, pgrid = ds4_embeddings(args.binary, args.mmproj, image_path, out,
                                                          args.min_tokens, args.max_tokens)
            exact_q, exact_grid = hf_embeddings_from_patches(quant, patches, pgrid)
            ref_q, grid_q = hf_embeddings(quant, config, args.snapshot, image_path, args.min_tokens, args.max_tokens)
            ref_o, grid_o = hf_embeddings(original, config, args.snapshot, image_path, args.min_tokens, args.max_tokens)
            result = {"image": image_path, "grid": grid, "shape": list(actual.shape),
                      "implementation": compare(actual, exact_q, grid, exact_grid, args.min_cos, args.max_abs),
                      "preprocessing": compare(actual, ref_q, grid, grid_q, args.min_cos, None),
                      "quantization_quality": compare(ref_q, ref_o, grid_q, grid_o, args.min_cos, None),
                      "gpu_vs_original": compare(actual, ref_o, grid, grid_o, args.min_cos, None)}
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump({"snapshot": args.snapshot, "mmproj": args.mmproj, "results": results}, f, indent=2)
    if any(not r["implementation"]["pass"] for r in results):
        sys.exit("FAIL")
    print("ok: implementation parity; quantization quality reported separately")


if __name__ == "__main__":
    main()
