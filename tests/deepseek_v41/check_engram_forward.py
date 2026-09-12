#!/usr/bin/env python3
"""Score DS4's engram block against the released reference maths on the real weights.

DS4 reads the sidecar; this reads the original checkpoint and derives the hash layout
from config.json, so a pass exercises the converter, the loader and the forward path at
once. Run tests/test_dsv41_ref with DS4_ENGRAM_DUMP set first, then point --dump here.
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-tools"))
import deepseek_v41_engram as conv  # noqa: E402

SNAP = ("/Volumes/WD_BLACK/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/"
        "snapshots/dba1be0a40aa45a94ad051997016db3960a90277")
SHARD = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}
TOKENS = [128000, 2, 1820, 4062, 14198, 39935]
DIM, HC = 5120, 4


def lcg_stream(n, seed=12345):
    """The same pseudo-random stream tests/test_dsv41_ref.c builds."""
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = ((seed >> 8) % 2001 - 1000) / 1000.0
    return out


def hash_ids(config, layer_index, tokens):
    """NgramHashState.forward for the last position, from config-derived tables."""
    primes = conv.engram_primes(config)
    flat = np.array([p for per_ngram in primes[layer_index] for p in per_ngram], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(flat[:-1])))
    mult = conv.engram_multipliers(config)[layer_index]
    token_map = conv.build_token_map(SNAP, config)
    pad = int(token_map[config["engram_pad_token_id"]])

    compressed = [int(token_map[t]) for t in tokens]
    pos, blocked, ids = len(tokens) - 1, False, []
    for shift in range(config["engram_max_ngram_size"]):
        if pos - shift < 0:
            blocked = True
        ids.append(pad if blocked else compressed[pos - shift])
    products = [np.int64(t) * mult[i] for i, t in enumerate(ids)]
    rolling, out = products[0], []
    for i in range(1, config["engram_max_ngram_size"]):
        rolling = np.bitwise_xor(rolling, products[i])
        out.append(rolling)
    n_head = config["engram_n_heads"]
    rows = np.empty(len(flat), dtype=np.int64)
    for i, roll in enumerate(out):
        for h in range(n_head):
            c = i * n_head + h
            rows[c] = roll % flat[c] + offsets[c]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="file written by DS4_ENGRAM_DUMP")
    parser.add_argument("--layer-index", type=int, default=0)
    args = parser.parse_args()

    config = conv.load_config(SNAP)
    layer = config["engram_layer_ids"][args.layer_index]
    path = os.path.join(SNAP, SHARD[layer])
    db = conv.SourceDB(SNAP, conv.validate_deepseek41_index, conv.validate_engram_sources)

    rows = hash_ids(config, args.layer_index, TOKENS)
    print(f"layer {layer}  hash rows {rows[:4].tolist()} ... {rows[-1]}")

    # --- the 24 looked-up rows, straight from the checkpoint
    head_dim = config["engram_head_dim"]
    emb = np.empty((len(rows), head_dim), dtype=np.float32)
    lut = conv.e4m3_lut()
    with safe_open(path, framework="pt") as fp:
        weight = fp.get_slice(f"layers.{layer}.engram.embed.weight")
        for i, row in enumerate(rows):
            raw = weight[int(row):int(row) + 1].view(torch.uint8).numpy().reshape(-1)
            scale = np.frombuffer(b"".join(db.iter_read(
                f"layers.{layer}.engram.embed.scale", int(row) * 8, 8)), np.uint8)
            emb[i] = lut[raw] * np.repeat(
                np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127), 32)

    # --- wkv, dequantized the same way the converter does
    wkv_shape = db.info(f"layers.{layer}.engram.wkv.weight")["shape"]
    payload = conv.wkv_bf16(db, layer, tuple(wkv_shape))
    wkv = (np.frombuffer(payload, np.uint16).astype(np.uint32) << 16)
    wkv = wkv.view(np.float32).reshape(wkv_shape)
    with safe_open(path, framework="pt") as fp:
        q_w = fp.get_tensor(f"layers.{layer}.engram.q_weight").to(torch.float32).numpy()
        k_w = fp.get_tensor(f"layers.{layer}.engram.k_weight").to(torch.float32).numpy()
    db.close()

    # --- Engram.forward, model.py lines 350-365
    kv = wkv @ emb.reshape(-1).astype(np.float32)
    key = kv[:HC * DIM].reshape(HC, DIM)
    value = kv[HC * DIM:]

    blob = np.fromfile(args.dump, dtype=np.float32)
    assert blob.size == HC * DIM * 2 + HC, blob.size
    x = blob[:HC * DIM].reshape(HC, DIM)
    got_gate = blob[HC * DIM:HC * DIM + HC]
    got_out = blob[HC * DIM + HC:].reshape(HC, DIM)

    # -ffast-math turns the C side's /1000.0f into a reciprocal multiply, so the
    # stream matches to a float ULP rather than bit for bit
    want_x = lcg_stream(HC * DIM).reshape(HC, DIM)
    x_d = np.abs(x - want_x).max()
    same_x = x_d <= 1e-7

    eps = 1e-20
    h = x.astype(np.float64)
    weight = q_w.astype(np.float64) * k_w.astype(np.float64)
    rstd = (1.0 / np.sqrt((h ** 2).mean(-1) + eps)) * \
           (1.0 / np.sqrt((key.astype(np.float64) ** 2).mean(-1) + eps))
    dot = (h * weight * key).sum(-1) * rstd * DIM ** -0.5
    gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)))
    out = h + gate[:, None] * value.astype(np.float64)[None, :]

    gate_d = np.abs(got_gate - gate).max()
    out_d = np.abs(got_out - out).max()
    rel = out_d / max(np.abs(out).max(), 1e-9)
    ok = same_x and gate_d < 2e-5 and rel < 2e-5
    print(f"stream input matches    : {same_x}  max|d| {x_d:.3e}")
    print(f"gate   ds4 {np.array2string(got_gate, precision=6)}")
    print(f"gate   ref {np.array2string(gate.astype(np.float32), precision=6)}")
    print(f"gate   max|d| {gate_d:.3e}")
    print(f"output max|d| {out_d:.3e}  rel {rel:.3e}  (|out| max {np.abs(out).max():.4g})")
    print("engram forward vs the released reference: " + ("MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
