#!/usr/bin/env python3
"""Check the engram sidecar transforms against the checkpoint, without writing 203 GB.

Reads a slice of the real engram tables and proves:
  1. the f8_e4m3 interleave puts the E8M0 exponent in front of its 32 E4M3 values;
  2. decoding those blocks matches torch's own float8_e4m3fn dequantization bit for bit;
  3. the wkv 32x32-blocked fp8 weights survive the bf16 round trip the converter asserts.
"""

import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-tools"))
import deepseek_v41_engram as conv  # noqa: E402
from glm53_quantize import SourceDB  # noqa: E402

SNAP = ("/Volumes/WD_BLACK/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/"
        "snapshots/dba1be0a40aa45a94ad051997016db3960a90277")
ROWS = 4096
fails = 0


def check(what, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(f"  {what:<34s} {'ok  ' if ok else 'FAIL'}  {detail}")


def torch_rows(shard, name, rows):
    with safe_open(os.path.join(SNAP, shard), framework="pt") as fp:
        return fp.get_slice(name)[0:rows]


def main():
    conv.load_config(SNAP)
    db = SourceDB(SNAP, conv.validate_deepseek41_index, conv.validate_engram_sources)
    try:
        # --- 1. interleave layout -------------------------------------------------
        blocks = next(conv.embed_chunks(db, 1, ROWS))
        blocks = np.frombuffer(blocks[: ROWS * 264], np.uint8).reshape(ROWS, 8, 33)
        raw_w = np.frombuffer(
            b"".join(db.iter_read("layers.1.engram.embed.weight", 0, ROWS * 256)), np.uint8
        ).reshape(ROWS, 8, 32)
        raw_s = np.frombuffer(
            b"".join(db.iter_read("layers.1.engram.embed.scale", 0, ROWS * 8)), np.uint8
        ).reshape(ROWS, 8)
        check("block scale byte", np.array_equal(blocks[:, :, 0], raw_s))
        check("block value bytes", np.array_equal(blocks[:, :, 1:], raw_w))
        check("row stride", blocks[0].nbytes == 264, f"{blocks[0].nbytes} bytes/row")

        # --- 2. decode vs torch's own fp8 -----------------------------------------
        lut = conv.e4m3_lut()
        scales = np.ldexp(np.float32(1.0), blocks[:, :, 0].astype(np.int32) - 127)
        ours = lut[blocks[:, :, 1:]] * scales[:, :, None]
        ours = ours.reshape(ROWS, 256)

        weight = torch_rows("model-00047-of-00048.safetensors",
                            "layers.1.engram.embed.weight", ROWS)
        theirs = weight.to(torch.float32).numpy()
        exps = np.ldexp(np.float32(1.0), raw_s.astype(np.int32) - 127)
        theirs = theirs * np.repeat(exps, 32, axis=1)
        check("decode vs torch float8_e4m3fn", np.array_equal(ours, theirs),
              f"{ROWS} rows, max|d|={np.abs(ours - theirs).max():.3e}")
        nonzero = np.count_nonzero(ours)
        check("rows are not all zero", nonzero > ROWS * 64,
              f"{nonzero}/{ours.size} nonzero, range [{ours.min():.4g}, {ours.max():.4g}]")

        # --- 3. wkv bf16 round trip ----------------------------------------------
        shape = db.info("layers.1.engram.wkv.weight")["shape"]
        payload = conv.wkv_bf16(db, 1, tuple(shape))
        got = np.frombuffer(payload, np.uint16).astype(np.uint32) << 16
        got = got.view(np.float32).reshape(shape)
        check("wkv bf16 payload size", len(payload) == shape[0] * shape[1] * 2,
              f"{len(payload)} bytes")
        want = torch_rows("model-00047-of-00048.safetensors",
                          "layers.1.engram.wkv.weight", 64).to(torch.float32).numpy()
        wscale = np.frombuffer(
            b"".join(db.iter_read("layers.1.engram.wkv.scale", 0, 2 * shape[1] // 32)), np.uint8
        ).reshape(2, shape[1] // 32)
        want = want * np.repeat(np.repeat(
            np.ldexp(np.float32(1.0), wscale.astype(np.int32) - 127), 32, axis=0), 32, axis=1)
        check("wkv vs torch float8_e4m3fn", np.array_equal(got[:64], want),
              f"64 rows, max|d|={np.abs(got[:64] - want).max():.3e}")
    finally:
        db.close()

    print("engram converter transforms: " +
          (f"{fails} failure(s)" if fails else "all checks passed"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
