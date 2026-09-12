#!/usr/bin/env python3
"""Score the DS4 V4.1 CPU reference against the mini-model oracle, stage by stage.

Runs ds4_test_dsv41_prefill over mini/mini.gguf through a small C driver and compares
layerN.attn / layerN.ffn / layerN.out and the final logits with oracle_prefill.npz.
Layers whose compress_ratio is non-zero need the compressor and indexer, which the
reference does not implement yet; --layers limits the comparison to the ones it does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int ds4_test_dsv41_prefill(const char *, const int *, unsigned, float *, float *, float *, float *, int32_t *, float *, const char *);
void ds4_test_dsv41_scan_stats(unsigned long long *);
int main(int argc, char **argv) {
    if (argc < 10) return 2;
    const char *gguf = argv[1];
    const char *out = argv[2];
    const unsigned dim = (unsigned)atoi(argv[3]);
    const unsigned hc = (unsigned)atoi(argv[4]);
    const unsigned vocab = (unsigned)atoi(argv[5]);
    const char *engram = argv[8][0] ? argv[8] : NULL;
    const unsigned n = (unsigned)(argc - 9);
    int *tok = malloc(n * sizeof(int));
    for (unsigned i = 0; i < n; i++) tok[i] = atoi(argv[9 + i]);
    float *logits = malloc((size_t)vocab * sizeof(float));
    float *attn = malloc((size_t)64 * n * dim * sizeof(float));
    float *ffn = malloc((size_t)64 * n * dim * sizeof(float));
    float *stream = malloc((size_t)64 * n * dim * hc * sizeof(float));
    const unsigned topk = (unsigned)atoi(argv[6]);
    int32_t *idxs = malloc((size_t)64 * n * topk * sizeof(int32_t));
    const unsigned hd = (unsigned)atoi(argv[7]);
    float *comp = calloc((size_t)128 * n * hd, sizeof(float));
    int nl = ds4_test_dsv41_prefill(gguf, tok, n, logits, attn, ffn, stream, idxs, comp, engram);
    FILE *f = fopen(out, "wb");
    if (!f) return 3;
    fwrite(&nl, sizeof(int), 1, f);
    fwrite(logits, sizeof(float), vocab, f);
    fwrite(attn, sizeof(float), (size_t)nl * n * dim, f);
    fwrite(ffn, sizeof(float), (size_t)nl * n * dim, f);
    fwrite(stream, sizeof(float), (size_t)nl * n * dim * hc, f);
    fwrite(idxs, sizeof(int32_t), (size_t)nl * n * topk, f);
    fwrite(comp, sizeof(float), (size_t)2 * nl * n * hd, f);
    unsigned long long stats[4]; ds4_test_dsv41_scan_stats(stats);
    fwrite(stats, sizeof(unsigned long long), 4, f);
    fclose(f);
    return 0;
}
"""


def build_driver(tmp):
    src = os.path.join(tmp, "driver.c")
    exe = os.path.join(tmp, "driver")
    with open(src, "w", encoding="utf-8") as fp:
        fp.write(DRIVER)
    objs = ["ds4_cpu_test_hooks.o", "ds4_image.o", "ds4_distributed.o",
            "ds4_tp.o", "ds4_ssd.o", "ds4_layer_pack.o"]
    cmd = ["cc", "-O2", "-o", exe, src] + [os.path.join(ROOT, o) for o in objs] + ["-lm", "-pthread"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return exe


def compare(name, got, want, tol, report):
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = max(np.abs(want).max(), 1e-9)
    ok = d.max() / scale <= tol
    report.append((name, ok, d.max(), d.max() / scale))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=os.path.join(HERE, "mini", "mini.gguf"))
    parser.add_argument("--oracle", default=os.path.join(HERE, "mini", "oracle_prefill.npz"))
    parser.add_argument("--config", default=os.path.join(HERE, "mini", "mini_config.json"))
    parser.add_argument("--layers", default="", help="comma-separated layers to score")
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument("--dump-idxs", default="")
    parser.add_argument("--engram", default="", help="engram sidecar GGUF")
    args = parser.parse_args()

    config = json.load(open(args.config, encoding="utf-8"))
    oracle = np.load(args.oracle)
    ids = oracle["input_ids"][0].tolist()
    dim, hc, vocab = config["dim"], config["hc_mult"], config["vocab_size"]
    n = len(ids)

    layers = ([int(x) for x in args.layers.split(",") if x != ""]
              if args.layers else
              [i for i, r in enumerate(config["compress_ratios"]) if r == 0])

    with tempfile.TemporaryDirectory() as tmp:
        exe = build_driver(tmp)
        dump = os.path.join(tmp, "dump.bin")
        topk = config["index_topk"]
        subprocess.run([exe, args.gguf, dump, str(dim), str(hc), str(vocab), str(topk),
                        str(config["head_dim"]), args.engram]
                       + [str(i) for i in ids], check=True, cwd=ROOT)
        with open(dump, "rb") as fp:
            nl = np.frombuffer(fp.read(4), np.int32)[0]
            logits = np.frombuffer(fp.read(4 * vocab), np.float32)
            attn = np.frombuffer(fp.read(4 * nl * n * dim), np.float32).reshape(nl, n, dim)
            ffn = np.frombuffer(fp.read(4 * nl * n * dim), np.float32).reshape(nl, n, dim)
            stream = np.frombuffer(fp.read(4 * nl * n * dim * hc), np.float32)
            stream = stream.reshape(nl, n, hc, dim)
            idxs = np.frombuffer(fp.read(4 * nl * n * topk), np.int32).reshape(nl, n, topk)
            hd = config["head_dim"]
            both = np.frombuffer(fp.read(4 * 2 * nl * n * hd), np.float32).reshape(2 * nl, n, hd)
            comp, scores = both[:nl], both[nl:]
            stats = np.frombuffer(fp.read(32), np.uint64)

    print(f"  index scans: {stats[0]} candidate-restricted, {stats[1]} full; "
          f"{stats[2]} positions scored vs {stats[3]} for a full scan "
          f"({100.0 * stats[2] / max(int(stats[3]), 1):.1f}%)")
    report_pre = []
    if args.dump_idxs:
        for il in [int(x) for x in args.dump_idxs.split(",")]:
            key = f"layer{il}.indexer_idxs"
            print(f"  {key}: ds4={idxs[il][23].tolist()}")
            if key in oracle:
                print(f"  {' ' * len(key)}  ref={oracle[key][0][23].astype(int).tolist()}")

    for il, r in enumerate(config["compress_ratios"]):
        key = f"layer{il}.compressor"
        if key in oracle:
            want = oracle[key][0]
            compare(key, comp[il][: want.shape[0]], want, args.tol, report_pre)

    report, fails = [], 0
    print(f"mini prefill: {n} tokens, {nl} layers, scoring layers {layers}")
    for il in layers:
        for kind, got in (("attn", attn[il]), ("ffn", ffn[il]), ("out", stream[il])):
            key = f"layer{il}.{kind}"
            if key not in oracle:
                continue
            want = oracle[key][0]
            if not compare(key, got, want, args.tol, report):
                fails += 1
    if len(layers) == nl and "logits" in oracle:
        if not compare("logits", logits, oracle["logits"][0], args.tol, report):
            fails += 1

    for name, ok, absd, reld in report_pre:
        print(f"  {name:16s} {'ok  ' if ok else 'FAIL'}  max|d|={absd:.3e}  rel={reld:.3e}")
    for name, ok, absd, reld in report:
        print(f"  {name:16s} {'ok  ' if ok else 'FAIL'}  max|d|={absd:.3e}  rel={reld:.3e}")
    print("v4.1 prefill reference: " + (f"{fails} failure(s)" if fails else "all scored stages match"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
