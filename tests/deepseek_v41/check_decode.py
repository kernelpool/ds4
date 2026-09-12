#!/usr/bin/env python3
"""Score the DS4 V4.1 CPU reference's decode steps against the mini-model oracle.

Prefills the same 24 tokens, then runs the oracle's own sampled continuation one token at a
time, comparing every layer's attn/ffn/stream and the logits at each step. Decode exercises
what prefill cannot: the sliding-window ring buffer, the partial compressor group carried
between steps, and the per-step indexer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int ds4_test_dsv41_decode(const char *, const int *, unsigned, const int *, unsigned,
                          float *, float *, float *, float *, int32_t *, float *,
                          const char *);
int main(int argc, char **argv) {
    const char *gguf = argv[1];
    const char *out = argv[2];
    const unsigned dim = (unsigned)atoi(argv[3]);
    const unsigned hc = (unsigned)atoi(argv[4]);
    const unsigned vocab = (unsigned)atoi(argv[5]);
    const unsigned np_ = (unsigned)atoi(argv[6]);
    const unsigned ns = (unsigned)atoi(argv[7]);
    const char *engram = argv[8][0] ? argv[8] : NULL;
    int *pre = malloc(np_ * sizeof(int));
    int *step = malloc(ns * sizeof(int));
    for (unsigned i = 0; i < np_; i++) pre[i] = atoi(argv[9 + i]);
    for (unsigned i = 0; i < ns; i++) step[i] = atoi(argv[9 + np_ + i]);
    float *logits = malloc((size_t)ns * vocab * sizeof(float));
    float *attn = malloc((size_t)ns * 64 * dim * sizeof(float));
    float *ffn = malloc((size_t)ns * 64 * dim * sizeof(float));
    float *stream = malloc((size_t)ns * 64 * dim * hc * sizeof(float));
    const unsigned tk = (unsigned)atoi(argv[9 + np_ + ns]);
    int32_t *idxs = malloc((size_t)ns * 64 * tk * sizeof(int32_t));
    int nl = ds4_test_dsv41_decode(gguf, pre, np_, step, ns, logits, attn, ffn, stream, idxs,
                                   NULL, engram);
    FILE *f = fopen(out, "wb");
    if (!f) return 3;
    fwrite(&nl, sizeof(int), 1, f);
    fwrite(logits, sizeof(float), (size_t)ns * vocab, f);
    fwrite(attn, sizeof(float), (size_t)ns * nl * dim, f);
    fwrite(ffn, sizeof(float), (size_t)ns * nl * dim, f);
    fwrite(stream, sizeof(float), (size_t)ns * nl * dim * hc, f);
    fwrite(idxs, sizeof(int32_t), (size_t)ns * nl * tk, f);
    fclose(f);
    return 0;
}
"""


def build_driver(tmp):
    src, exe = os.path.join(tmp, "d.c"), os.path.join(tmp, "d")
    with open(src, "w", encoding="utf-8") as fp:
        fp.write(DRIVER)
    objs = ["ds4_cpu_test_hooks.o", "ds4_image.o", "ds4_distributed.o",
            "ds4_tp.o", "ds4_ssd.o", "ds4_layer_pack.o"]
    subprocess.run(["cc", "-O2", "-o", exe, src] + [os.path.join(ROOT, o) for o in objs]
                   + ["-lm", "-pthread"], check=True, cwd=ROOT)
    return exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=os.path.join(HERE, "mini", "mini.gguf"))
    parser.add_argument("--oracle", default=os.path.join(HERE, "mini", "oracle_decode.npz"))
    parser.add_argument("--config", default=os.path.join(HERE, "mini", "mini_config.json"))
    parser.add_argument("--tol", type=float, default=2e-4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--engram", default="", help="engram sidecar GGUF")
    args = parser.parse_args()

    config = json.load(open(args.config, encoding="utf-8"))
    oracle = np.load(args.oracle)
    pre = oracle["prefill_ids"][0].tolist()
    n_steps = sum(1 for k in oracle.files if k.endswith(".id"))
    steps = [int(oracle[f"step{i}.id"][0, 0]) for i in range(n_steps)]
    dim, hc, vocab = config["dim"], config["hc_mult"], config["vocab_size"]

    with tempfile.TemporaryDirectory() as tmp:
        exe = build_driver(tmp)
        dump = os.path.join(tmp, "d.bin")
        subprocess.run([exe, args.gguf, dump, str(dim), str(hc), str(vocab),
                        str(len(pre)), str(n_steps), args.engram]
                       + [str(i) for i in pre] + [str(i) for i in steps]
                       + [str(config["index_topk"])], check=True, cwd=ROOT)
        with open(dump, "rb") as fp:
            nl = int(np.frombuffer(fp.read(4), np.int32)[0])
            logits = np.frombuffer(fp.read(4 * n_steps * vocab), np.float32).reshape(n_steps, vocab)
            attn = np.frombuffer(fp.read(4 * n_steps * nl * dim), np.float32).reshape(n_steps, nl, dim)
            ffn = np.frombuffer(fp.read(4 * n_steps * nl * dim), np.float32).reshape(n_steps, nl, dim)
            stream = np.frombuffer(fp.read(4 * n_steps * nl * dim * hc), np.float32)
            stream = stream.reshape(n_steps, nl, hc, dim)
            tk = config["index_topk"]
            idxs = np.frombuffer(fp.read(4 * n_steps * nl * tk), np.int32).reshape(n_steps, nl, tk)

    fails = 0
    print(f"mini decode: {len(pre)} prefill tokens, {n_steps} steps, {nl} layers")
    for s in range(n_steps):
        worst, where = 0.0, ""
        for il in range(nl):
            for kind, got in (("attn", attn[s][il]), ("ffn", ffn[s][il]), ("out", stream[s][il])):
                key = f"step{s}.layer{il}.{kind}"
                if key not in oracle:
                    continue
                want = oracle[key][0, 0]
                rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-9)
                if rel > worst:
                    worst, where = rel, key
        lg = oracle[f"step{s}.logits"][0]
        lrel = np.abs(logits[s] - lg).max() / max(np.abs(lg).max(), 1e-9)
        top_ok = int(np.argmax(logits[s])) == int(np.argmax(lg))
        ok = worst <= args.tol and lrel <= args.tol and top_ok
        if not ok:
            fails += 1
        print(f"  step{s} (pos {len(pre) + s})  {'ok  ' if ok else 'FAIL'}  "
              f"layers rel<={worst:.3e} ({where})  logits rel={lrel:.3e}  top1={'=' if top_ok else 'X'}")
        if args.verbose:
            for il in range(nl):
                row = []
                for kind, got in (("attn", attn[s][il]), ("ffn", ffn[s][il]), ("out", stream[s][il])):
                    key = f"step{s}.layer{il}.{kind}"
                    if key in oracle:
                        want = oracle[key][0, 0]
                        row.append(f"{kind}={np.abs(got - want).max() / max(np.abs(want).max(), 1e-9):.2e}")
                key = f"step{s}.layer{il}.indexer_idxs"
                if key in oracle:
                    row.append(f"idx_ds4={idxs[s][il].tolist()}")
                    row.append(f"idx_ref={oracle[key][0,0].astype(int).tolist()}")
                print(f"      layer{il}: " + "  ".join(row))
    print("v4.1 decode reference: " + (f"{fails} failing step(s)" if fails else "all steps match"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
