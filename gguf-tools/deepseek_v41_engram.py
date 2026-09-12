#!/usr/bin/env python3
"""Create or validate the standalone DeepSeek V4.1 Flash engram sidecar GGUF.

The engram tables are 203 GB of the checkpoint's 510 GB and are touched only 48 rows
at a time, so they live in their own file that DS4 maps without making resident.
Rows are copied value for value: the source's per-32 E8M0 exponents are interleaved
with the E4M3 values into DS4's f8_e4m3 blocks rather than requantized.

The sidecar also carries the n-gram hash layout, which the checkpoint does not ship:
the compressed token map (a tokenizer normalizer pipeline), the per-layer prime bucket
moduli, their offsets, and the numpy PCG64 multipliers. All three are reproduced here
and checked against config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import sys

import numpy as np

from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_VERSION,
    QTYPE_BF16,
    QTYPE_F8_E4M3,
    QTYPE_I32,
    QTYPE_I64,
    QTYPE_NAMES,
    SourceDB,
    TensorPlan,
    align,
    fail,
    kv_string,
    kv_u32,
    kv_u32_array,
    kv_u64,
    qtype_nbytes,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
    tensor_header,
)


SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ARCHITECTURE = "deepseek4-engram"

SOURCE_TENSORS = 96085
SOURCE_SHARDS = 48

# rows per streamed chunk of an embed table: 16 MB in, 16.5 MB out
EMBED_CHUNK_ROWS = 1 << 16


def validate_deepseek41_index(weight_map):
    if len(weight_map) != SOURCE_TENSORS:
        fail(f"expected {SOURCE_TENSORS} source tensors, found {len(weight_map)}")
    expected = {f"model-{i:05d}-of-{SOURCE_SHARDS:05d}.safetensors"
                for i in range(1, SOURCE_SHARDS + 1)}
    if set(weight_map.values()) != expected:
        fail("source shard inventory differs from the pinned checkpoint")


def validate_engram_sources(tensors):
    """Only the engram tensors are checked; the rest of the checkpoint is another file's job."""
    names = {name for name in tensors if ".engram." in name}
    if len(names) != 12:
        fail(f"expected 12 engram source tensors, found {len(names)}")


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
        "engram_layer_ids": [1, 14],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_vocab_size": 16000000,
        "engram_compressed_vocab_size": 99092,
        "engram_pad_token_id": 2,
        "engram_num_embeddings": [384006168, 384016682],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"unexpected config {key}: {config.get(key)!r}")
    return config


def is_prime(n):
    """Deterministic Miller-Rabin; these bases are exact for every n below 3.3e24."""
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n < 2:
        return False
    for base in bases:
        if n % base == 0:
            return n == base
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for base in bases:
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_primes(config):
    """One prime bucket modulus per (layer, n-gram size, head).

    engram.py restarts the scan at engram_vocab_size - 1 for every n-gram size but shares
    one `seen` set, so the moduli are simply the smallest distinct primes above that bound
    handed out in nesting order. Their per-layer sum must be the table's row count.
    """
    bound = config["engram_vocab_size"] - 1
    n_ngram = config["engram_max_ngram_size"] - 1
    seen, primes = set(), []
    for _ in config["engram_layer_ids"]:
        per_layer = []
        for _ in range(n_ngram):
            sizes, candidate = [], bound
            for _ in range(config["engram_n_heads"]):
                candidate += 1
                while not is_prime(candidate) or candidate in seen:
                    candidate += 1
                seen.add(candidate)
                sizes.append(candidate)
            per_layer.append(sizes)
        primes.append(per_layer)
    rows = [sum(p for per_ngram in layer for p in per_ngram) for layer in primes]
    if rows != config["engram_num_embeddings"]:
        fail(f"derived table rows {rows} != config engram_num_embeddings "
             f"{config['engram_num_embeddings']}")
    return primes


def engram_multipliers(config):
    """One odd multiplier per (layer, lookback), from numpy's PCG64 seeded per layer."""
    bound = max(1, (int(np.iinfo(np.int64).max) // config["engram_compressed_vocab_size"]) // 2)
    rows = []
    for layer_id in config["engram_layer_ids"]:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound,
                                    size=(config["engram_max_ngram_size"],), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


def build_token_map(hf_dir, config):
    """Collapse token ids that normalize alike, so " The", "the" and "THE" hash the same."""
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    if len(tokenizer) != config["vocab_size"]:
        fail(f"tokenizer has {len(tokenizer)} ids, config says {config['vocab_size']}")
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, np.zeros(len(tokenizer), dtype=np.int32)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    if len(key_to_new) != config["engram_compressed_vocab_size"]:
        fail(f"compressed vocab is {len(key_to_new)}, config says "
             f"{config['engram_compressed_vocab_size']}")
    return lookup


def hash_layout(hf_dir, config):
    primes = engram_primes(config)
    flat = np.array([[p for per_ngram in layer for p in per_ngram] for layer in primes],
                    dtype=np.int64)
    offsets = np.cumsum(np.pad(flat[:, :-1], ((0, 0), (1, 0))), axis=1)
    return {
        "engram.token_map": build_token_map(hf_dir, config),
        "engram.hash_multipliers": engram_multipliers(config),
        "engram.hash_primes": flat,
        "engram.hash_offsets": offsets,
    }


E4M3_NAN = 0x7F


def e4m3_lut():
    """E4M3FN: subnormals are m*2^-9, normals (1+m/8)*2^(e-7), no infinities."""
    index = np.arange(256, dtype=np.int32)
    exponent = (index >> 3) & 0xF
    mantissa = (index & 0x7).astype(np.float64)
    magnitude = np.where(exponent == 0,
                         mantissa * 2.0 ** -9,
                         (1.0 + mantissa / 8.0) * 2.0 ** (exponent.astype(np.float64) - 7))
    return np.where(index & 0x80, -magnitude, magnitude).astype(np.float32)


def reject_reserved_nan(values, name):
    if np.any((values & 0x7F) == E4M3_NAN):
        fail(f"{name}: source holds the E4M3FN pattern reserved for NaN")


def embed_chunks(db, layer, rows):
    """Interleave [rows, 256] E4M3 values with [rows, 8] E8M0 exponents into f8_e4m3 blocks."""
    weight = f"layers.{layer}.engram.embed.weight"
    scale = f"layers.{layer}.engram.embed.scale"
    for start in range(0, rows, EMBED_CHUNK_ROWS):
        count = min(EMBED_CHUNK_ROWS, rows - start)
        values = np.frombuffer(
            b"".join(db.iter_read(weight, start * 256, count * 256)), np.uint8
        ).reshape(count, 8, 32)
        reject_reserved_nan(values, weight)
        exponents = np.frombuffer(
            b"".join(db.iter_read(scale, start * 8, count * 8)), np.uint8
        ).reshape(count, 8, 1)
        yield np.concatenate((exponents, values), axis=2).tobytes()


def wkv_bf16(db, layer, shape):
    """Dequantize the 32x32-blocked fp8 wkv to bf16, which holds it exactly (4 significant bits)."""
    weight = f"layers.{layer}.engram.wkv.weight"
    scale = f"layers.{layer}.engram.wkv.scale"
    rows, cols = shape
    values = np.frombuffer(db.read(weight), np.uint8).reshape(rows, cols)
    reject_reserved_nan(values, weight)
    exponents = np.frombuffer(db.read(scale), np.uint8).reshape(rows // 32, cols // 32)
    scales = np.ldexp(np.float32(1.0), exponents.astype(np.int32) - 127)
    dequantized = e4m3_lut()[values] * np.repeat(np.repeat(scales, 32, axis=0), 32, axis=1)
    bits = dequantized.view(np.uint32)
    if np.any(bits & 0xFFFF):
        fail(f"{weight}: dequantized values do not round-trip through bf16")
    return (bits >> 16).astype(np.uint16).tobytes()


def build_plan(db, config, derived):
    plan, payloads = [], {}
    for name, array in derived.items():
        qtype = QTYPE_I32 if array.dtype == np.int32 else QTYPE_I64
        payloads[name] = array.tobytes()
        plan.append(TensorPlan(name=name, shape=tuple(reversed(array.shape)), qtype=qtype,
                               role="hash_layout"))
    for index, layer in enumerate(config["engram_layer_ids"]):
        rows = config["engram_num_embeddings"][index]
        for suffix in ("q_weight", "k_weight"):
            name = f"layers.{layer}.engram.{suffix}"
            info = db.info(name)
            if info["dtype"] != "BF16" or info["shape"] != [4, config["hidden_size"]]:
                fail(f"{name}: unexpected source layout {info['dtype']} {info['shape']}")
            plan.append(TensorPlan(name=name, shape=tuple(reversed(info["shape"])),
                                   qtype=QTYPE_BF16, role="engram_gate", source=name,
                                   raw_copy=True))
        name = f"layers.{layer}.engram.wkv.weight"
        info = db.info(name)
        expect = [config["engram_head_dim"] * 100, config["hidden_size"] * 6 // 5]
        if info["dtype"] != "F8_E4M3" or info["shape"] != expect:
            fail(f"{name}: unexpected source layout {info['dtype']} {info['shape']}")
        plan.append(TensorPlan(name=name, shape=tuple(reversed(info["shape"])), qtype=QTYPE_BF16,
                               role="engram_wkv", source=name, transform="wkv"))
        name = f"layers.{layer}.engram.embed.weight"
        info = db.info(name)
        if info["dtype"] != "F8_E4M3" or info["shape"] != [rows, config["engram_head_dim"]]:
            fail(f"{name}: unexpected source layout {info['dtype']} {info['shape']}")
        scale = db.info(f"layers.{layer}.engram.embed.scale")
        if scale["dtype"] != "F8_E8M0" or scale["shape"] != [rows, config["engram_head_dim"] // 32]:
            fail(f"{name}: unexpected source scale layout {scale['dtype']} {scale['shape']}")
        plan.append(TensorPlan(name=name, shape=(config["engram_head_dim"], rows),
                               qtype=QTYPE_F8_E4M3, role="engram_embed", source=name,
                               transform="embed"))
    offset = 0
    for item in plan:
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        item.offset = offset
        offset += align(item.nbytes)
    return plan, payloads


def item_chunks(db, config, item, payloads):
    if item.transform == "embed":
        layer = int(item.name.split(".")[1])
        return embed_chunks(db, layer, item.shape[1])
    if item.transform == "wkv":
        layer = int(item.name.split(".")[1])
        return iter((wkv_bf16(db, layer, (item.shape[1], item.shape[0])),))
    if item.source is not None:
        return db.iter_read(item.source)
    return iter((payloads[item.name],))


def sidecar_metadata(config, source_revision, derived):
    prefix = ARCHITECTURE
    token_map = derived["engram.token_map"]
    return [
        kv_string("general.architecture", ARCHITECTURE),
        kv_string("general.name", "DeepSeek V4.1 Flash Engram"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", source_revision),
        kv_string(f"{prefix}.checkpoint_variant", "flash-v41"),
        kv_u32_array(f"{prefix}.layers", config["engram_layer_ids"]),
        kv_u32_array(f"{prefix}.row_counts", config["engram_num_embeddings"]),
        kv_u32(f"{prefix}.max_ngram_size", config["engram_max_ngram_size"]),
        kv_u32(f"{prefix}.ngram_count", config["engram_max_ngram_size"] - 1),
        kv_u32(f"{prefix}.head_count", config["engram_n_heads"]),
        kv_u32(f"{prefix}.head_dim", config["engram_head_dim"]),
        kv_u32(f"{prefix}.vocab_size", config["engram_vocab_size"]),
        kv_u32(f"{prefix}.compressed_vocab_size", config["engram_compressed_vocab_size"]),
        kv_u32(f"{prefix}.pad_token_id", config["engram_pad_token_id"]),
        kv_u32(f"{prefix}.compressed_pad_id", int(token_map[config["engram_pad_token_id"]])),
        kv_u32(f"{prefix}.language.block_count",
               config["num_hidden_layers"] + config["num_nextn_predict_layers"]),
        kv_u32(f"{prefix}.language.embedding_length", config["hidden_size"]),
        kv_u64(f"{prefix}.language.vocab_size", config["vocab_size"]),
    ]


def layout(plan, metadata):
    header_bytes = 4 + 4 + 8 + 8
    header_bytes += sum(len(record) for record in metadata)
    header_bytes += sum(len(tensor_header(item)) for item in plan)
    data_offset = align(header_bytes)
    data_bytes = sum(align(item.nbytes) for item in plan)
    return data_offset, data_bytes


def print_summary(plan, metadata):
    data_offset, data_bytes = layout(plan, metadata)
    print(f"tensors: {len(plan)}")
    print(f"metadata_records: {len(metadata)}")
    print(f"metadata_bytes: {data_offset}")
    print(f"tensor_bytes: {sum(item.nbytes for item in plan)}")
    print(f"file_bytes: {data_offset + data_bytes}")
    roles = {}
    for item in plan:
        roles[item.role] = roles.get(item.role, 0) + item.nbytes
    for role, size in sorted(roles.items()):
        print(f"role_bytes: {role} {size}")
    for item in plan:
        print(f"tensor: {item.name} {QTYPE_NAMES[item.qtype]} {list(item.shape)} {item.nbytes}")
    return data_offset, data_bytes


def create_gguf(path, plan, metadata, db, config, payloads, overwrite):
    data_offset, data_bytes = print_summary(plan, metadata)
    required = data_offset + data_bytes + 4 * (1 << 30)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free < required:
        fail(f"insufficient free space: need output plus reserve {required}, have {free}")
    if os.path.exists(path) and not overwrite:
        fail(f"output exists: {path}; use --overwrite")
    partial = path + ".partial"
    if os.path.exists(partial):
        if not overwrite:
            fail(f"partial output exists: {partial}; use --overwrite")
        os.unlink(partial)

    with open(partial, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", GGUF_VERSION, len(plan), len(metadata)))
        for record in metadata:
            fp.write(record)
        for item in plan:
            fp.write(tensor_header(item))
        if fp.tell() > data_offset:
            fail("GGUF header exceeds its planned data offset")
        fp.write(bytes(data_offset - fp.tell()))
        for index, item in enumerate(plan, 1):
            if fp.tell() != data_offset + item.offset:
                fail(f"{item.name}: output offset mismatch")
            written = 0
            for chunk in item_chunks(db, config, item, payloads):
                fp.write(chunk)
                written += len(chunk)
                if item.nbytes > (1 << 30) and written % (1 << 33) < len(chunk):
                    print(f"{item.name}: {written}/{item.nbytes} bytes",
                          file=sys.stderr, flush=True)
            if written != item.nbytes:
                fail(f"{item.name}: wrote {written} bytes, expected {item.nbytes}")
            fp.write(bytes(align(item.nbytes) - item.nbytes))
            print(f"wrote tensors: {index}/{len(plan)} ({item.name})", file=sys.stderr, flush=True)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(partial, path)


def read_metadata_value(fp, value_type):
    if value_type == GGUF_STRING:
        return read_gguf_string(fp, "GGUF metadata string")
    if value_type == GGUF_UINT32:
        return read_u32(fp, "GGUF metadata uint32")
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
    skip_gguf_value(fp, value_type)
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gguf(path, plan, expected_metadata, db, config, payloads,
                  verify_payload, expected_sha256):
    expected_values = {}
    for record in expected_metadata:
        reader = io.BytesIO(record)
        key = read_gguf_string(reader, "expected metadata key")
        value_type = read_u32(reader, "expected metadata type")
        expected_values[key] = read_metadata_value(reader, value_type)

    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version != GGUF_VERSION:
            fail(f"expected GGUF v{GGUF_VERSION}, got v{version}")
        tensor_count = read_u64(fp, "GGUF tensor count")
        metadata_count = read_u64(fp, "GGUF metadata count")
        if tensor_count != len(plan):
            fail(f"tensor count {tensor_count} != expected {len(plan)}")
        if metadata_count != len(expected_metadata):
            fail(f"metadata count {metadata_count} != expected {len(expected_metadata)}")

        actual_values = {}
        for _ in range(metadata_count):
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            actual_values[key] = read_metadata_value(fp, value_type)
        if actual_values != expected_values:
            fail("GGUF metadata differs from the pinned conversion metadata")

        for index, item in enumerate(plan):
            name = read_gguf_string(fp, f"tensor {index} name")
            rank = read_u32(fp, f"tensor {index} rank")
            shape = tuple(read_u64(fp, f"tensor {index} dimension") for _ in range(rank))
            qtype = read_u32(fp, f"tensor {index} type")
            offset = read_u64(fp, f"tensor {index} offset")
            if (name, shape, qtype, offset) != (item.name, item.shape, item.qtype, item.offset):
                fail(f"tensor {index} header differs from the conversion plan: {name}")

        data_offset, data_bytes = layout(plan, expected_metadata)
        actual_size = os.fstat(fp.fileno()).st_size
        if actual_size != data_offset + data_bytes:
            fail(f"file size {actual_size} != expected {data_offset + data_bytes}")
        verified = 0
        if verify_payload:
            for index, item in enumerate(plan, 1):
                fp.seek(data_offset + item.offset)
                for source in item_chunks(db, config, item, payloads):
                    if read_exact(fp, len(source), item.name) != source:
                        fail(f"{item.name}: payload differs from the official source")
                    verified += len(source)
                padding = read_exact(fp, align(item.nbytes) - item.nbytes, f"{item.name} padding")
                if any(padding):
                    fail(f"{item.name}: nonzero alignment padding")
                print(f"verified payloads: {index}/{len(plan)} ({item.name})",
                      file=sys.stderr, flush=True)

    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        fail(f"SHA-256 {digest} != expected {expected_sha256.lower()}")
    print(f"validated {path}: {len(plan)} tensors, {actual_size} bytes")
    print(f"SHA-256: {digest}")
    if verify_payload:
        print(f"source payloads matched byte for byte: {verified} bytes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="official DeepSeek-V4.1-Flash snapshot")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out", help="create this engram GGUF")
    output.add_argument("--validate", metavar="GGUF", help="validate an existing engram GGUF")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--verify-payload", action="store_true")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.source_revision != SOURCE_REVISION:
        parser.error(f"unsupported source revision: {args.source_revision}")
    if not args.dry_run and not args.out and not args.validate:
        parser.error("one of --out, --validate, or --dry-run is required")
    if args.verify_payload and not args.validate:
        parser.error("--verify-payload requires --validate")
    if args.expected_sha256 and not args.validate:
        parser.error("--expected-sha256 requires --validate")
    return args


def main():
    args = parse_args()
    config = load_config(args.hf)
    db = SourceDB(args.hf, validate_deepseek41_index, validate_engram_sources)
    try:
        derived = hash_layout(args.hf, config)
        plan, payloads = build_plan(db, config, derived)
        metadata = sidecar_metadata(config, args.source_revision, derived)
        if args.dry_run:
            print_summary(plan, metadata)
            print("token_map_sha256: " +
                  hashlib.sha256(payloads["engram.token_map"]).hexdigest())
        elif args.out:
            create_gguf(args.out, plan, metadata, db, config, payloads, args.overwrite)
            print(f"deepseek4-engram: wrote {args.out}", file=sys.stderr)
        else:
            validate_gguf(args.validate, plan, metadata, db, config, payloads,
                          args.verify_payload, args.expected_sha256)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"deepseek4-engram: error: {error}", file=sys.stderr)
        raise SystemExit(1)
