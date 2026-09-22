# MiMo-V2.6 Flash

[Back to README](../README.md)

MiMo-V2.6 Flash uses the llama.cpp `mimo2` GGUF architecture and a
dedicated Metal graph: 48 layers alternating global attention with
128-token sliding-window attention (per-head sinks on the windowed layers),
one dense layer followed by 256-expert MoE layers with a sigmoid router,
three chained MTP blocks, and the shared Qwen2 tokenizer with ChatML turns.
The port is Metal only.

## Run

```sh
make
./ds4 --model MiMo-V2.6-Flash-MXFP4.gguf
```

The MXFP4 release keeps the checkpoint's own MXFP4 experts (the attention
and dense weights are Q8_0); it is the lossless option. `--nothink` disables
thinking. The server
exposes `mimo-v2.6-flash`, `mimo-v2.6-flash-chat` and
`mimo-v2.6-flash-reasoner` aliases; tool calls use the model's
`<tool_call><function=...><parameter=...>` format, and tool results are
`tool` turns. The chat template puts tool schemas in a leading system turn
of their own and opens the assistant turn with `<think>`, which the model
otherwise emits itself.

## Speculative decoding

Two drafters are available; use one of them.

`--mtp` runs the MTP blocks embedded in the main GGUF. Each block is a
sliding-window attention layer with a dense FFN chained on the previous
block's residual. A cycle drafts one token and verifies it in one target
pass, which decodes fastest on Apple Silicon; `DS4_MIMO_MTP_DEPTH=2` or `3`
chains the deeper blocks for more drafts per cycle.

`--mtp-model MiMo-V2.6-Flash-DFlash-Q8_0.gguf` runs the DFlash drafter
instead: five Qwen3-style layers over features taken from five target
layers draft a block of seven tokens per cycle, of which two are verified
by default (`DS4_MIMO_MTP_DEPTH=1..7`). The sidecar uses the llama.cpp
`dflash` layout plus the mask embedding DS4 needs. The built-in MTP
drafter is the faster of the two here.

Both drafters verify against the target's logits, so temperature-zero
output follows plain decoding; the batched verifier's reduction order can
differ from one-token decode, so a near tie in a long greedy continuation
may resolve differently (see [speculative decoding](SPECULATIVE_DECODING.md)).
For non-zero temperature, `--mtp-exact-sampling` preserves the target
distribution. `--mtp-timing` prints the verify cycles and accepted drafts
at exit.

## Vision

Images go through the model's own vision tower, loaded from a separate
encoder GGUF: `--vision MiMo-V2.6-Flash-Vision-F32.gguf`. The F32 encoder
reproduces the checkpoint's BF16 tower; a Q8_0 encoder is smaller but
measurably less exact. The CLI accepts `/read image.png`; the server accepts
`image_url` parts. Each image is resized to multiples of 32 pixels within 64
to 1024 tokens (`DS4_MIMO_IMAGE_MAX_TOKENS` raises the cap), normalised with
the checkpoint's mean and standard deviation, and encoded on the GPU: 28
blocks alternating full attention with a 64-token banded window and
per-head sinks, the column-major reordering of the window blocks, and the
merger that maps 2x2 patch groups to text embeddings. The image tokens take
plain sequential positions in the text model. `make test-mimo-vision`
compares the tower with the Hugging Face implementation on the same GGUF
weights (it feeds DS4's own patches to both, so resizing differences do not
hide tower differences) and reports GGUF-versus-original quality
separately.

## Quality

The release GGUFs are scored on official continuations collected from the
Xiaomi platform with thinking disabled, using the shared scorer:

```sh
gguf-tools/quality-testing/score_official MiMo-V2.6-Flash-MXFP4.gguf \
  gguf-tools/quality-testing/mimo-v2.6-flash-20260922/manifest.tsv /tmp/mimo.tsv 4096
```

The fixture directory's README records the collection settings and its
`results/` directory the Metal reference scores; the platform returns no
logprobs, so only NLL, first-token match and greedy prefix length apply.
Every speed change must reproduce the reference TSV (`validate_scores.py
--strict-identical`) and the greedy texts of `--mtp` and `--mtp-model` on
the reference prompts must stay identical to plain decoding.

## Limits

Tensor parallelism, SSD streaming, DSpark, CUDA, native session batching
and disk KV checkpoints are not supported yet; the server decodes MiMo
slots in order. There is no Q2 release yet.

## Conversion

`gguf-tools/mimo26_quantize.py` reads the released checkpoint (fp8
attention, MXFP4 experts) and writes the main GGUF and the DFlash sidecar;
`gguf-tools/mimo26_vision.py` writes the vision encoder in llama.cpp's clip
`mimovl` layout; see [GGUF conversion](../gguf-tools/README.md#mimo-v26-flash).
The converter de-interleaves the tensor-parallel chunks of the fused QKV
projection and repacks the MXFP4 experts into ggml blocks without
requantizing them.

## Testing

`make test-mimo-mtp` (`DS4_TEST_MODEL=<gguf>`, optionally
`DS4_TEST_DFLASH=<sidecar>`) checks that speculative cycles, including
rewinds and both sampling modes, reproduce plain greedy decoding;
`DS4_MIMO_SPEC_DRAFTS=FILE` feeds drafts by position so the accept paths
run on any model. `make test-mimo-vision` (`DS4_MIMO_SNAPSHOT`,
`DS4_MIMO_MMPROJ`, `DS4_MIMO_IMAGE`) compares the vision tower with the
Hugging Face tower on the same GGUF weights, feeding DS4's own patches to
both so resizing differences do not hide tower differences, and reports
GGUF-versus-original quality separately.
