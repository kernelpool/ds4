# MiMo-V2.6 Flash

[Back to README](../README.md)

MiMo-V2.6 Flash uses the llama.cpp `mimo2` GGUF architecture and a
dedicated Metal graph: 48 layers alternating global attention with
128-token sliding-window attention (per-head sinks on the windowed layers),
one dense layer followed by 256-expert MoE layers with a sigmoid router,
three MTP blocks, and the shared Qwen2 tokenizer with ChatML turns.
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
`mimo-v2.6-flash-reasoner` aliases (`mimo-v2.6-pro...` for Pro); tool calls use the model's
`<tool_call><function=...><parameter=...>` format, and tool results are
`tool` turns. The chat template puts tool schemas in a leading system turn
of their own and opens the assistant turn with `<think>`, which the model
otherwise emits itself.

## Pro

MiMo-V2.6 Pro RL (70 layers, 6144 wide, 384 experts, 128 query heads) runs
on the same graph. Its MXFP4 GGUF does not fit one Mac, so it runs over
[tensor parallelism between two Macs](DISTRIBUTED.md#tensor-parallelism-between-two-macs):
each rank keeps half of the attention heads (their QKV rows and output
columns are copied into rank-local buffers at startup), the KV cache of
those heads, half of the routed experts and half of the vocabulary head.
Decode gates release inside the command buffer, which is committed every two
layers. `--mtp` on both ranks drafts with the MTP blocks, which each rank
runs whole; the verify rows cross the RDMA verify window, and the vocabulary
head stays whole on both ranks because the drafts follow its argmax. Exact
sampling decodes plainly under tensor parallelism, and vision, DFlash and
session batching stay off.

## Speculative decoding

Two drafters are available; use one of them.

`--mtp` runs the MTP blocks embedded in the main GGUF. Each block is a
sliding-window attention layer with a dense FFN over the target's last
hidden state before its final norm and the embedding of the token it
follows; block k drafts k + 1 tokens past the same target row (the blocks
are not chained, as trained). A cycle drafts two or three tokens and
verifies them in one target pass: three while the first two drafts land
often enough to pay for the extra verify row (typically code and reasoning),
two otherwise. `DS4_MIMO_MTP_DEPTH=1..3` fixes the drafts per cycle.

`--mtp-model MiMo-V2.6-Flash-DFlash-Q8_0.gguf` runs the DFlash drafter
instead: five Qwen3-style layers over features taken from five target
layers draft a block of seven tokens per cycle, of which two are verified
by default (`DS4_MIMO_MTP_DEPTH=1..7`). The sidecar uses the llama.cpp
`dflash` layout plus the mask embedding DS4 needs. The built-in MTP
drafter is the faster of the two here.

Both drafters verify against the target's logits, and the verify rows keep
the one-token arithmetic: the Q8 projections run up to four rows through the
single-row kernel's reduction, reading the weights once, and attention
splits each row's keys as that row's own decode would. Temperature-zero
output therefore equals plain decoding bit for bit.
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
Xiaomi platform with thinking disabled, using the shared scorer (Pro:
`mimo-v2.6-pro-20260923`, with the tensor-parallel options added):

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

SSD streaming, DSpark and CUDA are not supported yet; see [Pro](#pro) for
what tensor parallelism leaves out.
The server's disk KV cache saves and restores MiMo sessions (the MTP blocks
replay their rows after a restore; the DFlash context of the last window
positions is saved with the session). `ds4-server --batched-session N` decodes the slots
together on shared transients, reading the experts once per batch; the
drafters stay off while batching, as for the other families. Each slot's
logits equal its single-session decode bit for bit, because the batched Q8
projections keep the single-row kernel; `DS4_MIMO_BATCH_MM=1` uses the
multi-row kernels instead for more throughput at the cost of that identity.
Likewise, prefill attention gives the per-row kernel's output bit for bit;
`DS4_MIMO_ATTN=tiled` selects a tiled kernel that is faster on long prompts
but not bit-identical to it.
Prefill is not chunk-invariant at the last bit, so two identical prompts
whose prefills were split differently (the server interleaves 128-token
quanta while other slots generate) can part at a near tie;
`--mixed-prefill-quantum` sets that split. There is no Q2 release yet.

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
run on any model. `make test-mimo-verify-exact` (`DS4_TEST_MODEL=<gguf>`)
checks that every verify block's logits equal plain decoding bit for bit at
each draft depth, with and without rejected drafts. `make test-mimo-vision` (`DS4_MIMO_SNAPSHOT`,
`DS4_MIMO_MMPROJ`, `DS4_MIMO_IMAGE`) compares the vision tower with the
Hugging Face tower on the same GGUF weights, feeding DS4's own patches to
both so resizing differences do not hide tower differences, and reports
GGUF-versus-original quality separately.
