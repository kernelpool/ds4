# Models and Vision

[README](../README.md) | [Getting started](../README.md#start-here)

DwarfStar is not a general GGUF runner. Use the download targets below: other
GGUFs may have unsupported tensor layouts, metadata, or quantization mixes.
Run `./download_model.sh --help` for filenames and all available targets.

Main-model downloads update `ds4flash.gguf`. Encoders, draft models, packaged
FP8 weights, and split PRO pieces do not. Pass `-m FILE` to avoid depending on
which model was downloaded last.

Some downloads require the Hugging Face CLI; the script prints installation
instructions when needed. Authentication is optional for public weights;
your cached Hugging Face token or `HF_TOKEN` is used when present.

## DeepSeek V4

| Target | Use |
| --- | --- |
| `ds4f-q2` | Flash 0731, about 81 GiB; starting point for 96/128 GB systems |
| `ds4f-q2-q4` | Mostly Q2, with the last six routed-expert layers at Q4; needs more memory |
| `ds4f-q4` | Flash 0731 Q4; larger-memory or distributed systems |
| `ds4f-mxfp4` | Native MXFP4 routed experts; larger-memory or distributed systems |
| `pro-q2-imatrix` | PRO 0813; 512 GB resident target, or SSD streaming |
| `pro-q4-split` | Both PRO Q4 pieces for pipeline execution |

```sh
./download_model.sh ds4f-q2
./ds4
```

The Flash Q2 recipe spends most of its compression on routed experts:
IQ2_XXS gate/up and Q2_K down. Other components use higher precision, including
Q8 projections, shared experts and output, plus F16/F32 tensors. They are not
all untouched source weights. The imatrix guides the routed quantization.

The MXFP4 recipe preserves DeepSeek's released MXFP4 routed experts without
requantizing them. Metal and CUDA support it. Blackwell uses native FP4 matrix
instructions and FP4 activations for batched expert work; CUDA decode and
other CUDA architectures use Q8 activations. ROCm also has a resident MXFP4
path, including pipeline execution for models too large for one host.

To build weights rather than download them, see [GGUF tools](../gguf-tools/README.md).

## DeepSeek V4.1 Flash

DeepSeek V4.1 Flash runs on its own Metal path: 40 trunk layers with four-stream
hyper-connections, sliding-window attention over the last 128 tokens plus compressed
attention over latents the two-level indexer selects, the hashed n-gram engram in two
layers, 384-expert MoE with a shared expert, and three DSpark draft stages. There is no
download target; build the three GGUFs from the Hugging Face checkpoint with the tools in
`gguf-tools/`. The backbone converter takes its metadata and tensor list from a
header-only template, so that is generated first:

```sh
python3 gguf-tools/deepseek_v41_template.py --hf ../DeepSeek-V4.1-Flash \
  --out gguf/DeepSeek-V4.1-Flash-template.gguf
gguf-tools/deepseek4-quantize --hf ../DeepSeek-V4.1-Flash \
  --template gguf/DeepSeek-V4.1-Flash-template.gguf --experts mxfp4 \
  --out gguf/DeepSeek-V4.1-Flash-MXFP4.gguf                              # about 306 GB
python3 gguf-tools/deepseek_v41_engram.py --hf ../DeepSeek-V4.1-Flash \
  --out gguf/DeepSeek-V4.1-Flash-engram.gguf                             # about 203 GB
gguf-tools/deepseek4-quantize --hf ../DeepSeek-V4.1-Flash \
  --dspark-support --dspark-heads-only --out gguf/DeepSeek-V4.1-Flash-dspark.gguf
```

The backbone keeps the released MXFP4 routed experts and holds projections, shared experts
and the output at Q8_0; it is resident, so it needs a Mac with more unified memory than
its size. The engram sidecar is memory-mapped and read a few rows per token, so it only
has to be on fast storage. The DSpark file carries just the draft heads, since the three
draft blocks are already in the backbone.

```sh
./ds4 -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --engram gguf/DeepSeek-V4.1-Flash-engram.gguf --ctx 32768
./ds4 -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --engram gguf/DeepSeek-V4.1-Flash-engram.gguf \
  --mtp-model gguf/DeepSeek-V4.1-Flash-dspark.gguf --dspark --temp 0
./ds4 -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --engram gguf/DeepSeek-V4.1-Flash-engram.gguf \
  --mtp-model gguf/DeepSeek-V4.1-Flash-dspark.gguf --dspark --mtp-exact-sampling
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --engram gguf/DeepSeek-V4.1-Flash-engram.gguf \
  --mtp-model gguf/DeepSeek-V4.1-Flash-dspark.gguf --dspark --ctx 65536 --kv-disk-dir ~/.ds4/server-kv
```

Pass the engram sidecar with `--engram`: the model loads without it, but its two engram
layers then skip their lookups and the output degrades. `--dspark` with the heads file
enables the drafter, which speeds up greedy decoding. At non-zero temperature it keeps
target-matching greedy drafts like the other models; `--mtp-exact-sampling` accepts each
draft with the target's own probability instead and preserves the sampling distribution.
`--dspark-confidence` prunes drafts below a confidence; for this model the default of 0.6
is the measured optimum in both modes, and `DS4_DSPARK_STATS=1` prints acceptance counts
at exit.

Two Macs can run it with tensor parallelism (see [DISTRIBUTED.md](DISTRIBUTED.md)): the
trunk's routed experts are split between the ranks and everything else is replicated, so
each rank holds half the experts plus the dense weights and the engram sidecar. Pass the
same `--engram`, `--mtp-model` and DSpark options to the worker and the coordinator.

Prefill runs in chunks of 512 tokens by default; `--prefill-chunk` changes it. A larger
chunk helps very long prompts a little more, a smaller one trims the per-session scratch
that scales with it, and the output does not depend on the choice. The
compressed caches, index keys and selections stay on the GPU, so long contexts do not
move data per token. Activations and the matrix kernels stay in F32 (`DS4_DSV41_HALF_MM=1`
selects the half-precision tiles for comparison). Disk KV checkpoints and live prefix
reuse work as for the other models; a checkpoint carries the window, the compressed caches,
the open compressor groups and the draft rings, and a session with another prefill chunk
can load it.

For A/B checks, `DS4_DSV41_TRACE=1` prints each pass's host and GPU time,
`DS4_DSV41_FULL_SCAN=1` scores every compressed position in the index layers after the
candidate source instead of the candidate blocks, `DS4_DSV41_RADIX_MULTI=1` and
`DS4_DSV41_COMPACT_SINGLE=1` select the multi-dispatch selection kernels,
`DS4_METAL_Q8_MV_EXT=1` and `DS4_METAL_F16_MV_EXT=1` the generic small-batch matvecs, and
`DS4_DSV41_FLUSH_LAYERS=0` disables the mid-pass command flushes. `make tests/test_dsv41_metal
&& ./tests/test_dsv41_metal` compares every V4.1 kernel with a CPU transcription, and
`./tests/test_dsv41_layer <backbone> <engram> <dspark>` checks each layer kind and the
drafter against the CPU reference on the released weights. Without arguments it runs a
six-layer random mini model against the reference implementation's own outputs; those
files are generated, not tracked: `tests/deepseek_v41/make_mini_model.py --snapshot DIR`
(torch 2.4 and the checkpoint's `inference/` code) writes the weights and oracles into
`tests/deepseek_v41/mini`, and `make_mini_gguf.py --mini tests/deepseek_v41/mini --out
tests/deepseek_v41/mini/mini.gguf` packs the GGUF the test loads.

Metal only. Vision is not supported.

## GLM 5.3 Flash

| Target | Approximate file size | Use |
| --- | ---: | --- |
| `glm53-q2` | 90 GiB | One 128 GB Mac or DGX Spark; ROCm also supported |
| `glm53-q4` | 178 GiB | Larger Mac, two 128 GB Macs, or SSD streaming |
| `glm53-fp8` | 305 GiB | Packaged native weights only; inference not implemented |

```sh
./download_model.sh glm53-q2
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --ctx 32768
```

GLM 5.3 Flash has recurrent KDA layers, sparse DSA attention, hyper-connections,
and a built-in MTP block. The Q2 file uses imatrix-guided IQ2_XXS gate/up and
Q2_K down experts. Q4 is the higher-precision alternative.

Q2 is close enough to a 128 GB machine's memory budget that other workloads
and context size matter. Follow the [Metal](METAL.md), [Spark](DGX_SPARK.md),
or [Strix Halo](STRIX_HALO.md) starting configuration for your host.

Ordinary decode is the default. Enable the embedded draft block with `--mtp`:

```sh
./ds4-agent -m gguf/GLM-5.3-Flash-Q2.gguf --mtp --ctx 50000
```

No second model file is needed. See [sampling behavior](SPECULATIVE_DECODING.md)
before choosing between the default opportunistic mode and exact sampling.

## Full GLM 5.3 and GLM 5.2

Full GLM 5.3 Q2 is about 197 GiB. Use a sufficiently large machine or streaming:

```sh
./download_model.sh glm53-full-q2
./ds4 --ssd-streaming
```

GLM 5.2 downloads are `glm-antirez-iq2xxs`, `glm-antirez-q2`,
`glm-antirez-q4`, and the 11-shard `glm-unsloth-q4`. They are much larger
than GLM 5.3 Flash; choose memory capacity before choosing the quantization.

GLM runs on Metal, CUDA and ROCm. The routed paths include IQ2_XXS, Q2_K, and
Q4_K, with additional mixed layouts supported by the tested GGUFs. Two-Mac
ownership-aware TP accepts IQ2_XXS, Q2_K, and Q4_K gate/up layouts; this does
not mean every Q4 model fits two 128 GB machines. Use a tested artifact, not
an arbitrary combination of supported tensor types.

GLM uses graph-selected prefill chunks and does not accept `--prefill-chunk`
or an external `--mtp-model`. It currently requires `--power 100`.
Directional steering is supported for GLM 5.3, not GLM 5.2.

## Qwen3.8 Flash Next

Qwen3.8-Flash-Next (`qwen4exp` in GGUF terms) runs on its own Metal graph:
36 gated delta-net layers and 12 gated GQA layers with the QSA block-sparse
indexer, four-stream hyper-connections, the hashed per-layer n-gram table,
512-expert MoE with a shared expert, and the built-in MTP block. Build the
GGUF from the Hugging Face checkpoint with the converter in `gguf-tools/`;
the stock `ggml-org` Q8_0 GGUF also loads once its two parts are merged with
`llama-gguf-split --merge`:

```sh
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-Q8.gguf --outtype q8_0            # about 192 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q4_K \
  --tensor-type ffn_up_exps=Q4_K --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q4K.gguf Q8_0  # about 124 GB
```

The Q8 file is the reference build: every weight at 8 bits except the QSA
indexer projections, which stay at the released BF16. The Q4K file
requantizes the expert gate/up projections to Q4_K and the n-gram table to
Q4_0; keep the `hc_=f16` override so the hyper-connection mixers are not
requantized, which would slow prefill. Lower-bit tiers are future work. Both
need a Mac with more unified memory than the file size.

```sh
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 32768
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --mtp --temp 0
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 65536 --kv-disk-dir ~/.ds4/server-kv
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
./ds4-agent -m gguf/Qwen3.8-Flash-Next-Q8.gguf
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q4K.gguf --mtp --mtp-exact-sampling --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

The MTP block is inside the same GGUF; `--mtp` enables it and speeds up
greedy decoding. At non-zero temperature it keeps target-matching greedy
drafts, like the GLM path, which skews sampled output toward the greedy
choice; `--mtp-exact-sampling` preserves the ordinary sampling distribution
at a smaller speedup. Prefill runs in 8192-token chunks
(`DS4_QWEN4_PREFILL_CHUNK` overrides it; the transient buffers scale with the
chunk size). For A/B checks, `DS4_QWEN4_NO_FUSE=1` selects the unfused decode
kernels, `DS4_QWEN4_NO_IDX_SELECT=1` the argsort top-k and
`DS4_QWEN4_NO_ATTN_MM=1` the per-token attention kernel for prefill batches;
`DS4_QWEN4_TIMING=1` prints stage timings.

`--batched-session N` keeps N sessions resident; their decode steps run one
after another rather than as one grouped batch, so it buys concurrency, not
throughput. Thinking is on by default with the model's `xhigh` reasoning
instruction; `reasoning_effort` `low`, `medium` or `xhigh` selects the model
card's levels (`chat_template_kwargs` with `enable_thinking` and
`reasoning_effort` is accepted too), the server's `qwen3.8-flash-next-chat`
alias disables thinking and `qwen3.8-flash-next-reasoner` forces it. Tool
calls use the model's native `<tool_call><function=...><parameter=...>` format
in both the server and the agent. Disk KV checkpoints and live prefix reuse
work as for the other models; the recurrent state travels with the checkpoint,
so a session cannot be rewound to an arbitrary earlier position and a shorter
prompt is prefilled again.

The model's native window is 262144 tokens. For longer prompts set
`DS4_QWEN4_YARN_FACTOR=4`, which
applies the static YaRN scaling from the model card, up to about 1M tokens;
use a factor of 2 for 512k. Static YaRN costs a little accuracy on short
prompts, so leave it off otherwise.

Images go through the model's Qwen3-VL vision tower. Pass llama.cpp's mmproj
file (`ggml-org/Qwen3.8-Flash-Next-GGUF` ships a Q8_0 one, or run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation.

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and a Q8_0/Q4_0/MXFP4/F16/F32 n-gram table.
Multi-node tensor parallelism is not implemented yet.

## Vision

PNG and JPEG input works in the CLI, native agent, and HTTP server on Metal,
single-GPU CUDA, and ROCm. The encoder must match the model.

### DeepSeek Flash Vision Experimental

Vision Experimental is a different language checkpoint from Flash 0731.
The main download includes its encoder:

```sh
./download_model.sh ds4f-vision-q2
./ds4 --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

Larger targets are `ds4f-vision-q2-q4` and `ds4f-vision-mxfp4`.
`ds4f-vision-encoder` downloads just the encoder when the language GGUF is
already present. Vision Experimental has its own [DSpark drafter](SPECULATIVE_DECODING.md).

### GLM 5.3 Flash

The text GGUF stays the same. Download and add the encoder explicitly:

```sh
./download_model.sh glm53-vision
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf \
  --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
```

Use `/read image.png` in `ds4`, or start `ds4-agent` with the same `--vision`
argument to enable `view_image`. Agent sessions containing images cannot yet
be saved with `/save`.

For two-Mac TP, pass the same encoder on both ranks. The coordinator encodes
the image and sends the projected visual tokens to the worker.
For HTTP image formats and limits, see [serving](SERVER.md#images).
