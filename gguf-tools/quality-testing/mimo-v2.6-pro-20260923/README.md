# MiMo-V2.6 Pro official continuations (2026-09-23)

100 greedy continuations of `prompts.jsonl` from the Xiaomi platform
(`https://api.xiaomimimo.com/v1/chat/completions`, model `mimo-v2.6-pro`,
`chat_template_kwargs.enable_thinking=false`, no system message, 24 tokens,
temperature 0). The platform returns no logprobs, so the `api_*` columns are
empty; only NLL, first-token match and greedy prefix length apply.

Collected with the Flash fixture's command
(`mimo-v2.6-flash-20260922/README.md`) with `--model mimo-v2.6-pro`.

Scored over two-Mac tensor parallelism, with the worker started first:

```sh
./ds4 --model MiMo-V2.6-Pro-RL-MXFP4.gguf --ctx 4096 \
  --tensor-parallel --role worker --coordinator HOST 9911 --transport rdma
gguf-tools/quality-testing/score_official MiMo-V2.6-Pro-RL-MXFP4.gguf \
  gguf-tools/quality-testing/mimo-v2.6-pro-20260923/manifest.tsv /tmp/mimo-pro.tsv 4096 \
  --tensor-parallel --role coordinator --listen HOST 9911 --transport rdma
```

`results/` holds the reference score TSV (Metal, two M3 Ultras); the
numbers are recorded in `QA_BEFORE_RELEASES.md`.
