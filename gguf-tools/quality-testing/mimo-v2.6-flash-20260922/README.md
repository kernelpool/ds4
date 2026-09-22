# MiMo-V2.6 Flash official continuations (2026-09-22)

100 greedy continuations of `prompts.jsonl` from the Xiaomi platform
(`https://api.xiaomimimo.com/v1/chat/completions`, model `mimo-v2.6-flash`,
`chat_template_kwargs.enable_thinking=false`, no system message, 24 tokens,
temperature 0). The platform returns no logprobs, so the `api_*` columns are
empty; only NLL, first-token match and greedy prefix length apply.

Collected with:

```sh
export XIAOMI_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --out gguf-tools/quality-testing/mimo-v2.6-flash-20260922 \
  --model mimo-v2.6-flash --endpoint https://api.xiaomimimo.com/v1/chat/completions \
  --api-key-env XIAOMI_API_KEY --api-key-header api-key \
  --chat-template-kwargs '{"enable_thinking": false}' --thinking omit --reasoning-effort omit \
  --top-logprobs 0 --count 100 --max-tokens 24
```

Scored with:

```sh
gguf-tools/quality-testing/score_official MiMo-V2.6-Flash-MXFP4.gguf \
  gguf-tools/quality-testing/mimo-v2.6-flash-20260922/manifest.tsv /tmp/mimo.tsv 4096
```

`results/` holds the reference score TSV of each release (Metal, M3 Ultra);
the numbers are recorded in `QA_BEFORE_RELEASES.md`.
