# Official Quality Fixtures

This directory contains curated hosted-model continuation fixtures that are
safe to commit and use in release QA.

- `glm52-openrouter-100`: 100 GLM 5.2 OpenRouter continuations with API
  top-logprob slices.
- `glm53-flash-openrouter-zai-fp8-100`: 100 deterministic GLM 5.3 Flash
  continuations from OpenRouter's pinned Z.AI FP8 endpoint. That endpoint did
  not return logprobs.
- `flash`: 100 DeepSeek V4 Flash 0731 continuations from the official DeepSeek
  API, with API top-logprob slices.
- `pro`: 100 DeepSeek V4 PRO preview continuations with API top-logprob slices.
- `pro-0813`: 100 DeepSeek V4 PRO 0813 continuations with API top-logprob
  slices.
- `v41-flash`: 100 DeepSeek V4.1 Flash continuations from the official
  DeepSeek API (model alias `deepseek-flash`, 2026-09-10, thinking disabled,
  24 tokens). The API reported logprob 0.0 for every chosen token and -9999
  for the alternatives, so only the continuation tokens carry information;
  score with `--engram` so the model runs with its sidecar.

Each fixture directory contains:

- `prompts/case_*.txt`: exact user prompts.
- `continuations/case_*.txt`: deterministic hosted-model continuations.
- `responses/case_*.json`: raw hosted responses, including logprob slices.
- `manifest.tsv`: paths consumed by `score_official`.

DeepSeek V4 Flash smoke vectors are also tracked in `tests/test-vectors/` and
are run by `./ds4_test --logprob-vectors`.
