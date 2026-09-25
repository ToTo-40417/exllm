# EXLLM 5M — Model Card / モデルカード

## Model details

| Field | Value |
|---|---|
| Model | EXLLM |
| Version | 1.1.0-5m |
| Developer | ToTo |
| Language focus | Japanese |
| Parameters | 5,377,824 unique trainable parameters |
| Architecture | decoder-only Transformer |
| Layers | 6 |
| Hidden size | 288 |
| Attention heads | 9 |
| Head dimension | 32 |
| FFN size | 896 |
| FFN activation | ReLU |
| Normalization | RMSNorm, pre-norm |
| Context | 128 tokens |
| Vocabulary | 868 |
| Tokenizer | frequent-character + UTF-8 byte fallback, NFC |
| Position encoding | learned position embedding |
| Output head | tied to token embedding |
| Primary target | CASIO EX-word XD-B4800 / DATAPLUS 6 class device |
| License | Apache-2.0 |

The tied language-model head is the same parameter as the token embedding. State-dict tensor element totals can therefore look larger if a serializer lists both names. The unique parameter count is **5,377,824**.

日本語: EXLLM 5Mは、短い日本語対話を低資源端末上で生成するための実験的モデルです。汎用知識モデルではなく、未知の話題や長い指示、高リスク判断には使用できません。

## Design goals

EXLLM is designed to maximize usefulness under a very small memory/compute budget rather than to maximize broad knowledge. The architecture deliberately uses a small vocabulary, short context, tied embeddings, ReLU, and a byte fallback tokenizer to make a future fixed-point SH4 implementation practical.

The runtime separates tasks that should be deterministic from tasks that should be generated. In particular, simple integer arithmetic is handled by a tiny parser/calculator instead of trusting the language model to infer exact arithmetic.

## Intended use

Appropriate uses include:

- short Japanese greetings and UI conversation;
- short explanations of common words and computing terms present in the training scope;
- self-identification as EXLLM and developer identification as ToTo;
- offline capability statements;
- simple dictionary-assistant interaction on an electronic dictionary;
- research/education around tiny on-device language models.

## Out-of-scope use

EXLLM should not be treated as a general factual authority. It does not have live web access, current weather/news, personal data access, or reliable expert-level medical/legal/financial knowledge. It should not be used as the sole decision source for high-stakes matters.

The model has only ~5.38M parameters. Unseen concepts, complicated instructions, long reasoning tasks, nuanced translation, and open-ended factual questions can fail or fall back.

## Training provenance

The EXLLM project originates from random initialization. The 5M release was expanded from an earlier in-project EXLLM checkpoint by transplanting dimension-compatible parameters. No external pretrained checkpoint, including TinyJP/beta, was used. The published training JSONL files were generated for EXLLM and are included in `data/` for inspection. See `DATA_PROVENANCE.md`.

## Quantization

`weights/EXLLM-v1.1-5m-int8.bin` uses symmetric int8 quantization per output row for 2D weight matrices. RMSNorm vectors are stored in fp16 because their footprint is negligible and preserving them reduces unnecessary quantization error. `lm_head.weight` is an alias of `tok.weight` and is not duplicated in the EXLLM8 file.

The int8 package was dequantized back into the reference PyTorch model and passed the same project release gate as the fp32 candidate.

## Character robustness

The tokenizer has 256 byte tokens plus frequent Japanese character tokens and five structural special tokens. Unknown Unicode characters are encoded as their UTF-8 bytes, not as `<unk>`.

The reference generator masks illegal next bytes with an explicit UTF-8 state machine. Property testing in this release included random valid Unicode scalars, tokenizer round-trips, invalid UTF-8 edge cases, and mixed-Unicode prompt fuzzing.

## Evaluation interpretation

The included tests are **project regression tests**, not an external benchmark and not proof of general language understanding. They are intended to prevent regressions in the narrow product behavior EXLLM is designed to provide.
