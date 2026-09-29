# GGUF companions

EXLLM's embedded checkpoints use the custom EXLLM / EXLLM8 / EXQ12 stack. They are not Llama checkpoints and cannot be truthfully relabeled or losslessly converted to GGUF.

The PC artifacts are therefore separately trained Llama-compatible companions. They reuse the corresponding project's accepted data boundary and objective, but have different weights, tokenizer vocabulary, feed-forward structure, context metadata, and parameter count. Results from a GGUF companion must not be presented as measurements of the EX-word checkpoint.

## 2026-09-30 builds

| Model repository | GGUF | Parameters | Training / runtime context | RTX 3060 llama.cpp smoke |
|---|---|---:|---:|---|
| [EXLLM-JPTOEN](https://huggingface.co/ToTo-40417/EXLLM-JPTOEN) | `EXLLM-JPTOEN-F16.gguf`, SHA-256 `dc4f8031aa74e62595e0dd7bc00ed542320a953c1423ac1d2263a0adbcfad6ef` | 5,441,184 | 128 / 512 | `おはようをえいごでいうと？` → `good morning`, approximately 415 token/s |
| [EXLLM-ONI5M](https://huggingface.co/ToTo-40417/EXLLM-ONI5M) | `EXLLM-ONI5M-F16.gguf`, SHA-256 `7ea10747a117600d4bf2aed88fd9275a4a720e527dfbaca67364913900586c4e` | 5,441,184 | 128 / 512 | `こんにちは` → the expected ONI greeting, approximately 443 token/s |

Both builds use six Llama decoder layers, hidden size 288, FFN size 608, nine attention/KV heads, tied embeddings, a 1,024-piece SentencePiece tokenizer, and random initialization. The model repositories contain the exact training manifest and smoke log.

## Reproduction boundary

- [`tools/train_lmstudio_companion.py`](../tools/train_lmstudio_companion.py) trains the Llama-compatible checkpoint and records input hashes.
- [`tools/build_jptoen_gguf_dataset.py`](../tools/build_jptoen_gguf_dataset.py) deterministically expands the JPTOEN project-owned source lists into train and held-out-template validation files.
- ONI5M uses the published ONI5M training shards and validation file directly.
- `llama.cpp/convert_hf_to_gguf.py --outtype f16` performs the Hugging Face Llama checkpoint to GGUF serialization.
- LM Studio and llama.cpp should use a 512-token context for these companions. The source examples remain capped at 128 tokens.

The GGUF relationship is an interoperability companion, not numerical equivalence. EXQ12 device tests, GGUF PC tests, and their throughput figures remain separate evidence.
