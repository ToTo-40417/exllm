# Model catalog

EXLLM is the common architecture, conversion, evaluation, and reproducibility project. Models with materially different objectives are published as separate Hugging Face repositories and linked from this catalog.

| Model | Objective | Parameters / context | EXQ12 | EX-word app | Distribution |
|---|---|---|---|---|---|
| EXLLM 5M v1.1 | short Japanese responses, definitions, fallback, integer calculation routing | 5,377,824 / 128 | supported | `exllm-exword` v1.1.0 or later; v1.3.1 current | [ToTo-40417/EXLLM](https://huggingface.co/ToTo-40417/EXLLM) |
| EXLLM GGUF companion | PC/LM Studio compatibility; separately trained Llama-compatible companion | 5,441,184 / 128 training | not an EXQ12 conversion | not applicable | [ToTo-40417/EXLLM](https://huggingface.co/ToTo-40417/EXLLM) |
| EXLLM-JPTOEN | kana-input Japanese-to-English short-answer model; 100,003,620 processed-token checkpoint; separate Llama-compatible GGUF companion | 5,377,824 / 128 (EX-word); 5,441,184 / 512 runtime (GGUF) | verified on XD-B4800 | `exllm-exword` v1.3.1 recommended | [ToTo-40417/EXLLM-JPTOEN](https://huggingface.co/ToTo-40417/EXLLM-JPTOEN) |
| EXLLM-ONI5M | playful impatient-teacher persona; separate Llama-compatible GGUF companion | 5,377,824 / 128 (EX-word); 5,441,184 / 512 runtime (GGUF) | verified on XD-B4800 | `exllm-exword` v1.3.1 recommended | [ToTo-40417/EXLLM-ONI5M](https://huggingface.co/ToTo-40417/EXLLM-ONI5M) |

“Supported” means the artifact satisfies the EXQ12 structure and has been exercised with the stated app/runtime. It does not mean that every prompt produces a correct answer.

EXLLM-JPTOEN's 2,901,546 processed-token parent is an intermediate checkpoint, not a public model. Its hash remains in the lineage record, but its weights are not uploaded. Objective-specific models receive separate Hugging Face repositories and model cards without merging unlike capability claims into the base EXLLM model card. Each GGUF companion is separately trained and is not a format conversion of its EXQ12 counterpart.

Names follow [`NAMING_POLICY.md`](NAMING_POLICY.md). Token counts and parameter counts are metadata rather than name suffixes.

## Non-release experiments

| Experiment | Result | Publication status |
|---|---|---|
| General Japanese 5M / 100M processed-token pipeline validation | Training pipeline completed, but the semantic gate passed only 8/35 and the written-Japanese alignment did not match the kana-first EX-word input contract | rejected checkpoint; method and failure record only |

The general-Japanese experiment is documented under [`training/general-ja-5m/`](../training/general-ja-5m/). A rejected checkpoint is not a model variant and must not be uploaded to Hugging Face as though it passed the release gate.
