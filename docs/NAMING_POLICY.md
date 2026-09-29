# EXLLM model naming policy

Public model names use this form:

```text
EXLLM[base-generation]-PURPOSE[variant-generation]
```

- `EXLLM` identifies the shared project and first architecture generation.
- `base-generation` changes only when the base architecture contract changes, such as layers, hidden dimensions, vocabulary, tokenizer compatibility, or another runtime-breaking structural revision.
- `PURPOSE` is a short uppercase identifier for the model objective.
- `variant-generation` changes when additional training or alignment improves a model without changing the base architecture contract.
- The first generation on either axis omits the number `1`.

Examples:

| Name | Meaning |
|---|---|
| `EXLLM-JPTOEN` | first Japanese-to-English model on the first EXLLM architecture |
| `EXLLM-JPTOEN2` | second improved Japanese-to-English model on the same base architecture |
| `EXLLM2-JPTOEN` | first Japanese-to-English model on the second EXLLM architecture |
| `EXLLM2-JPTOEN4` | fourth Japanese-to-English variant on the second EXLLM architecture |

Processed-token counts, parameter counts, quantization, and checkpoint steps do not belong in the public model name. They remain explicit model-card and manifest fields.

Intermediate or rejected checkpoints do not receive public generation numbers. Their hashes and lineage may be retained for reproducibility without publishing their weights or presenting them as model releases.
