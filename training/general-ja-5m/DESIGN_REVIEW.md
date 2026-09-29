# General Japanese 5M design review

This note records which external review points were verified against the EXLLM
implementation and which remain experimental. It prevents suggestions from
silently becoming facts or fixed hyperparameters.

## 2026-09-29 result and stop decision

The full pipeline completed: 100,007,936 raw-language tokens, a three-epoch
canonical alignment run, checkpoint creation, and the release gate. This is
evidence that the model-generation mechanism works. It is not evidence that
the resulting model is useful.

The gate accepted 8/35 semantic cases and 200/200 fuzz cases, so the candidate
is rejected. Training is not being extended or restarted. The completed
artifacts remain available as reproducibility evidence. Before any new run,
device-facing prompts must be rebuilt for the kana-first input contract; simply
adding more compute to the kanji-oriented alignment is not an accepted remedy.

## Verified and adopted

- **Parameter count:** the model contains 5,377,824 unique trainable parameters.
  `5,443,105` is the byte size of the current EXQ12 artifact. The number was
  recalculated from `EXLLMConfig` and `model.parameters()`.
- **Token terminology:** 100M is a processed-input-token ceiling. Supervised
  prompts are masked by the existing collator, so processed and loss-bearing
  tokens are different and are now reported separately.
- **Tokenizer audit:** the fixed tokenizer uses UTF-8 byte fallback for
  non-atomic characters. Corpus manifests now report fallback use, tokens per
  Unicode character and context-limit incidence.
- **Legacy corpus:** 85,005 rows contain 12,968 unique prompt/answer pairs and
  1,329 prompts with multiple answers. Raw row count is not independent
  coverage; conflicting prompts cannot be replayed without resolution.
- **Evaluation isolation:** exact prompt non-overlap alone does not prevent
  semantic leakage. New data must be split by concept or source document before
  augmentation, and evaluation transformations must be distinct from training
  transformations.
- **Inspection points:** 5M, 20M, 50M and 100M processed tokens are explicit
  evaluation opportunities. The 100M ceiling is not a completion requirement.
- **Device input contract:** the current EX-word UI produces hiragana from
  romanized input and has a separate ASCII mode, but no kanji conversion.
  Device-facing alignment and evaluation therefore require deterministic
  kana-only prompt forms. Answers and raw language data remain normal
  kanji-kana mixed Japanese. Original and kana-converted forms are grouped
  together before splitting to prevent leakage.

## Considered but not used in this production generation

- Interleaved replay may be a reasonable defense against regression, but the
  proposed 5–10% and 10–15% ranges have not been validated on EXLLM. Rather than
  introduce an unverified mixture into the production run, this specification
  uses a complete canonical alignment stage after language training.
- Continuing from `EXLLM-v1.1-5m-release3.pt` remains the first controlled
  candidate. A matched pilot from a pre-alignment checkpoint may be useful, but
  no claim is made that either parent is better before evaluation.
- The 100M / 5,377,824 = 18.595 ratio is only a planning heuristic inspired by
  large-model scaling work. It is not evidence of a compute-optimal point for a
  5M model, a 128-token context, or this Japanese tokenizer.
- A compact on-device IME was considered, but it is not part of this production
  contract. Dictionary size, candidate selection, input effort and runtime
  memory must be measured before adopting it; model training must not assume
  that kanji entry exists today.

## Production contract

1. Record source URL, revision/retrieval date, license URL/status, attribution,
   original hash, normalized hash and transformation version for every shard.
2. Use the independently authored `validation.jsonl` suite separately from the
   legacy release gate; no training transform is applied to this suite.
3. Accept the output only after corpus manifests, token counters, the held-out
   suite and the legacy release gate are all available. A failed gate rejects
   the output; it does not trigger an improvised intermediate release.
4. Report the kana-only device suite separately from host-side written-Japanese
   evaluation. Do not use success on a kanji prompt as evidence for a prompt the
   current device UI cannot enter.
