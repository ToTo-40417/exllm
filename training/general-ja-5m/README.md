# EXLLM General Japanese 5M — development workspace

This directory defines the next EXLLM model experiment without replacing the
published `EXLLM-v1.1-5m` checkpoint.

## Run outcome

The 2026-09-29 run is stopped and is not a release candidate. No training
process remains active on the AI server, and its checkpoints and logs were
retained rather than deleted.

The run establishes that the AI-assisted project can assemble the corpus,
train the fixed 5,377,824-parameter architecture, perform canonical alignment,
and produce inspectable checkpoints end to end. The raw-language stage reached
100,007,936 processed tokens in 6,104 steps; canonical alignment completed 270
steps. This demonstrates pipeline operation, not model quality or suitability.

The final gate passed all 200 fuzz cases but only 8 of 35 semantic cases, so the
candidate was rejected. In addition, its alignment assumptions used ordinary
written Japanese with kanji, while the current EX-word UI accepts kana-first
questions. Continuing the same training would spend compute on a candidate
whose host-side prompt distribution does not match actual device input. No
automatic retry or further training is planned. A future run must first rebuild
alignment and held-out evaluation around the EX-word input contract below.

## Objective

- keep the current 5,377,824-trainable-parameter architecture and 128-token
  context (`5,443,105` is the byte size of the current EXQ12 artifact, not its
  parameter count);
- broaden short Japanese question coverage without weakening the known-answer,
  unknown-input, Unicode, and offline-safety behavior of the current model;
- preserve compatibility with the EXLLM8 and EXQ12 exporters and the EX-word
  fixed-memory runtime;
- use only project-owned or redistribution-compatible training records, with
  provenance recorded before a source enters the corpus.

The target is a ceiling of approximately 100 million **processed input tokens**,
or 18.595 processed tokens per trainable parameter. This ratio is only a
large-model scaling-law-inspired planning heuristic; it is not an experimentally
established optimum for EXLLM. Because supervised examples mask the prompt side,
the run must report processed tokens and loss-bearing tokens separately. It must
also report padding, truncation, document/example counts, unique counts and
exposure counts from the packed corpus rather than estimating them from file
size.

Transformer layers cannot be assigned independent datasets: every training
example updates all layers. The production generation therefore uses one fixed
two-stage specification:

1. **Language stage:** continue the release3 checkpoint for exactly 100M
   processed causal-language tokens from the deduplicated, document-split Aozora
   corpus. Documents are seen in packed form without padding or truncation.
2. **Canonical alignment stage:** make exactly three epochs over one
   deterministically selected answer for each of the 11,396 distinct legacy
   prompts. Shard priority is release fix, final consistency/balance/anchor,
   recovery, robustness/hardness, mix, then original train. Within the highest
   priority shard, frequency and then lexical order break ties.

### EX-word input contract

The current `exllm-exword` keyboard accepts romanized input and converts it to
hiragana, or accepts ASCII in `ABC` mode. It does not provide kanji conversion.
Device-facing alignment must therefore treat kana-only questions as the primary
form, for example `おはようをえいごでいうと？`, rather than assuming that a
user can enter `おはようを英語で言うと？`.

This constraint applies to user prompts, not to model answers or the language
stage. Answers remain natural Japanese with kanji and kana, and raw language
training remains in its original orthography. The alignment corpus must add a
deterministic kana reading for supported device prompts, retain the original
written prompt for non-device evaluation, and record the conversion tool and
version. Ambiguous readings require an authored reading; they must not be
silently guessed. Kana variants and their original forms stay in the same split
group so that conversion cannot leak an evaluation item into training.

Acceptance testing includes a separate device-realistic suite containing only
characters the current input UI can enter. Host-side results from kanji prompts
must not be reported as evidence of EX-word usability unless the equivalent
kana prompt also passes. Model responses are still evaluated in normal
kanji-kana mixed Japanese.

An on-device IME is a possible future runtime feature, but it is outside this
model's required contract. Its conversion dictionary, candidate UI, binary
size, and key-operation cost must be measured before it can replace kana-first
alignment.

This structure maximizes distinct language exposure first, then restores the
runtime's `USER`/`ASSIST` behavior without counting 72,037 duplicate legacy rows
as new knowledge. The 1,329 cross-shard prompt conflicts remain reported in the
selection manifest. Raw language text and instruction records remain separate
so the generation is reproducible.

The first candidate is a controlled continuation from the published
`EXLLM-v1.1-5m-release3.pt`. It is a separate checkpoint and never overwrites
the published model. This production specification does not branch into multiple
candidate checkpoints; intermediate snapshots, if retained, are recovery
artifacts rather than review releases.

## Reproducibility boundary

`experiment.json` pins the parent checkpoint, tokenizer, model configuration,
and initial corpus inputs by SHA-256. Generated checkpoints, optimizer state,
temporary corpora, and exports belong under `work/` and are intentionally
ignored by Git.

Before training:

```sh
python tools/audit_training_data.py \
  --manifest training/general-ja-5m/experiment.json \
  --output training/general-ja-5m/work/data-audit.json
```

The audit rejects malformed records and reports exact duplicates, conflicting
answers for the same prompt, category balance, token lengths, byte-fallback
usage, context-limit/truncation counts, and overlap between training and
validation prompts. Exact-prompt separation is necessary but insufficient:
new evaluation data must also be separated by concept/source-document identity
before any augmentation, and evaluation transformations must be distinct from
training transformations. Training must not begin if artifact
hashes differ from the manifest or if newly added data lacks a documented
source and redistribution basis.

## Planned gates

1. Baseline: record the current 5M checkpoint on both the existing backward-
   compatibility release gate and a separate held-out general-Japanese suite.
2. Candidate: train into `work/` with a new experiment ID and immutable log.
3. Regression: require all current release checks to remain green, then compare
   the general-Japanese categories (grammar, denoising, intent, facts,
   uncertainty, safety and long/noisy inputs). Run the device-realistic
   kana-only prompt suite separately and require answer quality in natural
   kanji-kana mixed Japanese. Training loss alone is not an acceptance test.
4. Export: produce EXLLM8 and EXQ12 artifacts, run their parsers, and verify
   hashes and byte counts.
5. Device: test the candidate separately on DATAPLUS 6 and 7. The published
   model remains available until the candidate passes both.

E130 is the environment-authoring and packaging host. It prepares the manifest,
audits and packs the corpus, and records hashes. GPU training runs in a dedicated
EXLLM environment on the local AI server from that hash-checked snapshot. The
server does not edit the source corpus, and only completed checkpoints, logs,
environment manifests, and evaluations are copied back.

## Data acceptance

`sources.json` is the allowlist. A source may be downloaded only after its
license, attribution, text availability, and redistribution status are recorded.
Restricted corpora and unclear web crawls stay excluded. Model weights from
other projects are never used; compatible public text may be used only under its
own terms.

The planned cumulative inspection points are 5M, 20M, 50M and 100M processed
tokens. A run may stop before the ceiling when held-out quality saturates or a
backward-compatibility gate regresses.

The current 85,005-line corpus is retained for lineage reproduction, but it is
not treated as 85,005 independent examples: the audit currently finds 12,968
unique prompt/answer pairs, 72,037 repeated rows, and 1,329 prompts with multiple
answers. New training shards must report these figures before entering a run.
