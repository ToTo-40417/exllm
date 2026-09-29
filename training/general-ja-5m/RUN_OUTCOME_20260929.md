# General Japanese 5M run outcome — 2026-09-29

## Decision

Training is stopped. The generated checkpoint is retained for technical audit
but rejected as a release candidate. No matching training process was active on
the AI server when the stop decision was applied, so no process required a
termination signal. No automatic restart was configured or initiated.

## What was established

- An AI-assisted workflow produced a complete, runnable model-training setup.
- Corpus acquisition and packing completed.
- The fixed 5,377,824-parameter EXLLM architecture trained successfully.
- Canonical instruction alignment completed.
- Checkpoints and a machine-readable release-gate result were produced.
- The job wrapper preserved commands, process state, logs and return codes.

These facts establish pipeline feasibility only. They do not establish answer
quality, factual reliability, EX-word usability, or release fitness.

## Recorded results

| Stage | Result |
|---|---:|
| Raw-language processed tokens | 100,007,936 |
| Raw-language steps | 6,104 |
| Raw-language validation loss | 2.8393710708618163 |
| Raw-language elapsed time | 489.434 s |
| Alignment processed tokens | 2,116,392 |
| Alignment loss-bearing tokens | 756,156 |
| Alignment steps | 270 |
| Alignment elapsed time | 12.710 s |
| Semantic gate | 8 / 35 |
| Fuzz gate | 200 / 200 |
| Release accepted | No |

## Why training was not continued

The current `exllm-exword` input UI converts romanized input to hiragana and
does not provide kanji conversion. The completed alignment was designed around
ordinary written Japanese. A question such as `おはようを英語で言うと？` therefore
becomes the device-enterable `おはようをえいごでいうと？`, which is not
guaranteed to preserve model behavior at this scale.

The failed semantic gate already rejects this candidate. Continuing the same
run would not correct the mismatch between training prompts and real input.
Any successor must use deterministic kana forms for device-facing prompts,
retain natural kanji-kana answers, and include a separate kana-only device
evaluation suite before GPU training begins.

## Preserved server artifacts

The server retains the completed raw-language checkpoint, final aligned
checkpoint, release-gate JSON, job status files and logs. They are evidence of
the run and must not be presented as released model artifacts.
