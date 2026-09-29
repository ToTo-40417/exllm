# EXLLM-JPTOEN data provenance and release boundary

## Scope

This note covers EXLLM-JPTOEN and its unpublished 2,901,546 processed-token parent. EXLLM-JPTOEN is the 100,003,620 processed-token continuation. The token count is metadata, not part of its public name.

## Recorded origin

- The 266-row Japanese-kana-to-English lexicon was assembled and edited by the project owner with generative-AI assistance.
- Prompt templates, noisy prompt wrappers, ambiguous-kana entries, nonce generation, and unknown-calibration lists were created inside the project.
- No third-party dictionary corpus, pretrained model checkpoint, or web crawl is recorded as an input to these two checkpoints.
- A CC0 Wikidata acquisition recipe exists elsewhere in the development workspace, but no snapshot, query hash, or raw-result hash from that recipe appears in the executed checkpoint manifests. The case study therefore does **not** describe these checkpoints as Wikidata-derived.

Short word-to-word correspondences are factual in nature, but this repository does not rely on that alone as a licensing conclusion. The publishable claim is narrower: the released project owner controls the assembled lists and templates and has chosen to place the project-authored material under the repository license.

## Generation path

The 100M continuation samples repeatedly from a small, fixed source set:

| Source | Rows | Training use |
|---|---:|---|
| release lexicon | 266 | known bare words and template-derived queries |
| ambiguous kana | 12 | unknown calibration |
| real unknown training list | 80 | unknown calibration |
| real unknown holdout | 40 | evaluation only |
| nonce strings | generated | unknown calibration |
| general prompts | 7 pairs in the script | greeting and OOD calibration |

The sampling mixture is deterministic for seed `40417001`. The resulting 100M count is processed non-padding tokens, not unique text. Repetition and template derivation account for most of the token budget.

## Files and hashes

Machine-readable row counts, data hashes, checkpoint hashes, evaluation hashes, and the exact mixture are recorded in [`training/exllm-jptoen-case-study.json`](../training/exllm-jptoen-case-study.json).

## Release boundary

The case-study documentation, aggregate results, selected device screenshots, and hashes may be published in this repository. The old intermediate checkpoint will not be published. EXLLM-JPTOEN weights and full training lists are not included in this preparation. Before its future Hugging Face release:

1. retain this provenance statement with the model;
2. include the exact list and script hashes;
3. identify the model as project-authored with generative-AI assistance;
4. do not claim use of Wikidata unless an executed, hash-pinned snapshot is actually used;
5. publish it as `EXLLM-JPTOEN` in a repository separate from the base EXLLM model;
6. retain the old checkpoint only as a non-public parent hash in the lineage.
