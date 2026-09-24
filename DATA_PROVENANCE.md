# Data and Weight Provenance

## Weight lineage

The EXLLM project originates from random initialization. The 5M release has the following lineage:

1. a new EXLLM architecture was defined;
2. the earlier in-project EXLLM checkpoint was trained from random initialization;
3. the 5M architecture was expanded by transplanting dimension-compatible parameters from that checkpoint;
4. training/fine-tuning used the project corpora under `data/`;
5. iterative hard-case corpora were added after regression failures;
6. the final fp32 candidate was quantized to the EXLLM8 int8 deployment format.

**No external pretrained checkpoint, including TinyJP/beta, was used.**

## Training data

The JSONL corpora included with this release were generated specifically for EXLLM. They contain short Japanese instruction/answer pairs for the intended electronic-dictionary assistant scope, plus hard cases for:

- greetings and polite interaction;
- EXLLM / ToTo identity;
- common dictionary definitions;
- offline/current-information limitations;
- unknown-term fallback;
- mixed Unicode and byte-fallback behavior;
- noise/fuzz-like inputs;
- calculator routing;
- regression anchors discovered during development.

No external public dataset was imported into the final zero-based training corpus.

## Licensing

The project-generated data in this release is distributed under Apache-2.0 together with the code and released weights, unless otherwise stated.

## Important limitation

Project-generated definitions and factual examples are small and curated for product behavior; they are not a comprehensive knowledge corpus. Inclusion of a statement in the training files is not a guarantee that it is universally correct or current.
