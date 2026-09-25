# Training and release reproducibility

The EXLLM project originates from random initialization. The published 5M
release was expanded from the earlier project-owned v1.0 checkpoint and then
trained in six recorded stages. No external pretrained checkpoint was used.

## Published lineage inputs

- `weights/EXLLM-v1.0.0.safetensors`: direct 4-layer parent of the 5M expansion.
- `config-v1.0.json`: architecture of that parent.
- `tokenizer.json`: tokenizer shared by the parent and 5M model.
- `data/*.jsonl`: all datasets used by the recorded 5M stages.
- `training/release-5m-stages.json`: exact stage order, hyperparameters, seeds,
  input/output names, and expected SHA-256 values.

These are project-created artifacts released under Apache-2.0. The parent is
not a third-party checkpoint.

## Replay the 5M lineage

Install the dependencies in `requirements.txt`, then inspect the commands
without running them:

```bash
python tools/replay_5m.py --dry-run
```

Run the complete recorded path from the v1.0 parent through `release3`:

```bash
python tools/replay_5m.py
```

Resume at a later stage when its input checkpoint already exists:

```bash
python tools/replay_5m.py --from-stage release2
```

The replay tool compares every generated checkpoint with the historical hash.
Commands, seeds, datasets, and hyperparameters are preserved, but PyTorch,
BLAS, CPU, and platform differences can prevent bit-for-bit identity. A hash
mismatch records numerical non-identity; it does not by itself prove that a
different training recipe was used.

## Stage summary

| Stage | Input | Data | Work | Batch | LR | Output |
|---|---|---|---:|---:|---:|---|
| expand-and-train-5m | v1.0 safetensors | `mix.jsonl` | 8 epochs | 128 | 5e-4 | `EXLLM-v1.1-5m.pt` |
| fix1 | 5M | `final_consistent.jsonl` | 800 steps | 64 | 1.2e-4 | `5m-fix1.pt` |
| final | fix1 | `v1_1_release_fix.jsonl` | 500 steps | 32 | 6e-5 | `5m-final.pt` |
| release | final | `v1_1_release_fix.jsonl` | 250 steps | 32 | 2.5e-5 | `5m-release.pt` |
| release2 | release | `v1_1_release_fix.jsonl` | 160 steps | 32 | 1.2e-5 | `5m-release2.pt` |
| release3 | release2 | `v1_1_release_fix.jsonl` | 100 steps | 32 | 6e-6 | `5m-release3.pt` |

The expansion uses seed `20260923`; corrective stages use seed `20260924` and
derive their sampling RNG from the input `global_step`. The machine-readable
manifest is authoritative for filenames, hashes, and full parameters.

## Corpus generation

`src.data_gen` creates the base corpus. The hard/recovery generators and their
generated JSONL outputs are retained under `src/` and `data/`. The historical
5M release used the published JSONL snapshots listed in the stage manifest;
regenerating a corpus is a separate experiment and is not required to replay
the released training stages.
