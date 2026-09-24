# Training / reproducibility notes

EXLLM v1.0.0 was trained from a fresh random initialization. No TinyJP/beta weights were used.

The development process was iterative:

1. generate the base corpus with `src.data_gen`;
2. train the 4-layer EXLLM architecture from random weights;
3. add hard paraphrases / robustness examples;
4. run semantic and Unicode regression tests;
5. generate focused recovery corpora for observed failures;
6. fine-tune in short, checkpointed stages;
7. teach raw-model arithmetic routing instead of forcing unreliable arithmetic memorization;
8. quantize the final candidate and rerun the release gate.

All corrective corpus generators are preserved as `src/make_recovery*.py`, and the generated JSONL files are included under `data/`.

A clean reference training starts with:

```bash
python -m src.data_gen data
python -m src.train --epochs 8 --batch 96 --out weights/base.pt
```

Then run the hard/recovery generators and use `python -m src.finetune ...` for the corrective stages. Exact bit-for-bit reproduction is not guaranteed across PyTorch/CPU implementations, and the development process used short checkpointed iterations rather than one monolithic training command.
