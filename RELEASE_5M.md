# EXLLM 5M v1.1 release

This release contains the 5,377,824-parameter EXLLM checkpoint and the model
artifacts used by the `v1.1.0` EX-word runtime. Device source, installation,
and measured XD-B4800 results are maintained in
[`exllm-exword`](https://github.com/ToTo-40417/exllm-exword).

## Architecture

- vocabulary: 868
- context: 128
- hidden size: 288
- layers: 6
- attention heads: 9 (head dimension 32)
- FFN size: 896
- device weights: row-wise int8; RMSNorm vectors: Q12 int16

## Release files

| File | Bytes | SHA-256 |
|---|---:|---|
| `weights/EXLLM-v1.1-5m-release3.pt` | 21,526,737 | `48ad00c6640689595a253fc949cad83f3000e14cccdf69275fcf245e85be3ceb` |
| `weights/EXLLM-v1.1-5m-int8.bin` | 5,443,105 | `3b2bf2d9f103cba714bc98976709c5ad34860c1ff55fed2fce91fc95afb5b34a` |
| Hugging Face `weights/model.q12` | 5,443,105 | `d64037fde791e5c0e48101bc1a8ab366a3287f1f36b4464879e36495ea7e5a53` |
| `weights/EXLLM-v1.0.0.safetensors` | 4,905,568 | `caad6a93aa2a8f6f2b525c4671dc8d2f8e4328d262a093172a55c672cd4e24c9` |

The v1.0 artifact is the project-owned direct parent used for dimensional
transplantation into the 5M architecture. The exact release-stage lineage is
recorded in `training/release-5m-stages.json`. Generate and verify the EX-word
artifact with `python tools/export_exq12.py`.

The EXLLM8 int8 quality gate passed all semantic, calculator, fuzz and
sampling cases. The fixed-point device runtime also produced the expected
Japanese answers. The device UI accepts
romaji keyboard input and redraws the answer as complete UTF-8 tokens arrive.
The menu distinguishes free input, example questions and device information;
its title does not contain a hard-coded device model name.
The generation screen preserves the submitted question above a divider and
streams the answer below it. Repeated SYMBOL presses cycle through Japanese
question/period/comma, exclamation and prolonged-sound marks.
The HISTORY key captures the complete 528×320 framebuffer as a timestamped
RGB565 BMP in the app `_USER` directory. The earlier experimental microSD/JPEG
path was removed after device testing; internal BMP storage is the stable
v1.1.0 behavior.
