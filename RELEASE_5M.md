# EXLLM 5M XD-B4800 build

This prerelease keeps the on-screen version string at `v1.0.0` and enlarges
the device model to 5,377,824 parameters.

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
| `weights/EXLLM-v1.1-5m-int8.bin` | 5,443,105 | `3b2bf2d9f103cba714bc98976709c5ad34860c1ff55fed2fce91fc95afb5b34a` |
| `../device_apps/exllm-public/model.q12` | 5,443,105 | `d64037fde791e5c0e48101bc1a8ab366a3287f1f36b4464879e36495ea7e5a53` |
| `../device_apps/exllm-public/exllm.d01` | 50,176 | `8aed597cc8fa8383fa2b1e0acb3ba670acb0198b3ecb4cecf2d01be3d6c63144` |

The EXLLM8 int8 quality gate passed all semantic, calculator, fuzz and
sampling cases. The separate fixed-point device reference also produced the
expected Japanese answers for all four bundled prompts. The device UI accepts
romaji keyboard input and redraws the answer as complete UTF-8 tokens arrive.
The menu distinguishes free input, example questions and device information;
its title does not contain a hard-coded device model name.
The generation screen preserves the submitted question above a divider and
streams the answer below it. Repeated SYMBOL presses cycle through Japanese
question/period/comma, exclamation and prolonged-sound marks.
The HISTORY key captures the complete 528×320 framebuffer. It appends a
timestamped JPEG under `crd0:/DCIM` when microSD is available, with a
timestamped RGB565 BMP in the app `_USER` directory as the automatic fallback.
