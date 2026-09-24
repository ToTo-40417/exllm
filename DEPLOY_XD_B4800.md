# XD-B4800 Deployment Notes

EXLLM v1.0.0 was shaped for the CASIO EX-word XD-B4800 / DATAPLUS 6 homebrew project, but the final C/SH4 runtime still has to be integrated and measured on the physical unit.

## Deployment weight footprint

`weights/EXLLM-v1.0.0-int8.bin` is about 1.25 MB and contains the model tensors once; the tied LM head is not duplicated.

At context length 128, a straightforward KV cache needs approximately:

- int8 K+V: `4 layers × 2 × 128 × 160 = 163,840 bytes`;
- int16 K+V: `327,680 bytes`.

Small activation/scratch buffers are additional. If all int8 weights are resident in RAM, the model + int8 KV cache starts around 1.42 MB before runtime code, allocator overhead, scales, framebuffer interaction, and scratch buffers. Streaming some weights from storage can reduce resident RAM at the cost of speed.

**The actual RAM available to a homebrew process on the XD-B4800 has not been established by this model build. Measure it on the unit before fixing allocator limits.**

## Approximate compute per generated token

With KV caching, dense weight projections are approximately:

- transformer blocks: ~1.065M multiply-accumulates/token;
- tied vocabulary projection: ~0.139M MAC/token;
- attention at full context 128: up to ~0.164M dot-product MAC/token.

So the full-context order of magnitude is ~1.37M MAC per generated token, excluding normalization, softmax, quantization/dequantization bookkeeping, sampling, and UI work.

## Required device kernels

A practical C runtime needs:

1. token + position embedding lookup;
2. RMSNorm fixed-point implementation;
3. int8 matrix-vector multiply with wider accumulator;
4. Q/K/V projection and KV cache;
5. causal attention and integer/fixed-point softmax approximation;
6. ReLU FFN;
7. tied vocabulary projection;
8. UTF-8-constrained token sampling;
9. the small deterministic integer calculator from `src/runtime.py`;
10. UI/key input integration.

## No-FPU consideration

The EXLLM8 interchange file stores per-row scales as float32. This is convenient for verification, but a no-FPU SH4 build should convert those scale values into fixed-point multipliers/shifts during the host-side deployment build. The model architecture itself does not require Python or PyTorch on the device.

## Recommended bring-up order

1. parse EXLLM8 and validate hashes/tensor shapes on the host;
2. implement int8 matvec and compare against Python;
3. implement embedding/RMSNorm/one block;
4. compare one-token logits against the reference model;
5. add KV cache and full generation;
6. port UTF-8 output constraints and calculator routing;
7. integrate the EX-word UI;
8. benchmark token latency and maximum resident memory on the actual XD-B4800.

Do not assume the published model package alone proves device execution: it proves a size/compute target and reference behavior. Physical-device validation remains the final deployment step.
