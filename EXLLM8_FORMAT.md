# EXLLM8 binary format v1

All integer fields are little-endian.

## File header

| Field | Type | Value |
|---|---|---|
| magic | 8 bytes | `EXLLM8\0\0` |
| format_version | uint32 | `1` |
| tensor_count | uint32 | number of serialized tensors |

## Tensor record

Each tensor is stored as:

1. `name_len`: uint16
2. `ndim`: uint8
3. `qtype`: uint8
4. `name`: `name_len` UTF-8 bytes
5. `dims`: `ndim` × uint32
6. `scale_count`: uint32
7. `data_bytes`: uint32
8. `scales`: `scale_count` × float32
9. `data`: `data_bytes` bytes

### qtype 1 — row-wise int8

Used for 2D matrices. If a weight is shaped `[rows, cols]`, one float32 symmetric scale is stored per row and the matrix data is signed int8.

Approximate reconstruction:

```text
weight[row, col] = int8_value[row, col] * scale[row]
```

Quantized values are limited to `[-127, 127]`.

### qtype 2 — float16

Used for the small RMSNorm vectors. `scale_count` is zero and `data` is IEEE-754 binary16.

## Weight tying

`lm_head.weight` is not stored. It aliases `tok.weight`. The companion manifest declares this alias explicitly.

## Deployment note

The float scales are an interchange representation. A SH4/no-FPU deployment should translate them into the fixed-point scale representation chosen by the C inference kernel.
