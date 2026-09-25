# EXQ12 deployment format

`model.q12` is produced deterministically from the published EXLLM8 file by
`tools/export_exq12.py`. It contains the same 39 tensors in the order stored by
EXLLM8. The EX-word runtime uses integer matrix weights and fixed-point scales;
the file is not GGUF and is not interchangeable with EXLLM8.

## Header

- 8 bytes: `EXQ12\0\0\0`
- little-endian `uint32`: format version (`1`)
- little-endian `uint32`: tensor count (`39` for the 5M release)

Each tensor record contains a little-endian `uint16` name length, `uint8`
dimension count, `uint8` qtype, UTF-8 name bytes, `uint32` dimensions, `uint32`
scale count, `uint32` payload byte count, optional scale data, and the payload.

- qtype `1`: signed int8 matrix payload; one signed Q20 `int32` scale per row.
- qtype `3`: signed Q12 `int16` vector payload; no separate scale table.

## Reproduce the release artifact

```bash
python tools/export_exq12.py
```

Expected result:

```text
bytes: 5443105
sha256: d64037fde791e5c0e48101bc1a8ab366a3287f1f36b4464879e36495ea7e5a53
tensors: 39
```
