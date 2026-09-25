#!/usr/bin/env python3
"""Convert the published EXLLM8 artifact to the EX-word EXQ12 format."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

MAGIC_IN = b"EXLLM8\0\0"
MAGIC_OUT = b"EXQ12\0\0\0"


def read_exllm8(path):
    raw = path.read_bytes()
    if raw[:8] != MAGIC_IN:
        raise ValueError("input is not EXLLM8")
    offset = 8
    version, count = struct.unpack_from("<II", raw, offset)
    offset += 8
    if version != 1:
        raise ValueError(f"unsupported EXLLM8 version: {version}")
    records = []
    for _ in range(count):
        name_len, ndim, qtype = struct.unpack_from("<HBB", raw, offset)
        offset += 4
        name = raw[offset : offset + name_len]
        offset += name_len
        dims = struct.unpack_from("<" + "I" * ndim, raw, offset)
        offset += 4 * ndim
        scale_count, payload_bytes = struct.unpack_from("<II", raw, offset)
        offset += 8
        scales = struct.unpack_from("<" + "f" * scale_count, raw, offset) if scale_count else ()
        offset += 4 * scale_count
        payload = raw[offset : offset + payload_bytes]
        offset += payload_bytes
        records.append((name, qtype, dims, scales, payload))
    if offset != len(raw):
        raise ValueError("trailing bytes in EXLLM8 input")
    return records


def convert(source, output):
    records = read_exllm8(source)
    with output.open("wb") as stream:
        stream.write(MAGIC_OUT)
        stream.write(struct.pack("<II", 1, len(records)))
        for name, qtype, dims, scales, payload in records:
            if qtype == 1:
                fixed_scales = [max(1, min(0x7FFFFFFF, round(scale * (1 << 20)))) for scale in scales]
                output_qtype = 1
            elif qtype == 2:
                values = struct.unpack("<" + "e" * (len(payload) // 2), payload)
                payload = struct.pack(
                    "<" + "h" * len(values),
                    *(max(-32768, min(32767, round(value * 4096))) for value in values),
                )
                fixed_scales = []
                output_qtype = 3
            else:
                raise ValueError(f"unsupported EXLLM8 qtype: {qtype}")
            stream.write(struct.pack("<HBB", len(name), len(dims), output_qtype))
            stream.write(name)
            stream.write(struct.pack("<" + "I" * len(dims), *dims))
            stream.write(struct.pack("<II", len(fixed_scales), len(payload)))
            if fixed_scales:
                stream.write(struct.pack("<" + "i" * len(fixed_scales), *fixed_scales))
            stream.write(payload)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("weights/EXLLM-v1.1-5m-int8.bin"))
    parser.add_argument("--output", type=Path, default=Path("weights/model.q12"))
    parser.add_argument(
        "--expect-sha256",
        default="d64037fde791e5c0e48101bc1a8ab366a3287f1f36b4464879e36495ea7e5a53",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_count = convert(args.input, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    result = {"output": str(args.output), "bytes": args.output.stat().st_size, "sha256": digest, "tensors": tensor_count}
    print(json.dumps(result, indent=2))
    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit("EXQ12 SHA-256 mismatch")


if __name__ == "__main__":
    main()
