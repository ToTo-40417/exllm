#!/usr/bin/env python3
"""Replace general.size_label in a GGUF file without touching tensor data."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING = 8
ARRAY = 9


class Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise ValueError("truncated GGUF")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> tuple[str, int, int]:
        length_pos = self.pos
        length = self.u64()
        value_pos = self.pos
        return self.take(length).decode("utf-8"), length_pos, value_pos


def skip_value(cursor: Cursor, value_type: int) -> None:
    if value_type in VALUE_SIZES:
        cursor.take(VALUE_SIZES[value_type])
    elif value_type == STRING:
        cursor.string()
    elif value_type == ARRAY:
        element_type = cursor.u32()
        count = cursor.u64()
        for _ in range(count):
            skip_value(cursor, element_type)
    else:
        raise ValueError(f"unsupported GGUF value type: {value_type}")


def replace_label(path: Path, label: str) -> tuple[str, str]:
    data = path.read_bytes()
    cursor = Cursor(data)
    if cursor.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = cursor.u32()
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    tensor_count = cursor.u64()
    metadata_count = cursor.u64()

    alignment = 32
    target = None
    old_label = None
    for _ in range(metadata_count):
        key, _, _ = cursor.string()
        value_type = cursor.u32()
        if key == "general.alignment":
            if value_type != 4:
                raise ValueError("general.alignment is not uint32")
            alignment = cursor.u32()
        elif key == "general.size_label":
            if value_type != STRING:
                raise ValueError("general.size_label is not a string")
            old_label, length_pos, value_pos = cursor.string()
            target = (length_pos, value_pos, len(old_label.encode("utf-8")))
        else:
            skip_value(cursor, value_type)

    for _ in range(tensor_count):
        cursor.string()
        dimensions = cursor.u32()
        cursor.take(8 * dimensions)
        cursor.take(4 + 8)

    old_data_offset = (cursor.pos + alignment - 1) // alignment * alignment
    if target is None or old_label is None:
        raise ValueError("general.size_label not found")

    encoded = label.encode("utf-8")
    length_pos, value_pos, old_length = target
    prefix = data[:length_pos] + struct.pack("<Q", len(encoded)) + encoded + data[value_pos + old_length:cursor.pos]
    new_data_offset = (len(prefix) + alignment - 1) // alignment * alignment
    output = prefix + bytes(new_data_offset - len(prefix)) + data[old_data_offset:]
    path.write_bytes(output)
    return old_label, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    old, new = replace_label(args.path, args.label)
    print(f"{args.path}: {old} -> {new}")


if __name__ == "__main__":
    main()
