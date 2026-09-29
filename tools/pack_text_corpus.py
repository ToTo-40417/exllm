#!/usr/bin/env python3
"""Deduplicate and pack raw Japanese documents into uint16 token streams."""

import argparse
import hashlib
import importlib.util
import json
from array import array
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
tokenizer_spec = importlib.util.spec_from_file_location("exllm_tokenizer", ROOT / "src/tokenizer.py")
tokenizer_module = importlib.util.module_from_spec(tokenizer_spec)
tokenizer_spec.loader.exec_module(tokenizer_module)
HybridTokenizer = tokenizer_module.HybridTokenizer
normalize_text = tokenizer_module.normalize_text


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_documents(paths, field):
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    value = json.loads(line).get(field, "")
                else:
                    value = line
                text = normalize_text(value) if isinstance(value, str) else ""
                if text:
                    yield path, line_number, text


def write_u16(stream, values):
    data = array("H", values)
    if sys.byteorder != "little":
        data.byteswap()
    data.tofile(stream)


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tokenizer_stats(tokenizer, text):
    """Measure tokenizer cost without confusing ordinary ASCII bytes with fallback."""
    encoded = tokenizer.encode_text(text)
    fallback_characters = 0
    fallback_byte_tokens = 0
    atomic_characters = 0
    ascii_byte_tokens = 0
    for character in text:
        if character in tokenizer.char_to_id:
            atomic_characters += 1
        elif ord(character) < 128:
            ascii_byte_tokens += 1
        else:
            fallback_characters += 1
            fallback_byte_tokens += len(character.encode("utf-8"))
    return encoded, {
        "unicode_characters": len(text),
        "atomic_characters": atomic_characters,
        "ascii_byte_tokens": ascii_byte_tokens,
        "non_ascii_fallback_characters": fallback_characters,
        "non_ascii_fallback_byte_tokens": fallback_byte_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.json")
    parser.add_argument("--field", default="text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-permyriad", type=int, default=100)
    parser.add_argument("--source-id", default="unspecified")
    args = parser.parse_args()
    if not 0 < args.validation_permyriad < 10000:
        raise SystemExit("--validation-permyriad must be between 1 and 9999")

    tokenizer = HybridTokenizer.load(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: args.output_dir / f"raw-{name}.u16" for name in ("train", "validation")}
    streams = {name: path.open("wb") for name, path in outputs.items()}

    seen = set()
    report = {
        "format": "little-endian uint16 token IDs",
        "tokenizer": str(args.tokenizer),
        "source_id": args.source_id,
        "inputs": [str(path) for path in args.inputs],
        "documents_seen": 0,
        "documents_unique": 0,
        "documents_duplicate": 0,
        "normalization": "NFC via src/tokenizer.py normalize_text",
        "splits": {
            "train": {"documents": 0, "processed_tokens": 0, "loss_bearing_tokens": 0, "padding_tokens": 0, "truncated_tokens": 0},
            "validation": {"documents": 0, "processed_tokens": 0, "loss_bearing_tokens": 0, "padding_tokens": 0, "truncated_tokens": 0},
        },
        "tokenizer_audit": {
            "unicode_characters": 0,
            "atomic_characters": 0,
            "ascii_byte_tokens": 0,
            "non_ascii_fallback_characters": 0,
            "non_ascii_fallback_byte_tokens": 0,
            "documents_at_or_above_128_tokens": 0,
        },
    }
    try:
        for _path, _line, text in iter_documents(args.inputs, args.field):
            report["documents_seen"] += 1
            digest = digest_text(text)
            if digest in seen:
                report["documents_duplicate"] += 1
                continue
            seen.add(digest)
            split = "validation" if int(digest[:8], 16) % 10000 < args.validation_permyriad else "train"
            encoded, stats = tokenizer_stats(tokenizer, text)
            tokens = [tokenizer.BOS, *encoded, tokenizer.EOS]
            if max(tokens) >= 65536:
                raise SystemExit("token ID does not fit uint16")
            write_u16(streams[split], tokens)
            report["documents_unique"] += 1
            report["splits"][split]["documents"] += 1
            report["splits"][split]["processed_tokens"] += len(tokens)
            report["splits"][split]["loss_bearing_tokens"] += len(tokens)
            for key, value in stats.items():
                report["tokenizer_audit"][key] += value
            if len(tokens) >= 128:
                report["tokenizer_audit"]["documents_at_or_above_128_tokens"] += 1
    finally:
        for stream in streams.values():
            stream.close()

    for split, path in outputs.items():
        report["splits"][split]["path"] = str(path)
        report["splits"][split]["bytes"] = path.stat().st_size
        report["splits"][split]["sha256"] = digest_file(path)
    characters = report["tokenizer_audit"]["unicode_characters"]
    text_tokens = sum(item["processed_tokens"] for item in report["splits"].values()) - 2 * report["documents_unique"]
    report["tokenizer_audit"]["text_tokens_per_unicode_character"] = text_tokens / max(1, characters)
    report["tokenizer_audit"]["non_ascii_fallback_character_rate"] = (
        report["tokenizer_audit"]["non_ascii_fallback_characters"] / max(1, characters)
    )
    manifest = args.output_dir / "raw-corpus.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
