#!/usr/bin/env python3
"""Build deterministic, redistributable JPTOEN companion train/validation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

KNOWN = [
    "{ja}をえいごでいうと？", "{ja}をえいごで", "{ja}はえいごでなに？",
    "{ja}ってえいごでは？", "{ja}をえいやくして", "{ja}のえいごは？",
    "{ja}をえいごにして", "{ja}のえいごをおしえて",
    "えいごでは{ja}をなんという？", "{ja}のえいごだけおしえて",
    "このことばをえいごにすると？ {ja}",
    "ほんやくしたいことばは{ja}です。えいごで？",
    "みじかくこたえて。{ja}はえいごで？",
]
NOISE = [
    "きょうはいいてんきです。{ja}をえいごでいうと？",
    "まえのはなしはむしして。{ja}だけえいごで",
    "ながいせつめいはいりません。{ja}はえいごでなに？",
    "こたえだけでいいです。{ja}をえいごで",
]
HELDOUT = [
    "{ja}をえいごにやくすと？", "えいやくすると{ja}はなに？",
    "{ja}のえいごひょうげんは？",
]
GENERAL = [
    ("こんにちは", "hello"), ("あなたは何というモデルですか？", "unknown"),
    ("オフラインとは何ですか？", "unknown"), ("きょうのてんきは？", "unknown"),
    ("これをせつめいして", "unknown"), ("なにができますか？", "unknown"),
    ("たすけてください", "unknown"),
]
SYL = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわがぎぐげござじずぜぞばびぶべぼぱぴぷぺぽ")


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pairs(path: Path) -> list[tuple[str, str]]:
    return [tuple(line.split("\t", 1)) for line in lines(path)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40417002)
    parser.add_argument("--nonce-count", type=int, default=4096)
    args = parser.parse_args()
    lexicon_path = args.source / "release_lexicon.tsv"
    ambiguous_path = args.source / "ambiguous_kana.txt"
    unknown_train_path = args.source / "real_unknown_train.txt"
    unknown_holdout_path = args.source / "real_unknown_holdout.txt"
    lexicon = pairs(lexicon_path)
    ambiguous = lines(ambiguous_path)
    unknown_train = lines(unknown_train_path)
    unknown_holdout = lines(unknown_holdout_path)
    blocked = {ja for ja, _ in lexicon} | set(ambiguous) | set(unknown_train) | set(unknown_holdout)
    train = []
    valid = []
    for ja, en in lexicon:
        train.append({"prompt": ja, "answer": en, "category": "known-bare"})
        train.extend({"prompt": template.format(ja=ja), "answer": en, "category": "known-query"} for template in KNOWN)
        train.extend({"prompt": template.format(ja=ja), "answer": en, "category": "known-noise"} for template in NOISE)
        valid.extend({"prompt": template.format(ja=ja), "answer": en, "category": "known-heldout-template"} for template in HELDOUT)
    for word in unknown_train + ambiguous:
        train.extend({"prompt": template.format(ja=word), "answer": "unknown", "category": "unknown-calibration"} for template in KNOWN)
    rng = random.Random(args.seed)
    nonces = set()
    while len(nonces) < args.nonce_count:
        value = "".join(rng.choice(SYL) for _ in range(rng.randint(3, 7)))
        if value not in blocked:
            nonces.add(value)
    for index, word in enumerate(sorted(nonces)):
        train.append({"prompt": KNOWN[index % len(KNOWN)].format(ja=word), "answer": "unknown", "category": "nonce-unknown"})
    train.extend({"prompt": prompt, "answer": answer, "category": "general"} for prompt, answer in GENERAL)
    for word in unknown_holdout:
        valid.append({"prompt": HELDOUT[0].format(ja=word), "answer": "unknown", "category": "real-unknown-heldout"})
    train.sort(key=lambda row: hashlib.sha256((row["prompt"] + "\0" + row["answer"]).encode()).hexdigest())
    valid.sort(key=lambda row: hashlib.sha256((row["prompt"] + "\0" + row["answer"]).encode()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    valid_path = args.out / "valid.jsonl"
    write(train_path, train)
    write(valid_path, valid)
    manifest = {
        "name": "EXLLM-JPTOEN GGUF companion dataset",
        "seed": args.seed,
        "selection": "deterministic expansion from project-owned lexicon and authored templates",
        "train_records": len(train),
        "validation_records": len(valid),
        "nonce_count": len(nonces),
        "sources": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (lexicon_path, ambiguous_path, unknown_train_path, unknown_holdout_path)
        ],
        "outputs": {
            "train.jsonl": {"bytes": train_path.stat().st_size, "sha256": digest(train_path)},
            "valid.jsonl": {"bytes": valid_path.stat().st_size, "sha256": digest(valid_path)},
        },
    }
    (args.out / "dataset-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
