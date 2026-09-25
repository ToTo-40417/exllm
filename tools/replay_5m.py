#!/usr/bin/env python3
"""Replay the recorded EXLLM 5M training stages."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command(stage):
    if stage["runner"] == "src.train_large":
        return [
            sys.executable, "-m", "src.train_large", "--epochs", str(stage["epochs"]),
            "--batch", str(stage["batch"]), "--threads", str(stage["threads"]),
            "--lr", str(stage["learning_rate"]), "--data", stage["data"],
            "--valid", stage["valid"], "--out", stage["output"],
            "--parent", stage["input"], "--parent-config", "config-v1.0.json",
            "--d-model", str(stage["d_model"]), "--layers", str(stage["layers"]),
            "--heads", str(stage["heads"]), "--d-ff", str(stage["d_ff"]),
        ]
    return [
        sys.executable, "-m", "src.finetune", "--in-ckpt", stage["input"],
        "--out-ckpt", stage["output"], "--data", stage["data"],
        "--steps", str(stage["steps"]), "--batch", str(stage["batch"]),
        "--lr", str(stage["learning_rate"]), "--threads", str(stage["threads"]),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "training/release-5m-stages.json")
    parser.add_argument("--from-stage", default="expand-and-train-5m")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for relative, expected in manifest["input_artifacts"].items():
        path = ROOT / relative
        if not path.exists() or digest(path) != expected:
            raise SystemExit(f"input artifact SHA-256 mismatch: {relative}")
    names = [stage["name"] for stage in manifest["stages"]]
    if args.from_stage not in names:
        raise SystemExit(f"unknown stage: {args.from_stage}")
    start = names.index(args.from_stage)
    parent = ROOT / manifest["stages"][start]["input"]
    if not parent.exists():
        raise SystemExit(f"missing stage input: {parent}")
    for index, stage in enumerate(manifest["stages"][start:], start=start):
        if not args.dry_run or index == start:
            input_path = ROOT / stage["input"]
            expected_input = manifest["parent"]["sha256"] if index == 0 else manifest["stages"][index - 1]["output_sha256"]
            actual_input = digest(input_path)
            if actual_input != expected_input:
                raise SystemExit(f"input SHA-256 mismatch for {stage['name']}: {actual_input}")
        cmd = command(stage)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)
        actual = digest(ROOT / stage["output"])
        expected = stage["output_sha256"]
        print(json.dumps({"stage": stage["name"], "sha256": actual, "expected": expected, "match": actual == expected}))
        if actual != expected:
            raise SystemExit(f"output SHA-256 mismatch after {stage['name']}")


if __name__ == "__main__":
    main()
