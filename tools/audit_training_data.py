#!/usr/bin/env python3
"""Audit an EXLLM experiment corpus without importing PyTorch."""

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = ("prompt", "answer", "category")


def load_tokenizer(path):
    spec = importlib.util.spec_from_file_location("exllm_tokenizer", ROOT / "src/tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridTokenizer.load(path)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_file(relative, split):
    path = ROOT / relative
    records = []
    errors = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"file": relative, "line": line_number, "error": str(exc)})
                continue
            missing = [field for field in REQUIRED_FIELDS if not isinstance(record.get(field), str) or not record[field]]
            if missing:
                errors.append({"file": relative, "line": line_number, "error": "missing/non-string fields", "fields": missing})
                continue
            records.append({
                "split": split,
                "file": relative,
                "line": line_number,
                "prompt": record["prompt"],
                "answer": record["answer"],
                "category": record["category"],
            })
    return records, errors


def summarize(records, tokenizer=None, context_tokens=128):
    pairs = collections.Counter((item["prompt"], item["answer"]) for item in records)
    prompts = collections.defaultdict(set)
    for item in records:
        prompts[item["prompt"]].add(item["answer"])
    categories = collections.Counter(item["category"] for item in records)
    prompt_lengths = [len(item["prompt"]) for item in records]
    answer_lengths = [len(item["answer"]) for item in records]
    conflicts = [
        {"prompt": prompt, "answers": sorted(answers)}
        for prompt, answers in prompts.items()
        if len(answers) > 1
    ]
    summary = {
        "records": len(records),
        "unique_prompt_answer_pairs": len(pairs),
        "duplicate_records": sum(count - 1 for count in pairs.values()),
        "unique_prompts": len(prompts),
        "conflicting_prompts": len(conflicts),
        "conflict_examples": conflicts[:50],
        "categories": dict(sorted(categories.items())),
        "character_lengths": {
            "prompt_max": max(prompt_lengths, default=0),
            "answer_max": max(answer_lengths, default=0),
            "prompt_mean": sum(prompt_lengths) / max(1, len(prompt_lengths)),
            "answer_mean": sum(answer_lengths) / max(1, len(answer_lengths)),
        },
    }
    if tokenizer is not None:
        processed = 0
        loss_bearing = 0
        padding = 0
        truncated = 0
        fallback_characters = 0
        fallback_byte_tokens = 0
        unicode_characters = 0
        at_context_limit = 0
        for item in records:
            prompt_tokens = tokenizer.encode_text(item["prompt"])
            answer_tokens = tokenizer.encode_text(item["answer"])
            untrimmed = 4 + len(prompt_tokens) + len(answer_tokens)
            sequence = tokenizer.encode_example(item["prompt"], item["answer"], context_tokens)
            processed += len(sequence)
            assist_index = sequence.index(tokenizer.ASSIST)
            loss_bearing += len(sequence) - assist_index - 1
            padding += context_tokens - len(sequence)
            truncated += max(0, untrimmed - len(sequence))
            at_context_limit += int(len(sequence) == context_tokens)
            for character in item["prompt"] + item["answer"]:
                unicode_characters += 1
                if ord(character) >= 128 and character not in tokenizer.char_to_id:
                    fallback_characters += 1
                    fallback_byte_tokens += len(character.encode("utf-8"))
        summary["token_accounting"] = {
            "processed_tokens": processed,
            "loss_bearing_tokens": loss_bearing,
            "padding_tokens_if_batched_to_context": padding,
            "truncated_tokens": truncated,
            "records_at_context_limit": at_context_limit,
        }
        summary["tokenizer_audit"] = {
            "unicode_characters": unicode_characters,
            "non_ascii_fallback_characters": fallback_characters,
            "non_ascii_fallback_byte_tokens": fallback_byte_tokens,
            "non_ascii_fallback_character_rate": fallback_characters / max(1, unicode_characters),
            "processed_tokens_per_unicode_character": processed / max(1, unicode_characters),
        }
    return summary


def verify_artifact(path_text, expected, label, failures):
    path = ROOT / path_text
    if not path.is_file():
        failures.append(f"missing {label}: {path_text}")
        return {"path": path_text, "exists": False, "sha256": None, "expected": expected, "match": False}
    actual = sha256(path)
    if actual != expected:
        failures.append(f"SHA-256 mismatch for {label}: {path_text}")
    return {"path": path_text, "exists": True, "sha256": actual, "expected": expected, "match": actual == expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(ROOT / manifest["runtime_contract"]["tokenizer"])
    context_tokens = manifest["runtime_contract"]["context_tokens"]
    failures = []
    artifacts = {
        "parent": verify_artifact(manifest["lineage"]["parent"], manifest["lineage"]["parent_sha256"], "parent", failures),
        "config": verify_artifact(manifest["runtime_contract"]["config"], manifest["runtime_contract"]["config_sha256"], "config", failures),
        "tokenizer": verify_artifact(manifest["runtime_contract"]["tokenizer"], manifest["runtime_contract"]["tokenizer_sha256"], "tokenizer", failures),
    }

    train_records = []
    valid_records = []
    errors = []
    files = []
    for split, target in (("train", train_records), ("validation", valid_records)):
        paths = list(manifest["corpus"][split])
        if split == "train":
            paths.extend(manifest["corpus"].get("new_records", []))
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                errors.append({"file": relative, "error": "missing file"})
                continue
            loaded, file_errors = load_file(relative, split)
            target.extend(loaded)
            errors.extend(file_errors)
            files.append({"split": split, "path": relative, "bytes": path.stat().st_size, "sha256": sha256(path), "records": len(loaded)})

    overlap = sorted(set(item["prompt"] for item in train_records) & set(item["prompt"] for item in valid_records))
    if errors:
        failures.append(f"{len(errors)} malformed or missing corpus entries")
    if overlap and not args.allow_overlap:
        failures.append(f"{len(overlap)} prompts overlap between train and validation")

    report = {
        "experiment_id": manifest["experiment_id"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "artifacts": artifacts,
        "files": files,
        "train": summarize(train_records, tokenizer, context_tokens),
        "validation": summarize(valid_records, tokenizer, context_tokens),
        "train_validation_prompt_overlap": {"count": len(overlap), "examples": overlap[:50]},
        "record_errors": errors[:100],
        "pass": not failures,
        "failures": failures,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
