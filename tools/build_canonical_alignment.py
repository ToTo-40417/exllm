#!/usr/bin/env python3
"""Build one deterministic target per prompt from ordered EXLLM JSONL shards."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="highest-priority shard first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidates = collections.defaultdict(list)
    input_report = []
    for priority, path in enumerate(args.inputs):
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                prompt = record["prompt"].strip()
                answer = record["answer"].strip()
                if prompt and answer:
                    candidates[prompt].append((priority, answer, record.get("category", "unknown"), path.name, line_number))
                    count += 1
        input_report.append({"path": str(path), "priority": priority, "records": count, "sha256": sha256(path)})

    selected = []
    conflicts = 0
    for prompt, values in candidates.items():
        answers = {value[1] for value in values}
        conflicts += int(len(answers) > 1)
        best_priority = min(value[0] for value in values)
        preferred = [value for value in values if value[0] == best_priority]
        counts = collections.Counter(value[1] for value in preferred)
        answer = sorted(counts, key=lambda item: (-counts[item], item))[0]
        winner = next(value for value in preferred if value[1] == answer)
        selected.append({"prompt": prompt, "answer": answer, "category": winner[2], "source_file": winner[3]})
    selected.sort(key=lambda item: hashlib.sha256(item["prompt"].encode()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "selection": "highest-priority shard; within that shard most frequent answer; lexical tie-break",
        "inputs": input_report,
        "unique_prompts": len(candidates),
        "prompts_with_multiple_answers_across_inputs": conflicts,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "output_records": len(selected),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
