#!/usr/bin/env python3
"""Build the post-hoc exact-intersection table for EXLLM-JPTOEN.

This does not run inference. It compares the unpublished legacy parent with
the public-candidate evaluation, retaining byte-identical prompt/answer pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUITES = (
    ("bare_known", "bare_known", "bare_known"),
    ("ambiguous_unknown", "ambiguous_unknown", "ambiguous_unknown"),
    ("general_ood", "ood", "general_ood"),
    ("device_regression", "device_regression", "device_regression"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(cases: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["prompt"], row["expected"]): row for row in cases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-parent", type=Path, required=True)
    ap.add_argument("--jptoen", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    old = json.loads(args.legacy_parent.read_text(encoding="utf-8"))
    new = json.loads(args.jptoen.read_text(encoding="utf-8"))
    old_groups = old["suites"]
    new_groups = new["groups"]
    suites: dict[str, dict] = {}

    for label, old_name, new_name in SUITES:
        a = index(old_groups[old_name]["cases"])
        b = index(new_groups[new_name]["cases"])
        keys = sorted(a.keys() & b.keys())
        cases = []
        for prompt, expected in keys:
            cases.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "legacy_parent": {"got": a[(prompt, expected)]["got"], "pass": bool(a[(prompt, expected)]["pass"])},
                    "exllm_jptoen": {"got": b[(prompt, expected)]["got"], "pass": bool(b[(prompt, expected)]["pass"])},
                }
            )
        suites[label] = {
            "total": len(cases),
            "legacy_parent_pass": sum(row["legacy_parent"]["pass"] for row in cases),
            "exllm_jptoen_pass": sum(row["exllm_jptoen"]["pass"] for row in cases),
            "cases": cases,
        }

    result = {
        "schema": "exllm-jptoen-common-eval-v1",
        "method": "post-hoc exact intersection of prompt and expected answer; no inference was rerun",
        "inputs": {
            "legacy_parent": {"published": False, "sha256": sha256(args.legacy_parent)},
            "exllm_jptoen": {"sha256": sha256(args.jptoen)},
        },
        "suites": suites,
        "limitations": [
            "Only exact prompt/expected intersections are comparable.",
            "Held-out-template and real-unknown suites have no exact common cases and are intentionally excluded.",
            "The two evaluations were produced at different development stages.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
