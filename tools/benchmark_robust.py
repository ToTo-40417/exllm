#!/usr/bin/env python3
"""Warm EXLLM repeatedly, then report raw and Tukey-filtered timing medians."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch
from benchmark_5m import generate_timed, load, sync


def quartiles(values):
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return q1, q3


def tukey_keep(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    q1, q3 = quartiles(values)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [row for row in rows if row[key] is not None and low <= row[key] <= high], low, high


def median(rows, key):
    return round(statistics.median(row[key] for row in rows if row[key] is not None), 3)


device = torch.device("cuda")
torch.cuda.empty_cache()
started = time.perf_counter()
model, tokenizer = load(ROOT / "weights" / "EXLLM-v1.1-5m-release3.pt", device)
sync(device)
load_seconds = time.perf_counter() - started
prompts = ["こんにちは", "あなたは何というモデルですか？", "RAMとは何ですか？", "オフラインとは何ですか？"]

# Exercise every prompt shape before recording. These runs are intentionally discarded.
warmups = []
for _ in range(10):
    for prompt in prompts:
        warmups.append(generate_timed(model, tokenizer, prompt, device))

runs = []
for _ in range(30):
    for prompt in prompts:
        runs.append(generate_timed(model, tokenizer, prompt, device))

ttft_kept, ttft_low, ttft_high = tukey_keep(runs, "ttft_ms")
speed_kept, speed_low, speed_high = tukey_keep(runs, "decode_tokens_per_second")
total_kept, total_low, total_high = tukey_keep(runs, "total_ms")

report = {
    "schema": "exllm-benchmark-robust-v1",
    "checkpoint": "EXLLM-v1.1-5m-release3.pt",
    "parameters": model.num_parameters(),
    "torch": torch.__version__,
    "load_seconds": round(load_seconds, 4),
    "gpu": torch.cuda.get_device_name(0),
    "warmup_runs": len(warmups),
    "measured_runs": len(runs),
    "first_warmup": warmups[0],
    "last_warmup": warmups[-1],
    "raw": {
        "median_ttft_ms": median(runs, "ttft_ms"),
        "median_total_ms": median(runs, "total_ms"),
        "median_decode_tokens_per_second": median(runs, "decode_tokens_per_second"),
    },
    "tukey_1_5_iqr": {
        "ttft": {"kept": len(ttft_kept), "removed": len(runs) - len(ttft_kept), "fence": [ttft_low, ttft_high], "median_ms": median(ttft_kept, "ttft_ms")},
        "total": {"kept": len(total_kept), "removed": len(runs) - len(total_kept), "fence": [total_low, total_high], "median_ms": median(total_kept, "total_ms")},
        "decode": {"kept": len(speed_kept), "removed": len(runs) - len(speed_kept), "fence": [speed_low, speed_high], "median_tokens_per_second": median(speed_kept, "decode_tokens_per_second")},
    },
    "runs": runs,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
