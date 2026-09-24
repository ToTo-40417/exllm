#!/usr/bin/env python3
"""Reproducible timing probe for the EXLLM 5M PyTorch checkpoint."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import bad_text  # noqa: E402
from src.model import EXLLM, EXLLMConfig  # noqa: E402
from src.tokenizer import HybridTokenizer, UTF8State  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load(checkpoint, device):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EXLLM(EXLLMConfig(**data["config"]))
    model.load_state_dict(data["model"])
    return model.eval().to(device), HybridTokenizer.load(ROOT / "tokenizer.json")


@torch.inference_mode()
def generate_timed(model, tok, prompt, device, max_new=48):
    ids = tok.encode_user(prompt)
    reserve = min(max_new, model.cfg.max_seq_len // 2)
    if len(ids) > model.cfg.max_seq_len - reserve:
        keep = model.cfg.max_seq_len - reserve - 3
        ids = [tok.BOS, tok.USER, *ids[2:-1][-max(1, keep):], tok.ASSIST]

    out, state, token_ms = [], UTF8State(), []
    sync(device)
    start = time.perf_counter_ns()
    ttft_ns = None
    for _ in range(min(max_new, model.cfg.max_seq_len - len(ids))):
        x = torch.tensor([ids + out], dtype=torch.long, device=device)
        logits = model(x)[0, -1].clone()
        logits[tok.PAD] = logits[tok.BOS] = logits[tok.USER] = logits[tok.ASSIST] = -1e30
        valid = torch.zeros_like(logits, dtype=torch.bool)
        if state.complete:
            valid[tok.EOS] = True
            for token in range(256):
                if state.accepts_byte(token):
                    valid[token] = True
            if tok.chars:
                valid[256:256 + len(tok.chars)] = True
        else:
            for token in range(256):
                if state.accepts_byte(token):
                    valid[token] = True
        logits[~valid] = -1e30
        token = int(torch.argmax(logits).item())
        sync(device)
        now = time.perf_counter_ns()
        if ttft_ns is None:
            ttft_ns = now - start
        token_ms.append((now - start) / 1e6 if len(token_ms) == 0 else 0.0)
        if token == tok.EOS:
            break
        if token < 256:
            state.push(token)
        elif not state.complete:
            continue
        out.append(token)
    sync(device)
    end = time.perf_counter_ns()
    while out:
        try:
            text = tok.decode(out)
            break
        except UnicodeDecodeError:
            out.pop()
    else:
        text = ""
    total_s = (end - start) / 1e9
    generated = len(out)
    decode_s = max(0.0, total_s - (ttft_ns or 0) / 1e9)
    return {
        "prompt": prompt,
        "input_tokens": len(ids),
        "output_tokens": generated,
        "ttft_ms": round((ttft_ns or 0) / 1e6, 3),
        "total_ms": round(total_s * 1000, 3),
        "decode_tokens_per_second": round(max(0, generated - 1) / decode_s, 3) if decode_s else None,
        "text": text.strip(),
        "bad_text": bad_text(text.strip()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "weights" / "EXLLM-v1.1-5m-release3.pt"))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    started = time.perf_counter()
    model, tok = load(args.checkpoint, device)
    sync(device)
    load_s = time.perf_counter() - started
    prompts = ["こんにちは", "あなたは何というモデルですか？", "RAMとは何ですか？", "オフラインとは何ですか？"]
    _ = generate_timed(model, tok, prompts[0], device, 16)  # warm-up
    runs = [generate_timed(model, tok, p, device) for p in prompts for _ in range(args.repeats)]
    report = {
        "schema": "exllm-benchmark-v1",
        "checkpoint": Path(args.checkpoint).name,
        "parameters": model.num_parameters(),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "load_seconds": round(load_s, 4),
        "gpu": None,
        "runs": runs,
        "summary": {
            "median_ttft_ms": round(statistics.median(x["ttft_ms"] for x in runs), 3),
            "median_total_ms": round(statistics.median(x["total_ms"] for x in runs), 3),
            "median_decode_tokens_per_second": round(statistics.median(x["decode_tokens_per_second"] for x in runs if x["decode_tokens_per_second"] is not None), 3),
        },
    }
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": prop.name,
            "vram_bytes": prop.total_memory,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
