"""Continue EXLLM causal-language training from packed uint16 token streams."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import EXLLM, EXLLMConfig


class PackedWindows(Dataset):
    def __init__(self, path, sequence_length):
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.sequence_length = sequence_length
        self.count = max(0, (len(self.tokens) - 1) // sequence_length)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index * self.sequence_length
        values = np.asarray(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        return torch.from_numpy(values[:-1].copy()), torch.from_numpy(values[1:].copy())


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for index, (inputs, labels) in enumerate(loader):
        if index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tokens += labels.numel()
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=0, help="stop after at least this many training tokens; 0 uses all epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    checkpoint = torch.load(args.parent, map_location="cpu", weights_only=False)
    config = EXLLMConfig(**checkpoint["config"])
    model = EXLLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    train_data = PackedWindows(args.train, config.max_seq_len)
    valid_data = PackedWindows(args.validation, config.max_seq_len)
    if not train_data or not valid_data:
        raise SystemExit("packed train and validation streams must each contain at least one full window")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", generator=generator)
    valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.03)
    total_steps = args.epochs * len(train_loader)
    if args.max_tokens:
        tokens_per_full_batch = args.batch * config.max_seq_len
        total_steps = min(total_steps, math.ceil(args.max_tokens / tokens_per_full_batch))
    warmup = max(20, int(total_steps * 0.03))
    global_step = 0
    processed_tokens = 0
    loss_bearing_tokens = 0
    started = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            if global_step >= total_steps:
                break
            global_step += 1
            progress = max(0.0, (global_step - warmup) / max(1, total_steps - warmup))
            scale = global_step / warmup if global_step <= warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_tokens = labels.numel()
            processed_tokens += batch_tokens
            loss_bearing_tokens += batch_tokens
            if global_step % args.log_every == 0:
                print(json.dumps({"step": global_step, "epoch": epoch, "loss": float(loss.detach()), "processed_tokens": processed_tokens, "loss_bearing_tokens": loss_bearing_tokens, "padding_tokens": 0, "truncated_tokens": 0, "tokens_per_sec": processed_tokens / max(1e-9, time.time() - started), "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        validation_loss = evaluate(model, valid_loader, device, amp_dtype, args.validation_batches)
        record = {"epoch": epoch, "step": global_step, "processed_tokens": processed_tokens, "loss_bearing_tokens": loss_bearing_tokens, "padding_tokens": 0, "truncated_tokens": 0, "validation_loss": validation_loss, "elapsed_sec": time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if global_step >= total_steps:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint.get("meta", {}))
    metadata.update({"experiment": "general-ja-5m-v1", "stage": "raw-language", "parent": str(args.parent), "seed": args.seed, "processed_tokens": processed_tokens, "loss_bearing_tokens": loss_bearing_tokens, "padding_tokens": 0, "truncated_tokens": 0})
    model.to("cpu")
    torch.save({"model": model.state_dict(), "config": config.__dict__, "meta": metadata, "history": history, "global_step": checkpoint.get("global_step", 0) + global_step}, args.output)
    print(json.dumps({"output": str(args.output), "processed_tokens": processed_tokens, "loss_bearing_tokens": loss_bearing_tokens, "padding_tokens": 0, "truncated_tokens": 0, "steps": global_step, "elapsed_sec": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
