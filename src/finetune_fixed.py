"""Fixed-tokenizer CUDA alignment for the General Japanese 5M production run."""

import argparse, json, math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import EXLLM, EXLLMConfig
from .tokenizer import HybridTokenizer
from .train import J, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ckpt", type=Path, required=True)
    ap.add_argument("--out-ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    config = EXLLMConfig(**checkpoint["config"])
    model = EXLLM(config); model.load_state_dict(checkpoint["model"]); model.to(device)
    tokenizer = HybridTokenizer.load(args.tokenizer)
    dataset = J(args.data, tokenizer, config.max_seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=generator,
                        collate_fn=lambda batch: collate(batch, tokenizer.PAD))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.02)
    total_steps = args.epochs * len(loader); step = 0; processed = loss_tokens = padding = 0; started = time.time()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in loader:
            step += 1; fraction = (step - 1) / max(1, total_steps - 1)
            lr = args.lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * fraction)))
            for group in optimizer.param_groups: group["lr"] = lr
            inputs = inputs.to(device); labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            processed += labels.numel(); loss_tokens += int((labels != -100).sum()); padding += int((labels == -100).sum())
            if step % 50 == 0:
                print(json.dumps({"step": step, "epoch": epoch, "loss": float(loss.detach()), "processed_tokens": processed,
                                  "loss_bearing_tokens": loss_tokens, "masked_or_padding_tokens": padding, "lr": lr}), flush=True)
    meta = dict(checkpoint.get("meta", {})); meta.update({"experiment": "general-ja-5m-v1", "stage": "canonical-alignment",
        "seed": args.seed, "epochs": args.epochs, "processed_tokens_alignment": processed, "loss_bearing_tokens_alignment": loss_tokens})
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True); model.to("cpu")
    torch.save({"model": model.state_dict(), "config": config.__dict__, "meta": meta,
                "global_step": checkpoint.get("global_step", 0) + step}, args.out_ckpt)
    print(json.dumps({"output": str(args.out_ckpt), "steps": step, "processed_tokens": processed,
                      "loss_bearing_tokens": loss_tokens, "elapsed_sec": time.time() - started}), flush=True)


if __name__ == "__main__": main()
