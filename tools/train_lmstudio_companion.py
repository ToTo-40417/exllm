#!/usr/bin/env python3
"""Train the LM Studio / llama.cpp companion release for EXLLM.

This is intentionally a separate, Llama-compatible checkpoint trained from
the EXLLM project data.  It is not a format conversion of the EX-word weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spm_pb2
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM


def records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield str(row["prompt"]), str(row["answer"])


class SentencePieceTokenizer:
    def __init__(self, model_file: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.pad_token_id = self.sp.pad_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()

    def __len__(self): return self.sp.vocab_size()
    def encode(self, text, add_special_tokens=False): return self.sp.encode(text, out_type=int, add_bos=add_special_tokens, add_eos=False)
    def decode(self, ids): return self.sp.decode([int(x) for x in ids])
    def convert_tokens_to_ids(self, token): return self.sp.piece_to_id(token)
    def save_pretrained(self, out):
        # The companion is trained as one user turn followed by one assistant
        # turn. Embedding this template in the converted GGUF lets LM Studio
        # and llama.cpp's chat endpoint reproduce that exact token sequence.
        chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>' + message['content'] + '<|assistant|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        Path(out, "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "LlamaTokenizer", "model_max_length": 128,
            "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>",
            "chat_template": chat_template,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_tokenizer(paths: list[Path], out: Path, vocab_size: int):
    # Stock Llama conversion in llama.cpp requires tokenizer.model. A
    # character-coverage=1 SentencePiece BPE keeps every emitted token valid
    # Unicode without relying on EXLLM's device-side UTF-8 state mask.
    corpus = out / "tokenizer-corpus.txt"
    with corpus.open("w", encoding="utf-8") as handle:
        for prompt, answer in records(paths):
            handle.write(unicodedata.normalize("NFC", f"<|user|>{prompt}<|assistant|>{answer}") + "\n")
    prefix = out / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), model_type="bpe",
        vocab_size=vocab_size, character_coverage=1.0,
        byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        user_defined_symbols=["<|user|>", "<|assistant|>"],
        normalization_rule_name="identity", hard_vocab_limit=False,
    )
    # SentencePiece USER_DEFINED pieces are not parsed as special tokens by
    # llama.cpp chat templates. Re-tag the two role markers as CONTROL pieces
    # without changing their IDs or any learned model weights.
    model_path = Path(str(prefix) + ".model")
    proto = spm_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    for piece in proto.pieces:
        if piece.piece in {"<|user|>", "<|assistant|>"}:
            piece.type = spm_pb2.ModelProto.SentencePiece.CONTROL
    model_path.write_bytes(proto.SerializeToString())
    corpus.unlink()
    tokenizer = SentencePieceTokenizer(model_path)
    tokenizer.save_pretrained(out)
    probe = "こんにちは。RAMとは何ですか？"
    probe_ids = tokenizer.encode(probe, add_special_tokens=False)
    if len(tokenizer) < 600 or not probe_ids or tokenizer.decode(probe_ids).strip() != probe:
        raise RuntimeError("SentencePiece tokenizer round-trip validation failed")
    return tokenizer


class Instructions(Dataset):
    def __init__(self, rows, tokenizer, max_len=128):
        self.items = []
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        user = tokenizer.convert_tokens_to_ids("<|user|>")
        assistant = tokenizer.convert_tokens_to_ids("<|assistant|>")
        for prompt, answer in rows:
            p = tokenizer.encode(prompt, add_special_tokens=False)
            a = tokenizer.encode(answer, add_special_tokens=False)
            prefix = [bos, user] + p + [assistant]
            room = max_len - len(a) - 1
            if room < 3:
                a = a[: max(1, max_len - 4)]
                room = max_len - len(a) - 1
            prefix = prefix[-room:]
            seq = prefix + a + [eos]
            labels = [-100] * len(prefix) + a + [eos]
            self.items.append((seq, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad):
    width = max(len(x) for x, _ in batch)
    ids, labels, masks = [], [], []
    for x, y in batch:
        n = width - len(x)
        ids.append(x + [pad] * n)
        labels.append(y + [-100] * n)
        masks.append([1] * len(x) + [0] * n)
    return tuple(torch.tensor(v, dtype=torch.long) for v in (ids, labels, masks))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, tokens = 0.0, 0
    for ids, labels, masks in loader:
        ids, labels, masks = ids.to(device), labels.to(device), masks.to(device)
        logits = model(input_ids=ids, attention_mask=masks).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        total += float(loss)
        tokens += int((labels[:, 1:] != -100).sum())
    return total / max(tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/EXLLM-0.005B-LMStudio"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(p for p in args.data_dir.glob("*.jsonl") if p.name != "valid.jsonl")
    valid_path = args.data_dir / "valid.jsonl"
    tokenizer = train_tokenizer(train_paths, args.out, args.vocab_size)
    train_rows = list(records(train_paths))
    valid_rows = list(records([valid_path]))
    train_set = Instructions(train_rows, tokenizer)
    valid_set = Instructions(valid_rows, tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, generator=generator, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    cfg = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=288, intermediate_size=608,
        num_hidden_layers=6, num_attention_heads=9, num_key_value_heads=9,
        # Training examples remain capped at 128 tokens.  The larger RoPE
        # window leaves room for LM Studio's chat wrapper and short history.
        max_position_embeddings=512, rms_norm_eps=1e-5, hidden_act="silu",
        attention_bias=False, mlp_bias=False, tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.03)
    steps = args.epochs * len(train_loader)
    warmup = max(20, int(steps * 0.04))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    history = []
    started = time.time()
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids, labels, masks in train_loader:
            step += 1
            if step <= warmup:
                scale = step / warmup
            else:
                q = (step - warmup) / max(1, steps - warmup)
                scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            ids, labels, masks = ids.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(input_ids=ids, attention_mask=masks, labels=labels)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step % 100 == 0:
                print(json.dumps({"step": step, "loss": float(loss), "seconds": time.time() - started}), flush=True)
        value = evaluate(model, valid_loader, device)
        record = {"epoch": epoch, "valid_loss": value, "seconds": time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if value < best:
            best = value
            model.save_pretrained(args.out, safe_serialization=True)
            tokenizer.save_pretrained(args.out)

    manifest = {
        "name": "EXLLM-0.005B-LMStudio-GGUF-companion",
        "relationship": "separately trained Llama-compatible companion; not converted EX-word weights",
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": {
            "family": "LlamaForCausalLM",
            "layers": 6,
            "hidden_size": 288,
            "intermediate_size": 608,
            "attention_heads": 9,
            "kv_heads": 9,
            "training_sequence_length": 128,
            "runtime_context_length": 512,
            "vocabulary_size": len(tokenizer),
        },
        "training_records_including_overlaps": len(train_rows),
        "validation_records": len(valid_rows),
        "epochs": args.epochs, "steps": steps, "best_validation_loss": best,
        "seed": args.seed, "history": history,
    }
    (args.out / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
