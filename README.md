---
license: apache-2.0
language:
  - ja
library_name: pytorch
pipeline_tag: text-generation
tags: [japanese, tiny-language-model, edge-ai, ex-word]
---

# EXLLM 5M

日本語 | [English](#english)

EXLLMは、CASIO EX-wordのような低資源端末で動かすためにゼロから学習した、約538万パラメータの小型日本語言語モデルです。XD-B4800上では、24.184 MHzの単一コアSH-4Aと整数推論ランタイムで実際に文章を生成します。

## 特徴

- 6層、hidden 288、9 attention heads、context 128 tokens
- 868語彙の文字＋UTF-8 byte fallback tokenizer
- PyTorch fp32 checkpointと行単位int8版を収録
- 電子辞書版はint8重み・Q12活性・整数演算で推論
- 短い日本語の挨拶、用語説明、簡単な質問向け
- 学習データ、重み、コードは本プロジェクトで作成

このモデルは汎用知識モデルではありません。未知の話題、長い指示、専門判断、最新情報には適しません。医療・法律・金融などの重要な判断には使用しないでください。

## 使い方

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python chat.py こんにちは
```

標準checkpointは`weights/EXLLM-v1.1-5m-release3.pt`です。LM Studioの標準llama.cpp backendはGGUF専用のため、この独自アーキテクチャを直接ロードできません。

電子辞書版、導入方法、実機ベンチマークは[`exllm-exword`](https://github.com/ToTo-40417/exllm-exword)を参照してください。

## ライセンス

コード、公開重み、付属データはApache License 2.0です。詳細は[`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)、[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)を参照してください。

## English

EXLLM is a tiny Japanese language model trained from scratch for highly constrained devices such as CASIO EX-word electronic dictionaries. Its 5.38-million-parameter model generates text on an XD-B4800 using an integer runtime on a single-core 24.184 MHz SH-4A processor.

### Highlights

- 6 layers, hidden size 288, 9 attention heads, 128-token context
- 868-token character vocabulary with UTF-8 byte fallback
- PyTorch fp32 checkpoint and row-wise int8 release
- Integer device runtime using int8 weights and Q12 activations
- Intended for short Japanese greetings, definitions, and simple questions
- Project-generated training data, weights, and code

This is not a general-purpose knowledge model. Do not rely on it for medical, legal, financial, current-information, or other high-stakes decisions.

### Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python chat.py こんにちは
```

The default checkpoint is `weights/EXLLM-v1.1-5m-release3.pt`. Standard LM Studio llama.cpp backends cannot load it directly because EXLLM is not a GGUF architecture. See [`exllm-exword`](https://github.com/ToTo-40417/exllm-exword) for the device runtime, installation notes, and physical-device benchmarks.

Released under Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
