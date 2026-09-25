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

## EX-word対応環境

電子辞書版は、利用する[`exword-template`](https://github.com/brain-hackers/exword-template)とlibexwordの対応範囲から、DATAPLUS 5 / 6 / 7を理論上の対象としています。実機で起動・推論・ベンチマークを確認したのはXD-B4800（DATAPLUS 6）のみです。他機種での動作は保証せず、DATAPLUS 5 / 7およびそれ以外の世代は実機未確認です。

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

標準checkpointは`weights/EXLLM-v1.1-5m-release3.pt`です。この独自アーキテクチャはLM Studioの標準llama.cpp backendで直接ロードできません。Hugging Faceでは、同じプロジェクトデータから別途学習したLlama互換GGUF companionを配布しています。作成スクリプトは[`tools/train_lmstudio_companion.py`](tools/train_lmstudio_companion.py)です。

モデルカードと配布用重みは[`ToTo-40417/EXLLM`](https://huggingface.co/ToTo-40417/EXLLM)、電子辞書版、導入方法、実機ベンチマークは[`exllm-exword`](https://github.com/ToTo-40417/exllm-exword)を参照してください。関連ツールとして[`exword-hardware-dump`](https://github.com/ToTo-40417/exword-hardware-dump)と[`exword-gnuboy-save-importer`](https://github.com/ToTo-40417/exword-gnuboy-save-importer)があります。

RTX 3060での40回warm-up後・120試行の生ログは[`benchmarks/rtx3060-cuda-robust-20260925.json`](benchmarks/rtx3060-cuda-robust-20260925.json)、学習・lineageの機械可読情報は[`training/release-5m.json`](training/release-5m.json)に収録しています。

## ライセンス

コード、公開重み、付属データはApache License 2.0です。詳細は[`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)、[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)を参照してください。

## English

EXLLM is a tiny Japanese language model trained from scratch for highly constrained devices such as CASIO EX-word electronic dictionaries. Its 5.38-million-parameter model generates text on an XD-B4800 using an integer runtime on a single-core 24.184 MHz SH-4A processor.

### EX-word compatibility

Based on the supported scope of [`exword-template`](https://github.com/brain-hackers/exword-template) and the libexword installation path, the device runtime theoretically targets DATAPLUS 5, 6, and 7. Boot, inference, and benchmark operation have been tested only on an XD-B4800 (DATAPLUS 6). Other models are not guaranteed; DATAPLUS 5, DATAPLUS 7, and all other generations remain untested on physical hardware.

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

The default checkpoint is `weights/EXLLM-v1.1-5m-release3.pt`. Standard LM Studio llama.cpp backends cannot load this custom architecture directly. Hugging Face therefore provides a separately trained, Llama-compatible GGUF companion built from the same project data; it is not a conversion of the embedded weights. Its training script is [`tools/train_lmstudio_companion.py`](tools/train_lmstudio_companion.py). The model card and release weights are available at [`ToTo-40417/EXLLM`](https://huggingface.co/ToTo-40417/EXLLM). See [`exllm-exword`](https://github.com/ToTo-40417/exllm-exword) for the device runtime and benchmarks, plus [`exword-hardware-dump`](https://github.com/ToTo-40417/exword-hardware-dump) and [`exword-gnuboy-save-importer`](https://github.com/ToTo-40417/exword-gnuboy-save-importer) for related tools.

The 40-warm-up/120-run RTX 3060 log is published as [`benchmarks/rtx3060-cuda-robust-20260925.json`](benchmarks/rtx3060-cuda-robust-20260925.json). Machine-readable training and lineage metadata is in [`training/release-5m.json`](training/release-5m.json).

Released under Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
