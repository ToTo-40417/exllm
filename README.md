---
license: apache-2.0
language:
  - ja
library_name: pytorch
pipeline_tag: text-generation
tags: [japanese, little-language-model, tiny-language-model, edge-ai, ex-word]
---

# EXLLM 5M

日本語 | [English](#english)

EXLLMは、CASIO EX-wordのような低資源端末を対象に、プロジェクト内データから学習した約538万パラメータの実験的な日本語言語モデルです。短い挨拶や定型質問を中心とする狭いモデルであり、汎用知識モデルではありません。XD-B4800（DATAPLUS 6）とXD-N6500（DATAPLUS 7）の単一コアSH-4A実機で整数推論を確認しています。

LLMとは、Little Language Modelの略です。

このGitHubリポジトリは、単一checkpointだけでなく、**共通5M級アーキテクチャ、学習来歴、EXQ12変換、共通評価、実機ベンチマーク、再現手順**を管理するプロジェクト本体です。学習目的や能力が異なるモデルはHugging Face上で別モデルとして公開し、[`docs/MODEL_CATALOG.md`](docs/MODEL_CATALOG.md)から相互リンクします。公開名は[`docs/NAMING_POLICY.md`](docs/NAMING_POLICY.md)の命名規則に従います。

## プロジェクト構成

| 領域 | 主な場所 |
|---|---|
| モデル一覧・用途・対応アプリ | [`docs/MODEL_CATALOG.md`](docs/MODEL_CATALOG.md) |
| LM Studio / llama.cpp用GGUF companion | [`docs/GGUF_COMPANIONS.md`](docs/GGUF_COMPANIONS.md) |
| 公開モデルの命名規則 | [`docs/NAMING_POLICY.md`](docs/NAMING_POLICY.md) |
| 学習・重みlineage・データ来歴 | [`training/`](training/), [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) |
| 0→1のモデル作成・権利確認プレイブック | [`docs/LLM_CREATION_PLAYBOOK_JA.md`](docs/LLM_CREATION_PLAYBOOK_JA.md) |
| EXLLM8 / EXQ12形式と変換 | [`EXLLM8_FORMAT.md`](EXLLM8_FORMAT.md), [`EXQ12_FORMAT.md`](EXQ12_FORMAT.md), [`tools/`](tools/) |
| 共通評価スクリプトと機械可読結果 | [`eval/`](eval/), [`benchmarks/`](benchmarks/) |
| 実機runtime・導入 | [`exllm-exword`](https://github.com/ToTo-40417/exllm-exword) |
| 固定5M級の学習量・一般化ケーススタディ | [`docs/case-studies/fixed-5m-training-and-generalization.md`](docs/case-studies/fixed-5m-training-and-generalization.md) |

## モデルと実行環境

| 用途 | モデル | 実行環境 |
|---|---|---|
| 電子辞書・参照実行 | EXLLM独自checkpoint、0.005377824B parameters | EX-word用整数runtime / PyTorch参照runtime |
| PCでの簡易実行 | 別途学習したLlama互換GGUF companion、0.005441184B parameters | LM Studio / llama.cpp |

GGUF companionは電子辞書用checkpointの形式変換ではなく、プロジェクトデータから別途学習したPC向けモデルです。重み、tokenizer、context、評価結果は共有しません。詳細は[`GGUF_COMPANIONS.md`](docs/GGUF_COMPANIONS.md)を参照してください。

## EX-word対応環境

電子辞書版は、依存する[`exword-template`](https://github.com/brain-hackers/exword-template)とlibexwordの対応範囲からDATAPLUS 5 / 6 / 7を想定しています。XD-B4800（DATAPLUS 6 / CY168）とXD-N6500（DATAPLUS 7 / CY460）で、起動・モデル選択・推論・ベンチマークを実機確認済みです。DATAPLUS 5は未検証であり、その他の機種・世代を含めて動作を保証しません。

## 特徴

- 6層、hidden 288、9 attention heads、context 128 tokens
- 868語彙の文字＋UTF-8 byte fallback tokenizer
- PyTorch fp32 checkpointと行単位int8版を収録
- 電子辞書版はint8重み・Q12活性・整数演算で推論
- 短い日本語の挨拶と学習済みに近い定型質問を対象
- プロジェクト内で作成した反復・template中心の学習データ
- v1.0親checkpointから5M release3までのstage manifestと再実行ツールを収録

このモデルは汎用知識モデルではありません。未知の話題、長い指示、専門判断、最新情報には適しません。医療・法律・金融などの重要な判断には使用しないでください。

## 使い方

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python chat.py こんにちは
```

標準checkpointは`weights/EXLLM-v1.1-5m-release3.pt`です。この独自アーキテクチャはLM Studioの標準llama.cpp backendで直接ロードできません。

### LM Studioで使う

LM Studioで`ToTo-40417/EXLLM`を検索し、`EXLLM-0.005B-LMStudio-F16.gguf`をダウンロードします。ロード時のcontext lengthは`512`に設定してください。ただし学習系列長は128 tokensであり、512 tokensにわたる品質を示す設定ではありません。

GGUF本体、SHA-256、詳細なモデルカード、学習manifestは[`ToTo-40417/EXLLM`](https://huggingface.co/ToTo-40417/EXLLM)を正本とします。

電子辞書版の導入方法と実機ベンチマークは[`exllm-exword`](https://github.com/ToTo-40417/exllm-exword)、EXQ12の事前検査は[`exllm-model-check`](https://github.com/ToTo-40417/exllm-model-check)を参照してください。その他の関連ツールは[`MODEL_CATALOG.md`](docs/MODEL_CATALOG.md)から辿れます。

開発背景と実機での動作は、Note記事「[高校生用の電子辞書でLLMを動かしてみた ―0.024GHz/0.016GB](https://note.com/joyful_beetle869/n/nbd1e26679b78)」で紹介しています。

学習・lineage・ベンチマークの機械可読記録は[`training/`](training/)と[`benchmarks/`](benchmarks/)にあります。固定5M級アーキテクチャの学習量と一般化は[`ケーススタディ`](docs/case-studies/fixed-5m-training-and-generalization.md)、EXQ12の再生成手順は[`EXQ12_FORMAT.md`](EXQ12_FORMAT.md)にまとめています。

公開派生モデルは[EXLLM-JPTOEN](https://huggingface.co/ToTo-40417/EXLLM-JPTOEN)と[EXLLM-ONI5M](https://huggingface.co/ToTo-40417/EXLLM-ONI5M)です。用途と対応runtimeは[`MODEL_CATALOG.md`](docs/MODEL_CATALOG.md)を参照してください。

## ライセンス

コード、公開重み、付属データはApache License 2.0です。詳細は[`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)、[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)を参照してください。

## English

EXLLM is an experimental 5M-class Japanese language model for constrained devices such as CASIO EX-word electronic dictionaries. It is centered on short greetings and template-adjacent questions, not general knowledge or unrestricted conversation. Integer inference has been verified on single-core SH-4A hardware in an XD-B4800 (DATAPLUS 6) and XD-N6500 (DATAPLUS 7).

Here, LLM stands for Little Language Model.

This GitHub repository manages the shared architecture, training and data lineage, EXQ12 conversion, evaluation, device benchmarks, and reproduction procedures. Purpose-specific releases are listed in the [`model catalog`](docs/MODEL_CATALOG.md).

### Models and runtimes

| Purpose | Model | Runtime |
|---|---|---|
| Embedded and reference inference | Custom EXLLM checkpoint, 0.005377824B parameters | EX-word integer runtime / PyTorch reference runtime |
| Convenient PC inference | Separately trained Llama-compatible GGUF companion, 0.005441184B parameters | LM Studio / llama.cpp |

The GGUF companion is separately trained—not a conversion of the embedded checkpoint. Its weights, tokenizer, context, and evaluation results do not transfer to the embedded model. See [`GGUF_COMPANIONS.md`](docs/GGUF_COMPANIONS.md).

### EX-word compatibility

The dependency scope of [`exword-template`](https://github.com/brain-hackers/exword-template) and libexword suggests DATAPLUS 5, 6, and 7. Boot, model selection, inference, and benchmarks have been physically verified only on an XD-B4800 (DATAPLUS 6 / CY168) and XD-N6500 (DATAPLUS 7 / CY460). DATAPLUS 5 and all other devices remain unverified and are not guaranteed.

### Highlights

- 6 layers, hidden size 288, 9 attention heads, 128-token context
- 868-token character vocabulary with UTF-8 byte fallback
- PyTorch fp32 checkpoint and row-wise int8 release
- Integer device runtime using int8 weights and Q12 activations
- Targets short Japanese greetings and prompt forms close to its training templates
- Project-authored, repetition- and template-centered training data
- Stage manifest and replay tool from the v1.0 parent checkpoint through 5M release3

This is not a general-purpose knowledge model. Do not rely on it for medical, legal, financial, current-information, or other high-stakes decisions.

### Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python chat.py こんにちは
```

The default checkpoint is `weights/EXLLM-v1.1-5m-release3.pt`. Standard LM Studio llama.cpp backends cannot load this custom architecture directly. Use [`exllm-exword`](https://github.com/ToTo-40417/exllm-exword) for the device runtime and [`exllm-model-check`](https://github.com/ToTo-40417/exllm-model-check) to validate EXQ12 files.

The development background and physical-device demonstration are covered in the Japanese Note article, “[高校生用の電子辞書でLLMを動かしてみた ―0.024GHz/0.016GB](https://note.com/joyful_beetle869/n/nbd1e26679b78).”

### LM Studio

Search for `ToTo-40417/EXLLM` in LM Studio and download `EXLLM-0.005B-LMStudio-F16.gguf`. Set the runtime context to `512`; training sequences were limited to 128 tokens, so this setting does not establish 512-token task quality.

The Hugging Face repository is canonical for the GGUF binary and model card. Machine-readable lineage and benchmarks are under [`training/`](training/) and [`benchmarks/`](benchmarks/); the [`fixed-5M case study`](docs/case-studies/fixed-5m-training-and-generalization.md) documents the scope and limits of the comparison. Public derivatives include [EXLLM-JPTOEN](https://huggingface.co/ToTo-40417/EXLLM-JPTOEN) and [EXLLM-ONI5M](https://huggingface.co/ToTo-40417/EXLLM-ONI5M).

Released under Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
