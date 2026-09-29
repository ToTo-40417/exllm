# 0→1 言語モデル作成プレイブック

Copyright 2026 ToTo

この文書は、ユーザーから「〇〇についての言語モデルを作りたい」と相談されたAIエージェントまたは開発者が、目的定義、権利確認、データ設計、学習、評価、配布までを一貫して実行するための手順書です。

本文と付属テンプレートは、このリポジトリの `LICENSE`（Apache License 2.0）に従います。原著作者の表示を保持したまま利用・改変できます。第三者資料、データ、コード、モデルにはそれぞれ個別の条件が適用され、本書のライセンスへ吸収されたり、ToToの成果物へ帰属が移ったりするものではありません。

> 本書は技術的な確認手順であり、個別案件の法律相談ではありません。権利関係が不明なデータを「学習用途だから問題ない」と推定して使用しないでください。

## 1. 完了条件

モデル作成は、weightファイルが生成された時点では完了しません。次の成果物がそろい、相互に矛盾しない状態を完成とします。

1. 目的・非目的・利用環境を記述した仕様書
2. architecture、tokenizer、context長、入出力形式を固定したruntime contract
3. 全データを追跡できるprovenance manifest
4. 学習集合から独立した評価集合とrelease gate
5. 再実行可能な学習recipe、環境情報、seed、hash、ログ
6. 変換済みweightと実行に必要な設定・tokenizer
7. model card、制限事項、ライセンス、第三者表示
8. 対象runtimeでの実機または同等環境検証

途中checkpointは障害復旧用です。仕様の異なる試作品を無計画に公開し、後から説明を継ぎ足す運用は避けます。

## 2. 最初にユーザーへ確認すること

AIエージェントは、データ収集や学習を始める前に、最低限次を確定します。

```yaml
model_request:
  subject: "何について答えるモデルか"
  target_users: "誰が使うか"
  languages: ["ja"]
  tasks: ["question-answering"]
  excluded_tasks: ["current-information", "medical-diagnosis"]
  runtime:
    hardware: "CPU/GPU/組込み機器"
    ram_limit_bytes: null
    storage_limit_bytes: null
    context_tokens: null
    latency_target: null
  distribution:
    publish_weights: true
    publish_dataset: false
    commercial_use_expected: false
  behavior:
    answer_length: "short"
    unknown_policy: "不明と明示する"
    safety_requirements: []
```

未回答項目がモデル構造、データ権利、費用、公開範囲を変える場合は、推測で進めません。

## 3. 仕様を先に固定する

### 3.1 runtimeから逆算する

次の順で決めます。

1. 実行環境の連続確保可能RAM、storage、演算命令、thread数を実測する
2. weight、KV cache、activation、tokenizer、UIを含むmemory budgetを作る
3. context長と最大出力token数を決める
4. 量子化形式とloaderを決める
5. その制約内でarchitectureを確定する

モデルparameter数だけで「動く」と判断してはいけません。ファイルを開けても、連続heap、temporary buffer、alignment、memory mappingの制約で失敗することがあります。

### 3.2 runtime contract

以下を機械可読に固定します。

```json
{
  "architecture": "decoder-only-transformer",
  "parameters": 0,
  "vocab_size": 0,
  "context_tokens": 0,
  "tokenizer_sha256": "",
  "config_sha256": "",
  "prompt_format": "BOS USER text ASSIST",
  "weight_formats": ["safetensors"],
  "target_runtimes": []
}
```

tokenizerやspecial tokenの番号を途中で変更すると、同じarchitectureでも既存weightと互換でなくなります。

## 4. 権利・ライセンスの入口審査

### 4.1 4つの成果物を分ける

ライセンスを一括で考えず、次を別々に管理します。

| 対象 | 例 | 確認事項 |
|---|---|---|
| コード | trainer、exporter、runtime | source license、linking条件、NOTICE、特許条項 |
| 入力データ | 本文、会話、ラベル | 著作権、契約、利用規約、個人情報、再配布条件 |
| 生成データ | paraphrase、QA、noise変換 | 元データの条件、生成手段の規約、人手修正の記録 |
| weight・model card | checkpoint、設定、説明 | 配布条件、上流weightの条件、第三者表示、名称・商標 |

コードがGPLやApacheであることは、入力データまで同じlicenseになることを意味しません。逆も同様です。

### 4.2 source acceptance gate

各sourceは次を満たすまで取得・学習対象に入れません。

```json
{
  "source_id": "stable-id",
  "upstream_url": "https://example.invalid/",
  "retrieved_at": "ISO-8601",
  "revision": "release/tag/commit/date",
  "license_name": "",
  "license_url": "",
  "rights_holder": "",
  "attribution": "",
  "allowed_uses": ["download", "transform", "train"],
  "redistribution": {
    "original": false,
    "normalized": false,
    "generated_records": false
  },
  "personal_data_review": "pass|fail|not-applicable",
  "original_sha256": "",
  "normalization_version": "",
  "normalized_sha256": "",
  "decision": "enabled|disabled|review-required",
  "notes": ""
}
```

必須情報が欠けていれば `review-required`、矛盾・禁止があれば `disabled` とします。

### 4.3 判断の基本

- 自作データ：作成者、作成方法、作成日、利用した資料を残す。
- public domain：著者だけでなく翻訳者、編集、底本、データベース部分も確認する。
- CC0：権利放棄の対象外となる商標、特許、第三者の権利、個人情報は別途確認する。
- CC BY：指定された表示、license link、変更表示を保持する。
- CC BY-SA：再配布する翻案物やdatabaseにshare-alikeが及ぶ範囲を確認する。判断できなければ隔離する。
- GPL系コード：copyright notice、license全文、source提供義務、組み合わせ方を確認する。第三者コードを自作と表示しない。
- 非営利限定：将来の商用利用可能性があるモデルには混ぜない。
- 利用規約のみで取得したwebデータ：copyright licenseとサービス利用規約を両方確認する。
- 規約・license不明：使用しない。

学習済みweightが入力データのlicenseを当然に継承する、または当然に継承しない、という一律の前提を置きません。入力の保存・再配布、memorization、出力、database権、契約条件を分け、公開地域と用途に応じて判断します。

### 4.4 帰属を守る

第三者素材を使える場合でも、次を削除しません。

- 原著作者・rights holder
- project/source名とcanonical URL
- license名とURL
- 変更点と変換履歴
- 元ファイルと変換後ファイルのhash

自分の成果物として主張できるのは、自分が作成したコード、設計、データ、選択・編集、学習結果など、実際に権利または許諾を持つ範囲だけです。他者の著作物や貢献を、自分の単独成果として表記してはいけません。

## 5. データ設計

### 5.1 「行数」ではなく情報量を測る

最低限、次を記録します。

- raw records / unique records
- unique documents / concepts / prompts / answers
- exact duplicateとnear duplicate
- 同一promptに対する競合回答
- processed tokens / loss-bearing tokens
- padding tokens / truncated tokens
- source別・category別token数
- tokenizer byte-fallback率
- tokens per Unicode character
- context上限到達率
- 各unique recordの露出回数

同じ例を8回並べても、8倍の知識にはなりません。反復は意図的な重み付けとしてのみ使用し、manifestで明示します。

### 5.2 splitは変換前に行う

元文書またはconceptをtrain/validation/testへ分割してから、paraphrase、noise付加、QA化を行います。同じ元文書から作った別表現をtrainとtestへ分けると、見かけ上の評価だけが上がります。

```text
source documents / concept IDs
              │
              ├── train groups ── training transformations
              ├── validation groups ── validation-only transformations
              └── test groups ── final evaluation-only transformations
```

### 5.3 tokenizer監査

小型モデルではtokenizerの非効率がcontextと速度を直接消費します。

1. 対象言語の文字分布を測る
2. unknown/fallback表現を確認する
3. source別のtokens/characterを測る
4. 長文・異体字・emoji・記号をround-tripする
5. runtime実装とtrainerが同じ正規化を使うことをhashで固定する

既存runtimeとの互換性が必要なら、効率が悪くてもtokenizerを変更しない判断があります。その場合はfallback率を制限事項として残します。

## 6. 学習recipe

### 6.1 parentを明示する

```yaml
lineage:
  origin: random-initialization | external-checkpoint | project-checkpoint
  parent_path: ""
  parent_sha256: ""
  architecture_change: false
  transplanted_parameters: false
  external_pretrained_checkpoint: false
```

「projectはrandom initializationから始まった」と「今回のcheckpointは既存project checkpointから継続した」は両立します。起源と直接の親を分けて書きます。

### 6.2 学習段階を役割で分ける

一般的な構成は次のとおりです。

1. language/domain continuation：対象分野の文章構造・語彙
2. structure transformation：分類、抽出、denoising、paraphrase、composition
3. instruction alignment：ユーザー入力と期待回答
4. regression repair：失敗例を小量で修正。ただし無限に継ぎ足さない

全layerが全exampleから更新されます。「layerごとに別データを載せる」とは解釈しません。役割分担はcurriculum、loss、sampling、adapterなどで設計します。

### 6.3 token accounting

`100M tokens`だけでは不十分です。少なくとも次をログへ出します。

```json
{
  "processed_tokens": 0,
  "loss_bearing_tokens": 0,
  "padding_tokens": 0,
  "truncated_tokens": 0,
  "unique_examples": 0,
  "example_exposures": 0
}
```

promptをlossからmaskするSFTでは、processed tokensとloss-bearing tokensが一致しません。

### 6.4 中間checkpointの扱い

中間checkpointは、停電、OOM、job中断からの復旧と学習曲線の観測に使用します。仕様が固まっている場合、これらを別製品やMVPとして公開しません。最終gateを通った単一成果物だけをrelease candidateとします。

## 7. 評価設計

評価は最低3層に分けます。

| gate | 目的 | 例 |
|---|---|---|
| backward compatibility | 既存機能を壊していない | identity、固定操作、既知回答 |
| target capability | 新しい目的を満たす | domain QA、文法、抽出、意図理解 |
| robustness/safety | 異常入力で破綻しない | noise、Unicode、未知語、長文、危険質問 |

training lossだけで採用しません。baselineとcandidateを同じ条件で測り、category別に比較します。単一の総合点で退行を隠さないようにします。

### release gateの例

```yaml
acceptance:
  runtime_load: pass
  tokenizer_roundtrip: pass
  backward_compatibility: "all required cases pass"
  target_capability: "baseline以上、重要categoryに退行なし"
  invalid_output_rate: 0
  memory_peak_bytes: "runtime budget以下"
  artifact_hash_verified: true
```

gate失敗時は成果物を不採用にします。評価問題だけを学習へ追加して再実行すると、test contaminationになるため、原因分類後に新しいtraining例と新しいheld-out評価を設計します。

## 8. 実行環境と運用

- source authoring、training、inference serverを分離する場合、それぞれの責務を固定する。
- source snapshotをhash付きでtraining hostへ送る。
- training hostがsource corpusを無断変更しない。
- 長時間jobはPID、command、cwd、開始・終了時刻、return code、stdout/stderrを保存する。
- GPU、driver、framework、precision、device名、実測tokens/sを保存する。
- 別用途のGPU modelを明示的にunloadしてから学習する。
- 失敗時に上書きせず、run IDごとにimmutable logを残す。

## 9. 配布物

推奨構成です。

```text
model-project/
├── README.md
├── MODEL_CARD.md
├── LICENSE
├── NOTICE
├── DATA_PROVENANCE.md
├── config.json
├── tokenizer.json
├── weights/
├── training/
│   ├── RECIPE.md
│   ├── experiment.json
│   ├── environment-lock.*
│   └── release-manifest.json
├── eval/
│   ├── benchmark-spec.json
│   └── results/
└── tools/
```

大きい・非再配布のdatasetをrepoへ入れられなくても、取得元、revision、filter、変換コード、hash、除外理由を公開すれば再現境界を明確にできます。取得権限が必要なsourceは、利用者自身が正規入手する手順を案内します。

## 10. AIエージェント用実行アルゴリズム

```text
INPUT: 「〇〇についての言語モデルを作りたい」

1. 目的、利用者、言語、task、runtime、公開範囲、禁止事項を聞く
2. runtime budgetを実測または保守的に定義する
3. architecture/tokenizer/context/formatをruntime contractとして固定する
4. 候補sourceごとに権利・規約・帰属・再配布・個人情報を審査する
5. 不明なsourceを除外し、provenance manifestを作る
6. concept/source単位でsplitする
7. trainだけにtraining transformationsを適用する
8. duplicate、conflict、token、fallback、truncationを監査する
9. baselineを全gateで測る
10. recipe、seed、hash、環境を固定する
11. 単一の本番runを開始し、復旧checkpointだけを保存する
12. 独立評価とruntime検証を行う
13. gate合格時のみweight、model card、provenance、licenseを公開する
14. gate不合格なら公開せず、原因と次回仕様を分離して記録する
```

## 11. 停止条件

次の場合、AIエージェントは勝手に補完せず停止します。

- sourceのlicenseまたは利用規約が確認できない
- 個人情報・機密情報の混入を否定できない
- userが権利を持たない非公開データの使用を求めている
- 上流weightの再配布条件が不明
- runtime制約内にarchitectureが収まらない
- test dataがtrainingへ混入している
- manifestと実ファイルのhashが一致しない
- 必須gateが失敗した

## 12. EXLLM General Japanese 5Mでの適用例

本プロジェクトでは次のように適用しています。

- architecture：5,377,824 trainable parameters、context 128、固定tokenizer
- parent：project-owned `EXLLM-v1.1-5m-release3.pt`
- language data：著者・翻訳者の権利条件をfilterした青空文庫作品
- language budget：100M processed tokens
- alignment：85,005行をそのまま反復せず、11,396 distinct promptsへ正規化して3 epochs
- accounting：processed / loss-bearing / padding / truncationを分離
- evaluation：既存release gate、独立汎用日本語集合、Unicode/fuzz、DATAPLUS 6/7実機
- distribution：sourceごとの条件を保持し、第三者素材をEXLLM単独の著作物として表示しない

現行`exllm-exword`の自由入力はローマ字からひらがなへ変換するか、ASCIIを入力する方式で、漢字IMEは持ちません。そのため実機向けalignmentでは、質問側に決定的なかな表記を用意します。例えば「おはようを英語で言うと？」だけでなく、実機で入力できる「おはようをえいごでいうと？」を主要形として学習・評価します。回答側はかなに制限せず、自然な漢字かな交じり文を維持します。

自動変換で読みが曖昧な場合は勝手に確定せず、人手で読みを指定します。原表記とかな表記は同じsplit groupに置き、変換後の同一質問がtrain/test間に分かれることを防ぎます。漢字入力の成績と実機かな入力の成績は別々に報告します。IME再実装は、変換辞書容量、候補選択UI、操作量、実行時メモリを実測するまで将来案とします。

青空文庫では、著作権の切れた作品ファイルは複製・再配布・共有・翻案等が認められていますが、翻訳作品は翻訳者の権利も確認する必要があります。書誌データは本文と別条件です。本プロジェクトはこの区別をfilterとprovenanceへ反映します。

## 参考となる一次資料

- GNU GPL v2: <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
- Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>
- Creative Commons licenses: <https://creativecommons.org/share-your-work/cclicenses/>
- CC0 1.0 Universal: <https://creativecommons.org/publicdomain/zero/1.0/>
- 青空文庫収録ファイルの取り扱い規準: <https://www.aozora.gr.jp/guide/kijyunn.html>
- Wikimedia Foundation Terms of Use: <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>
