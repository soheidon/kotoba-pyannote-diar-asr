# SPEC.md

# kotoba-pyannote-diar-asr 仕様書

## 1. 概要

本アプリは、Docker / Windows 11 環境で動作する日本語音声文字起こしツールである。  
主な目的は、単一の長時間音声ファイルを入力として、

1. 音声を 16kHz mono WAV に変換し
2. pyannote により話者分離を行い
3. Kotoba-Whisper により各セグメントを文字起こしし
4. 話者ラベル付きの VTT / TXT を出力する

ことである。

本アプリは、研究インタビューや聞き取り音声の一次文字起こしを主用途とする。  
ただし、最終成果物の品質は本アプリ単体で完結させることを前提とせず、必要に応じて Zoom VTT との突合や LLM による文脈補正を後段で行う構成を想定する。

---

## 2. 設計方針

### 2.1 基本方針
本アプリは、ASR 単体の極限的な精度追求よりも、以下を重視する。

- 長時間音声でも完走しやすいこと
- 例外時に原因追跡しやすいこと
- GPU メモリ使用を明示的に制御できること
- 辞書機能を実験的・補助的に扱えること
- 後段の Zoom VTT 突合・LLM 修正と競合しないこと

### 2.2 辞書機能の位置づけ
辞書機能は本アプリの標準動作ではなく、**オプション機能**として扱う。  
デフォルト運用は「辞書なし」である。

理由は以下の通りである。

- Whisper 系の事前ヒントにはトークン長制約がある
- 事後補正は誤補正リスクを持つ
- 実運用では Zoom VTT 突合 + LLM 修正のほうが文脈的に安全な場合が多い
- 辞書機能は案件や語彙に依存して効果が大きく変動する

したがって、辞書機能は「常時オン」ではなく、必要に応じて Layer A / Layer B を個別に試す構成とする。

---

## 3. 想定動作環境

### 3.1 OS / 実行環境
- Windows 11
- Docker Desktop
- NVIDIA GPU 利用環境
- `docker run --gpus all` で起動

### 3.2 主なソフトウェア構成
- Python 3.11 系
- PyTorch
- `pyannote.audio`
- `transformers`
- `huggingface_hub`
- `pydub`
- `ffmpeg`
- `tqdm`

### 3.3 主なモデル
- 話者分離: `pyannote/speaker-diarization-3.1`
- ASR: `kotoba-tech/kotoba-whisper-v2.2`

---

## 4. 入出力仕様

### 4.1 入力
入力は音声ファイル 1 本である。  
想定する主な入力形式は m4a だが、ffmpeg で読み込める音声形式であれば拡張可能である。

指定方法:
- `INPUT_FILENAME` 環境変数でファイル名を指定
- 実体は `WORK_SOURCE` 配下に置く

### 4.2 出力
出力は `WORK_OUTPUT` 配下に保存する。  
`WORK_OUTPUT` が未指定の場合は `WORK_SOURCE/output` を用いる。

出力ファイル:
- `<base>.vtt`
- `<base>.txt`

必要に応じて辞書デバッグ用ログ:
- `<base>.dict.log`

### 4.3 VTT 出力
VTT は話者ラベル付きで出力する。  
各セグメントは `<v speaker>` 形式で出力する。

### 4.4 TXT 出力
TXT は 1 行 1 セグメントの簡易出力とし、各行の先頭に話者ラベルを付す。

---

## 5. 処理フロー

### 5.1 概要
処理フローは以下の通りである。

1. 入力ファイル存在確認
2. ffmpeg による 16kHz mono WAV 変換
3. pyannote 話者分離モデルのロード
4. 話者分離の実行
5. 話者分離用リソースの解放
6. 必要なら辞書読み込み
7. Kotoba-Whisper ASR pipeline ロード
8. 必要なら Layer A 用 prompt_ids 生成
9. セグメント単位 ASR
10. 必要なら Layer B 補正
11. VTT / TXT 書き出し
12. 一時ディレクトリ削除

### 5.2 音声変換
入力音声は ffmpeg を subprocess で呼び出し、16kHz mono WAV に変換する。  
変換後の WAV は一時ディレクトリに保存する。

### 5.3 話者分離
pyannote により話者分離を行う。  
`NUM_SPEAKERS` が指定されていれば固定話者数として扱い、未指定なら自動推定とする。

### 5.4 ASR
話者分離で得られた各セグメントを WAV として切り出し、Kotoba-Whisper pipeline に渡す。

---

## 6. GPU メモリ管理

### 6.1 方針
話者分離後は、ASR 実行前に不要になった話者分離リソースを解放し、GPU メモリを明示的に整理する。

### 6.2 実施内容
- 話者分離オブジェクトへの参照を外す
- `gc.collect()` を呼ぶ
- CUDA 利用時は `torch.cuda.empty_cache()` を呼ぶ
- 解放前後の VRAM 状況をログに出す

### 6.3 目的
- pyannote と Kotoba-Whisper の連続使用による VRAM 圧迫を緩和する
- 4060 Ti 16GB クラスでも安定運用しやすくする
- メモリエラー時の切り分けを容易にする

---

## 7. 辞書機能

## 7.1 概要
辞書機能は TSV 形式の固有名詞辞書を用いる。  
形式は以下を基本とする。

`正式表記<TAB>よみ`

例:

- 宇多津    うたづ
- 宇多津町  うたづちょう
- 浜一番丁  はまいちばんちょう

本辞書は一般辞書ではなく、**案件固有の固有名詞辞書**として扱う。

---

## 7.2 Layer A: pre_asr_hint

### 7.2.1 目的
Whisper に事前ヒントを与え、重要な正式表記を優先候補として持たせる。

### 7.2.2 実装方針
- `prompt_ids` を `generate_kwargs` に渡す
- 文字列 `initial_prompt` は用いない
- 正式表記のみをスペース区切りで列挙する
- `token budget` を超えない範囲で greedy に採用する
- 環境変数でオンオフ可能とする

### 7.2.3 制約
Layer A は Whisper 側のデコーダ長制約を受けるため、辞書全件を投入する用途には向かない。  
したがって、Layer A は「少数の優先語 shortlist」を渡す機能として扱う。

### 7.2.4 fallback
Layer A 有効時でも、以下のような prompt 関連エラーが発生した場合は、

- `prompt_ids`
- `max_target_positions`
- `decoder_input_ids`
- `model_kwargs`

などを含む例外

Layer A を無効化し、同一セグメントを prompt なしで再試行する。  
以後のセグメントも Layer A を無効化したまま継続する。

---

## 7.3 Layer B: post_asr_correction

### 7.3.1 目的
ASR 出力後に、辞書の `よみ -> 正式表記` を用いて補正する。

### 7.3.2 実装方針
- 読みの長い順にソートして置換する
- 短い読みが長い読みに含まれる場合の誤置換を抑える
- 辞書件数制限は設けず、全件使用可能とする
- 環境変数でオンオフ可能とする

### 7.3.3 制約
Layer B は文脈を見ない単純置換である。  
そのため、誤補正リスクがある。  
常用は推奨せず、必要に応じて `.dict.log` により検証する。

---

## 7.4 デフォルト運用
デフォルトでは以下とする。

- `USE_LAYER_A=0`
- `USE_LAYER_B=0`

すなわち、通常運用は辞書なし版を基準とする。

理由:
- 辞書なし最新版が比較上もっとも安定した
- Layer A / Layer B は用途依存性が高い
- 本格的な固有名詞補正は後段の LLM に任せたほうが安全な場合が多い

---

## 8. 辞書ログ

### 8.1 目的
辞書機能の挙動をあとから確認可能にする。

### 8.2 出力
`DEBUG_GLOSSARY_LOG=1` のとき、`<base>.dict.log` を出力する。

### 8.3 Layer A ログ
記録項目例:
- 辞書総件数
- token budget
- 採用件数
- 採用された正式表記一覧
- `prompt_ids` の shape
- Layer A 有効 / 無効
- fallback の発生有無と理由

### 8.4 Layer B ログ
記録項目例:
- セグメント番号
- 話者ラベル
- ヒットした reading
- 置換後の formal
- 必要に応じて短いテキスト断片

### 8.5 コンソールとの役割分担
- コンソール: 要約ログ、進行状況、重大警告
- `.dict.log`: 辞書機能の詳細記録

---

## 9. 環境変数

### 9.1 基本
- `HF_TOKEN`
- `INPUT_FILENAME`
- `NUM_SPEAKERS`
- `MODEL_DIAR`
- `MODEL_ASR`
- `ASR_LANGUAGE`
- `ASR_TASK`

### 9.2 パス関連
- `WORK_SOURCE`
- `WORK_OUTPUT`
- `WORK_HF_CACHE`
- `WORK_TMP`

### 9.3 辞書関連
- `DICT_PATH`
- `USE_LAYER_A`
- `USE_LAYER_B`
- `GLOSSARY_TOKEN_BUDGET`
- `LAYER_A_FALLBACK`
- `DEBUG_GLOSSARY_LOG`

---

## 10. 運用方針

### 10.1 通常運用
通常運用は辞書なしとする。

想定:
- `USE_LAYER_A=0`
- `USE_LAYER_B=0`

### 10.2 実験運用
必要に応じて以下を個別に試す。

- Layer A のみ
- Layer B のみ
- 両方
- `.dict.log` 詳細化

### 10.3 推奨比較パターン
辞書機能の評価時は少なくとも以下を比較する。

1. 旧版
2. 新版・辞書なし
3. 新版・Layer A のみ
4. 新版・Layer B のみ
5. 新版・両方

---

## 11. 非目標

本アプリは以下を現時点の目標としない。

- 辞書全件を Whisper に事前投入すること
- IME 辞書のような強制変換
- 音声と正解転記を用いた LoRA / fine-tuning
- 文脈理解を伴う高度補正を ASR 前段で完結させること

これらは、本アプリの後段である

- Zoom VTT 突合
- LLM による文脈補正

で扱うことを優先する。

---

## 12. 今後の拡張候補

- Layer A 用 shortlist 辞書の分離
- Layer B の誤置換対策強化
- 辞書ヒット統計の集計
- fallback 発生率の自動記録
- Zoom VTT 突合ツールとの連携強化

---

## 13. 結論

本アプリは、**辞書なし最新版を基準とする安定運用版**であり、  
辞書機能は Layer A / Layer B として切り替え可能な補助機能として扱う。

固有名詞補正を ASR 前段だけで解決しようとせず、  
後段の Zoom VTT 突合および LLM による文脈補正と組み合わせて、  
全体として実用的なインタビュー文字起こしを実現することを目標とする。

---

# kotoba-pyannote-diar-asr Specification

## 1. Overview

This application is a Docker / Windows 11 based Japanese speech transcription tool.  
Its primary purpose is to take a single long-form audio file as input and:

1. convert the audio to 16kHz mono WAV,
2. perform speaker diarization with pyannote,
3. transcribe each segment with Kotoba-Whisper, and
4. output speaker-labeled VTT / TXT files.

The main use case is first-pass transcription for research interviews and spoken recordings.  
However, the application is not designed with the assumption that final transcript quality must be completed within this tool alone. When necessary, downstream alignment with Zoom VTT and context-aware correction by an LLM are assumed.

---

## 2. Design Policy

### 2.1 Core Policy
This application prioritizes the following over extreme ASR-only accuracy optimization:

- the ability to complete long-form audio processing reliably,
- ease of error tracing when exceptions occur,
- explicit control of GPU memory usage,
- treating glossary features as experimental / auxiliary options,
- non-conflict with downstream Zoom VTT alignment and LLM-based correction.

### 2.2 Position of Glossary Features
Glossary features are treated as **optional features**, not as default behavior.

The default operating mode is **without glossary features**.

This is for the following reasons:

- Whisper-style pre-ASR hints are constrained by token length,
- post-ASR correction carries a risk of incorrect replacements,
- in practical use, Zoom VTT alignment + LLM correction is often safer at the contextual level,
- the effectiveness of glossary features varies greatly depending on the task and vocabulary.

Therefore, glossary features are not treated as always-on standard functions, but as optional components where Layer A / Layer B can be tested independently when needed.

---

## 3. Intended Runtime Environment

### 3.1 OS / Runtime Environment
- Windows 11
- Docker Desktop
- NVIDIA GPU-enabled environment
- launched via `docker run --gpus all`

### 3.2 Main Software Components
- Python 3.11 series
- PyTorch
- `pyannote.audio`
- `transformers`
- `huggingface_hub`
- `pydub`
- `ffmpeg`
- `tqdm`

### 3.3 Main Models
- Diarization: `pyannote/speaker-diarization-3.1`
- ASR: `kotoba-tech/kotoba-whisper-v2.2`

---

## 4. Input / Output Specification

### 4.1 Input
The input is a single audio file.  
The primary expected format is m4a, though other ffmpeg-readable audio formats can be supported.

Specification method:
- the file name is specified via the `INPUT_FILENAME` environment variable,
- the actual file is placed under `WORK_SOURCE`.

### 4.2 Output
Outputs are saved under `WORK_OUTPUT`.  
If `WORK_OUTPUT` is not specified, `WORK_SOURCE/output` is used.

Output files:
- `<base>.vtt`
- `<base>.txt`

Optional glossary debug log:
- `<base>.dict.log`

### 4.3 VTT Output
The VTT output includes speaker labels.  
Each segment is written using `<v speaker>` style annotations.

### 4.4 TXT Output
The TXT output is a simple one-line-per-segment format, with a speaker label at the beginning of each line.

---

## 5. Processing Flow

### 5.1 Overview
The processing flow is as follows:

1. verify input file existence,
2. convert to 16kHz mono WAV using ffmpeg,
3. load the pyannote diarization model,
4. run diarization,
5. release diarization resources,
6. load glossary if needed,
7. load the Kotoba-Whisper ASR pipeline,
8. build Layer A prompt IDs if needed,
9. run ASR per segment,
10. apply Layer B correction if needed,
11. write VTT / TXT outputs,
12. remove temporary directories.

### 5.2 Audio Conversion
The input audio is converted to 16kHz mono WAV by directly invoking ffmpeg via subprocess.  
The converted WAV is saved in a temporary directory.

### 5.3 Diarization
Speaker diarization is performed with pyannote.  
If `NUM_SPEAKERS` is specified, it is treated as a fixed speaker count; otherwise, a more automatic estimation path is used.

### 5.4 ASR
Each diarized segment is cut into a WAV chunk and passed to the Kotoba-Whisper pipeline.

---

## 6. GPU Memory Management

### 6.1 Policy
After diarization, unused diarization resources are explicitly released before ASR begins, and GPU memory is proactively cleaned up.

### 6.2 Implemented Actions
- remove references to diarization objects,
- call `gc.collect()`,
- call `torch.cuda.empty_cache()` when CUDA is available,
- log VRAM status before and after release.

### 6.3 Purpose
- reduce VRAM pressure when chaining pyannote and Kotoba-Whisper,
- improve stability even on GPUs such as the RTX 4060 Ti 16GB class,
- make memory-related failures easier to diagnose.

---

## 7. Glossary Features

### 7.1 Overview
Glossary features use a TSV-format proper noun glossary.  
The expected format is:

`formal<TAB>reading`

Example:

- 宇多津    うたづ
- 宇多津町  うたづちょう
- 浜一番丁  はまいちばんちょう

This glossary is not treated as a general-purpose dictionary, but as a **task-specific proper noun glossary**.

---

## 7.2 Layer A: pre_asr_hint

### 7.2.1 Purpose
Provide pre-ASR hints to Whisper so that important formal spellings are more likely to be considered.

### 7.2.2 Implementation Policy
- pass `prompt_ids` via `generate_kwargs`,
- do not use string-based `initial_prompt`,
- list only formal spellings separated by spaces,
- adopt only as many entries as fit within the token budget using a greedy strategy,
- make activation controllable via environment variables.

### 7.2.3 Constraints
Layer A is subject to Whisper decoder length constraints and is therefore not suitable for injecting the entire glossary.  
Accordingly, Layer A should be treated as a mechanism for passing a **small shortlist of high-priority terms**.

### 7.2.4 Fallback
Even when Layer A is enabled, if prompt-related errors occur, such as exceptions containing:

- `prompt_ids`
- `max_target_positions`
- `decoder_input_ids`
- `model_kwargs`

then Layer A is disabled, the same segment is retried without prompts, and all following segments continue with Layer A disabled.

---

## 7.3 Layer B: post_asr_correction

### 7.3.1 Purpose
After ASR output is generated, apply corrections using the glossary mapping from `reading -> formal`.

### 7.3.2 Implementation Policy
- sort entries by descending reading length before replacement,
- reduce incorrect replacements where shorter readings are contained inside longer readings,
- impose no dictionary-size limit; all entries may be used,
- make activation controllable via environment variables.

### 7.3.3 Constraints
Layer B is a context-free string replacement mechanism.  
Therefore, it carries a risk of incorrect correction.  
It is not recommended as a default always-on mode; when used, it should be validated via `.dict.log`.

---

## 7.4 Default Operation
The default settings are:

- `USE_LAYER_A=0`
- `USE_LAYER_B=0`

In other words, normal operation uses the glossary-free mode as the baseline.

Reasons:
- the latest glossary-free version was the most stable in comparison,
- Layer A / Layer B are highly task-dependent,
- more reliable proper noun correction can often be handled later by an LLM.

---

## 8. Glossary Logging

### 8.1 Purpose
Make glossary behavior reviewable after execution.

### 8.2 Output
When `DEBUG_GLOSSARY_LOG=1`, the application writes `<base>.dict.log`.

### 8.3 Layer A Logging
Example recorded items:
- total glossary size,
- token budget,
- number of adopted entries,
- adopted formal spellings,
- `prompt_ids` shape,
- whether Layer A was enabled or disabled,
- whether fallback occurred and why.

### 8.4 Layer B Logging
Example recorded items:
- segment number,
- speaker label,
- matched reading,
- replacement formal spelling,
- optionally a short text fragment.

### 8.5 Role Split Between Console and Log File
- Console: summary logs, progress, major warnings
- `.dict.log`: detailed glossary-related records

---

## 9. Environment Variables

### 9.1 Core
- `HF_TOKEN`
- `INPUT_FILENAME`
- `NUM_SPEAKERS`
- `MODEL_DIAR`
- `MODEL_ASR`
- `ASR_LANGUAGE`
- `ASR_TASK`

### 9.2 Path-Related
- `WORK_SOURCE`
- `WORK_OUTPUT`
- `WORK_HF_CACHE`
- `WORK_TMP`

### 9.3 Glossary-Related
- `DICT_PATH`
- `USE_LAYER_A`
- `USE_LAYER_B`
- `GLOSSARY_TOKEN_BUDGET`
- `LAYER_A_FALLBACK`
- `DEBUG_GLOSSARY_LOG`

---

## 10. Operational Policy

### 10.1 Normal Operation
Normal operation is glossary-free.

Expected settings:
- `USE_LAYER_A=0`
- `USE_LAYER_B=0`

### 10.2 Experimental Operation
The following can be tested independently when needed:

- Layer A only,
- Layer B only,
- both,
- `.dict.log` detail mode.

### 10.3 Recommended Comparison Patterns
When evaluating glossary features, compare at least:

1. old version,
2. new version without glossary,
3. new version with Layer A only,
4. new version with Layer B only,
5. new version with both layers.

---

## 11. Non-Goals

The following are not current goals of this application:

- injecting the entire glossary into Whisper in advance,
- IME-like forced conversion,
- LoRA / fine-tuning using paired audio and ground-truth transcripts,
- completing context-aware advanced correction solely in the ASR front stage.

These are instead handled, as much as possible, in downstream stages such as:

- Zoom VTT alignment,
- LLM-based contextual correction.

---

## 12. Future Extension Candidates

- separate shortlist glossary for Layer A,
- stronger false-replacement protection in Layer B,
- aggregated glossary hit statistics,
- automatic recording of fallback frequency,
- stronger integration with the Zoom VTT alignment tool.

---

## 13. Conclusion

This application is a **stable operational baseline centered on the glossary-free latest version**,  
while glossary features are treated as optional auxiliary components that can be switched on and off as Layer A / Layer B.

Rather than trying to solve proper noun correction entirely in the ASR front stage,  
the intended goal is to combine this application with downstream Zoom VTT alignment and LLM-based contextual correction  
to achieve practically usable interview transcription as a whole.