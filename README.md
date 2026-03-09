# kotoba-pyannote-diar-asr

Windows 11 + Docker Desktop（WSL2）+ NVIDIA GPU 環境で、**pyannote による話者分離**と **Kotoba Whisper による日本語文字起こし**を連携して実行するための Docker ベースのツールである。

このリポジトリは、次の一連の処理を行う。

1. 入力音声を **16kHz / mono の WAV** に変換する  
2. `pyannote/speaker-diarization-3.1` で **話者分離**を行う  
3. 分割した各区間を `kotoba-tech/kotoba-whisper-v2.2` で **文字起こし**する  
4. 話者ラベル付きの **TXT** と **VTT** を出力する  

> 注意: 音声ファイル、文字起こし結果、ログ、キャッシュ、トークンは Git にコミットしてはいけない。個人情報・機密情報が含まれやすいためである。本リポジトリは、それらを Git 管理外に置く前提で設計している。

---

## 前提環境

- Windows 11
- Docker Desktop（WSL2 backend）
- NVIDIA GPU + 比較的新しいドライバ
- PowerShell

### 簡易動作確認

```powershell
docker context show
wsl -l -v
nvidia-smi

# Docker 上で GPU が見えているか確認
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
````

---

## 推奨ディレクトリ構成

モデルキャッシュや一時 WAV により I/O が増えるため、保存先によって体感速度が変わる。

* **SSD 強推奨**

  * Hugging Face キャッシュ（`hf_cache`）
  * 一時ファイル（`tmp`）
* **HDD でも可（遅くなる可能性あり）**

  * 入力音声
  * 出力テキスト

例:

* 入力/出力: `D:\asr\work\source`
* 辞書: `D:\asr\work\dict`
* HF キャッシュ: `D:\asr\hf_cache`
* 一時ファイル: `D:\asr\tmp`

作成例:

```powershell
$base = "D:\asr"
New-Item -ItemType Directory -Force -Path "$base\work\source" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\work\output" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\work\dict" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\hf_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\tmp" | Out-Null
```

---

## Hugging Face トークン（HF_TOKEN）

`pyannote/speaker-diarization-3.1` は Hugging Face 上で利用規約への同意が必要なことが多い。
事前にブラウザでログインし、モデルページで利用規約に同意しておくこと。

その後、`HF_TOKEN` をユーザー環境変数として設定する。

```powershell
# 実際のトークンを入れる（公開しないこと）
$token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ユーザー環境変数として保存
[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")

# 新しい PowerShell を開き直して確認
$env:HF_TOKEN
```

---

## ビルド

```powershell
cd <このリポジトリのフォルダ>
docker build -t kotoba-diar-asr:cu126 .
```

---

## 実行

入力音声（例: `meeting_sample.m4a`）を `D:\asr\work\source\` に置く。

### 通常運用（辞書なし・推奨）

辞書関連機能は**デフォルトでオフ**であり、このモードを安定した基準版とする。

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

### 辞書を使う場合（任意）

辞書機能は補助機能であり、必要時のみ有効にする。

| 目的                   | 追加するオプション                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Layer A のみ試す（推論前ヒント） | `-v D:\asr\work\dict:/work/dict` `-e DICT_PATH="/work/dict/glossary_confirmed.tsv"` `-e USE_LAYER_A=1` |
| Layer B のみ試す（推論後補正）  | `-v D:\asr\work\dict:/work/dict` `-e DICT_PATH="/work/dict/glossary_confirmed.tsv"` `-e USE_LAYER_B=1` |
| 両方試す                 | `-e USE_LAYER_A=1` `-e USE_LAYER_B=1`（加えて `DICT_PATH` と volume mount）                                  |
| 辞書ログを出す              | `-e DEBUG_GLOSSARY_LOG=1`                                                                              |

Layer A + Layer B + 辞書ログの例:

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -e USE_LAYER_A=1 `
  -e USE_LAYER_B=1 `
  -e DEBUG_GLOSSARY_LOG=1 `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\dict:/work/dict `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e DICT_PATH="/work/dict/glossary_confirmed.tsv" `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

`WORK_OUTPUT` を省略した場合は、既定で `WORK_SOURCE/output` を使う。

出力例:

* `D:\asr\work\output\meeting_sample.txt`
* `D:\asr\work\output\meeting_sample.vtt`
* `D:\asr\work\output\meeting_sample.dict.log`
  （`DEBUG_GLOSSARY_LOG=1` かつ辞書レイヤー有効時のみ）

---

## 環境変数

### 基本

* `HF_TOKEN`: Hugging Face トークン
* `INPUT_FILENAME`: 入力音声ファイル名
* `NUM_SPEAKERS`: 話者数（例: `2`、空または `auto` で自動寄り）
* `MODEL_DIAR`: 話者分離モデル（既定: `pyannote/speaker-diarization-3.1`）
* `MODEL_ASR`: ASR モデル（既定: `kotoba-tech/kotoba-whisper-v2.2`）
* `ASR_LANGUAGE`: ASR 言語（既定: `japanese`）
* `ASR_TASK`: ASR タスク（既定: `transcribe`）

### パス関連

* `WORK_SOURCE`: 入力ディレクトリ（既定: `/work/source`）
* `WORK_OUTPUT`: 出力ディレクトリ（既定: `WORK_SOURCE/output`）
* `WORK_HF_CACHE`: Hugging Face キャッシュ（既定: `/work/hf_cache`）
* `WORK_TMP`: 一時ファイル保存先（既定: `/work/tmp`）

### 辞書関連

* `DICT_PATH`: 辞書 TSV のパス（`正式表記<TAB>よみ`）
* `USE_LAYER_A`: `1` で Layer A 有効、既定 `0`
* `USE_LAYER_B`: `1` で Layer B 有効、既定 `0`
* `LAYER_A_FALLBACK`: `1` で Layer A エラー時に自動無効化、既定 `1`
* `DEBUG_GLOSSARY_LOG`: `1` で `*.dict.log` 出力、既定 `0`
* `GLOSSARY_TOKEN_BUDGET`: Layer A の token budget、既定 `100`

### デバッグ用

* `TORCH_LOAD_WEIGHTS_ONLY`: `1` で `torch.load(..., weights_only=False)` パッチを無効化

---

## 辞書機能について

辞書機能は**デフォルトでオフ**である。
通常運用では、まず辞書なし版を基準とする。

### Layer A: 推論前ヒント

* Whisper に `prompt_ids` を渡す
* 少数の優先語だけを対象にする
* token budget 制約があるため、辞書全件投入には向かない
* エラー時は fallback で自動無効化可能

### Layer B: 推論後補正

* `よみ -> 正式表記` の単純置換
* 全件辞書を使用可能
* 文脈を見ないため、誤補正の可能性がある
* 必要なら `.dict.log` で置換内容を確認する

### 推奨方針

* 通常運用: `USE_LAYER_A=0`, `USE_LAYER_B=0`
* 実験的に Layer A / Layer B を個別に試す
* 本格的な文脈補正は、後段の Zoom VTT 突合や LLM 修正で扱う

---

## ログ

### 通常ログ

コンソールには進行状況、警告、エラー要約を出す。

### 辞書ログ

`DEBUG_GLOSSARY_LOG=1` のとき、`*.dict.log` を出力する。
ここには少なくとも以下を記録する。

* Layer A の採用語
* token budget
* prompt_ids の shape
* Layer A fallback の有無
* Layer B で適用した `reading -> formal`

---

## 補足

* 話者分離後に GPU メモリ解放を行い、ASR 実行前の VRAM 圧迫を抑えている
* 一部環境でのモデルロード問題に備え、`torch.load(..., weights_only=False)` の互換パッチを含む
* 音声や文字起こしは個人情報・機密情報を含みやすいため、リポジトリ外で管理することを推奨する

---

# README (English)

# kotoba-pyannote-diar-asr

A Docker-based tool for running **speaker diarization with pyannote** and **Japanese ASR with Kotoba Whisper** on **Windows 11 + Docker Desktop (WSL2) + NVIDIA GPU**.

This repository performs the following workflow:

1. Convert the input audio to **16kHz mono WAV**
2. Run **speaker diarization** with `pyannote/speaker-diarization-3.1`
3. Transcribe each diarized segment with `kotoba-tech/kotoba-whisper-v2.2`
4. Export speaker-labeled **TXT** and **VTT**

> Note: Do **not** commit audio files, transcripts, logs, caches, or tokens. This repository is designed on the assumption that such artifacts remain outside Git.

---

## Requirements

* Windows 11
* Docker Desktop (WSL2 backend)
* NVIDIA GPU with a reasonably recent driver
* PowerShell

### Quick sanity check

```powershell
docker context show
wsl -l -v
nvidia-smi

# Check whether Docker can see the GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## Recommended storage layout

This pipeline performs substantial I/O for model caches and temporary WAV chunks, so storage placement affects practical speed.

* **Strongly recommended on SSD**

  * Hugging Face cache (`hf_cache`)
  * Temporary files (`tmp`)
* **Can be on HDD (possibly slower)**

  * Input audio
  * Output texts

Example paths:

* Input/Output: `D:\asr\work\source`
* Dictionary: `D:\asr\work\dict`
* HF cache: `D:\asr\hf_cache`
* Temp: `D:\asr\tmp`

Create directories:

```powershell
$base = "D:\asr"
New-Item -ItemType Directory -Force -Path "$base\work\source" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\work\output" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\work\dict" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\hf_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\tmp" | Out-Null
```

---

## Hugging Face token (HF_TOKEN)

`pyannote/speaker-diarization-3.1` often requires accepting the model terms on Hugging Face.
Sign in on the Hugging Face website and accept the terms on the model page first.

Then set `HF_TOKEN` as a user environment variable:

```powershell
# Put your real token here (do not share it)
$token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Persist as a user environment variable
[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")

# Open a new PowerShell and confirm
$env:HF_TOKEN
```

---

## Build

```powershell
cd <path-to-this-repository>
docker build -t kotoba-diar-asr:cu126 .
```

---

## Run

Place an input audio file (for example, `meeting_sample.m4a`) in `D:\asr\work\source\`.

### Normal operation (no dictionary, recommended)

Dictionary-related features are **off by default**. This mode should be treated as the stable baseline.

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

### With dictionary features (optional)

Dictionary features are optional and should only be enabled when needed.

| Goal                                   | Add these options                                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Try Layer A only (pre-ASR hint)        | `-v D:\asr\work\dict:/work/dict` `-e DICT_PATH="/work/dict/glossary_confirmed.tsv"` `-e USE_LAYER_A=1` |
| Try Layer B only (post-ASR correction) | `-v D:\asr\work\dict:/work/dict` `-e DICT_PATH="/work/dict/glossary_confirmed.tsv"` `-e USE_LAYER_B=1` |
| Try both layers                        | `-e USE_LAYER_A=1` `-e USE_LAYER_B=1` (plus `DICT_PATH` and dictionary volume mount)                   |
| Write dictionary debug log             | `-e DEBUG_GLOSSARY_LOG=1`                                                                              |

Example with both layers enabled and dictionary log output:

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -e USE_LAYER_A=1 `
  -e USE_LAYER_B=1 `
  -e DEBUG_GLOSSARY_LOG=1 `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\dict:/work/dict `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e DICT_PATH="/work/dict/glossary_confirmed.tsv" `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

If `WORK_OUTPUT` is omitted, the default output path is `WORK_SOURCE/output`.

Example outputs:

* `D:\asr\work\output\meeting_sample.txt`
* `D:\asr\work\output\meeting_sample.vtt`
* `D:\asr\work\output\meeting_sample.dict.log`
  (only when `DEBUG_GLOSSARY_LOG=1` and at least one dictionary layer is enabled)

---

## Environment variables

### Core

* `HF_TOKEN`: Hugging Face token
* `INPUT_FILENAME`: input audio file name
* `NUM_SPEAKERS`: number of speakers (for example `2`; empty or `auto` for more automatic behavior)
* `MODEL_DIAR`: diarization model (default: `pyannote/speaker-diarization-3.1`)
* `MODEL_ASR`: ASR model (default: `kotoba-tech/kotoba-whisper-v2.2`)
* `ASR_LANGUAGE`: ASR language (default: `japanese`)
* `ASR_TASK`: ASR task (default: `transcribe`)

### Paths

* `WORK_SOURCE`: input directory (default: `/work/source`)
* `WORK_OUTPUT`: output directory (default: `WORK_SOURCE/output`)
* `WORK_HF_CACHE`: Hugging Face cache directory (default: `/work/hf_cache`)
* `WORK_TMP`: temp directory (default: `/work/tmp`)

### Dictionary-related

* `DICT_PATH`: dictionary TSV path (`formal<TAB>reading`)
* `USE_LAYER_A`: enable Layer A when set to `1`, default `0`
* `USE_LAYER_B`: enable Layer B when set to `1`, default `0`
* `LAYER_A_FALLBACK`: automatically disable Layer A on prompt-related errors when set to `1`, default `1`
* `DEBUG_GLOSSARY_LOG`: write `*.dict.log` when set to `1`, default `0`
* `GLOSSARY_TOKEN_BUDGET`: token budget for Layer A, default `100`

### Debugging

* `TORCH_LOAD_WEIGHTS_ONLY`: disable the `torch.load(..., weights_only=False)` compatibility patch when set to `1`

---

## About dictionary features

Dictionary features are **off by default**.
The recommended baseline is the dictionary-free mode.

### Layer A: pre-ASR hint

* Passes `prompt_ids` to Whisper
* Intended for a small shortlist of high-priority terms
* Not suitable for injecting the whole dictionary
* Can be automatically disabled by fallback on prompt-related errors

### Layer B: post-ASR correction

* Applies simple `reading -> formal` replacement
* Can use the full dictionary
* Does not look at context and may produce incorrect replacements
* Should be validated with `.dict.log` if enabled

### Recommended usage

* Normal operation: `USE_LAYER_A=0`, `USE_LAYER_B=0`
* Test Layer A and Layer B separately when needed
* Handle heavier context-sensitive correction in downstream Zoom VTT alignment and LLM-based post-editing

---

## Logs

### Standard logs

The console prints progress, warnings, and error summaries.

### Dictionary log

When `DEBUG_GLOSSARY_LOG=1`, the tool writes `*.dict.log`.
This log may include:

* Layer A selected terms
* token budget
* `prompt_ids` shape
* whether Layer A fallback occurred
* `reading -> formal` replacements applied by Layer B

---

## Notes

* GPU memory is explicitly released after diarization and before ASR to reduce peak VRAM usage
* The repository includes a compatibility patch for `torch.load(..., weights_only=False)` to avoid model-loading issues in some environments
* Audio and transcript files may contain private or sensitive information and should be managed outside the repository