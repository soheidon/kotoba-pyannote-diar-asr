# kotoba-pyannote-diar-asr
Dockerized speaker diarization + Japanese ASR pipeline (**pyannote + Kotoba Whisper**) for **Windows 11 (WSL2) + NVIDIA GPU**.

This repository runs an end-to-end workflow:

1. Convert the input audio to **16kHz mono WAV**
2. Run **speaker diarization** with `pyannote/speaker-diarization-3.1`
3. Transcribe each diarized segment with **Kotoba Whisper** (`kotoba-tech/kotoba-whisper-v2.2`)
4. Export speaker-labeled **TXT** and **SRT**

> Note: Do **NOT** commit audio files, transcripts, logs, caches, or tokens. This repo is designed to keep those artifacts outside Git by default.

---

## Requirements

- Windows 11
- Docker Desktop (WSL2 backend)
- NVIDIA GPU + recent driver
- PowerShell

### Quick sanity check

```powershell
docker context show
wsl -l -v
nvidia-smi

# Check GPU visibility inside Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
````

---

## Storage layout (recommended)

This pipeline performs a lot of I/O for model caches and temporary WAV chunks.

* **Recommended on SSD (strongly):**

  * Hugging Face cache (`hf_cache`)
  * Temporary files (`tmp`)
* **HDD is acceptable (may be slower):**

  * Input audio and output texts

Example paths (change as you like):

* Input/Output: `D:\asr\work\source`
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

`pyannote/speaker-diarization-3.1` often requires accepting its terms on Hugging Face.
Log in on the website, open the model page, and accept the terms first.

Then set `HF_TOKEN` as a **user environment variable**:

```powershell
# Put your actual token here (do NOT commit/share it)
$token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Persist as a user environment variable
[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")

# Open a NEW PowerShell and confirm:
$env:HF_TOKEN
```

---

## Build

```powershell
cd <path-to-this-repo>
docker build -t kotoba-diar-asr:cu126 .
```

---

## Run

Put an audio file (example: `meeting_sample.m4a`) into `D:\asr\work\source\`.

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\dict:/work/dict `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e DICT_PATH="/work/dict/glossary_confirmed.tsv" `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

Omit `-e DICT_PATH=...` and the `-v D:\asr\work\dict:/work/dict` mount for dictionary-off mode.  
Omit `-e WORK_OUTPUT=...` and `-v D:\asr\work\output:/work/output` to use `WORK_SOURCE/output` (e.g. `D:\asr\work\source\output`).

Outputs will be written to:

* `D:\asr\work\output\meeting_sample.txt`
* `D:\asr\work\output\meeting_sample.vtt`

---

## Configuration (environment variables)

* `HF_TOKEN`: Hugging Face token (often required)
* `WORK_SOURCE`: input directory (default: `/work/source`)
* `WORK_OUTPUT`: output directory (default: `WORK_SOURCE/output`). Set to `/work/output` for a sibling output dir.
* `INPUT_FILENAME`: input audio file name under `WORK_SOURCE`
* `NUM_SPEAKERS`: e.g. `2` (empty or `auto` for auto mode)
* `DICT_PATH`: path to glossary TSV (formal<TAB>reading). Omit for dictionary-off mode.
* `GLOSSARY_TOKEN_BUDGET`: token budget for pre-ASR hint (default: 100). Keeps prompt within Whisper's limit.
* `MODEL_DIAR`: diarization model (default: `pyannote/speaker-diarization-3.1`)
* `MODEL_ASR`: ASR model (default: `kotoba-tech/kotoba-whisper-v2.2`)
* `TORCH_LOAD_WEIGHTS_ONLY`: set to `1` to disable the torch.load patch (debug use)

---

## Dictionary mode (optional)

Set `DICT_PATH` to a TSV file with `formal<TAB>reading` per line (e.g. `宇多津\tうたづ`).
When set, the pipeline:
1. Uses the glossary as a `prompt_ids` hint for Whisper (greedy selection within token budget)
2. Applies post-inference correction (reading → formal) on ASR output

A `*.dict.log` file is written to the output folder (e.g. `2026-03-06-MIYAGAWA.dict.log`) with detailed records of Layer A (adopted terms, token budget) and Layer B (each replacement: seg, speaker, reading, formal). Use it to verify dictionary behavior.

GPU memory is released after diarization, before ASR loading, to reduce peak VRAM usage.

## Notes

* VRAM usage is reduced by releasing diarization resources before loading Whisper.
* This repo includes a compatibility patch that forces `torch.load(..., weights_only=False)` to avoid certain loading issues depending on your environment.
* Do not store private audio/transcripts in this repository.

---

# 使用法

Windows 11 + Docker Desktop（WSL2）+ NVIDIA GPU 環境で、**pyannote による話者分離**と **Kotoba Whisper による文字起こし**を連携して実行するためのリポジトリである。

処理の流れは次の通りである。

1. 入力音声を **16kHz / mono の WAV** に変換する
2. `pyannote/speaker-diarization-3.1` で **話者分離**を行う
3. 分割区間ごとに `kotoba-tech/kotoba-whisper-v2.2` で **文字起こし**する
4. 話者ラベル付きの **TXT** と **SRT** を出力する

> 注意：音声ファイル、議事録、ログ、キャッシュ、トークンはコミットしてはいけない。個人情報・機密情報が混ざりやすいためである。本リポジトリはそれらをGit追跡から外す前提で構成している。

---

## 前提

* Windows 11
* Docker Desktop（WSL2 backend）
* NVIDIA GPU + ドライバ
* PowerShell

### 動作確認（最低限）

```powershell
docker context show
wsl -l -v
nvidia-smi

# Docker上でGPUが見えるか
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## ストレージ設計（おすすめ）

モデルキャッシュと一時WAVによりI/Oが増えるため、置き場所で体感速度が変わる。

* **SSD推奨（強い）**

  * Hugging Face キャッシュ（`hf_cache`）
  * 一時ファイル（`tmp`）
* **HDDでも成立しやすい（遅くなる可能性あり）**

  * 入力音声・出力（SRT/TXT）

例（任意に変更してよい）：

* 入力/出力：`D:\asr\work\source`
* HFキャッシュ：`D:\asr\hf_cache`
* 一時ファイル：`D:\asr\tmp`

作成コマンド：

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

`pyannote/speaker-diarization-3.1` は、Hugging Face 上で利用規約同意が必要になることが多い。
事前にブラウザでログインし、モデルページで同意しておくこと。

そのうえで `HF_TOKEN` をユーザー環境変数として登録する。

```powershell
# 自分のトークンを入れる（絶対に公開しない）
$token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ユーザー環境変数として永続化
[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")

# 新しいPowerShellを開き直して確認
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

入力音声（例：`meeting_sample.m4a`）を `D:\asr\work\source\` に置く。

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\dict:/work/dict `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e DICT_PATH="/work/dict/glossary_confirmed.tsv" `
  -e WORK_OUTPUT="/work/output" `
  kotoba-diar-asr:cu126
```

辞書を使わない場合は `-e DICT_PATH=...` と `-v D:\asr\work\dict:/work/dict` を省く。  
`WORK_SOURCE/output` に出力したい場合は `-e WORK_OUTPUT=...` と `-v D:\asr\work\output:/work/output` を省く。

出力先：

* `D:\asr\work\output\meeting_sample.txt`
* `D:\asr\work\output\meeting_sample.vtt`

---

## 設定（環境変数）

* `HF_TOKEN`：Hugging Face トークン（必須になりやすい）
* `WORK_SOURCE`：入力ディレクトリ（既定：`/work/source`）
* `WORK_OUTPUT`：出力ディレクトリ（既定：`WORK_SOURCE/output`）。`/work/output` で兄弟フォルダに出力可
* `INPUT_FILENAME`：入力音声ファイル名（`WORK_SOURCE` からの相対）
* `NUM_SPEAKERS`：話者数（例：`2`。空や `auto` で推定寄り）
* `DICT_PATH`：辞書 TSV のパス（正式表記<TAB>よみ）。未指定で辞書なしモード
* `GLOSSARY_TOKEN_BUDGET`：推論前ヒントのトークンバジェット（既定：100）
* `MODEL_DIAR`：話者分離モデル（既定：`pyannote/speaker-diarization-3.1`）
* `MODEL_ASR`：ASRモデル（既定：`kotoba-tech/kotoba-whisper-v2.2`）
* `TORCH_LOAD_WEIGHTS_ONLY`：`1` で torch.load パッチを抑止（切り分け用）

---

## 辞書ありモード（任意）

`DICT_PATH` に TSV ファイル（正式表記<TAB>よみ、例：`宇多津\tうたづ`）のパスを指定すると：
1. 辞書を Whisper の `prompt_ids` ヒントとして使用（トークンバジェット内で greedy 採用）
2. ASR 出力に対して推論後補正（読み→正式表記）を適用

出力フォルダに `*.dict.log`（例：`2026-03-06-MIYAGAWA.dict.log`）が出力され、Layer A の採用語・Layer B の各置換が記録される。

話者分離終了後、ASR 開始前に GPU メモリを解放し、ピーク VRAM 使用量を抑えている。

## 補足

* 話者分離終了後に GPU メモリ解放を行うため、ピーク VRAM は従来より抑えられている。
* 環境によってモデルロードが失敗するケースがあるため、互換性目的で `torch.load(..., weights_only=False)` を強制するパッチを入れている。
* 音声や議事録は個人情報・機密情報が混ざりやすいので、リポジトリ外で管理するのが安全である。



