# kotoba-pyannote-diar-asr / Kotoba Whisper と pyannote を連携して話者分離＋文字起こしをする(Windows+Docker+NVIDIA GPU)
End-to-end pipeline for speaker diarization (pyannote) and Japanese transcription (Kotoba Whisper) using Docker on Windows 11 (WSL2) with NVIDIA GPU; outputs TXT/SRT with speaker labels.
Windows 11（WSL2）+ Docker + NVIDIA GPUで、pyannote話者分離→Kotoba Whisper文字起こしを一括実行し、話者ラベル付きTXT/SRTを生成する。 

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
New-Item -ItemType Directory -Force -Path "$base\work\source\output" | Out-Null
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
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  kotoba-diar-asr:cu126
```

Outputs will be written to:

* `D:\asr\work\source\output\meeting_sample.txt`
* `D:\asr\work\source\output\meeting_sample.srt`

---

## Configuration (environment variables)

* `HF_TOKEN`: Hugging Face token (often required)
* `INPUT_FILENAME`: input audio file name under `/work/source`
* `NUM_SPEAKERS`: e.g. `2` (empty or `auto` for auto mode)
* `MODEL_DIAR`: diarization model (default: `pyannote/speaker-diarization-3.1`)
* `MODEL_ASR`: ASR model (default: `kotoba-tech/kotoba-whisper-v2.2`)
* `TORCH_LOAD_WEIGHTS_ONLY`: set to `1` to disable the torch.load patch (debug use)

---

## Notes

* VRAM usage can be high because diarization and ASR models are loaded in a single run.
* This repo includes a compatibility patch that forces `torch.load(..., weights_only=False)` to avoid certain loading issues depending on your environment.
* Do not store private audio/transcripts in this repository.

---

# 日本語（後半）

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
New-Item -ItemType Directory -Force -Path "$base\work\source\output" | Out-Null
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
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  kotoba-diar-asr:cu126
```

出力先：

* `D:\asr\work\source\output\meeting_sample.txt`
* `D:\asr\work\source\output\meeting_sample.srt`

---

## 設定（環境変数）

* `HF_TOKEN`：Hugging Face トークン（必須になりやすい）
* `INPUT_FILENAME`：入力音声ファイル名（`/work/source` からの相対）
* `NUM_SPEAKERS`：話者数（例：`2`。空や `auto` で推定寄り）
* `MODEL_DIAR`：話者分離モデル（既定：`pyannote/speaker-diarization-3.1`）
* `MODEL_ASR`：ASRモデル（既定：`kotoba-tech/kotoba-whisper-v2.2`）
* `TORCH_LOAD_WEIGHTS_ONLY`：`1` で torch.load パッチを抑止（切り分け用）

---

## 補足

* 話者分離とASRを同一実行で回すため、VRAM消費は大きくなりやすい。
* 環境によってモデルロードが失敗するケースがあるため、互換性目的で `torch.load(..., weights_only=False)` を強制するパッチを入れている。
* 音声や議事録は個人情報・機密情報が混ざりやすいので、リポジトリ外で管理するのが安全である。

```

