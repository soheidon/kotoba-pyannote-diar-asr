# ================================
# diar_asr.py  (Docker版 / Dドライブキャッシュ・tmp対応 / エラー追跡強化版)
#  - pyannote diarization + kotoba-whisper ASR
#  - PyTorch 2.6+ "weights_only=True" 既定変更による UnpicklingError を回避
#  - 入力(m4a等)→WAV変換は ffmpeg を subprocess で直叩き
#  - エラー時の Traceback 取得と VRAM 使用量のロギングを追加
#  - 出力を VTT および TXT 形式に変更
# ================================

import os
import time
import datetime
import tempfile
import shutil
import warnings
import subprocess
import traceback
from pathlib import Path

# ---- quiet warnings ----
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================================
# Paths (Docker volume mount前提)
# ================================
WORK_SOURCE = Path(os.getenv("WORK_SOURCE", "/work/source"))
WORK_OUTPUT = WORK_SOURCE / "output"
WORK_HF     = Path(os.getenv("WORK_HF_CACHE", "/work/hf_cache"))
WORK_TMP    = Path(os.getenv("WORK_TMP", "/work/tmp"))

WORK_OUTPUT.mkdir(parents=True, exist_ok=True)
WORK_HF.mkdir(parents=True, exist_ok=True)
WORK_TMP.mkdir(parents=True, exist_ok=True)

# Hugging Face cache / Transformers cache をD側に寄せる
os.environ.setdefault("HF_HOME", str(WORK_HF))
os.environ.setdefault("HF_HUB_CACHE", str(WORK_HF / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(WORK_HF / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORK_HF))

# temp dir をD側に寄せる
os.environ.setdefault("TMPDIR", str(WORK_TMP))
os.environ.setdefault("TEMP", str(WORK_TMP))
os.environ.setdefault("TMP", str(WORK_TMP))

def log(msg: str):
    print(msg, flush=True)

# ================================
# Torch import + PyTorch 2.6+ workaround
# ================================
import torch

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([torch.torch_version.TorchVersion])
except Exception:
    pass

try:
    from torch.serialization import add_safe_globals
    from pyannote.audio.core.task import Specifications, Problem
    add_safe_globals([Specifications, Problem])
except Exception:
    pass

_force_legacy = os.getenv("TORCH_LOAD_WEIGHTS_ONLY", "0").strip().lower() in ("0", "false", "no", "off")
if _force_legacy:
    _orig_torch_load = torch.load

    def _torch_load_legacy(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_legacy
    log("[INFO] Patched torch.load(weights_only=False) for PyTorch 2.6+ compatibility.")
else:
    log("[INFO] torch.load patch disabled (TORCH_LOAD_WEIGHTS_ONLY!=0).")

# ================================
# HF login
# ================================
from huggingface_hub import login, HfApi

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    log("[WARN] 環境変数 HF_TOKEN が未設定です。docker run に -e HF_TOKEN=... を付けてください。")
else:
    login(token=hf_token)

try:
    who = HfApi().whoami(token=hf_token or None)
    who_str = who.get("name") or who.get("email") or who.get("username") or "Unknown"
    log(f"[OK] HF login: {who_str}")
except Exception:
    log("[WARN] whoami 取得に失敗。すでに認証済みであれば問題ありません。")

# ================================
# Settings
# ================================
INPUT_FILENAME = os.getenv("INPUT_FILENAME", "2026-02-11-utazu.m4a")
INPUT_FILE = WORK_SOURCE / INPUT_FILENAME

NUM_SPEAKERS_ENV = os.getenv("NUM_SPEAKERS", "").strip()
NUM_SPEAKERS = None if NUM_SPEAKERS_ENV in ("", "none", "null", "auto") else int(NUM_SPEAKERS_ENV)

MODEL_DIAR = os.getenv("MODEL_DIAR", "pyannote/speaker-diarization-3.1")
MODEL_ASR  = os.getenv("MODEL_ASR",  "kotoba-tech/kotoba-whisper-v2.2")

LANG = os.getenv("ASR_LANGUAGE", "japanese")
TASK = os.getenv("ASR_TASK", "transcribe")
LANG_KW = {"language": LANG, "task": TASK}

device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
log(f"[INFO] device={'cuda:0' if device==0 else 'cpu'}, torch_dtype={torch_dtype}")

log(f"[CFG] INPUT_FILENAME='{INPUT_FILENAME}'")
log(f"[CFG] NUM_SPEAKERS_ENV='{NUM_SPEAKERS_ENV}' -> NUM_SPEAKERS={NUM_SPEAKERS}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"入力が見つかりません: {INPUT_FILE}")

# ================================
# Imports for diarization + ASR
# ================================
from transformers import pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm.auto import tqdm

# ================================
# Convert to WAV (16k/mono)
# ================================
log("[STEP] 入力を WAV(16kHz mono) に変換...")
tmp_in_dir = tempfile.mkdtemp(prefix="wav_", dir=str(WORK_TMP))
wav16 = Path(tmp_in_dir) / "input_16k_mono.wav"

cmd = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-y",
    "-i", str(INPUT_FILE),
    "-ac", "1",
    "-ar", "16000",
    str(wav16),
]
subprocess.run(cmd, check=True)

audio16 = AudioSegment.from_file(str(wav16))
log("[OK] 変換完了")

# ================================
# Diarization
# ================================
log("[STEP] 話者分離モデルをロード...")
try:
    dia = Pipeline.from_pretrained(MODEL_DIAR, use_auth_token=hf_token)
    if dia is None:
        raise RuntimeError("Pipelineのロードに失敗しました。モデル名やHF_TOKENの権限を確認してください。")
    
    if device == 0:
        dia.to(torch.device("cuda"))
        
    if hasattr(dia, "embedding_batch_size"):
        dia.embedding_batch_size = 1
except Exception as e:
    log(f"[FATAL ERROR] モデルロード中に例外発生:\n{traceback.format_exc()}")
    raise

log("[STEP] 話者分離を実行...")
t0 = time.time()
try:
    if device == 0:
        log(f"[VRAM] Allocated before DIAR: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    if NUM_SPEAKERS is None:
        diar = dia(str(wav16))
    else:
        diar = dia(
            str(wav16), 
            num_speakers=int(NUM_SPEAKERS), 
            min_speakers=int(NUM_SPEAKERS), 
            max_speakers=int(NUM_SPEAKERS)
        )

    segments = list(diar.itertracks(yield_label=True))
    log(f"[OK] 話者分離: {len(segments)} セグメント ({time.time()-t0:.2f}s)")

    labels = sorted({lbl for _, _, lbl in segments})
    log(f"[DIAR] labels={labels} N={len(labels)}")

except Exception as e:
    log(f"[FATAL ERROR] 話者分離の実行中に例外発生:\n{traceback.format_exc()}")
    raise

# ================================
# ASR pipeline
# ================================
log("[STEP] Whisper をロード...")
try:
    asr = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ASR,
        device=device,
        trust_remote_code=True,
        return_timestamps=True,
        torch_dtype=torch_dtype,
    )
except Exception as e:
    log(f"[FATAL ERROR] Whisperモデルロード中に例外発生:\n{traceback.format_exc()}")
    raise

log("[STEP] セグメントごとにASR中...")

tmp_seg_dir = tempfile.mkdtemp(prefix="seg_", dir=str(WORK_TMP))
results = []

try:
    for idx, (turn, _, spk) in enumerate(tqdm(segments, desc="ASR", leave=True), 1):
        st, ed = float(turn.start), float(turn.end)
        if ed <= st:
            continue

        seg = audio16[int(st * 1000) : int(ed * 1000)]
        seg_path = Path(tmp_seg_dir) / f"seg_{idx:04d}.wav"
        seg.export(str(seg_path), format="wav")

        out = asr(str(seg_path), return_timestamps=True, generate_kwargs=LANG_KW)
        text = (out.get("text") or "").strip()
        results.append({"speaker": spk, "start": st, "end": ed, "text": text})
except Exception as e:
    log(f"[FATAL ERROR] ASR推論中に例外発生:\n{traceback.format_exc()}")
    raise

# ================================
# Write outputs
# ================================
base = INPUT_FILE.stem

def fmt_vtt(t):
    td = datetime.timedelta(seconds=float(t))
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    ms = td.microseconds // 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

vtt_lines = ["WEBVTT\n"]
for i, r in enumerate(results, 1):
    vtt_lines.append(
        f"{i}\n{fmt_vtt(r['start'])} --> {fmt_vtt(r['end'])}\n<v {r['speaker']}>{r['text']}</v>\n"
    )

(WORK_OUTPUT / f"{base}.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")

with (WORK_OUTPUT / f"{base}.txt").open("w", encoding="utf-8") as f:
    for r in results:
        f.write(f"[{r['speaker']}] {r['text']}\n")

# ================================
# Cleanup
# ================================
shutil.rmtree(tmp_seg_dir, ignore_errors=True)
shutil.rmtree(tmp_in_dir, ignore_errors=True)

log(f"[OK] 出力: {WORK_OUTPUT}")
log("[DONE] 完了")