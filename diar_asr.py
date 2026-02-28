import os
import time
import datetime
import tempfile
import shutil
import warnings
import subprocess
import gc
from pathlib import Path

# ---- logging / warnings -------------------------------------------------------
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def log(msg: str):
    print(msg, flush=True)

# ---- path settings ------------------------------------------------------------
# コンテナ内パス。ホスト側の実フォルダは docker run の -v で割り当てる。
WORK_SOURCE = Path(os.getenv("WORK_SOURCE", "/work/source"))
WORK_OUTPUT = WORK_SOURCE / "output"
WORK_HF     = Path(os.getenv("WORK_HF_CACHE", "/work/hf_cache"))
WORK_TMP    = Path(os.getenv("WORK_TMP", "/work/tmp"))

WORK_OUTPUT.mkdir(parents=True, exist_ok=True)
WORK_HF.mkdir(parents=True, exist_ok=True)
WORK_TMP.mkdir(parents=True, exist_ok=True)

# Hugging Face cache and temp redirection
# 注：この3つ（source/hf_cache/tmp）を別ドライブへ逃がすのが狙いである。
os.environ.update({
    "HF_HOME": str(WORK_HF),
    "HF_HUB_CACHE": str(WORK_HF / "hub"),
    "TRANSFORMERS_CACHE": str(WORK_HF / "transformers"),
    "XDG_CACHE_HOME": str(WORK_HF),
    "TMPDIR": str(WORK_TMP), "TEMP": str(WORK_TMP), "TMP": str(WORK_TMP),
})

import torch

# ---- PyTorch load workaround --------------------------------------------------
# 注：環境や組合せによって、weights_only の既定挙動が原因でロードが落ちることがある。
#     互換性重視で weights_only=False を強制する（切り分けは env で可能）。
try:
    from torch.serialization import add_safe_globals
    from pyannote.audio.core.task import Specifications, Problem
    add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem])
except Exception:
    pass

if os.getenv("TORCH_LOAD_WEIGHTS_ONLY", "0").lower() in ("0", "false"):
    _orig_torch_load = torch.load
    torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})
    log("[INFO] Patched torch.load(weights_only=False)")

from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# ---- parameters ---------------------------------------------------------------
INPUT_FILENAME = os.getenv("INPUT_FILENAME", "meeting_sample.m4a")
INPUT_FILE = WORK_SOURCE / INPUT_FILENAME

NUM_SPEAKERS_ENV = os.getenv("NUM_SPEAKERS", "").strip()
NUM_SPEAKERS = None if NUM_SPEAKERS_ENV in ("", "none", "auto") else int(NUM_SPEAKERS_ENV)

MODEL_DIAR = os.getenv("MODEL_DIAR", "pyannote/speaker-diarization-3.1")
MODEL_ASR  = os.getenv("MODEL_ASR",  "kotoba-tech/kotoba-whisper-v2.2")

device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if device == 0 else torch.float32

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

from transformers import pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm.auto import tqdm

# ---- step 1: convert to 16k mono wav -----------------------------------------
log("[STEP] Converting input to WAV(16kHz mono)...")
tmp_in_dir = tempfile.mkdtemp(prefix="wav_", dir=str(WORK_TMP))
wav16 = Path(tmp_in_dir) / "input_16k_mono.wav"

subprocess.run(
    ["ffmpeg", "-nostdin", "-y", "-i", str(INPUT_FILE), "-ac", "1", "-ar", "16000", str(wav16)],
    check=True
)

audio16 = AudioSegment.from_file(str(wav16))

# ---- step 2: diarization ------------------------------------------------------
log("[STEP] Loading Diarization model...")
dia = Pipeline.from_pretrained(MODEL_DIAR, use_auth_token=hf_token)
if device == 0:
    dia.to(torch.device("cuda"))

log("[STEP] Running Diarization...")
t0 = time.time()
params = {"num_speakers": NUM_SPEAKERS, "min_speakers": NUM_SPEAKERS, "max_speakers": NUM_SPEAKERS} if NUM_SPEAKERS else {}
diar = dia(str(wav16), **params)

segments = list(diar.itertracks(yield_label=True))
log(f"[OK] Diarization finished: {len(segments)} segments ({time.time()-t0:.2f}s)")

# ---- VRAM release before ASR --------------------------------------------------
del dia
gc.collect()
if device == 0:
    torch.cuda.empty_cache()

# ---- step 3: ASR per segment --------------------------------------------------
log("[STEP] Loading ASR model...")
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ASR,
    device=device,
    trust_remote_code=True,
    return_timestamps=True,
    torch_dtype=torch_dtype,
)

log("[STEP] Running ASR per segment...")
tmp_seg_dir = tempfile.mkdtemp(prefix="seg_", dir=str(WORK_TMP))

results = []
for idx, (turn, _, spk) in enumerate(tqdm(segments, desc="ASR")):
    st, ed = float(turn.start), float(turn.end)
    if ed <= st:
        continue

    seg_path = Path(tmp_seg_dir) / f"seg_{idx:04d}.wav"
    audio16[int(st*1000):int(ed*1000)].export(str(seg_path), format="wav")

    out = asr(str(seg_path), generate_kwargs={"language": "japanese", "task": "transcribe"})
    results.append({
        "speaker": spk,
        "start": st,
        "end": ed,
        "text": out.get("text", "").strip(),
    })

# ---- output -------------------------------------------------------------------
base = INPUT_FILE.stem

def fmt(t):
    td = datetime.timedelta(seconds=float(t))
    return f"{td.seconds//3600:02}:{(td.seconds//60)%60:02}:{td.seconds%60:02},{td.microseconds//1000:03}"

srt_body = [
    f"{i}\n{fmt(r['start'])} --> {fmt(r['end'])}\n{r['speaker']}: {r['text']}"
    for i, r in enumerate(results, 1)
]

(WORK_OUTPUT / f"{base}.srt").write_text("\n\n".join(srt_body), encoding="utf-8")
(WORK_OUTPUT / f"{base}.txt").write_text("\n".join([f"[{r['speaker']}] {r['text']}" for r in results]), encoding="utf-8")

# ---- cleanup ------------------------------------------------------------------
shutil.rmtree(tmp_seg_dir, ignore_errors=True)
shutil.rmtree(tmp_in_dir, ignore_errors=True)

log(f"[DONE] Outputs saved to {WORK_OUTPUT}")
