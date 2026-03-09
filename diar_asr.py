# ================================
# diar_asr.py  (Docker版 / Dドライブキャッシュ・tmp対応 / エラー追跡強化版)
#  - pyannote diarization + kotoba-whisper ASR
#  - PyTorch 2.6+ "weights_only=True" 既定変更による UnpicklingError を回避
#  - 入力(m4a等)→WAV変換は ffmpeg を subprocess で直叩き
#  - エラー時の Traceback 取得と VRAM 使用量のロギングを追加
#  - 出力を VTT および TXT 形式に変更
#  - 話者分離終了後の GPU メモリ解放
#  - 辞書ありモード（オプション・デフォルトオフ）
#    USE_LAYER_A=1 で Layer A（pre_asr_hint）、USE_LAYER_B=1 で Layer B（post_asr_correction）
#    通常運用は両方オフで安定。必要時のみ有効化して比較・実験可能
# ================================

import gc
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
WORK_OUTPUT_ENV = os.getenv("WORK_OUTPUT", "").strip()
WORK_OUTPUT = Path(WORK_OUTPUT_ENV) if WORK_OUTPUT_ENV else (WORK_SOURCE / "output")
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
# GPU メモリ解放
# ================================
def _log_gpu_memory(device: int, log_fn) -> None:
    """GPU メモリ状況をログに出す。"""
    if device == 0:
        try:
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            log_fn(f"[VRAM] Allocated={alloc:.2f} MB, Reserved={reserved:.2f} MB")
        except Exception:
            pass


def _release_gpu_memory(device: int, log_fn) -> None:
    """GPU メモリをクリアする。
    呼び出し側で dia = None 等で参照を絶ってから呼ぶこと。
    """
    log_fn("[STEP] GPUメモリを解放...")
    _log_gpu_memory(device, log_fn)
    gc.collect()
    if device == 0:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _log_gpu_memory(device, log_fn)
    log_fn("[OK] GPU 解放完了")


# ================================
# 辞書（TSV: 正式表記<TAB>よみ）
# ================================
def load_glossary_tsv(path: Path, log_fn) -> list[tuple[str, str]] | None:
    """TSV から辞書を読み込む。失敗時は None を返す。
    戻り値: [(正式表記, よみ), ...]
    """
    if not path or not path.exists():
        if path:
            log_fn(f"[WARN] 辞書ファイルが見つかりません: {path}。辞書なしモードで継続します。")
        return None
    entries: list[tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                formal, reading = parts[0].strip(), parts[1].strip()
                if formal and reading and formal not in ("表記", "用語"):
                    entries.append((formal, reading))
        log_fn(f"[DICT] 辞書読み込み: {path} -> {len(entries)} 件")
        return entries if entries else None
    except Exception as e:
        log_fn(f"[WARN] 辞書読み込み失敗: {e}。辞書なしモードで継続します。")
        return None


def build_prompt_ids_with_budget(
    entries: list[tuple[str, str]],
    asr_pipeline,
    token_budget: int,
    log_fn,
    dict_log_fn=None,
):
    """
    【Layer A: pre_asr_hint】
    辞書エントリから token budget に収まる範囲で prompt_ids を生成する。
    processor か tokenizer のどちらか利用可能な方を使ってトークン化する。
    正式表記をスペース区切りで greedy に追加し、budget 超過した時点で打ち切る。
    Whisper の max_target_positions (448) 超過を事前に防ぐ。
    """
    try:
        if hasattr(asr_pipeline, "processor") and hasattr(
            asr_pipeline.processor, "get_prompt_ids"
        ):
            prompt_helper = asr_pipeline.processor
        elif hasattr(asr_pipeline, "tokenizer") and hasattr(
            asr_pipeline.tokenizer, "get_prompt_ids"
        ):
            prompt_helper = asr_pipeline.tokenizer
        else:
            reason = "processor/tokenizer に get_prompt_ids が見つからない"
            log_fn(f"[DICT][Layer A] {reason}。pre_asr_hint を無効化します。")
            if dict_log_fn is not None:
                dict_log_fn(f"[DICT][Layer A] enabled=false reason={reason}")
            return None

        target_device = asr_pipeline.model.device
        selected_formals: list[str] = []

        for formal, _ in entries:
            candidate_text = " ".join(selected_formals + [formal])
            try:
                candidate_ids = prompt_helper.get_prompt_ids(
                    candidate_text, return_tensors="pt"
                )
            except TypeError:
                ids = prompt_helper.get_prompt_ids(candidate_text)
                if isinstance(ids, (list, tuple)):
                    candidate_ids = torch.tensor([ids], dtype=torch.long)
                else:
                    candidate_ids = (
                        ids
                        if hasattr(ids, "unsqueeze")
                        else torch.tensor([ids], dtype=torch.long)
                    )
            if candidate_ids.shape[-1] > token_budget:
                break
            selected_formals.append(formal)

        if not selected_formals:
            reason = f"1件目からバジェット({token_budget}token)超過"
            log_fn(f"[DICT][Layer A] {reason}。pre_asr_hint を無効化します。")
            if dict_log_fn is not None:
                dict_log_fn(f"[DICT][Layer A] glossary_total={len(entries)}")
                dict_log_fn(f"[DICT][Layer A] token_budget={token_budget}")
                dict_log_fn(f"[DICT][Layer A] enabled=false reason={reason}")
            return None

        prompt_text = " ".join(selected_formals)
        try:
            prompt_ids = prompt_helper.get_prompt_ids(
                prompt_text, return_tensors="pt"
            ).to(target_device)
        except TypeError:
            ids = prompt_helper.get_prompt_ids(prompt_text)
            if isinstance(ids, (list, tuple)):
                prompt_ids = torch.tensor([ids], dtype=torch.long).to(target_device)
            else:
                prompt_ids = (
                    ids.to(target_device)
                    if hasattr(ids, "to")
                    else torch.tensor([ids], dtype=torch.long).to(target_device)
                )
        # コンソールには要約のみ
        log_fn(
            f"[DICT][Layer A] prompt_ids 生成成功: "
            f"{len(selected_formals)}件/{len(entries)}件を採用"
        )
        # 詳細は .dict.log へ
        if dict_log_fn is not None:
            dict_log_fn(f"[DICT][Layer A] glossary_total={len(entries)}")
            dict_log_fn(f"[DICT][Layer A] token_budget={token_budget}")
            dict_log_fn(f"[DICT][Layer A] selected_count={len(selected_formals)}")
            dict_log_fn(
                f"[DICT][Layer A] selected_formals={','.join(selected_formals)}"
            )
            dict_log_fn(f"[DICT][Layer A] prompt_ids_shape={list(prompt_ids.shape)}")
            dict_log_fn("[DICT][Layer A] enabled=true")
        return prompt_ids

    except Exception:
        reason = "prompt_ids 生成例外"
        log_fn(f"[DICT][Layer A] {reason}。post_asr_correction のみで継続します。")
        if dict_log_fn is not None:
            dict_log_fn(f"[DICT][Layer A] enabled=false reason={reason}")
            dict_log_fn(traceback.format_exc())
        return None


def apply_glossary_correction(
    text: str,
    entries: list[tuple[str, str]],
    log_fn,
    *,
    seg_idx: int | None = None,
    speaker: str | None = None,
    dict_log_fn=None,
) -> str:
    """【Layer B: post_asr_correction】
    ASR 出力に対して読み→正式表記の補正を行う。
    長い読みを先に置換して、短い読みが長い読みに含まれる場合の誤置換を防ぐ。
    辞書あり時は常に有効。
    dict_log_fn がある場合は各置換を .dict.log に記録する。
    """
    if not text or not entries:
        return text
    result = text
    sorted_entries = sorted(entries, key=lambda x: len(x[1]), reverse=True)
    replacements: list[tuple[str, str]] = []
    for formal, reading in sorted_entries:
        if reading in result:
            result = result.replace(reading, formal)
            replacements.append((reading, formal))
    if replacements and dict_log_fn is not None and seg_idx is not None and speaker:
        for reading, formal in replacements:
            dict_log_fn(
                f"[DICT][Layer B] seg={seg_idx} speaker={speaker} "
                f"reading='{reading}' formal='{formal}'"
            )
    return result


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

# 辞書: 未指定なら辞書なしモード
DICT_PATH_ENV = os.getenv("DICT_PATH", "").strip()
DICT_PATH = Path(DICT_PATH_ENV) if DICT_PATH_ENV else None
# 辞書機能はオプション。デフォルトは両方オフ（辞書なし相当で安定運用）
USE_LAYER_A = os.getenv("USE_LAYER_A", "0").strip() in ("1", "true", "yes", "on")
USE_LAYER_B = os.getenv("USE_LAYER_B", "0").strip() in ("1", "true", "yes", "on")
LAYER_A_FALLBACK = os.getenv("LAYER_A_FALLBACK", "1").strip() in ("1", "true", "yes", "on")
DEBUG_GLOSSARY_LOG = os.getenv("DEBUG_GLOSSARY_LOG", "0").strip() in ("1", "true", "yes", "on")
# Layer A 用トークンバジェット（USE_LAYER_A=1 のときのみ使用）
GLOSSARY_TOKEN_BUDGET = int(os.getenv("GLOSSARY_TOKEN_BUDGET", "100"))

device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
log(f"[INFO] device={'cuda:0' if device==0 else 'cpu'}, torch_dtype={torch_dtype}")

log(f"[CFG] WORK_SOURCE={WORK_SOURCE}")
log(f"[CFG] WORK_OUTPUT={WORK_OUTPUT}")
log(f"[CFG] INPUT_FILENAME='{INPUT_FILENAME}'")
log(f"[CFG] NUM_SPEAKERS_ENV='{NUM_SPEAKERS_ENV}' -> NUM_SPEAKERS={NUM_SPEAKERS}")
log(f"[CFG] DICT_PATH={DICT_PATH if DICT_PATH else '(未指定)'}")
log(f"[CFG] USE_LAYER_A={USE_LAYER_A} USE_LAYER_B={USE_LAYER_B} LAYER_A_FALLBACK={LAYER_A_FALLBACK} DEBUG_GLOSSARY_LOG={DEBUG_GLOSSARY_LOG}")
log(f"[CFG] GLOSSARY_TOKEN_BUDGET={GLOSSARY_TOKEN_BUDGET}")

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
dia = None
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
    try:
        if dia is not None:
            dia = None
        _release_gpu_memory(device, log)
    except Exception:
        pass
    raise

# 話者分離終了後、ASR 開始前に GPU メモリを解放
if dia is not None:
    dia = None
_release_gpu_memory(device, log)

# ================================
# 辞書読み込み（オプション: USE_LAYER_A / USE_LAYER_B で有効化）
# ================================
base = INPUT_FILE.stem
dict_log_file = None
dict_log_fn = None

glossary_entries: list[tuple[str, str]] | None = None
if DICT_PATH:
    glossary_entries = load_glossary_tsv(DICT_PATH, log)

if not glossary_entries:
    if DICT_PATH:
        log("[DICT] 辞書ファイル未読み込みまたは空。辞書機能は使いません。")
    else:
        log("[DICT] DICT_PATH 未指定。辞書機能は無効です。")
elif not (USE_LAYER_A or USE_LAYER_B):
    log(f"[DICT] 辞書読み込み済み（{len(glossary_entries)}件）。USE_LAYER_A=0, USE_LAYER_B=0 のため辞書機能はオフです。")
else:
    if USE_LAYER_A:
        log(f"[DICT][Layer A] pre_asr_hint を有効化（token_budget={GLOSSARY_TOKEN_BUDGET}）")
    if USE_LAYER_B:
        log(f"[DICT][Layer B] post_asr_correction を有効化（全{len(glossary_entries)}件）")
    if DEBUG_GLOSSARY_LOG:
        dict_log_path = WORK_OUTPUT / f"{base}.dict.log"
        try:
            dict_log_file = open(dict_log_path, "w", encoding="utf-8")
            def _write_dict_log(msg: str) -> None:
                dict_log_file.write(msg + "\n")
                dict_log_file.flush()
            dict_log_fn = _write_dict_log
            dict_log_fn(f"# {base}.dict.log - 辞書補正・事前ヒントの記録")
            dict_log_fn(f"# USE_LAYER_A={USE_LAYER_A} USE_LAYER_B={USE_LAYER_B}")
            dict_log_fn(f"# 辞書総件数: {len(glossary_entries)}")
        except OSError as e:
            log(f"[WARN] dict.log を開けません: {dict_log_path} ({e})")

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

# ================================
# 辞書 Layer A: pre_asr_hint（USE_LAYER_A=1 のときのみ）
# ================================
pre_asr_prompt_ids = None
if glossary_entries and USE_LAYER_A:
    pre_asr_prompt_ids = build_prompt_ids_with_budget(
        glossary_entries, asr, GLOSSARY_TOKEN_BUDGET, log, dict_log_fn=dict_log_fn
    )
    if pre_asr_prompt_ids is not None:
        log("[DICT][Layer A] prompt_ids を generate_kwargs に渡します")
    else:
        log("[DICT][Layer A] prompt_ids 生成失敗。Layer A なしで継続します")

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

        # prompt_ids は transformers Whisper の正式パラメータ。
        # initial_prompt（文字列）は model_kwargs で弾かれるため使用しない。
        gen_kw = dict(LANG_KW)
        if pre_asr_prompt_ids is not None:
            gen_kw["prompt_ids"] = pre_asr_prompt_ids
        try:
            out = asr(str(seg_path), return_timestamps=True, generate_kwargs=gen_kw)
        except Exception as e_asr:
            error_msg = str(e_asr)
            is_prompt_error = any(
                kw in error_msg
                for kw in [
                    "prompt_ids",
                    "max_target_positions",
                    "decoder_input_ids",
                    "model_kwargs",
                ]
            )
            if (
                pre_asr_prompt_ids is not None
                and is_prompt_error
                and LAYER_A_FALLBACK
            ):
                log(
                    f"\n[WARN] ASR推論でプロンプト長起因と思われるエラー発生。"
                    f"Layer Aを無効化して再試行します。"
                )
                if dict_log_fn is not None:
                    dict_log_fn(
                        f"[DICT][Layer A] fallback_at_segment={idx} "
                        f"reason={error_msg[:200]}"
                    )
                pre_asr_prompt_ids = None  # type: ignore
                gen_kw_retry = dict(LANG_KW)
                out = asr(
                    str(seg_path),
                    return_timestamps=True,
                    generate_kwargs=gen_kw_retry,
                )
            else:
                raise e_asr
        text = (out.get("text") or "").strip()
        if glossary_entries and USE_LAYER_B:
            text = apply_glossary_correction(
                text,
                glossary_entries,
                log,
                seg_idx=idx,
                speaker=spk,
                dict_log_fn=dict_log_fn,
            )
        results.append({"speaker": spk, "start": st, "end": ed, "text": text})
except Exception as e:
    log(f"[FATAL ERROR] ASR推論中に例外発生:\n{traceback.format_exc()}")
    raise

# ================================
# Write outputs
# ================================

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
if dict_log_file is not None:
    try:
        dict_log_file.close()
        log(f"[DICT] 辞書ログ: {WORK_OUTPUT / f'{base}.dict.log'}")
    except Exception:
        pass

shutil.rmtree(tmp_seg_dir, ignore_errors=True)
shutil.rmtree(tmp_in_dir, ignore_errors=True)

log(f"[OK] 出力: {WORK_OUTPUT}")
log("[DONE] 完了")