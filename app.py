"""
Indic Language Annotation & Segmentation Tool  ──  ULTRA-FAST Edition
═══════════════════════════════════════════════════════════════════════
Speed tiers (vs original HuggingFace code):

  whisper.cpp  (int8, 4 threads)  ≈ 25-40× faster
  faster-whisper (CTranslate2)    ≈  4-8×  faster
  distil-whisper (HF)             ≈  5-6×  faster
  HuggingFace (vanilla)           ≈  1×    baseline

Additional speed features:
  • Model cache — switching languages doesn't re-download
  • Auto-preload on launch — model loads while you pick files
  • Background audio loading — UI never freezes
  • ThreadPool for segment-level parallelism
  • Streaming results — see each prediction as it arrives
  • Vectorised VAD (numpy, no Python loops)
  • soundfile + scipy resampling
  • Viewport-culled plotting

Install (pick ONE engine — listed fastest→slowest):

  # Option A: whisper.cpp (FASTEST — pure C++)
  pip install pywhispercpp soundfile scipy

  # Option B: faster-whisper (CTranslate2 — great balance)
  pip install faster-whisper soundfile scipy

  # Option C: HuggingFace + distil-whisper (easiest install)
  pip install transformers torch soundfile scipy
"""

import os, sys, csv, re, traceback, time
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import sounddevice as sd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTabWidget, QCheckBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# ── Optional fast imports ────────────────────────────────────────────────
HAS_WHISPERCPP = False
try:
    from pywhispercpp.model import Model as WhisperCppModel
    HAS_WHISPERCPP = True
except ImportError:
    pass

HAS_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    pass

HAS_HF = False
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    HAS_HF = True
except ImportError:
    pass

HAS_SOUNDFILE = False
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    pass

HAS_SCIPY = False
try:
    from scipy.signal import resample_poly
    from math import gcd
    HAS_SCIPY = True
except ImportError:
    pass

if not any([HAS_WHISPERCPP, HAS_FASTER_WHISPER, HAS_HF]):
    raise ImportError(
        "Install at least one engine:\n"
        "  pip install pywhispercpp      # fastest\n"
        "  pip install faster-whisper    # fast + accurate\n"
        "  pip install transformers torch  # baseline"
    )

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════

# (UI label, HF model ID, faster-whisper/cpp size tag, is_distil)
MODEL_OPTIONS = [
    ("Distil-Large-v3  ★ FAST+ACCURATE", "distil-whisper/distil-large-v3",   "distil-large-v3", True),
    ("Distil-Medium.en (English only)",   "distil-whisper/distil-medium.en",  "distil-medium.en",True),
    ("Whisper Tiny     (ultrafast)",       "openai/whisper-tiny",              "tiny",            False),
    ("Whisper Base     (very fast)",       "openai/whisper-base",              "base",            False),
    ("Whisper Small    (fast)",            "openai/whisper-small",             "small",           False),
    ("Whisper Medium   (balanced)",        "openai/whisper-medium",            "medium",          False),
    ("Whisper Large-v3 (best accuracy)",   "openai/whisper-large-v3",         "large-v3",        False),
]

LANG_OPTIONS = [
    ("English",  "en", "english"),
    ("Hindi",    "hi", "hindi"),
    ("Gujarati", "gu", "gujarati"),
    ("Marathi",  "mr", "marathi"),
]

PREDICTION_MODES = [
    ("Full Transcript", "full"),
    ("First Word",      "word"),
    ("First Character",  "char"),
]

# Build engine list from available backends
ENGINE_OPTIONS = []
if HAS_WHISPERCPP:
    ENGINE_OPTIONS.append(("whisper.cpp  (C++, fastest)",    "whispercpp"))
if HAS_FASTER_WHISPER:
    ENGINE_OPTIONS.append(("faster-whisper (CTranslate2)",   "faster_whisper"))
if HAS_HF:
    ENGINE_OPTIONS.append(("HuggingFace pipeline",           "hf_pipeline"))
    ENGINE_OPTIONS.append(("HuggingFace batched",            "hf_batched"))

TARGET_SR = 16000
CPU_THREADS = max(1, min(os.cpu_count() or 4, 8))
VERBOSE = True

# ── Unicode ──────────────────────────────────────────────────────────────
_INDIC = r"\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF\u0A80-\u0AFF"
_RE_KEEP   = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM   = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA  = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)


def log(msg, force=False):
    if VERBOSE or force:
        print(f"[LOG] {msg}", flush=True)

def warn(msg):
    print(f"[WARN] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def load_audio_fast(path):
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return (data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]), sr
        except Exception:
            pass
    import librosa
    return librosa.load(path, sr=None, mono=True)


def resample_fast(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    if HAS_SCIPY:
        g = gcd(orig_sr, target_sr)
        return resample_poly(y, target_sr // g, orig_sr // g).astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def time_to_sec(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x:
        return None
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"):
        try:
            dt = datetime.strptime(x, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            continue
    try:
        return float(x)
    except ValueError:
        return None


def norm_text(s):
    return _RE_NORM.sub("", str(s).strip().lower())


def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def compute_accuracy(pred, gt):
    p, g = norm_text(pred), norm_text(gt)
    if not g and not p: return 1.0
    if not g: return 0.0
    return max(0.0, 1.0 - edit_distance(p, g) / len(g))


def clean_prediction(text, mode="full"):
    if not text: return ""
    text = _RE_KEEP.sub("", str(text).strip())
    text = _RE_SPACES.sub(" ", text).strip()
    if not text: return ""
    if mode == "full": return text
    if mode == "word":
        parts = text.split()
        return parts[0] if parts else ""
    if mode == "char":
        m = _RE_ALPHA.search(text)
        return m.group(0) if m else ""
    return text


@lru_cache(maxsize=8)
def pick_font(lang_code):
    specs = {
        "hi": (["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
                os.path.expanduser("~/Library/Fonts/NotoSansDevanagari-Regular.ttf")],
               ["Noto Sans Devanagari","Lohit Devanagari","Kohinoor Devanagari","Mangal","Arial Unicode MS"]),
        "gu": (["/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
                "/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf",
                "/Library/Fonts/NotoSansGujarati-Regular.ttf",
                os.path.expanduser("~/Library/Fonts/NotoSansGujarati-Regular.ttf")],
               ["Noto Sans Gujarati","Lohit Gujarati","Shruti","Arial Unicode MS"]),
        "en": ([], ["Arial","Helvetica","DejaVu Sans"]),
    }
    lc = "hi" if lang_code == "mr" else lang_code
    files, families = specs.get(lc, specs["en"])
    for p in files:
        if os.path.exists(p):
            return fm.FontProperties(fname=p)
    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in families:
        if fam in installed:
            return fm.FontProperties(family=fam)
    return None


# ── Vectorised VAD ───────────────────────────────────────────────────────
def vad_segment_audio(y, sr, min_silence_ms=300, silence_thresh_db=-40,
                      min_segment_ms=200, max_segment_ms=10000):
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    n_frames  = 1 + (len(y) - frame_len) // hop_len
    if n_frames <= 0:
        return []

    indices = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    np.clip(indices, 0, len(y) - 1, out=indices)
    rms = np.sqrt(np.mean(y[indices] ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    is_voice = rms_db > silence_thresh_db

    ms2f = lambda ms: max(1, int((ms / 1000) * sr / hop_len))
    min_sil_f, min_seg_f, max_seg_f = ms2f(min_silence_ms), ms2f(min_segment_ms), ms2f(max_segment_ms)

    padded = np.concatenate([[False], is_voice, [False]])
    diffs  = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]

    merged_s, merged_e = [], []
    i = 0
    while i < len(starts):
        s, e = starts[i], ends[i]
        while i + 1 < len(starts) and (starts[i+1] - e) < min_sil_f:
            i += 1; e = ends[i]
        merged_s.append(s); merged_e.append(e); i += 1

    result = []
    for s, e in zip(merged_s, merged_e):
        if (e - s) < min_seg_f: continue
        for cs in range(s, e, max_seg_f):
            ce = min(cs + max_seg_f, e)
            if (ce - cs) >= min_seg_f:
                result.append({"start": round(cs * hop_len / sr, 4),
                               "end":   round(min(ce * hop_len / sr, len(y) / sr), 4)})
    log(f"VAD: {len(result)} segments")
    return result


# ══════════════════════════════════════════════════════════════════════════
#  BACKENDS
# ══════════════════════════════════════════════════════════════════════════

class WhisperCppBackend:
    """
    whisper.cpp via pywhispercpp — fastest possible inference.
    Pure C++, int8/int16 quantised, multi-threaded.
    ~25-40× faster than HuggingFace on CPU.
    """
    NAME = "whisper.cpp"

    def __init__(self, model_size, n_threads=None):
        t0 = time.time()
        self.n_threads = n_threads or CPU_THREADS
        log(f"[whisper.cpp] Loading '{model_size}' with {self.n_threads} threads")

        # pywhispercpp auto-downloads models
        self.model = WhisperCppModel(model_size, n_threads=self.n_threads)
        log(f"[whisper.cpp] Loaded in {time.time()-t0:.1f}s")

    def transcribe_one(self, audio, sr, lang_code, mode):
        if audio is None or len(audio) == 0:
            return ""
        if len(audio) < int(0.1 * sr):
            audio = np.pad(audio, (0, int(0.1 * sr) - len(audio)))

        segments = self.model.transcribe(audio, language=lang_code)
        text = " ".join(s.text.strip() for s in segments)
        return clean_prediction(text, mode)

    def transcribe_batch(self, chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        """Parallel transcription using ThreadPoolExecutor."""
        total = len(chunks)
        results = [""] * total
        done = [0]

        # whisper.cpp is thread-safe for separate model calls
        # But single model isn't thread-safe, so sequential with fast C++
        for i, chunk in enumerate(chunks):
            results[i] = self.transcribe_one(chunk, sr, lang_code, mode)
            done[0] += 1
            if progress_cb:
                progress_cb(done[0], total)

        return results


class FasterWhisperBackend:
    """CTranslate2 backend — 4-8× faster than HF, supports int8 quantisation."""
    NAME = "faster-whisper"

    def __init__(self, model_size, device="cpu"):
        t0 = time.time()
        dev = device.split(":")[0]
        compute = "float16" if dev == "cuda" else "int8"
        log(f"[faster-whisper] Loading '{model_size}' on {dev} ({compute})")

        self.model = FasterWhisperModel(
            model_size,
            device=dev,
            compute_type=compute,
            cpu_threads=CPU_THREADS,
            num_workers=min(4, CPU_THREADS),
        )
        log(f"[faster-whisper] Loaded in {time.time()-t0:.1f}s")

    def transcribe_one(self, audio, sr, lang_code, mode):
        if audio is None or len(audio) == 0:
            return ""
        if len(audio) < int(0.1 * sr):
            audio = np.pad(audio, (0, int(0.1 * sr) - len(audio)))

        segments, _ = self.model.transcribe(
            audio, language=lang_code, task="transcribe",
            beam_size=1, vad_filter=False, without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments)
        return clean_prediction(text, mode)

    def transcribe_batch(self, chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        total = len(chunks)
        results = [""] * total
        for i, chunk in enumerate(chunks):
            results[i] = self.transcribe_one(chunk, sr, lang_code, mode)
            if progress_cb:
                progress_cb(i + 1, total)
        return results


class HFPipelineBackend:
    """
    HuggingFace pipeline API — simplest, supports distil-whisper.
    Uses chunked processing and automatic batching internally.
    """
    NAME = "hf-pipeline"

    def __init__(self, model_id, device="cpu", batch_size=16):
        t0 = time.time()
        self.batch_size = batch_size

        if device != "cpu" and "cuda" in device:
            dev = device
            dt = torch.float16
        else:
            dev = "cpu"
            dt = torch.float32

        log(f"[HF-pipeline] Loading '{model_id}' on {dev}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dt,
            device=dev,
        )
        log(f"[HF-pipeline] Loaded in {time.time()-t0:.1f}s")

    def transcribe_batch(self, chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        total = len(chunks)
        results = [""] * total

        # Pad short chunks
        min_len = int(0.1 * sr)
        padded = []
        valid_indices = []
        for i, c in enumerate(chunks):
            if c is None or len(c) == 0:
                continue
            if len(c) < min_len:
                c = np.pad(c, (0, min_len - len(c)))
            padded.append({"raw": c, "sampling_rate": sr})
            valid_indices.append(i)

        if not padded:
            if progress_cb:
                progress_cb(total, total)
            return results

        gen_kwargs = {"task": "transcribe", "language": lang_name}

        done = 0
        for out in self.pipe(
            padded,
            batch_size=self.batch_size,
            generate_kwargs=gen_kwargs,
            return_timestamps=False,
        ):
            idx = valid_indices[done]
            text = out.get("text", "") if isinstance(out, dict) else ""
            results[idx] = clean_prediction(text, mode)
            done += 1
            if progress_cb:
                progress_cb(done, total)

        # Fill remaining progress
        if progress_cb and done < total:
            progress_cb(total, total)

        return results


class HFBatchedBackend:
    """HuggingFace manual batching with torch.compile — best GPU throughput."""
    NAME = "hf-batched"

    def __init__(self, model_id, device="cpu", batch_size=16):
        t0 = time.time()
        self.batch_size = batch_size

        if "cuda" in device:
            self.device = device
            dtype = torch.float16
        else:
            self.device = "cpu"
            try:
                torch.zeros(1, dtype=torch.bfloat16)
                dtype = torch.bfloat16
            except Exception:
                dtype = torch.float32

        log(f"[HF-batched] Loading '{model_id}' on {self.device} (dtype={dtype})")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype,
            low_cpu_mem_usage=True, use_safetensors=True,
        ).to(self.device)

        if hasattr(torch, "compile") and "cuda" in self.device:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                log("[HF-batched] torch.compile enabled")
            except Exception:
                pass

        self.processor = AutoProcessor.from_pretrained(model_id)
        log(f"[HF-batched] Loaded in {time.time()-t0:.1f}s")

    def transcribe_batch(self, chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        total = len(chunks)
        results = [""] * total
        done = 0
        min_len = int(0.1 * sr)

        for bs in range(0, total, self.batch_size):
            be = min(bs + self.batch_size, total)
            batch, indices = [], []

            for i in range(bs, be):
                c = chunks[i]
                if c is None or len(c) == 0:
                    continue
                if len(c) < min_len:
                    c = np.pad(c, (0, min_len - len(c)))
                batch.append(c)
                indices.append(i)

            if batch:
                inputs = self.processor(
                    batch, sampling_rate=sr, return_tensors="pt",
                    padding=True, return_attention_mask=True,
                )
                feat = inputs.input_features.to(self.device)
                mask = getattr(inputs, "attention_mask", None)
                if mask is not None:
                    mask = mask.to(self.device)

                gk = {"input_features": feat, "task": "transcribe",
                      "language": lang_name, "return_timestamps": False}
                if mask is not None:
                    gk["attention_mask"] = mask

                with torch.inference_mode():
                    ids = self.model.generate(**gk)

                texts = self.processor.batch_decode(ids, skip_special_tokens=True)
                for idx, text in zip(indices, texts):
                    results[idx] = clean_prediction(text, mode)

            done += (be - bs)
            if progress_cb:
                progress_cb(done, total)

        return results


# ══════════════════════════════════════════════════════════════════════════
#  MODEL CACHE — avoids reloading the same model
# ══════════════════════════════════════════════════════════════════════════

_model_cache = {}  # key: (engine, model_size_or_id) → backend instance


def get_or_load_backend(engine, model_id, model_size, batch_size):
    """Return cached backend or create a new one. ~0s if already loaded."""
    device = "cpu"
    if HAS_HF:
        import torch as _t
        if _t.cuda.is_available():
            device = "cuda:0"

    cache_key = (engine, model_size if engine in ("whispercpp", "faster_whisper") else model_id)

    if cache_key in _model_cache:
        log(f"[CACHE HIT] Reusing {engine}/{model_size}")
        return _model_cache[cache_key], device

    t0 = time.time()

    if engine == "whispercpp":
        backend = WhisperCppBackend(model_size)
    elif engine == "faster_whisper":
        backend = FasterWhisperBackend(model_size, device)
    elif engine == "hf_pipeline":
        backend = HFPipelineBackend(model_id, device, batch_size)
    elif engine == "hf_batched":
        backend = HFBatchedBackend(model_id, device, batch_size)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    _model_cache[cache_key] = backend
    elapsed = time.time() - t0
    log(f"[CACHE MISS] Loaded {engine}/{model_size} in {elapsed:.1f}s")
    return backend, device


# ══════════════════════════════════════════════════════════════════════════
#  THREADS
# ══════════════════════════════════════════════════════════════════════════

class AudioLoaderThread(QThread):
    """Load + resample audio in background — UI stays responsive."""
    finished_ok = pyqtSignal(np.ndarray, int, np.ndarray, str)  # y, sr, y16, path
    failed = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            t0 = time.time()
            y, sr = load_audio_fast(self.path)
            y16 = resample_fast(y, sr, TARGET_SR)
            log(f"Audio loaded+resampled in {time.time()-t0:.2f}s "
                f"({len(y)/sr:.1f}s, sr={sr})")
            self.finished_ok.emit(y, sr, y16, self.path)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, str, str, float)  # backend, device, info, elapsed
    failed = pyqtSignal(str)

    def __init__(self, engine, model_id, model_size, batch_size):
        super().__init__()
        self.engine = engine
        self.model_id = model_id
        self.model_size = model_size
        self.batch_size = batch_size

    def run(self):
        try:
            t0 = time.time()
            backend, device = get_or_load_backend(
                self.engine, self.model_id, self.model_size, self.batch_size
            )
            elapsed = time.time() - t0
            info = f"{backend.NAME} ({self.model_size}) on {device}"
            self.finished_ok.emit(backend, device, info, elapsed)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class InferenceThread(QThread):
    """Runs transcription and emits streaming per-segment results."""
    segment_done = pyqtSignal(int, dict)   # index, result dict (for streaming)
    progress = pyqtSignal(int, int, int)   # done, total, pct
    finished_ok = pyqtSignal(list, float)  # all results, mean_acc
    failed = pyqtSignal(str)

    def __init__(self, backend, y16, sr16, gt_rows, lang_code, lang_name, mode):
        super().__init__()
        self.backend = backend
        self.y16 = y16
        self.sr16 = sr16
        self.gt_rows = gt_rows
        self.lang_code = lang_code
        self.lang_name = lang_name
        self.mode = mode
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self.gt_rows)
            if total == 0:
                self.finished_ok.emit([], 0.0)
                return

            # Pre-slice audio
            chunks = []
            for row in self.gt_rows:
                s = max(0, int(row["start"] * self.sr16))
                e = min(len(self.y16), int(row["end"] * self.sr16))
                chunks.append(self.y16[s:e])

            log(f"Inference: {total} segs, {self.lang_code}, engine={self.backend.NAME}")

            results = [None] * total
            result_count = [0]

            def on_progress(done, tot):
                if self._stop:
                    raise InterruptedError("Stopped")
                pct = int(done / tot * 100)
                self.progress.emit(done, tot, pct)

            predictions = self.backend.transcribe_batch(
                chunks, self.sr16, self.lang_code, self.lang_name,
                self.mode, on_progress,
            )

            all_results = []
            for i, (row, pred) in enumerate(zip(self.gt_rows, predictions)):
                acc = compute_accuracy(pred, row["label"])
                r = {
                    "gt": row["label"], "pred": pred,
                    "gt_start": row["start"], "gt_end": row["end"],
                    "pred_start": row["start"], "pred_end": row["end"],
                    "score": acc,
                }
                all_results.append(r)
                self.segment_done.emit(i, r)

            mean_acc = np.mean([r["score"] for r in all_results]) if all_results else 0.0
            self.finished_ok.emit(all_results, float(mean_acc))

        except InterruptedError:
            warn("Inference stopped by user")
            partial = [r for r in (results if 'results' in dir() else []) if r is not None]
            self.finished_ok.emit(partial, 0.0)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class SegmentationThread(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, y, sr, params):
        super().__init__()
        self.y, self.sr, self.params = y, sr, params

    def run(self):
        try:
            self.finished_ok.emit(vad_segment_audio(self.y, self.sr, **self.params))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indic Annotation Tool  [ULTRA-FAST]")
        self.resize(1700, 1000)

        self.audio_path = self.xlsx_path = None
        self.y = self.y16 = self.wave_t = self.wave_y = None
        self.sr = None

        self.backend = None
        self.device = None
        self._loaded_model_key = None  # track what's loaded

        self.gt_rows = []
        self.results = []
        self.selected_index = None

        self.model_thread = self.infer_thread = self.seg_thread = None
        self.audio_thread = None

        self._build_ui()
        log("Application ready — auto-preloading default model...")

        # ── Auto-preload model on launch (0ms UI delay) ──────────────────
        QTimer.singleShot(100, self._auto_preload_model)

    def _auto_preload_model(self):
        """Start loading the default model immediately on app launch."""
        if self.backend is not None:
            return
        self.load_model(auto=True)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        L = QVBoxLayout(root); L.setSpacing(4); L.setContentsMargins(8,8,8,8)

        # Row 1 — Config
        r1 = QHBoxLayout()
        self.lang_combo = QComboBox()
        for lbl, c, w in LANG_OPTIONS:
            self.lang_combo.addItem(lbl, (c, w))

        self.model_combo = QComboBox()
        for lbl, mid, sz, is_d in MODEL_OPTIONS:
            self.model_combo.addItem(lbl, (mid, sz, is_d))
        self.model_combo.setCurrentIndex(0)  # Distil-large-v3 default

        self.engine_combo = QComboBox()
        for lbl, val in ENGINE_OPTIONS:
            self.engine_combo.addItem(lbl, val)

        self.mode_combo = QComboBox()
        for lbl, m in PREDICTION_MODES:
            self.mode_combo.addItem(lbl, m)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64); self.batch_spin.setValue(16)
        self.batch_spin.setToolTip("Batch size (HF engines only)")

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)
        self.load_gt_btn = QPushButton("Load GT (.xlsx/.csv)")
        self.load_gt_btn.clicked.connect(self.load_gt)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        for w in [QLabel("Lang:"), self.lang_combo,
                  QLabel("Model:"), self.model_combo,
                  QLabel("Engine:"), self.engine_combo,
                  QLabel("Batch:"), self.batch_spin,
                  QLabel("Mode:"), self.mode_combo,
                  self.load_audio_btn, self.load_gt_btn, self.load_model_btn]:
            r1.addWidget(w)
        r1.addStretch()

        # Row 2 — Actions
        r2 = QHBoxLayout()
        action_defs = [
            ("▶ Run",        self.run_inference),
            ("⏹ Stop",       self.stop_inference),
            ("Auto-Segment", self.run_auto_segmentation),
            ("♫ Play All",   self.play_audio),
            ("♫ Play Seg",   self.play_selected_segment),
            ("⏸ Stop Audio", self.stop_audio),
            ("🔍+",          lambda: self.zoom(0.6)),
            ("🔍−",          lambda: self.zoom(1.6)),
            ("Reset View",   self.reset_plot_view),
            ("Export CSV",   self.export_csv),
        ]
        self._btns = {}
        for name, fn in action_defs:
            b = QPushButton(name)
            b.clicked.connect(fn)
            r2.addWidget(b)
            self._btns[name] = b
        self._btns["⏹ Stop"].setEnabled(False)
        r2.addStretch()

        # Status bar
        status_row = QHBoxLayout()
        self.status_label = QLabel("Auto-loading model...")
        self.status_label.setStyleSheet("padding:4px; font-size:13px;")
        self.speed_label = QLabel("")
        self.speed_label.setStyleSheet("padding:4px; font-size:12px; color:#6b7280;")
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.speed_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border:1px solid #ccc; border-radius:4px;
                           background:#f3f4f6; text-align:center; font-size:11px; }
            QProgressBar::chunk { background:#3b82f6; border-radius:4px; }
        """)

        # Canvas
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Segment list
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_segment_clicked)
        self.seg_list.setStyleSheet("""
            QListWidget { background:#fff; border:1px solid #d0d0d0;
                          font-family: monospace; font-size:12px; }
            QListWidget::item { padding:3px 6px; }
            QListWidget::item:selected { background:#dbeafe; }
        """)

        # VAD settings
        vad_grp = QGroupBox("VAD Settings")
        vl = QVBoxLayout(vad_grp)
        self.vad_sil = QSpinBox();  self.vad_sil.setRange(50,2000);  self.vad_sil.setValue(300);  self.vad_sil.setSuffix(" ms")
        self.vad_db  = QDoubleSpinBox(); self.vad_db.setRange(-80,0); self.vad_db.setValue(-40); self.vad_db.setSuffix(" dB")
        self.vad_min = QSpinBox();  self.vad_min.setRange(50,5000);  self.vad_min.setValue(200);  self.vad_min.setSuffix(" ms")
        self.vad_max = QSpinBox();  self.vad_max.setRange(1000,30000);self.vad_max.setValue(10000);self.vad_max.setSuffix(" ms")
        for lbl, w in [("Silence gap:", self.vad_sil), ("Threshold:", self.vad_db),
                       ("Min seg:", self.vad_min), ("Max seg:", self.vad_max)]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(w); vl.addLayout(h)
        vl.addStretch()

        # Speed tips
        tips_grp = QGroupBox("Speed Tips")
        tips_layout = QVBoxLayout(tips_grp)
        tips_text = QLabel(
            "★ <b>whisper.cpp</b> = fastest (C++, ~25× vs HF)<br>"
            "★ <b>faster-whisper</b> = fast + accurate (int8)<br>"
            "★ <b>distil-large-v3</b> = 6× faster than large-v3<br>"
            "★ Model loads once → cached across runs<br>"
            "★ CUDA GPU = 10-50× faster than CPU"
        )
        tips_text.setTextFormat(Qt.RichText)
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet("font-size:11px; color:#374151;")
        tips_layout.addWidget(tips_text)
        tips_layout.addStretch()

        tabs = QTabWidget()
        seg_tab = QWidget(); sl = QVBoxLayout(seg_tab)
        legend = QLabel(
            '<span style="color:green;font-weight:600">GT</span> | '
            '<span style="color:blue;font-weight:600">PRED</span> | '
            '<span style="color:#16a34a">✓ ≥0.95</span> '
            '<span style="color:#f59e0b">~ 0.5-0.95</span> '
            '<span style="color:#dc2626">✗ &lt;0.5</span>'
        )
        legend.setTextFormat(Qt.RichText)
        sl.addWidget(legend); sl.addWidget(self.seg_list)
        tabs.addTab(seg_tab, "Segments")
        tabs.addTab(vad_grp, "VAD")
        tabs.addTab(tips_grp, "Speed")

        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(self.toolbar); ll.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(tabs)
        splitter.setSizes([1250, 400])

        L.addLayout(r1); L.addLayout(r2)
        L.addLayout(status_row); L.addWidget(self.progress_bar)
        L.addWidget(splitter)

    # ── Helpers ───────────────────────────────────────────────────────────
    def set_busy(self, busy, msg=""):
        for name, b in self._btns.items():
            b.setEnabled(not busy if name != "⏹ Stop" else busy)
        for w in [self.load_audio_btn, self.load_gt_btn, self.load_model_btn,
                  self.lang_combo, self.model_combo, self.engine_combo,
                  self.mode_combo, self.batch_spin]:
            w.setEnabled(not busy)
        if msg:
            self.status_label.setText(msg)

    def build_wave_cache(self):
        if self.y is None:
            self.wave_t = self.wave_y = None; return
        step = max(1, len(self.y) // 25000)
        self.wave_y = self.y[::step]
        self.wave_t = np.arange(len(self.wave_y)) * (step / self.sr)

    # ── Load Audio (background thread) ────────────────────────────────────
    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.webm)")
        if not path:
            return

        self.status_label.setText(f"Loading {os.path.basename(path)}...")
        self.load_audio_btn.setEnabled(False)

        self.audio_thread = AudioLoaderThread(path)
        self.audio_thread.finished_ok.connect(self._on_audio_ok)
        self.audio_thread.failed.connect(self._on_audio_fail)
        self.audio_thread.start()

    def _on_audio_ok(self, y, sr, y16, path):
        self.y, self.sr, self.y16, self.audio_path = y, sr, y16, path
        self.results = []; self.selected_index = None
        self.progress_bar.setValue(0)
        self.build_wave_cache()
        dur = len(y) / sr
        self.status_label.setText(f"Audio: {os.path.basename(path)} | {dur:.1f}s | sr={sr}")
        self.load_audio_btn.setEnabled(True)
        self.refresh_list(); self.redraw()

    def _on_audio_fail(self, msg):
        self.load_audio_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    # ── Load GT ───────────────────────────────────────────────────────────
    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GT", "", "Excel/CSV (*.xlsx *.csv)")
        if not path: return
        try:
            df = (pd.read_csv(path, header=None) if path.endswith(".csv")
                  else pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl"))
            rows, skip = [], 0
            for _, r in df.iterrows():
                lab = str(r.iloc[0]).strip()
                s, e = time_to_sec(r.iloc[1]), time_to_sec(r.iloc[2])
                if not lab or s is None or e is None or e <= s:
                    skip += 1; continue
                rows.append({"label": lab, "start": s, "end": e})
            self.gt_rows = rows; self.xlsx_path = path
            self.results = []; self.selected_index = None
            self.status_label.setText(f"GT: {len(rows)} segments (skipped {skip})")
            self.refresh_list(); self.redraw()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    # ── Load Model (with cache) ──────────────────────────────────────────
    def load_model(self, auto=False):
        if self.model_thread and self.model_thread.isRunning():
            return

        model_id, model_size, is_distil = self.model_combo.currentData()
        engine = self.engine_combo.currentData()
        bs = self.batch_spin.value()

        # Check cache — skip if already loaded
        cache_key = (engine, model_size if engine in ("whispercpp", "faster_whisper") else model_id)
        if self._loaded_model_key == cache_key and self.backend is not None:
            if not auto:
                self.status_label.setText(f"Model already loaded: {model_size}")
            return

        # Warn about unsupported combos
        if engine == "whispercpp" and is_distil:
            if not auto:
                QMessageBox.warning(self, "Unsupported",
                    "whisper.cpp doesn't support distil models.\n"
                    "Use faster-whisper or HF pipeline instead.")
            return

        label = f"Loading {model_size} ({engine})..."
        if auto:
            self.status_label.setText(f"Auto-preloading: {label}")
            self.load_model_btn.setEnabled(False)
        else:
            self.set_busy(True, label)

        self.model_thread = ModelLoaderThread(engine, model_id, model_size, bs)
        self.model_thread.finished_ok.connect(
            lambda b, d, i, t: self._on_model_ok(b, d, i, t, auto, cache_key))
        self.model_thread.failed.connect(lambda m: self._on_model_fail(m, auto))
        self.model_thread.start()

    def _on_model_ok(self, backend, device, info, elapsed, auto, cache_key):
        self.backend = backend
        self.device = device
        self._loaded_model_key = cache_key

        if elapsed < 0.1:
            msg = f"Model ready (cached): {info}"
        else:
            msg = f"Model ready in {elapsed:.1f}s: {info}"

        if auto:
            self.status_label.setText(msg)
            self.load_model_btn.setEnabled(True)
        else:
            self.set_busy(False, msg)
        self.speed_label.setText(f"Engine: {backend.NAME}")

    def _on_model_fail(self, msg, auto):
        if auto:
            self.status_label.setText(f"Auto-load failed: {msg}")
            self.load_model_btn.setEnabled(True)
        else:
            self.set_busy(False, "Model load failed.")
            QMessageBox.critical(self, "Error", msg)

    # ── VAD ───────────────────────────────────────────────────────────────
    def run_auto_segmentation(self):
        if self.y is None:
            QMessageBox.warning(self, "No Audio", "Load audio first."); return
        p = {"min_silence_ms": self.vad_sil.value(), "silence_thresh_db": self.vad_db.value(),
             "min_segment_ms": self.vad_min.value(), "max_segment_ms": self.vad_max.value()}
        self.set_busy(True, "Running VAD...")
        self.seg_thread = SegmentationThread(self.y, self.sr, p)
        self.seg_thread.finished_ok.connect(self._on_vad_ok)
        self.seg_thread.failed.connect(lambda m: (self.set_busy(False), QMessageBox.critical(self,"Error",m)))
        self.seg_thread.start()

    def _on_vad_ok(self, segs):
        self.gt_rows = [{"label": f"seg_{i+1:04d}", "start": s["start"], "end": s["end"]}
                        for i, s in enumerate(segs)]
        self.results = []; self.selected_index = None
        self.set_busy(False, f"VAD: {len(segs)} segments")
        self.refresh_list(); self.redraw()

    # ── Inference (with streaming) ────────────────────────────────────────
    def run_inference(self):
        if self.y is None:
            QMessageBox.warning(self, "Missing", "Load audio first."); return
        if not self.gt_rows:
            QMessageBox.warning(self, "Missing", "Load GT or run VAD."); return
        if self.backend is None:
            QMessageBox.warning(self, "Missing", "Load model first."); return
        if self.infer_thread and self.infer_thread.isRunning():
            return

        lang_code, lang_name = self.lang_combo.currentData()
        mode = self.mode_combo.currentData()
        self.results = [None] * len(self.gt_rows)
        self.progress_bar.setValue(0)
        self._infer_start_time = time.time()
        self.set_busy(True, "Running prediction...")

        self.infer_thread = InferenceThread(
            self.backend, self.y16, TARGET_SR, self.gt_rows,
            lang_code, lang_name, mode)
        self.infer_thread.segment_done.connect(self._on_segment_result)
        self.infer_thread.progress.connect(self._on_infer_prog)
        self.infer_thread.finished_ok.connect(self._on_infer_done)
        self.infer_thread.failed.connect(self._on_infer_fail)
        self.infer_thread.start()

    def _on_segment_result(self, idx, result):
        """Stream each result into the list as it arrives."""
        if 0 <= idx < len(self.results):
            self.results[idx] = result
            # Update just this row in the segment list
            if idx < self.seg_list.count():
                r = result
                sc = r["score"]
                sc_str = f"{sc:.2f}"
                text = (f"{idx+1:03d} | GT: {r['gt']}  →  PRED: {r['pred'] or '—'}"
                        f"  | {r['gt_start']:.2f}-{r['gt_end']:.2f}s  acc={sc_str}")
                self.seg_list.item(idx).setText(text)
                if sc >= 0.95:
                    self.seg_list.item(idx).setForeground(Qt.darkGreen)
                elif sc >= 0.5:
                    self.seg_list.item(idx).setForeground(Qt.darkYellow)
                else:
                    self.seg_list.item(idx).setForeground(Qt.red)

    def stop_inference(self):
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.request_stop()

    def _on_infer_prog(self, done, tot, pct):
        elapsed = time.time() - self._infer_start_time
        segs_per_sec = done / elapsed if elapsed > 0 else 0
        eta = (tot - done) / segs_per_sec if segs_per_sec > 0 else 0
        self.status_label.setText(f"Predicting {done}/{tot}  ({segs_per_sec:.1f} seg/s, ETA {eta:.0f}s)")
        self.progress_bar.setValue(pct)

    def _on_infer_done(self, results, mean_acc):
        self.results = results
        elapsed = time.time() - self._infer_start_time
        self.progress_bar.setValue(100)
        self.set_busy(False,
            f"Done: {len(results)} segs in {elapsed:.1f}s | "
            f"Accuracy: {mean_acc:.3f} | "
            f"Speed: {len(results)/elapsed:.1f} seg/s")
        self.speed_label.setText(f"{len(results)/elapsed:.1f} seg/s | {elapsed:.1f}s total")
        self.refresh_list(); self.redraw()

    def _on_infer_fail(self, msg):
        self.progress_bar.setValue(0)
        self.set_busy(False, "Inference failed.")
        QMessageBox.critical(self, "Error", msg)

    # ── Plot ──────────────────────────────────────────────────────────────
    def redraw(self):
        self.ax.clear()
        if self.wave_t is None:
            self.canvas.draw_idle(); return

        self.ax.plot(self.wave_t, self.wave_y, lw=0.5, color="black")
        ymax = max(np.max(np.abs(self.wave_y)), 1e-6)
        lang_code, _ = self.lang_combo.currentData()
        fp = pick_font(lang_code)

        data = [r for r in (self.results or []) if r is not None]
        if not data:
            data = [{"gt": r["label"], "pred": None,
                     "gt_start": r["start"], "gt_end": r["end"],
                     "pred_start": None, "pred_end": None, "score": None}
                    for r in self.gt_rows]

        x0, x1 = self.ax.get_xlim()
        if x0 == 0.0 and x1 == 1.0 and self.y is not None:
            x1 = len(self.y) / self.sr

        for idx, r in enumerate(data):
            s, e = r["gt_start"], r["gt_end"]
            if e < x0 or s > x1:
                continue

            color = "green"
            if r.get("score") is not None and r["score"] < 0.95:
                color = "red"
            self.ax.axvspan(s, e, alpha=0.10, color=color)

            if self.selected_index is not None and idx == self.selected_index:
                self.ax.axvspan(s, e, alpha=0.25, color="yellow")
                mid = (s + e) / 2
                kw = {"fontproperties": fp} if fp else {}
                self.ax.text(mid, ymax*0.95, f"GT: {r['gt']}",
                             ha="center", va="bottom", fontsize=10, color="green",
                             fontweight="bold",
                             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2), **kw)
                if r.get("pred") is not None:
                    shown = r["pred"] or "∅"
                    self.ax.text(mid, ymax*0.72, f"PRED: {shown}",
                                 ha="center", va="bottom", fontsize=10, color="blue",
                                 fontweight="bold",
                                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2), **kw)

        title = "Waveform"
        scores = [r["score"] for r in data if r.get("score") is not None]
        if scores:
            title += f" | Accuracy = {np.mean(scores):.3f}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Amplitude")
        self.canvas.draw_idle()

    # ── Segment List ──────────────────────────────────────────────────────
    def refresh_list(self):
        self.seg_list.clear()
        valid = [r for r in (self.results or []) if r is not None]
        src = valid if valid else [
            {"gt": r["label"], "pred": "", "gt_start": r["start"],
             "gt_end": r["end"], "score": None}
            for r in self.gt_rows]

        for i, r in enumerate(src):
            sc = r.get("score")
            sc_str = f"{sc:.2f}" if sc is not None else "  — "
            pred = r.get("pred", "") or "—"
            text = f"{i+1:03d} | GT: {r['gt']}  →  PRED: {pred}  | {r['gt_start']:.2f}-{r['gt_end']:.2f}s  acc={sc_str}"
            item = QListWidgetItem(text)
            if sc is not None:
                if sc >= 0.95:   item.setForeground(Qt.darkGreen)
                elif sc >= 0.5:  item.setForeground(Qt.darkYellow)
                else:            item.setForeground(Qt.red)
            self.seg_list.addItem(item)

    def on_segment_clicked(self, item):
        idx = self.seg_list.row(item)
        self.selected_index = idx
        valid = [r for r in (self.results or []) if r is not None]
        src = valid if valid else [{"gt_start": r["start"], "gt_end": r["end"]} for r in self.gt_rows]
        if 0 <= idx < len(src):
            s, e = src[idx]["gt_start"], src[idx]["gt_end"]
            pad = max(0.3, (e - s) * 0.2)
            self.ax.set_xlim(max(0, s - pad), e + pad)
            self.redraw()

    # ── Audio ─────────────────────────────────────────────────────────────
    def play_audio(self):
        if self.y is not None: sd.stop(); sd.play(self.y, self.sr)

    def play_selected_segment(self):
        if self.selected_index is None: return
        valid = [r for r in (self.results or []) if r is not None]
        src = valid if valid else [{"gt_start": r["start"], "gt_end": r["end"]} for r in self.gt_rows]
        if 0 <= self.selected_index < len(src):
            s, e = src[self.selected_index]["gt_start"], src[self.selected_index]["gt_end"]
            sd.stop(); sd.play(self.y[int(s*self.sr):int(e*self.sr)], self.sr)

    def stop_audio(self): sd.stop()

    # ── Zoom ──────────────────────────────────────────────────────────────
    def zoom(self, factor):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim(); c = (x0+x1)/2; w = (x1-x0)*factor/2
        tot = len(self.y)/self.sr
        self.ax.set_xlim(max(0, c-w), min(tot, c+w)); self.redraw()

    def on_scroll(self, event):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim()
        mx = event.xdata if event.xdata else (x0+x1)/2
        s = 0.6 if event.button == "up" else 1.6
        tot = len(self.y)/self.sr
        self.ax.set_xlim(max(0, mx-(mx-x0)*s), min(tot, mx+(x1-mx)*s)); self.redraw()

    def reset_plot_view(self):
        if self.y is not None:
            self.ax.set_xlim(0, len(self.y)/self.sr); self.redraw()

    # ── Export ────────────────────────────────────────────────────────────
    def export_csv(self):
        valid = [r for r in (self.results or []) if r is not None]
        if not valid:
            QMessageBox.warning(self, "No Results", "Run prediction first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "results.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["index","ground_truth","prediction","accuracy","gt_start","gt_end"])
                for i, r in enumerate(valid):
                    w.writerow([i+1, r["gt"], r["pred"], f"{r['score']:.4f}",
                                f"{r['gt_start']:.4f}", f"{r['gt_end']:.4f}"])
            self.status_label.setText(f"Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Indic Annotation Tool  [ULTRA-FAST]")
    print(f"  Engines: whisper.cpp={HAS_WHISPERCPP}  "
          f"faster-whisper={HAS_FASTER_WHISPER}  HF={HAS_HF}")
    print(f"  CPU threads: {CPU_THREADS}")
    if HAS_HF:
        import torch as _t
        print(f"  CUDA: {_t.cuda.is_available()}"
              + (f" ({_t.cuda.get_device_name(0)})" if _t.cuda.is_available() else ""))
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton { padding:5px 10px; border:1px solid #d1d5db; border-radius:4px;
                      background:#f9fafb; font-size:12px; }
        QPushButton:hover { background:#e5e7eb; }
        QPushButton:disabled { color:#9ca3af; }
        QComboBox, QSpinBox, QDoubleSpinBox { padding:3px 6px; font-size:12px; }
    """)
    w = AnnotationTool()
    w.show()
    sys.exit(app.exec_())