#!/usr/bin/env python3
"""
Indic Speech Annotation & Segmentation Tool
============================================
Languages : English, Hindi, Gujarati, Marathi
Engines   : faster-whisper (primary), HuggingFace transformers (fallback)

Install:
    pip install faster-whisper soundfile scipy numpy pandas \
                matplotlib PyQt5 sounddevice openpyxl

If faster-whisper is unavailable, install HuggingFace:
    pip install transformers torch
"""

# ══════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════
import os
import sys
import csv
import re
import time
import traceback
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import sounddevice as sd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTabWidget,
)

# ── Optional: fast audio I/O ─────────────────────────────────────────────
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Engine availability ──────────────────────────────────────────────────
_HAS_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel as _FWModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    pass

_HAS_HF = False
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    _HAS_HF = True
except ImportError:
    pass

if not _HAS_FASTER_WHISPER and not _HAS_HF:
    print("ERROR: Install at least one engine:")
    print("  pip install faster-whisper        # recommended")
    print("  pip install transformers torch     # fallback")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

SR16 = 16000  # Whisper expects 16 kHz

# (display, hf_model_id, faster_whisper_size)
MODELS = [
    ("tiny   — fastest, low accuracy",      "openai/whisper-tiny",     "tiny"),
    ("base   — very fast",                  "openai/whisper-base",     "base"),
    ("small  — fast, decent accuracy",      "openai/whisper-small",    "small"),
    ("medium — balanced",                   "openai/whisper-medium",   "medium"),
    ("large-v3 — best accuracy",            "openai/whisper-large-v3", "large-v3"),
]

# (display, iso_code, hf_full_name)
# iso_code  → used by faster-whisper
# hf_name   → used by HuggingFace generate()
LANGUAGES = [
    ("English",  "en", "english"),
    ("Hindi",    "hi", "hindi"),
    ("Gujarati", "gu", "gujarati"),
    ("Marathi",  "mr", "marathi"),
]

PRED_MODES = [
    ("Full Transcript", "full"),
    ("First Word",      "word"),
    ("First Character",  "char"),
]

# Available engines
ENGINES = []
if _HAS_FASTER_WHISPER:
    ENGINES.append(("faster-whisper (recommended)", "faster_whisper"))
if _HAS_HF:
    ENGINES.append(("HuggingFace transformers",     "huggingface"))

CPU_THREADS = max(1, min(os.cpu_count() or 4, 8))


# ══════════════════════════════════════════════════════════════════════════
#  TEXT & AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════

# Unicode: Devanagari + Devanagari Extended + Vedic + Gujarati
_INDIC = r"\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF\u0A80-\u0AFF"
_RE_CLEAN  = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM   = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA  = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_text(text, mode="full"):
    """Clean transcription output."""
    if not text:
        return ""
    text = _RE_CLEAN.sub("", str(text).strip())
    text = _RE_SPACES.sub(" ", text).strip()
    if not text:
        return ""
    if mode == "word":
        return text.split()[0] if text.split() else ""
    if mode == "char":
        m = _RE_ALPHA.search(text)
        return m.group(0) if m else ""
    return text


def norm_text(s):
    """Normalise for comparison: lowercase, remove non-alphanumeric."""
    return _RE_NORM.sub("", str(s).strip().lower())


def edit_distance(a, b):
    """Levenshtein distance."""
    na, nb = len(a), len(b)
    if na == 0: return nb
    if nb == 0: return na
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, nb + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[nb]


def accuracy(pred, gt):
    """Character-level accuracy: 1 - CER, clamped [0,1]."""
    p, g = norm_text(pred), norm_text(gt)
    if not g and not p: return 1.0
    if not g: return 0.0
    return max(0.0, 1.0 - edit_distance(p, g) / len(g))


def load_audio(path):
    """Load audio file as mono float32 + its sample rate."""
    if _HAS_SF:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return y.astype(np.float32), sr
        except Exception:
            pass
    # Fallback to librosa
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), sr


def resample(y, orig_sr, target_sr):
    """Resample audio. scipy polyphase > librosa."""
    if orig_sr == target_sr:
        return y
    if _HAS_SCIPY:
        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(y, target_sr // g, orig_sr // g).astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def parse_time(x):
    """Parse timestamp string → seconds."""
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


@lru_cache(maxsize=8)
def get_font(lang_code):
    """Find a font that can render the given language."""
    mapping = {
        "hi": (["NotoSansDevanagari-Regular.ttf", "Lohit-Devanagari.ttf"],
               ["Noto Sans Devanagari", "Lohit Devanagari", "Mangal", "Arial Unicode MS"]),
        "gu": (["NotoSansGujarati-Regular.ttf", "Lohit-Gujarati.ttf"],
               ["Noto Sans Gujarati", "Lohit Gujarati", "Shruti", "Arial Unicode MS"]),
        "en": ([], ["Arial", "Helvetica", "DejaVu Sans"]),
    }
    lc = "hi" if lang_code == "mr" else lang_code
    filenames, families = mapping.get(lc, mapping["en"])

    search_dirs = [
        "/usr/share/fonts", "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for fn in filenames:
                if fn in files:
                    return fm.FontProperties(fname=os.path.join(root, fn))

    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in families:
        if fam in installed:
            return fm.FontProperties(family=fam)
    return None


# ══════════════════════════════════════════════════════════════════════════
#  VAD SEGMENTATION (fully vectorised)
# ══════════════════════════════════════════════════════════════════════════

def vad_segment(y, sr, silence_ms=300, thresh_db=-40, min_ms=200, max_ms=10000):
    """Energy-based VAD. Returns list of {start, end} dicts (in seconds)."""
    frame_len = int(sr * 0.025)  # 25 ms
    hop = int(sr * 0.010)        # 10 ms
    n_frames = 1 + (len(y) - frame_len) // hop
    if n_frames <= 0:
        return []

    # Compute RMS per frame
    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop
    np.clip(idx, 0, len(y) - 1, out=idx)
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)

    voiced = rms_db > thresh_db

    # Find voiced runs
    pad = np.concatenate([[False], voiced, [False]])
    diff = np.diff(pad.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Merge short silences
    sil_f = max(1, int(silence_ms / 1000 * sr / hop))
    min_f = max(1, int(min_ms / 1000 * sr / hop))
    max_f = max(1, int(max_ms / 1000 * sr / hop))

    merged = []
    i = 0
    while i < len(starts):
        s, e = starts[i], ends[i]
        while i + 1 < len(starts) and (starts[i+1] - e) < sil_f:
            i += 1
            e = ends[i]
        merged.append((s, e))
        i += 1

    # Split long segments, filter short, convert to seconds
    result = []
    for s, e in merged:
        if (e - s) < min_f:
            continue
        for cs in range(s, e, max_f):
            ce = min(cs + max_f, e)
            if (ce - cs) >= min_f:
                result.append({
                    "start": round(cs * hop / sr, 4),
                    "end":   round(min(ce * hop / sr, len(y) / sr), 4),
                })
    return result


# ══════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

class TranscriptionEngine:
    """Unified interface wrapping faster-whisper or HuggingFace."""

    def __init__(self):
        self.engine_type = None   # "faster_whisper" or "huggingface"
        self.model = None
        self.processor = None     # HF only
        self.device = None        # HF only
        self._loaded_key = None   # (engine_type, model_size)

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self, engine_type, hf_model_id, fw_size, progress_fn=None):
        """
        Load (or reuse) a model.
        Returns (success: bool, message: str, elapsed: float)
        """
        cache_key = (engine_type, fw_size)
        if self._loaded_key == cache_key and self.model is not None:
            return True, f"Already loaded: {fw_size}", 0.0

        t0 = time.time()
        self.model = None
        self.processor = None
        self.engine_type = engine_type

        try:
            if engine_type == "faster_whisper":
                self._load_faster_whisper(fw_size)
            elif engine_type == "huggingface":
                self._load_huggingface(hf_model_id)
            else:
                return False, f"Unknown engine: {engine_type}", 0.0

            self._loaded_key = cache_key
            elapsed = time.time() - t0
            return True, f"Loaded {fw_size} ({engine_type}) in {elapsed:.1f}s", elapsed

        except Exception as e:
            traceback.print_exc()
            self.model = None
            return False, str(e), time.time() - t0

    def _load_faster_whisper(self, size):
        log(f"Loading faster-whisper model: {size}")
        self.model = _FWModel(
            size,
            device="cuda" if (_HAS_HF and torch.cuda.is_available()) else "cpu",
            compute_type="float16" if (_HAS_HF and torch.cuda.is_available()) else "int8",
            cpu_threads=CPU_THREADS,
        )
        log(f"faster-whisper model loaded: {size}")

    def _load_huggingface(self, model_id):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        log(f"Loading HuggingFace model: {model_id} on {device} ({dtype})")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,       # ← correct param name (NOT "dtype")
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        log(f"HuggingFace model loaded: {model_id}")

    def transcribe(self, audio_16k, lang_iso, lang_name, mode="full"):
        """
        Transcribe a single audio segment.

        Args:
            audio_16k: float32 numpy array at 16kHz
            lang_iso:  ISO 639-1 code ("gu", "hi", "mr", "en")
            lang_name: full name ("gujarati", "hindi", "marathi", "english")
            mode:      "full", "word", or "char"

        Returns:
            str: cleaned transcription
        """
        if self.model is None:
            return ""
        if audio_16k is None or len(audio_16k) == 0:
            return ""

        # Pad very short segments (< 0.1s) to avoid errors
        min_samples = int(0.1 * SR16)
        if len(audio_16k) < min_samples:
            audio_16k = np.pad(audio_16k, (0, min_samples - len(audio_16k)))

        # Ensure float32
        audio_16k = audio_16k.astype(np.float32)

        if self.engine_type == "faster_whisper":
            return self._transcribe_fw(audio_16k, lang_iso, mode)
        else:
            return self._transcribe_hf(audio_16k, lang_name, mode)

    def _transcribe_fw(self, audio, lang_iso, mode):
        """faster-whisper transcription."""
        # transcribe() returns (generator, info) — MUST consume the generator!
        segments_generator, info = self.model.transcribe(
            audio,
            language=lang_iso,     # ISO code: "gu", "hi", "mr", "en"
            task="transcribe",
            beam_size=1,           # greedy = fastest
            best_of=1,
            vad_filter=False,      # we do our own VAD
            without_timestamps=True,
            condition_on_previous_text=False,
        )

        # CRITICAL: consume the generator by converting to list
        segments_list = list(segments_generator)

        # Join all segment texts
        text = " ".join(seg.text.strip() for seg in segments_list if seg.text.strip())
        return clean_text(text, mode)

    def _transcribe_hf(self, audio, lang_name, mode):
        """HuggingFace transcription."""
        inputs = self.processor(
            audio,
            sampling_rate=SR16,
            return_tensors="pt",
            return_attention_mask=True,
        )

        features = inputs.input_features.to(self.device)
        attn_mask = getattr(inputs, "attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        gen_kwargs = {
            "input_features": features,
            "task": "transcribe",
            "language": lang_name,  # full name: "gujarati", "hindi", etc.
            "return_timestamps": False,
        }
        if attn_mask is not None:
            gen_kwargs["attention_mask"] = attn_mask

        with torch.inference_mode():
            ids = self.model.generate(**gen_kwargs)

        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return clean_text(text, mode)


# Global engine instance (persists across model switches)
ENGINE = TranscriptionEngine()


# ══════════════════════════════════════════════════════════════════════════
#  BACKGROUND THREADS
# ══════════════════════════════════════════════════════════════════════════

class AudioLoadThread(QThread):
    """Load + resample audio without freezing UI."""
    done = pyqtSignal(np.ndarray, int, np.ndarray, str)
    error = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            t0 = time.time()
            y, sr = load_audio(self.path)
            y16 = resample(y, sr, SR16)
            log(f"Audio loaded: {len(y)/sr:.1f}s, sr={sr} → 16kHz in {time.time()-t0:.2f}s")
            self.done.emit(y, sr, y16, self.path)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class ModelLoadThread(QThread):
    """Load model without freezing UI."""
    done = pyqtSignal(bool, str, float)  # success, message, elapsed
    error = pyqtSignal(str)

    def __init__(self, engine_type, hf_id, fw_size):
        super().__init__()
        self.engine_type = engine_type
        self.hf_id = hf_id
        self.fw_size = fw_size

    def run(self):
        try:
            ok, msg, elapsed = ENGINE.load(self.engine_type, self.hf_id, self.fw_size)
            self.done.emit(ok, msg, elapsed)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class PredictionThread(QThread):
    """Run transcription on all segments."""
    segment_result = pyqtSignal(int, str, float)  # index, prediction, accuracy
    progress = pyqtSignal(int, int)                # done, total
    done = pyqtSignal(list, float, float)          # results, mean_acc, elapsed
    error = pyqtSignal(str)

    def __init__(self, y16, segments, lang_iso, lang_name, mode):
        super().__init__()
        self.y16 = y16
        self.segments = segments     # list of {label, start, end}
        self.lang_iso = lang_iso
        self.lang_name = lang_name
        self.mode = mode
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            t0 = time.time()
            total = len(self.segments)
            results = []

            log(f"Starting prediction: {total} segments, "
                f"lang={self.lang_iso}/{self.lang_name}, mode={self.mode}")

            for i, seg in enumerate(self.segments):
                if self._stop:
                    log("Prediction stopped by user")
                    break

                # Extract audio chunk
                s_samp = max(0, int(seg["start"] * SR16))
                e_samp = min(len(self.y16), int(seg["end"] * SR16))
                chunk = self.y16[s_samp:e_samp]

                # Transcribe
                pred = ENGINE.transcribe(
                    chunk, self.lang_iso, self.lang_name, self.mode
                )
                acc = accuracy(pred, seg["label"])

                results.append({
                    "index": i,
                    "gt": seg["label"],
                    "pred": pred,
                    "start": seg["start"],
                    "end": seg["end"],
                    "accuracy": acc,
                })

                # Emit per-segment result for streaming UI update
                self.segment_result.emit(i, pred, acc)
                self.progress.emit(i + 1, total)

                if (i + 1) % 5 == 0 or (i + 1) == total:
                    log(f"  [{i+1}/{total}] GT='{seg['label']}' → PRED='{pred}' acc={acc:.3f}")

            elapsed = time.time() - t0
            mean_acc = np.mean([r["accuracy"] for r in results]) if results else 0.0

            log(f"Prediction complete: {len(results)} segments in {elapsed:.1f}s, "
                f"mean accuracy={mean_acc:.3f}")

            self.done.emit(results, float(mean_acc), elapsed)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class VADThread(QThread):
    done = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, y, sr, params):
        super().__init__()
        self.y, self.sr, self.params = y, sr, params

    def run(self):
        try:
            segs = vad_segment(self.y, self.sr, **self.params)
            self.done.emit(segs)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indic Speech Annotation Tool")
        self.resize(1700, 1000)

        # ── State ─────────────────────────────────────────────────────────
        self.audio_path = None
        self.y = None           # original sample rate
        self.sr = None
        self.y16 = None         # 16 kHz for Whisper
        self.wave_t = None      # downsampled time axis for plotting
        self.wave_y = None      # downsampled amplitude for plotting

        self.segments = []      # list of {label, start, end}
        self.results = []       # list of {index, gt, pred, start, end, accuracy}
        self.selected_idx = None

        # Active threads
        self._audio_thread = None
        self._model_thread = None
        self._pred_thread = None
        self._vad_thread = None
        self._pred_start_time = None

        self._build_ui()

        # Auto-load default model after UI is up
        QTimer.singleShot(200, self._auto_load_model)

    # ══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── Row 1: Configuration ──────────────────────────────────────────
        row1 = QHBoxLayout()

        self.cb_lang = QComboBox()
        for display, iso, name in LANGUAGES:
            self.cb_lang.addItem(display, (iso, name))

        self.cb_model = QComboBox()
        for display, hf_id, fw_size in MODELS:
            self.cb_model.addItem(display, (hf_id, fw_size))
        self.cb_model.setCurrentIndex(2)  # small as default

        self.cb_engine = QComboBox()
        for display, key in ENGINES:
            self.cb_engine.addItem(display, key)

        self.cb_mode = QComboBox()
        for display, key in PRED_MODES:
            self.cb_mode.addItem(display, key)

        self.btn_load_audio = QPushButton("Load Audio")
        self.btn_load_audio.clicked.connect(self.on_load_audio)

        self.btn_load_gt = QPushButton("Load Ground Truth")
        self.btn_load_gt.clicked.connect(self.on_load_gt)

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.on_load_model)

        for w in [QLabel("Language:"), self.cb_lang,
                  QLabel("Model:"), self.cb_model,
                  QLabel("Engine:"), self.cb_engine,
                  QLabel("Output:"), self.cb_mode,
                  self.btn_load_audio, self.btn_load_gt, self.btn_load_model]:
            row1.addWidget(w)
        row1.addStretch()

        # ── Row 2: Actions ────────────────────────────────────────────────
        row2 = QHBoxLayout()

        self.btn_run = QPushButton("▶ Run Prediction")
        self.btn_run.clicked.connect(self.on_run_prediction)
        self.btn_run.setStyleSheet("font-weight: bold; color: #16a34a;")

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.clicked.connect(self.on_stop_prediction)
        self.btn_stop.setEnabled(False)

        self.btn_vad = QPushButton("Auto-Segment (VAD)")
        self.btn_vad.clicked.connect(self.on_run_vad)

        self.btn_play_all = QPushButton("♫ Play All")
        self.btn_play_all.clicked.connect(lambda: self._play(self.y, self.sr))

        self.btn_play_seg = QPushButton("♫ Play Segment")
        self.btn_play_seg.clicked.connect(self.on_play_segment)

        self.btn_stop_audio = QPushButton("⏸ Stop Audio")
        self.btn_stop_audio.clicked.connect(sd.stop)

        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(0.6))

        self.btn_zoom_out = QPushButton("🔍−")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(1.6))

        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self._reset_view)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.on_export_csv)

        for w in [self.btn_run, self.btn_stop, self.btn_vad,
                  self.btn_play_all, self.btn_play_seg, self.btn_stop_audio,
                  self.btn_zoom_in, self.btn_zoom_out, self.btn_reset,
                  self.btn_export]:
            row2.addWidget(w)
        row2.addStretch()

        # ── Status bar ────────────────────────────────────────────────────
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Starting up...")
        self.lbl_status.setStyleSheet("padding: 4px; font-size: 13px;")
        self.lbl_speed = QLabel("")
        self.lbl_speed.setStyleSheet("color: #6b7280; font-size: 12px;")
        status_row.addWidget(self.lbl_status, 1)
        status_row.addWidget(self.lbl_speed)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(18)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc; border-radius: 4px;
                background: #f3f4f6; text-align: center; font-size: 11px;
            }
            QProgressBar::chunk { background: #3b82f6; border-radius: 4px; }
        """)

        # ── Waveform plot ─────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # ── Right panel ───────────────────────────────────────────────────
        # Segment list
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_seg_clicked)
        self.seg_list.setStyleSheet("""
            QListWidget {
                background: #fff; border: 1px solid #d0d0d0;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
            }
            QListWidget::item { padding: 3px 6px; }
            QListWidget::item:selected { background: #dbeafe; }
        """)

        # VAD settings
        vad_box = QGroupBox("VAD Parameters")
        vad_lay = QVBoxLayout(vad_box)

        self.sp_silence = QSpinBox()
        self.sp_silence.setRange(50, 2000); self.sp_silence.setValue(300)
        self.sp_silence.setSuffix(" ms")

        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(-80, 0); self.sp_thresh.setValue(-40)
        self.sp_thresh.setSuffix(" dB")

        self.sp_min_seg = QSpinBox()
        self.sp_min_seg.setRange(50, 5000); self.sp_min_seg.setValue(200)
        self.sp_min_seg.setSuffix(" ms")

        self.sp_max_seg = QSpinBox()
        self.sp_max_seg.setRange(1000, 30000); self.sp_max_seg.setValue(10000)
        self.sp_max_seg.setSuffix(" ms")

        for label, widget in [
            ("Min silence gap:", self.sp_silence),
            ("Silence threshold:", self.sp_thresh),
            ("Min segment:", self.sp_min_seg),
            ("Max segment:", self.sp_max_seg),
        ]:
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            vad_lay.addLayout(h)
        vad_lay.addStretch()

        # Tabs
        tabs = QTabWidget()

        seg_tab = QWidget()
        seg_lay = QVBoxLayout(seg_tab)
        legend = QLabel(
            '<span style="color:green; font-weight:bold;">■ GT</span>  '
            '<span style="color:blue; font-weight:bold;">■ PRED</span>  '
            '| Colour: '
            '<span style="color:#16a34a;">✓≥95%</span>  '
            '<span style="color:#ea580c;">~50-95%</span>  '
            '<span style="color:#dc2626;">✗&lt;50%</span>'
        )
        legend.setTextFormat(Qt.RichText)
        seg_lay.addWidget(legend)
        seg_lay.addWidget(self.seg_list)

        tabs.addTab(seg_tab, "Segments")
        tabs.addTab(vad_box, "VAD Settings")

        # Layout
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self.toolbar)
        left_lay.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setSizes([1250, 400])

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(status_row)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(splitter)

    # ══════════════════════════════════════════════════════════════════════
    #  UI HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _set_busy(self, busy, msg=""):
        """Disable/enable interactive controls."""
        enabled = not busy
        for w in [self.btn_load_audio, self.btn_load_gt, self.btn_load_model,
                  self.btn_run, self.btn_vad, self.btn_export,
                  self.cb_lang, self.cb_model, self.cb_engine, self.cb_mode]:
            w.setEnabled(enabled)
        self.btn_stop.setEnabled(busy)
        if msg:
            self.lbl_status.setText(msg)

    def _build_wave_cache(self):
        """Downsample waveform for fast plotting."""
        if self.y is None:
            self.wave_t = self.wave_y = None
            return
        max_pts = 25000
        step = max(1, len(self.y) // max_pts)
        self.wave_y = self.y[::step]
        self.wave_t = np.arange(len(self.wave_y)) * (step / self.sr)

    def _play(self, y, sr):
        if y is not None and sr:
            sd.stop()
            sd.play(y, sr)

    # ══════════════════════════════════════════════════════════════════════
    #  LOAD AUDIO
    # ══════════════════════════════════════════════════════════════════════

    def on_load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.webm *.aac)")
        if not path:
            return

        self.lbl_status.setText(f"Loading {os.path.basename(path)}...")
        self.btn_load_audio.setEnabled(False)

        self._audio_thread = AudioLoadThread(path)
        self._audio_thread.done.connect(self._on_audio_loaded)
        self._audio_thread.error.connect(self._on_audio_error)
        self._audio_thread.start()

    def _on_audio_loaded(self, y, sr, y16, path):
        self.y, self.sr, self.y16, self.audio_path = y, sr, y16, path
        self.results = []
        self.selected_idx = None
        self.progress.setValue(0)
        self._build_wave_cache()

        dur = len(y) / sr
        self.lbl_status.setText(f"Audio: {os.path.basename(path)}  ({dur:.1f}s, sr={sr})")
        self.btn_load_audio.setEnabled(True)
        self._refresh_list()
        self._redraw()

    def _on_audio_error(self, msg):
        self.btn_load_audio.setEnabled(True)
        QMessageBox.critical(self, "Audio Load Error", msg)

    # ══════════════════════════════════════════════════════════════════════
    #  LOAD GROUND TRUTH
    # ══════════════════════════════════════════════════════════════════════

    def on_load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Truth", "",
            "Spreadsheets (*.xlsx *.csv *.tsv)")
        if not path:
            return

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, header=None)
            elif path.endswith(".tsv"):
                df = pd.read_csv(path, header=None, sep="\t")
            else:
                df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")

            segs = []
            skipped = 0
            for _, row in df.iterrows():
                label = str(row.iloc[0]).strip()
                start = parse_time(row.iloc[1])
                end = parse_time(row.iloc[2])

                if not label or start is None or end is None or end <= start:
                    skipped += 1
                    continue

                segs.append({"label": label, "start": start, "end": end})

            self.segments = segs
            self.results = []
            self.selected_idx = None
            self.progress.setValue(0)

            self.lbl_status.setText(
                f"Ground truth: {len(segs)} segments loaded"
                + (f" ({skipped} skipped)" if skipped else ""))
            self._refresh_list()
            self._redraw()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "GT Load Error", str(e))

    # ══════════════════════════════════════════════════════════════════════
    #  LOAD MODEL
    # ══════════════════════════════════════════════════════════════════════

    def on_load_model(self):
        if self._model_thread and self._model_thread.isRunning():
            return

        hf_id, fw_size = self.cb_model.currentData()
        engine = self.cb_engine.currentData()

        self._set_busy(True, f"Loading model: {fw_size} ({engine})...")

        self._model_thread = ModelLoadThread(engine, hf_id, fw_size)
        self._model_thread.done.connect(self._on_model_loaded)
        self._model_thread.error.connect(self._on_model_error)
        self._model_thread.start()

    def _auto_load_model(self):
        """Called on app startup to preload default model."""
        if ENGINE.is_loaded:
            self.lbl_status.setText("Model already loaded. Load audio to begin.")
            return

        hf_id, fw_size = self.cb_model.currentData()
        engine = self.cb_engine.currentData()
        self.lbl_status.setText(f"Auto-loading {fw_size}...")
        self.btn_load_model.setEnabled(False)

        self._model_thread = ModelLoadThread(engine, hf_id, fw_size)
        self._model_thread.done.connect(self._on_model_loaded)
        self._model_thread.error.connect(self._on_model_error)
        self._model_thread.start()

    def _on_model_loaded(self, ok, msg, elapsed):
        self._set_busy(False)
        if ok:
            self.lbl_status.setText(f"✓ {msg}")
            self.lbl_speed.setText(f"Engine: {ENGINE.engine_type}")
        else:
            self.lbl_status.setText(f"✗ Model load failed: {msg}")
            QMessageBox.critical(self, "Model Error", msg)

    def _on_model_error(self, msg):
        self._set_busy(False)
        self.lbl_status.setText(f"✗ Model error: {msg}")
        QMessageBox.critical(self, "Model Error", msg)

    # ══════════════════════════════════════════════════════════════════════
    #  VAD AUTO-SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════

    def on_run_vad(self):
        if self.y is None:
            QMessageBox.warning(self, "No Audio", "Load an audio file first.")
            return

        params = {
            "silence_ms": self.sp_silence.value(),
            "thresh_db": self.sp_thresh.value(),
            "min_ms": self.sp_min_seg.value(),
            "max_ms": self.sp_max_seg.value(),
        }
        self._set_busy(True, "Running VAD segmentation...")

        self._vad_thread = VADThread(self.y, self.sr, params)
        self._vad_thread.done.connect(self._on_vad_done)
        self._vad_thread.error.connect(lambda m: (
            self._set_busy(False), QMessageBox.critical(self, "VAD Error", m)))
        self._vad_thread.start()

    def _on_vad_done(self, raw_segs):
        self.segments = [
            {"label": f"seg_{i+1:04d}", "start": s["start"], "end": s["end"]}
            for i, s in enumerate(raw_segs)
        ]
        self.results = []
        self.selected_idx = None
        self._set_busy(False, f"VAD found {len(raw_segs)} segments")
        self._refresh_list()
        self._redraw()

    # ══════════════════════════════════════════════════════════════════════
    #  RUN PREDICTION
    # ══════════════════════════════════════════════════════════════════════

    def on_run_prediction(self):
        if self.y16 is None:
            QMessageBox.warning(self, "No Audio", "Load an audio file first.")
            return
        if not self.segments:
            QMessageBox.warning(self, "No Segments",
                                "Load ground truth or run auto-segmentation first.")
            return
        if not ENGINE.is_loaded:
            QMessageBox.warning(self, "No Model", "Load a model first.")
            return
        if self._pred_thread and self._pred_thread.isRunning():
            return

        lang_iso, lang_name = self.cb_lang.currentData()
        mode = self.cb_mode.currentData()

        self.results = []
        self.progress.setValue(0)
        self._pred_start_time = time.time()
        self._set_busy(True, "Running prediction...")

        # Pre-populate segment list with pending state
        self._refresh_list()

        self._pred_thread = PredictionThread(
            self.y16, self.segments, lang_iso, lang_name, mode)
        self._pred_thread.segment_result.connect(self._on_segment_result)
        self._pred_thread.progress.connect(self._on_pred_progress)
        self._pred_thread.done.connect(self._on_pred_done)
        self._pred_thread.error.connect(self._on_pred_error)
        self._pred_thread.start()

    def _on_segment_result(self, idx, pred, acc):
        """Update single segment in list as results stream in."""
        if idx < self.seg_list.count():
            seg = self.segments[idx]
            text = self._format_seg_text(idx, seg["label"], pred, seg["start"], seg["end"], acc)
            item = self.seg_list.item(idx)
            item.setText(text)
            if acc >= 0.95:
                item.setForeground(Qt.darkGreen)
            elif acc >= 0.5:
                item.setForeground(Qt.darkYellow)
            else:
                item.setForeground(Qt.red)

    def _on_pred_progress(self, done, total):
        pct = int(done / total * 100) if total > 0 else 0
        self.progress.setValue(pct)

        elapsed = time.time() - self._pred_start_time
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        self.lbl_status.setText(
            f"Predicting {done}/{total}  ({speed:.1f} seg/s, ETA {eta:.0f}s)")

    def _on_pred_done(self, results, mean_acc, elapsed):
        self.results = results
        self.progress.setValue(100)

        n = len(results)
        speed = n / elapsed if elapsed > 0 else 0
        self._set_busy(False,
            f"✓ Done: {n} segments in {elapsed:.1f}s  |  "
            f"Mean accuracy: {mean_acc:.3f}  |  "
            f"Speed: {speed:.1f} seg/s")
        self.lbl_speed.setText(f"{speed:.1f} seg/s")
        self._refresh_list()
        self._redraw()

    def _on_pred_error(self, msg):
        self.progress.setValue(0)
        self._set_busy(False, f"✗ Prediction failed: {msg}")
        QMessageBox.critical(self, "Prediction Error", msg)

    def on_stop_prediction(self):
        if self._pred_thread and self._pred_thread.isRunning():
            self._pred_thread.request_stop()
            self.lbl_status.setText("Stopping...")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT
    # ══════════════════════════════════════════════════════════════════════

    def _redraw(self):
        self.ax.clear()
        if self.wave_t is None:
            self.canvas.draw_idle()
            return

        self.ax.plot(self.wave_t, self.wave_y, lw=0.5, color="black")
        ymax = max(np.max(np.abs(self.wave_y)), 1e-6)

        lang_iso, _ = self.cb_lang.currentData()
        fp = get_font(lang_iso)

        # Get current viewport
        x0, x1 = self.ax.get_xlim()
        if x0 == 0.0 and x1 == 1.0 and self.y is not None:
            x1 = len(self.y) / self.sr

        # Draw segment regions (only visible ones)
        for i, seg in enumerate(self.segments):
            s, e = seg["start"], seg["end"]
            if e < x0 or s > x1:
                continue  # Skip off-screen

            # Colour based on accuracy (if results exist)
            color = "green"
            if self.results and i < len(self.results):
                acc = self.results[i]["accuracy"]
                color = "#16a34a" if acc >= 0.95 else ("#ea580c" if acc >= 0.5 else "#dc2626")

            self.ax.axvspan(s, e, alpha=0.12, color=color)

            # Highlight selected
            if self.selected_idx is not None and i == self.selected_idx:
                self.ax.axvspan(s, e, alpha=0.3, color="yellow")
                mid = (s + e) / 2
                fkw = {"fontproperties": fp} if fp else {}
                bbox = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2)

                self.ax.text(mid, ymax * 0.95, f"GT: {seg['label']}",
                             ha="center", va="bottom", fontsize=10,
                             color="green", fontweight="bold", bbox=bbox, **fkw)

                if self.results and i < len(self.results):
                    pred = self.results[i]["pred"] or "∅"
                    acc = self.results[i]["accuracy"]
                    self.ax.text(mid, ymax * 0.72,
                                 f"PRED: {pred}  (acc={acc:.2f})",
                                 ha="center", va="bottom", fontsize=10,
                                 color="blue", fontweight="bold", bbox=bbox, **fkw)

        # Title
        title = "Waveform"
        if self.results:
            mean = np.mean([r["accuracy"] for r in self.results])
            title += f"  |  Mean accuracy = {mean:.3f}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    #  SEGMENT LIST
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_seg_text(idx, gt, pred, start, end, acc):
        pred_str = pred if pred else "—"
        if acc is not None:
            return f"{idx+1:03d} │ GT: {gt}  →  PRED: {pred_str}  │ {start:.2f}-{end:.2f}s  acc={acc:.2f}"
        else:
            return f"{idx+1:03d} │ GT: {gt}  →  PRED: ...  │ {start:.2f}-{end:.2f}s"

    def _refresh_list(self):
        self.seg_list.clear()
        for i, seg in enumerate(self.segments):
            gt = seg["label"]
            start, end = seg["start"], seg["end"]

            if self.results and i < len(self.results):
                r = self.results[i]
                text = self._format_seg_text(i, gt, r["pred"], start, end, r["accuracy"])
                item = QListWidgetItem(text)
                acc = r["accuracy"]
                if acc >= 0.95:
                    item.setForeground(Qt.darkGreen)
                elif acc >= 0.5:
                    item.setForeground(Qt.darkYellow)
                else:
                    item.setForeground(Qt.red)
            else:
                text = self._format_seg_text(i, gt, None, start, end, None)
                item = QListWidgetItem(text)

            self.seg_list.addItem(item)

    def on_seg_clicked(self, item):
        idx = self.seg_list.row(item)
        self.selected_idx = idx

        if 0 <= idx < len(self.segments):
            seg = self.segments[idx]
            s, e = seg["start"], seg["end"]
            pad = max(0.3, (e - s) * 0.3)
            self.ax.set_xlim(max(0, s - pad), e + pad)
            self._redraw()

    # ══════════════════════════════════════════════════════════════════════
    #  PLAYBACK
    # ══════════════════════════════════════════════════════════════════════

    def on_play_segment(self):
        if self.selected_idx is None or self.y is None:
            return
        if 0 <= self.selected_idx < len(self.segments):
            seg = self.segments[self.selected_idx]
            s = int(seg["start"] * self.sr)
            e = int(seg["end"] * self.sr)
            sd.stop()
            sd.play(self.y[s:e], self.sr)

    # ══════════════════════════════════════════════════════════════════════
    #  ZOOM
    # ══════════════════════════════════════════════════════════════════════

    def _zoom(self, factor):
        if self.y is None:
            return
        x0, x1 = self.ax.get_xlim()
        c = (x0 + x1) / 2
        w = (x1 - x0) * factor / 2
        total = len(self.y) / self.sr
        self.ax.set_xlim(max(0, c - w), min(total, c + w))
        self._redraw()

    def _on_scroll(self, event):
        if self.y is None:
            return
        x0, x1 = self.ax.get_xlim()
        mx = event.xdata if event.xdata else (x0 + x1) / 2
        f = 0.6 if event.button == "up" else 1.6
        total = len(self.y) / self.sr
        self.ax.set_xlim(
            max(0, mx - (mx - x0) * f),
            min(total, mx + (x1 - mx) * f))
        self._redraw()

    def _reset_view(self):
        if self.y is not None:
            self.ax.set_xlim(0, len(self.y) / self.sr)
            self._redraw()

    # ══════════════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════════════

    def on_export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run prediction first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results.csv", "CSV (*.csv)")
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["index", "ground_truth", "prediction",
                            "accuracy", "start_sec", "end_sec"])
                for r in self.results:
                    w.writerow([
                        r["index"] + 1,
                        r["gt"],
                        r["pred"],
                        f"{r['accuracy']:.4f}",
                        f"{r['start']:.4f}",
                        f"{r['end']:.4f}",
                    ])
            self.lbl_status.setText(f"Exported: {path}")
            log(f"CSV exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Indic Speech Annotation Tool")
    print(f"  Engines available:")
    if _HAS_FASTER_WHISPER:
        print(f"    ✓ faster-whisper (CTranslate2, int8, {CPU_THREADS} threads)")
    else:
        print(f"    ✗ faster-whisper  → pip install faster-whisper")
    if _HAS_HF:
        cuda = torch.cuda.is_available()
        gpu = f" ({torch.cuda.get_device_name(0)})" if cuda else ""
        print(f"    ✓ HuggingFace transformers (CUDA={cuda}{gpu})")
    else:
        print(f"    ✗ HuggingFace  → pip install transformers torch")
    print(f"  Audio I/O: soundfile={'✓' if _HAS_SF else '✗'}  scipy={'✓' if _HAS_SCIPY else '✗'}")
    print("=" * 65)

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton {
            padding: 5px 10px;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            background: #f9fafb;
            font-size: 12px;
        }
        QPushButton:hover { background: #e5e7eb; }
        QPushButton:pressed { background: #d1d5db; }
        QPushButton:disabled { color: #9ca3af; background: #f3f4f6; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            padding: 3px 6px; font-size: 12px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())