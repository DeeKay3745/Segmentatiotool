#!/usr/bin/env python3
"""
Indic Speech Annotation & Segmentation Tool v3
================================================
Purpose-built for Indian languages with proper script output.

Engines (in order of Indic accuracy):
  1. AI4Bharat IndicConformer-600M  — best for Gu/Hi/Mr (correct script!)
  2. faster-whisper                 — fast but may output wrong script
  3. HuggingFace Whisper            — fallback

Install:
    pip install transformers torchaudio torch soundfile scipy \
                numpy pandas matplotlib PyQt5 sounddevice openpyxl

For faster-whisper (optional):
    pip install faster-whisper
"""

import os, sys, csv, re, time, traceback
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

# ── Imports with availability flags ──────────────────────────────────────
try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import torch
    import torchaudio
    from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor
    _HF = True
except ImportError:
    _HF = False

try:
    from faster_whisper import WhisperModel as _FWModel
    _FW = True
except ImportError:
    _FW = False

if not _HF and not _FW:
    print("ERROR: pip install transformers torch torchaudio  (or)  pip install faster-whisper")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
SR16 = 16000
CPU_THR = max(1, min(os.cpu_count() or 4, 8))

# ── Models ───────────────────────────────────────────────────────────────
# (display, engine_key, model_id_or_size, notes)
MODELS = []
if _HF:
    MODELS += [
        ("★ IndicConformer-600M (BEST for Indic)",
         "indicconformer", "ai4bharat/indic-conformer-600m-multilingual",
         "AI4Bharat – all 22 Indian languages, correct script output"),
    ]
if _FW:
    MODELS += [
        ("faster-whisper tiny  (fastest)",    "fw", "tiny",     "May output wrong script for Indic"),
        ("faster-whisper base",               "fw", "base",     "May output wrong script for Indic"),
        ("faster-whisper small",              "fw", "small",    "May output wrong script for Indic"),
        ("faster-whisper medium",             "fw", "medium",   "Better Indic but still may mix scripts"),
        ("faster-whisper large-v3",           "fw", "large-v3", "Best Whisper, but not Indic-native"),
    ]
if _HF:
    MODELS += [
        ("HF Whisper small",    "hf_whisper", "openai/whisper-small",    "May output wrong script"),
        ("HF Whisper medium",   "hf_whisper", "openai/whisper-medium",   "May output wrong script"),
        ("HF Whisper large-v3", "hf_whisper", "openai/whisper-large-v3", "Best generic Whisper"),
    ]

LANGUAGES = [
    ("Gujarati",  "gu", "gujarati"),
    ("Hindi",     "hi", "hindi"),
    ("Marathi",   "mr", "marathi"),
    ("English",   "en", "english"),
    ("Bengali",   "bn", "bengali"),
    ("Tamil",     "ta", "tamil"),
    ("Telugu",    "te", "telugu"),
    ("Kannada",   "kn", "kannada"),
    ("Malayalam", "ml", "malayalam"),
    ("Punjabi",   "pa", "punjabi"),
    ("Odia",      "or", "odia"),
    ("Assamese",  "as", "assamese"),
]

PRED_MODES = [
    ("Full Transcript", "full"),
    ("First Word",      "word"),
    ("First Character",  "char"),
]

# ── Unicode ──────────────────────────────────────────────────────────────
_INDIC = (
    r"\u0900-\u097F"   # Devanagari
    r"\uA8E0-\uA8FF"   # Devanagari Extended
    r"\u0A80-\u0AFF"   # Gujarati
    r"\u0980-\u09FF"   # Bengali
    r"\u0B00-\u0B7F"   # Odia
    r"\u0B80-\u0BFF"   # Tamil
    r"\u0C00-\u0C7F"   # Telugu
    r"\u0C80-\u0CFF"   # Kannada
    r"\u0D00-\u0D7F"   # Malayalam
    r"\u0A00-\u0A7F"   # Gurmukhi (Punjabi)
)
_RE_CLEAN  = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM   = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA  = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def clean_text(text, mode="full"):
    if not text: return ""
    text = _RE_CLEAN.sub("", str(text).strip())
    text = _RE_SPACES.sub(" ", text).strip()
    if not text: return ""
    if mode == "word":
        return text.split()[0] if text.split() else ""
    if mode == "char":
        m = _RE_ALPHA.search(text)
        return m.group(0) if m else ""
    return text


def norm_text(s):
    return _RE_NORM.sub("", str(s).strip().lower())


def edit_distance(a, b):
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
    p, g = norm_text(pred), norm_text(gt)
    if not g and not p: return 1.0
    if not g: return 0.0
    return max(0.0, 1.0 - edit_distance(p, g) / len(g))


def load_audio(path):
    if _SF:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return y.astype(np.float32), sr
        except Exception:
            pass
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), sr


def resample_audio(y, orig_sr, target_sr):
    if orig_sr == target_sr: return y
    if _SCIPY:
        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(y, target_sr // g, orig_sr // g).astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def parse_time(x):
    if pd.isna(x): return None
    x = str(x).strip()
    if not x: return None
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"):
        try:
            dt = datetime.strptime(x, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            continue
    try: return float(x)
    except ValueError: return None


@lru_cache(maxsize=12)
def get_font(lang_code):
    mapping = {
        "hi": (["NotoSansDevanagari-Regular.ttf","Lohit-Devanagari.ttf"],
               ["Noto Sans Devanagari","Lohit Devanagari","Mangal","Arial Unicode MS"]),
        "gu": (["NotoSansGujarati-Regular.ttf","Lohit-Gujarati.ttf"],
               ["Noto Sans Gujarati","Lohit Gujarati","Shruti","Arial Unicode MS"]),
        "bn": (["NotoSansBengali-Regular.ttf"], ["Noto Sans Bengali","Arial Unicode MS"]),
        "ta": (["NotoSansTamil-Regular.ttf"], ["Noto Sans Tamil","Arial Unicode MS"]),
        "te": (["NotoSansTelugu-Regular.ttf"], ["Noto Sans Telugu","Arial Unicode MS"]),
        "kn": (["NotoSansKannada-Regular.ttf"], ["Noto Sans Kannada","Arial Unicode MS"]),
        "ml": (["NotoSansMalayalam-Regular.ttf"], ["Noto Sans Malayalam","Arial Unicode MS"]),
        "pa": (["NotoSansGurmukhi-Regular.ttf"], ["Noto Sans Gurmukhi","Arial Unicode MS"]),
        "en": ([], ["Arial","Helvetica","DejaVu Sans"]),
    }
    lc = "hi" if lang_code == "mr" else lang_code
    filenames, families = mapping.get(lc, mapping["en"])
    for d in ["/usr/share/fonts", "/Library/Fonts",
              os.path.expanduser("~/Library/Fonts"),
              os.path.expanduser("~/.local/share/fonts")]:
        if not os.path.isdir(d): continue
        for root, _, files in os.walk(d):
            for fn in filenames:
                if fn in files:
                    return fm.FontProperties(fname=os.path.join(root, fn))
    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in families:
        if fam in installed: return fm.FontProperties(family=fam)
    return None


# ── VAD ──────────────────────────────────────────────────────────────────
def vad_segment(y, sr, silence_ms=300, thresh_db=-40, min_ms=200, max_ms=10000):
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    n_frames = 1 + (len(y) - frame_len) // hop
    if n_frames <= 0: return []
    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop
    np.clip(idx, 0, len(y) - 1, out=idx)
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = rms_db > thresh_db
    pad = np.concatenate([[False], voiced, [False]])
    diff = np.diff(pad.astype(np.int8))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    sil_f = max(1, int(silence_ms / 1000 * sr / hop))
    min_f = max(1, int(min_ms / 1000 * sr / hop))
    max_f = max(1, int(max_ms / 1000 * sr / hop))
    merged = []
    i = 0
    while i < len(starts):
        s, e = starts[i], ends[i]
        while i + 1 < len(starts) and (starts[i+1] - e) < sil_f:
            i += 1; e = ends[i]
        merged.append((s, e)); i += 1
    result = []
    for s, e in merged:
        if (e - s) < min_f: continue
        for cs in range(s, e, max_f):
            ce = min(cs + max_f, e)
            if (ce - cs) >= min_f:
                result.append({"start": round(cs * hop / sr, 4),
                               "end": round(min(ce * hop / sr, len(y) / sr), 4)})
    return result


# ══════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION ENGINE
# ══════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self):
        self.backend = None       # the actual model object
        self.engine_key = None    # "indicconformer", "fw", "hf_whisper"
        self.processor = None     # HF whisper only
        self.device = None
        self._cache_key = None

    @property
    def is_loaded(self):
        return self.backend is not None

    def load(self, engine_key, model_id):
        cache_key = (engine_key, model_id)
        if self._cache_key == cache_key and self.backend is not None:
            return True, f"Already loaded: {model_id}", 0.0

        t0 = time.time()
        self.backend = self.processor = None
        self.engine_key = engine_key

        try:
            if engine_key == "indicconformer":
                self._load_indicconformer(model_id)
            elif engine_key == "fw":
                self._load_faster_whisper(model_id)
            elif engine_key == "hf_whisper":
                self._load_hf_whisper(model_id)
            else:
                return False, f"Unknown engine: {engine_key}", 0.0

            self._cache_key = cache_key
            elapsed = time.time() - t0
            return True, f"Loaded {model_id} in {elapsed:.1f}s", elapsed
        except Exception as e:
            traceback.print_exc()
            self.backend = None
            return False, str(e), time.time() - t0

    # ── IndicConformer ────────────────────────────────────────────────────
    def _load_indicconformer(self, model_id):
        log(f"Loading IndicConformer: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        # Move to device if the model supports .to()
        if hasattr(self.backend, 'to'):
            try:
                self.backend = self.backend.to(self.device)
            except Exception:
                self.device = "cpu"
        log(f"IndicConformer loaded on {self.device}")

    # ── faster-whisper ────────────────────────────────────────────────────
    def _load_faster_whisper(self, size):
        has_cuda = False
        try:
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass
        device = "cuda" if has_cuda else "cpu"
        compute = "float16" if has_cuda else "int8"
        log(f"Loading faster-whisper: {size} on {device} ({compute})")
        self.backend = _FWModel(size, device=device, compute_type=compute,
                                cpu_threads=CPU_THR)
        log(f"faster-whisper loaded: {size}")

    # ── HF Whisper ────────────────────────────────────────────────────────
    def _load_hf_whisper(self, model_id):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log(f"Loading HF Whisper: {model_id} on {device}")
        self.backend = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype,
            low_cpu_mem_usage=True, use_safetensors=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        log(f"HF Whisper loaded: {model_id}")

    # ── Transcribe ────────────────────────────────────────────────────────
    def transcribe(self, audio_16k, lang_iso, lang_name, mode="full"):
        if self.backend is None or audio_16k is None or len(audio_16k) == 0:
            return ""

        # Pad short segments
        min_samp = int(0.1 * SR16)
        if len(audio_16k) < min_samp:
            audio_16k = np.pad(audio_16k, (0, min_samp - len(audio_16k)))
        audio_16k = audio_16k.astype(np.float32)

        if self.engine_key == "indicconformer":
            return self._tx_indicconformer(audio_16k, lang_iso, mode)
        elif self.engine_key == "fw":
            return self._tx_fw(audio_16k, lang_iso, mode)
        elif self.engine_key == "hf_whisper":
            return self._tx_hf(audio_16k, lang_name, mode)
        return ""

    def _tx_indicconformer(self, audio, lang_iso, mode):
        """IndicConformer API: model(wav_tensor, lang_code, decode_type)"""
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, N]
        if self.device and self.device != "cpu":
            wav = wav.to(self.device)

        # Use CTC decoding (faster) — alternatives: "rnnt"
        with torch.inference_mode():
            text = self.backend(wav, lang_iso, "ctc")

        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return clean_text(str(text), mode)

    def _tx_fw(self, audio, lang_iso, mode):
        segments_gen, info = self.backend.transcribe(
            audio, language=lang_iso, task="transcribe",
            beam_size=1, best_of=1, vad_filter=False,
            without_timestamps=True, condition_on_previous_text=False,
        )
        segments_list = list(segments_gen)  # MUST consume generator
        text = " ".join(s.text.strip() for s in segments_list if s.text.strip())
        return clean_text(text, mode)

    def _tx_hf(self, audio, lang_name, mode):
        inputs = self.processor(audio, sampling_rate=SR16,
                                return_tensors="pt", return_attention_mask=True)
        feat = inputs.input_features.to(self.device)
        mask = getattr(inputs, "attention_mask", None)
        if mask is not None: mask = mask.to(self.device)
        gk = {"input_features": feat, "task": "transcribe",
              "language": lang_name, "return_timestamps": False}
        if mask is not None: gk["attention_mask"] = mask
        with torch.inference_mode():
            ids = self.backend.generate(**gk)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return clean_text(text, mode)


ENGINE = Engine()


# ══════════════════════════════════════════════════════════════════════════
#  THREADS
# ══════════════════════════════════════════════════════════════════════════
class AudioLoadThread(QThread):
    done = pyqtSignal(np.ndarray, int, np.ndarray, str)
    error = pyqtSignal(str)
    def __init__(self, path): super().__init__(); self.path = path
    def run(self):
        try:
            y, sr = load_audio(self.path)
            y16 = resample_audio(y, sr, SR16)
            self.done.emit(y, sr, y16, self.path)
        except Exception as e:
            traceback.print_exc(); self.error.emit(str(e))


class ModelLoadThread(QThread):
    done = pyqtSignal(bool, str, float)
    error = pyqtSignal(str)
    def __init__(self, engine_key, model_id):
        super().__init__()
        self.engine_key, self.model_id = engine_key, model_id
    def run(self):
        try:
            ok, msg, elapsed = ENGINE.load(self.engine_key, self.model_id)
            self.done.emit(ok, msg, elapsed)
        except Exception as e:
            traceback.print_exc(); self.error.emit(str(e))


class PredictionThread(QThread):
    seg_result = pyqtSignal(int, str, float)
    progress = pyqtSignal(int, int)
    done = pyqtSignal(list, float, float)
    error = pyqtSignal(str)

    def __init__(self, y16, segments, lang_iso, lang_name, mode):
        super().__init__()
        self.y16, self.segments = y16, segments
        self.lang_iso, self.lang_name, self.mode = lang_iso, lang_name, mode
        self._stop = False

    def request_stop(self): self._stop = True

    def run(self):
        try:
            t0 = time.time()
            total = len(self.segments)
            results = []
            log(f"Prediction: {total} segs, lang={self.lang_iso}, engine={ENGINE.engine_key}")

            for i, seg in enumerate(self.segments):
                if self._stop:
                    log("Stopped by user"); break

                s = max(0, int(seg["start"] * SR16))
                e = min(len(self.y16), int(seg["end"] * SR16))
                chunk = self.y16[s:e]

                pred = ENGINE.transcribe(chunk, self.lang_iso, self.lang_name, self.mode)
                acc = accuracy(pred, seg["label"])
                results.append({"i": i, "gt": seg["label"], "pred": pred,
                                "start": seg["start"], "end": seg["end"], "acc": acc})

                self.seg_result.emit(i, pred, acc)
                self.progress.emit(i + 1, total)

                if (i + 1) % 5 == 0 or (i + 1) == total:
                    log(f"  [{i+1}/{total}] GT='{seg['label']}' → PRED='{pred}' acc={acc:.3f}")

            elapsed = time.time() - t0
            mean_acc = np.mean([r["acc"] for r in results]) if results else 0.0
            self.done.emit(results, float(mean_acc), elapsed)
        except Exception as e:
            traceback.print_exc(); self.error.emit(str(e))


class VADThread(QThread):
    done = pyqtSignal(list)
    error = pyqtSignal(str)
    def __init__(self, y, sr, p): super().__init__(); self.y, self.sr, self.p = y, sr, p
    def run(self):
        try: self.done.emit(vad_segment(self.y, self.sr, **self.p))
        except Exception as e: traceback.print_exc(); self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indic Speech Annotation Tool v3")
        self.resize(1700, 1000)

        self.audio_path = None
        self.y = self.y16 = self.wave_t = self.wave_y = None
        self.sr = None
        self.segments, self.results = [], []
        self.selected_idx = None
        self._threads = {}
        self._pred_t0 = None
        self._timer = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        L = QVBoxLayout(root); L.setSpacing(4); L.setContentsMargins(8,8,8,8)

        # ── Row 1 ─────────────────────────────────────────────────────────
        r1 = QHBoxLayout()
        self.cb_lang = QComboBox()
        for d, iso, name in LANGUAGES:
            self.cb_lang.addItem(d, (iso, name))

        self.cb_model = QComboBox()
        for d, ek, mid, note in MODELS:
            self.cb_model.addItem(d, (ek, mid))
        self.cb_model.setToolTip("★ IndicConformer outputs correct Gujarati/Hindi/Marathi script")

        self.cb_mode = QComboBox()
        for d, k in PRED_MODES:
            self.cb_mode.addItem(d, k)

        self.btn_audio = QPushButton("Load Audio")
        self.btn_audio.clicked.connect(self.on_load_audio)
        self.btn_gt = QPushButton("Load Ground Truth")
        self.btn_gt.clicked.connect(self.on_load_gt)
        self.btn_model = QPushButton("⬇ Load Model")
        self.btn_model.clicked.connect(self.on_load_model)
        self.btn_model.setStyleSheet(
            "QPushButton{font-weight:bold;color:white;background:#2563eb;"
            "border:1px solid #1d4ed8;padding:5px 14px;border-radius:4px}"
            "QPushButton:hover{background:#1d4ed8}"
            "QPushButton:disabled{color:#9ca3af;background:#f3f4f6;border:1px solid #d1d5db;font-weight:normal}")

        for w in [QLabel("Language:"), self.cb_lang,
                  QLabel("Model:"), self.cb_model,
                  QLabel("Output:"), self.cb_mode,
                  self.btn_audio, self.btn_gt, self.btn_model]:
            r1.addWidget(w)
        r1.addStretch()

        # ── Row 2 ─────────────────────────────────────────────────────────
        r2 = QHBoxLayout()
        actions = [
            ("▶ Run Prediction", self.on_run, "font-weight:bold;color:#16a34a"),
            ("⏹ Stop", self.on_stop, ""),
            ("Auto-Segment (VAD)", self.on_vad, ""),
            ("♫ Play All", lambda: self._play(self.y, self.sr), ""),
            ("♫ Play Seg", self.on_play_seg, ""),
            ("⏸ Stop Audio", sd.stop, ""),
            ("🔍+", lambda: self._zoom(0.5), ""),
            ("🔍−", lambda: self._zoom(2.0), ""),
            ("Fit Segments", self._fit_segments, "font-weight:bold;color:#2563eb"),
            ("Reset View", self._reset_view, ""),
            ("Export CSV", self.on_export, ""),
        ]
        self._btns = {}
        for name, fn, style in actions:
            b = QPushButton(name)
            b.clicked.connect(fn)
            if style: b.setStyleSheet(f"QPushButton{{{style}}}")
            r2.addWidget(b)
            self._btns[name] = b
        self._btns["⏹ Stop"].setEnabled(False)
        r2.addStretch()

        # ── Status ────────────────────────────────────────────────────────
        sr = QHBoxLayout()
        self.lbl_status = QLabel("Step 1: Select model & click 'Load Model'")
        self.lbl_status.setStyleSheet("padding:4px;font-size:13px")
        self.lbl_speed = QLabel("")
        self.lbl_speed.setStyleSheet("color:#6b7280;font-size:12px")
        sr.addWidget(self.lbl_status, 1); sr.addWidget(self.lbl_speed)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100); self.pbar.setValue(0); self.pbar.setFixedHeight(18)
        self.pbar.setStyleSheet(
            "QProgressBar{border:1px solid #ccc;border-radius:4px;background:#f3f4f6;"
            "text-align:center;font-size:11px}"
            "QProgressBar::chunk{background:#3b82f6;border-radius:4px}")

        # ── Canvas (dual panel: main waveform + overview minimap) ─────────
        self.fig = plt.figure(figsize=(14, 6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        self.ax = self.fig.add_subplot(gs[0])      # main waveform
        self.ax_mini = self.fig.add_subplot(gs[1])  # overview minimap
        self.fig.set_tight_layout(False)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._view_box = None  # rectangle on minimap showing current view

        # ── Right panel ───────────────────────────────────────────────────
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_seg_clicked)
        self.seg_list.setStyleSheet(
            "QListWidget{background:#fff;border:1px solid #d0d0d0;"
            "font-family:Consolas,Courier New,monospace;font-size:12px}"
            "QListWidget::item{padding:3px 6px}"
            "QListWidget::item:selected{background:#dbeafe}")

        vad_box = QGroupBox("VAD Parameters")
        vl = QVBoxLayout(vad_box)
        self.sp_sil = QSpinBox(); self.sp_sil.setRange(50,2000); self.sp_sil.setValue(300); self.sp_sil.setSuffix(" ms")
        self.sp_db = QDoubleSpinBox(); self.sp_db.setRange(-80,0); self.sp_db.setValue(-40); self.sp_db.setSuffix(" dB")
        self.sp_min = QSpinBox(); self.sp_min.setRange(50,5000); self.sp_min.setValue(200); self.sp_min.setSuffix(" ms")
        self.sp_max = QSpinBox(); self.sp_max.setRange(1000,30000); self.sp_max.setValue(10000); self.sp_max.setSuffix(" ms")
        for lbl, w in [("Silence gap:", self.sp_sil), ("Threshold:", self.sp_db),
                       ("Min seg:", self.sp_min), ("Max seg:", self.sp_max)]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(w); vl.addLayout(h)
        vl.addStretch()

        # Model info tab
        info_box = QGroupBox("Model Guide")
        il = QVBoxLayout(info_box)
        info = QLabel(
            "<b>★ Recommended for Gujarati/Hindi/Marathi:</b><br>"
            "<span style='color:#16a34a'>AI4Bharat IndicConformer-600M</span><br>"
            "• Trained on all 22 Indian languages<br>"
            "• Outputs <b>correct script</b> (ગુજરાતી, हिन्दी, मराठी)<br>"
            "• Uses CTC decoding (fast)<br><br>"
            "<b>Why Whisper gives wrong script:</b><br>"
            "• Whisper small/medium confuse Gujarati → Hindi<br>"
            "• Even large-v3 may mix Devanagari & Gujarati<br>"
            "• Whisper is not fine-tuned for Indian languages<br><br>"
            "<b>Available Indic ASR Models:</b><br>"
            "1. <b>IndicConformer-600M</b> — AI4Bharat (best)<br>"
            "2. <b>IndicWhisper</b> — fine-tuned Whisper<br>"
            "3. <b>Meta MMS-1B</b> — 1000+ languages<br>"
            "4. <b>Whisper large-v3</b> — generic multilingual<br>"
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px;color:#374151")
        il.addWidget(info); il.addStretch()

        tabs = QTabWidget()
        seg_tab = QWidget(); sl = QVBoxLayout(seg_tab)
        legend = QLabel(
            '<span style="color:green;font-weight:bold">■ GT</span>  '
            '<span style="color:blue;font-weight:bold">■ PRED</span>  | '
            '<span style="color:#16a34a">✓≥95%</span>  '
            '<span style="color:#ea580c">~50-95%</span>  '
            '<span style="color:#dc2626">✗&lt;50%</span>')
        legend.setTextFormat(Qt.RichText)
        sl.addWidget(legend); sl.addWidget(self.seg_list)
        tabs.addTab(seg_tab, "Segments")
        tabs.addTab(vad_box, "VAD")
        tabs.addTab(info_box, "Model Guide")

        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(self.toolbar); ll.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(tabs)
        splitter.setSizes([1250, 400])

        L.addLayout(r1); L.addLayout(r2)
        L.addLayout(sr); L.addWidget(self.pbar); L.addWidget(splitter)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _set_busy(self, busy, msg=""):
        en = not busy
        for n, b in self._btns.items():
            b.setEnabled(busy if n == "⏹ Stop" else en)
        for w in [self.btn_audio, self.btn_gt, self.btn_model,
                  self.cb_lang, self.cb_model, self.cb_mode]:
            w.setEnabled(en)
        if msg: self.lbl_status.setText(msg)

    def _build_wave(self):
        if self.y is None:
            self.wave_t = self.wave_y = self.wave_env = None
            return
        n = len(self.y)
        # Downsampled waveform for fast plotting
        max_pts = 30000
        step = max(1, n // max_pts)
        self.wave_y = self.y[::step]
        self.wave_t = np.arange(len(self.wave_y)) * (step / self.sr)

        # RMS envelope (for overview when zoomed out)
        env_hop = max(1, n // 2000)  # ~2000 points for envelope
        n_env = n // env_hop
        if n_env > 0:
            trimmed = self.y[:n_env * env_hop].reshape(n_env, env_hop)
            self.wave_env_pos = np.sqrt(np.mean(trimmed ** 2, axis=1))
            self.wave_env_neg = -self.wave_env_pos
            self.wave_env_t = (np.arange(n_env) + 0.5) * env_hop / self.sr
        else:
            self.wave_env_pos = self.wave_env_neg = self.wave_env_t = None

        self.audio_duration = n / self.sr
        log(f"Wave cache: {len(self.wave_y)} pts, envelope: {n_env} pts, duration: {self.audio_duration:.1f}s")

    def _play(self, y, sr):
        if y is not None and sr: sd.stop(); sd.play(y, sr)

    # ── Load Audio ────────────────────────────────────────────────────────
    def on_load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.webm *.aac)")
        if not path: return
        self.lbl_status.setText(f"Loading {os.path.basename(path)}...")
        self.btn_audio.setEnabled(False)
        t = AudioLoadThread(path)
        t.done.connect(self._audio_ok); t.error.connect(self._audio_err)
        self._threads["audio"] = t; t.start()

    def _audio_ok(self, y, sr, y16, path):
        self.y, self.sr, self.y16, self.audio_path = y, sr, y16, path
        self.results = []; self.selected_idx = None; self.pbar.setValue(0)
        self._build_wave()
        self.lbl_status.setText(f"Audio: {os.path.basename(path)} ({len(y)/sr:.1f}s, sr={sr})")
        self.btn_audio.setEnabled(True)
        self._refresh_list(); self._redraw()

    def _audio_err(self, msg):
        self.btn_audio.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    # ── Load GT ───────────────────────────────────────────────────────────
    def on_load_gt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open GT", "",
            "Spreadsheets (*.xlsx *.csv *.tsv)")
        if not path: return
        try:
            if path.endswith(".csv"): df = pd.read_csv(path, header=None)
            elif path.endswith(".tsv"): df = pd.read_csv(path, header=None, sep="\t")
            else: df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
            segs, skip = [], 0
            for _, row in df.iterrows():
                lab = str(row.iloc[0]).strip()
                s, e = parse_time(row.iloc[1]), parse_time(row.iloc[2])
                if not lab or s is None or e is None or e <= s: skip += 1; continue
                segs.append({"label": lab, "start": s, "end": e})
            self.segments = segs; self.results = []; self.selected_idx = None
            self.lbl_status.setText(f"GT: {len(segs)} segments" +
                                    (f" ({skip} skipped)" if skip else ""))
            self._refresh_list(); self._redraw()
        except Exception as e:
            traceback.print_exc(); QMessageBox.critical(self, "Error", str(e))

    # ── Load Model ────────────────────────────────────────────────────────
    def on_load_model(self):
        t = self._threads.get("model")
        if t and t.isRunning():
            QMessageBox.information(self, "Loading",
                "Model is still downloading. Please wait.\n"
                "Check the status bar for progress.")
            return

        ek, mid = self.cb_model.currentData()
        self._set_busy(True, f"Loading {mid}...")
        self.btn_model.setText("Loading...")
        self.btn_model.setEnabled(False)

        self._load_t0 = time.time()
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: self.lbl_status.setText(
            f"Loading model... {int(time.time()-self._load_t0)}s elapsed"))
        self._timer.start(1000)

        t = ModelLoadThread(ek, mid)
        t.done.connect(self._model_ok); t.error.connect(self._model_err)
        self._threads["model"] = t; t.start()

    def _stop_timer(self):
        if self._timer: self._timer.stop(); self._timer = None

    def _model_ok(self, ok, msg, elapsed):
        self._stop_timer()
        self._set_busy(False)
        self.btn_model.setEnabled(True)
        if ok:
            self.btn_model.setText("✓ Model Loaded")
            self.btn_model.setStyleSheet(
                "QPushButton{font-weight:bold;color:white;background:#16a34a;"
                "border:1px solid #15803d;padding:5px 14px;border-radius:4px}"
                "QPushButton:hover{background:#15803d}")
            self.lbl_status.setText(f"✓ {msg}")
            self.lbl_speed.setText(f"Engine: {ENGINE.engine_key}")
        else:
            self.btn_model.setText("⬇ Load Model")
            self.lbl_status.setText(f"✗ Failed: {msg}")
            QMessageBox.critical(self, "Error",
                f"Failed to load model:\n\n{msg}\n\n"
                "• Check internet (first-time download)\n"
                "• Try a smaller model\n"
                "• Check terminal for details")

    def _model_err(self, msg):
        self._stop_timer(); self._set_busy(False)
        self.btn_model.setText("⬇ Load Model"); self.btn_model.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    # ── VAD ───────────────────────────────────────────────────────────────
    def on_vad(self):
        if self.y is None: QMessageBox.warning(self, "", "Load audio first."); return
        p = {"silence_ms": self.sp_sil.value(), "thresh_db": self.sp_db.value(),
             "min_ms": self.sp_min.value(), "max_ms": self.sp_max.value()}
        self._set_busy(True, "Running VAD...")
        t = VADThread(self.y, self.sr, p)
        t.done.connect(self._vad_ok)
        t.error.connect(lambda m: (self._set_busy(False), QMessageBox.critical(self,"Error",m)))
        self._threads["vad"] = t; t.start()

    def _vad_ok(self, segs):
        self.segments = [{"label": f"seg_{i+1:04d}", "start": s["start"], "end": s["end"]}
                         for i, s in enumerate(segs)]
        self.results = []; self.selected_idx = None
        self._set_busy(False, f"VAD: {len(segs)} segments")
        self._refresh_list(); self._redraw()

    # ── Run Prediction ────────────────────────────────────────────────────
    def on_run(self):
        if self.y16 is None: QMessageBox.warning(self, "", "Load audio."); return
        if not self.segments: QMessageBox.warning(self, "", "Load GT or run VAD."); return
        if not ENGINE.is_loaded: QMessageBox.warning(self, "", "Load model first."); return
        t = self._threads.get("pred")
        if t and t.isRunning(): return

        iso, name = self.cb_lang.currentData()
        mode = self.cb_mode.currentData()
        self.results = []; self.pbar.setValue(0); self._pred_t0 = time.time()
        self._set_busy(True, "Running prediction...")
        self._refresh_list()

        t = PredictionThread(self.y16, self.segments, iso, name, mode)
        t.seg_result.connect(self._on_seg_result)
        t.progress.connect(self._on_pred_prog)
        t.done.connect(self._on_pred_done)
        t.error.connect(self._on_pred_err)
        self._threads["pred"] = t; t.start()

    def _on_seg_result(self, idx, pred, acc):
        if idx < self.seg_list.count():
            seg = self.segments[idx]
            item = self.seg_list.item(idx)
            item.setText(f"{idx+1:03d} │ GT: {seg['label']}  →  PRED: {pred or '—'}"
                        f"  │ {seg['start']:.2f}-{seg['end']:.2f}s  acc={acc:.2f}")
            if acc >= 0.95: item.setForeground(Qt.darkGreen)
            elif acc >= 0.5: item.setForeground(Qt.darkYellow)
            else: item.setForeground(Qt.red)

    def _on_pred_prog(self, done, total):
        pct = int(done / total * 100) if total else 0
        self.pbar.setValue(pct)
        elapsed = time.time() - self._pred_t0
        spd = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / spd if spd > 0 else 0
        self.lbl_status.setText(f"Predicting {done}/{total} ({spd:.1f} seg/s, ETA {eta:.0f}s)")

    def _on_pred_done(self, results, mean_acc, elapsed):
        self.results = results
        self.pbar.setValue(100)
        n = len(results)
        spd = n / elapsed if elapsed > 0 else 0
        self._set_busy(False,
            f"✓ Done: {n} segs in {elapsed:.1f}s | Accuracy: {mean_acc:.3f} | {spd:.1f} seg/s")
        self.lbl_speed.setText(f"{spd:.1f} seg/s")
        self._refresh_list(); self._redraw()

    def _on_pred_err(self, msg):
        self.pbar.setValue(0); self._set_busy(False, f"✗ {msg}")
        QMessageBox.critical(self, "Error", msg)

    def on_stop(self):
        t = self._threads.get("pred")
        if t and t.isRunning(): t.request_stop()

    # ── Plot ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_time(seconds):
        """Format seconds as mm:ss or hh:mm:ss."""
        s = int(seconds)
        if s >= 3600:
            return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
        return f"{s//60}:{s%60:02d}"

    def _fit_segments(self):
        """Auto-zoom to the region containing all segments."""
        if not self.segments or self.y is None:
            self._reset_view(); return
        s_min = min(seg["start"] for seg in self.segments)
        s_max = max(seg["end"] for seg in self.segments)
        pad = max(1.0, (s_max - s_min) * 0.05)
        self._redraw(xlim=(max(0, s_min - pad), s_max + pad))

    def _redraw(self, xlim=None):
        """Redraw waveform + minimap. Pass xlim=(x0,x1) for zoom."""
        self.ax.clear()
        self.ax_mini.clear()

        if self.wave_t is None:
            self.canvas.draw_idle(); return

        dur = getattr(self, 'audio_duration', len(self.y) / self.sr) if self.y is not None else 1.0
        iso, _ = self.cb_lang.currentData()
        fp = get_font(iso)

        # ── Determine view range ─────────────────────────────────────────
        if xlim is None:
            # Default: fit to segments if available, else full audio
            if self.segments:
                s_min = min(seg["start"] for seg in self.segments)
                s_max = max(seg["end"] for seg in self.segments)
                pad = max(1.0, (s_max - s_min) * 0.05)
                xlim = (max(0, s_min - pad), s_max + pad)
            else:
                xlim = (0, dur)

        x0, x1 = xlim
        view_span = x1 - x0

        # ── Main waveform ────────────────────────────────────────────────
        # Use envelope when zoomed out (>30s visible), raw waveform when zoomed in
        if view_span > 30 and self.wave_env_t is not None:
            # Envelope (filled area)
            mask = (self.wave_env_t >= x0) & (self.wave_env_t <= x1)
            t_vis = self.wave_env_t[mask]
            pos_vis = self.wave_env_pos[mask]
            neg_vis = self.wave_env_neg[mask]
            if len(t_vis) > 0:
                self.ax.fill_between(t_vis, neg_vis, pos_vis,
                                     color="#3b82f6", alpha=0.4, linewidth=0)
                self.ax.plot(t_vis, pos_vis, lw=0.5, color="#1d4ed8", alpha=0.7)
                self.ax.plot(t_vis, neg_vis, lw=0.5, color="#1d4ed8", alpha=0.7)
        else:
            # Raw waveform (zoomed in)
            mask = (self.wave_t >= x0 - 1) & (self.wave_t <= x1 + 1)
            t_vis = self.wave_t[mask]
            y_vis = self.wave_y[mask]
            if len(t_vis) > 0:
                self.ax.plot(t_vis, y_vis, lw=0.6, color="#1e293b")

        ymax = 1.05  # fixed amplitude scale

        # ── Draw segments ────────────────────────────────────────────────
        for i, seg in enumerate(self.segments):
            s, e = seg["start"], seg["end"]
            if e < x0 or s > x1: continue

            # Color based on accuracy
            if self.results and i < len(self.results):
                a = self.results[i]["acc"]
                if a >= 0.95:
                    clr, bar_clr = "#dcfce7", "#16a34a"
                elif a >= 0.5:
                    clr, bar_clr = "#fef3c7", "#d97706"
                else:
                    clr, bar_clr = "#fecaca", "#dc2626"
            else:
                clr, bar_clr = "#e0f2fe", "#0284c7"

            # Segment background
            self.ax.axvspan(s, e, alpha=0.25, color=clr, zorder=1)
            # Bottom bar (always visible even when zoomed out)
            self.ax.plot([s, e], [-ymax*0.92, -ymax*0.92], lw=4,
                        color=bar_clr, solid_capstyle="butt", zorder=3)
            # Tick marks at segment boundaries
            self.ax.axvline(s, lw=0.5, color=bar_clr, alpha=0.4, zorder=2)

            # Selected segment highlight
            if self.selected_idx is not None and i == self.selected_idx:
                self.ax.axvspan(s, e, alpha=0.15, color="#facc15", zorder=2)
                mid = (s + e) / 2
                fkw = {"fontproperties": fp} if fp else {}
                bbox = dict(facecolor="white", alpha=0.9, edgecolor="#d1d5db",
                           boxstyle="round,pad=0.3")
                self.ax.text(mid, ymax*0.88, f"GT: {seg['label']}",
                    ha="center", va="bottom", fontsize=10, color="#15803d",
                    fontweight="bold", bbox=bbox, zorder=5, **fkw)
                if self.results and i < len(self.results):
                    r = self.results[i]
                    self.ax.text(mid, ymax*0.65,
                        f"PRED: {r['pred'] or '∅'}  (acc={r['acc']:.2f})",
                        ha="center", va="bottom", fontsize=10, color="#1d4ed8",
                        fontweight="bold", bbox=bbox, zorder=5, **fkw)

            # Show segment labels when zoomed in enough
            elif view_span < 15 and (e - s) > view_span * 0.02:
                mid = (s + e) / 2
                fkw = {"fontproperties": fp} if fp else {}
                label = seg["label"]
                if len(label) > 8:
                    label = label[:7] + "…"
                self.ax.text(mid, ymax*0.85, label, ha="center", va="bottom",
                            fontsize=8, color="#374151", alpha=0.8, zorder=4, **fkw)

        # ── Axis formatting ──────────────────────────────────────────────
        title = "Waveform"
        if self.results:
            mean_acc = np.mean([r["acc"] for r in self.results])
            n_good = sum(1 for r in self.results if r["acc"] >= 0.95)
            title += f"  │  Accuracy: {mean_acc:.3f}  │  {n_good}/{len(self.results)} correct (≥95%)"

        self.ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        self.ax.set_ylabel("Amplitude", fontsize=9)
        self.ax.set_ylim(-ymax, ymax)
        self.ax.set_xlim(x0, x1)
        self.ax.grid(axis="x", alpha=0.15, linestyle="--")

        # Time axis formatting (mm:ss for long audio)
        if view_span > 120:
            from matplotlib.ticker import FuncFormatter
            self.ax.xaxis.set_major_formatter(FuncFormatter(
                lambda val, pos: self._fmt_time(val)))
            self.ax.set_xlabel("Time (mm:ss)", fontsize=9)
        else:
            self.ax.set_xlabel("Time (seconds)", fontsize=9)

        # ── Minimap (overview) ───────────────────────────────────────────
        if self.wave_env_t is not None and len(self.wave_env_t) > 0:
            self.ax_mini.fill_between(self.wave_env_t,
                                      self.wave_env_neg, self.wave_env_pos,
                                      color="#94a3b8", alpha=0.5, linewidth=0)
        else:
            self.ax_mini.plot(self.wave_t, self.wave_y, lw=0.3, color="#94a3b8")

        # Segment markers on minimap
        for i, seg in enumerate(self.segments):
            s, e = seg["start"], seg["end"]
            clr = "#16a34a"
            if self.results and i < len(self.results):
                a = self.results[i]["acc"]
                clr = "#16a34a" if a >= 0.95 else ("#d97706" if a >= 0.5 else "#dc2626")
            self.ax_mini.axvspan(s, e, alpha=0.3, color=clr)

        # View rectangle on minimap
        rect_y = self.ax_mini.get_ylim()
        from matplotlib.patches import Rectangle
        ry0, ry1 = rect_y
        self.ax_mini.add_patch(Rectangle(
            (x0, ry0), x1 - x0, ry1 - ry0,
            linewidth=2, edgecolor="#2563eb", facecolor="#dbeafe",
            alpha=0.3, zorder=10))

        self.ax_mini.set_xlim(0, dur)
        self.ax_mini.set_yticks([])
        self.ax_mini.set_xlabel("")
        self.ax_mini.tick_params(axis="x", labelsize=7)

        # Time formatting on minimap too
        if dur > 120:
            from matplotlib.ticker import FuncFormatter
            self.ax_mini.xaxis.set_major_formatter(FuncFormatter(
                lambda val, pos: self._fmt_time(val)))

        self.canvas.draw_idle()

    # ── Segment List ──────────────────────────────────────────────────────
    def _refresh_list(self):
        self.seg_list.clear()
        for i, seg in enumerate(self.segments):
            if self.results and i < len(self.results):
                r = self.results[i]
                text = (f"{i+1:03d} │ GT: {seg['label']}  →  PRED: {r['pred'] or '—'}"
                       f"  │ {seg['start']:.2f}-{seg['end']:.2f}s  acc={r['acc']:.2f}")
                item = QListWidgetItem(text)
                if r["acc"] >= 0.95: item.setForeground(Qt.darkGreen)
                elif r["acc"] >= 0.5: item.setForeground(Qt.darkYellow)
                else: item.setForeground(Qt.red)
            else:
                text = f"{i+1:03d} │ GT: {seg['label']}  →  PRED: ...  │ {seg['start']:.2f}-{seg['end']:.2f}s"
                item = QListWidgetItem(text)
            self.seg_list.addItem(item)

    def on_seg_clicked(self, item):
        self.selected_idx = self.seg_list.row(item)
        if 0 <= self.selected_idx < len(self.segments):
            seg = self.segments[self.selected_idx]
            pad = max(0.3, (seg["end"] - seg["start"]) * 0.3)
            x0 = max(0, seg["start"] - pad)
            x1 = seg["end"] + pad
            self._redraw(xlim=(x0, x1))

    def on_play_seg(self):
        if self.selected_idx is not None and self.y is not None:
            seg = self.segments[self.selected_idx]
            sd.stop()
            sd.play(self.y[int(seg["start"]*self.sr):int(seg["end"]*self.sr)], self.sr)

    # ── Zoom ──────────────────────────────────────────────────────────────
    def _zoom(self, f):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim(); c = (x0+x1)/2; w = (x1-x0)*f/2
        tot = len(self.y)/self.sr
        new_x0 = max(0, c-w)
        new_x1 = min(tot, c+w)
        self._redraw(xlim=(new_x0, new_x1))

    def _on_scroll(self, ev):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim()
        mx = ev.xdata if ev.xdata else (x0+x1)/2
        f = 0.6 if ev.button == "up" else 1.6
        tot = len(self.y)/self.sr
        new_x0 = max(0, mx-(mx-x0)*f)
        new_x1 = min(tot, mx+(x1-mx)*f)
        self._redraw(xlim=(new_x0, new_x1))

    def _reset_view(self):
        if self.y is not None:
            self._redraw(xlim=(0, len(self.y)/self.sr))

    # ── Export ────────────────────────────────────────────────────────────
    def on_export(self):
        if not self.results: QMessageBox.warning(self, "", "Run prediction first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "results.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["index","ground_truth","prediction","accuracy","start","end"])
                for r in self.results:
                    w.writerow([r["i"]+1, r["gt"], r["pred"], f"{r['acc']:.4f}",
                                f"{r['start']:.4f}", f"{r['end']:.4f}"])
            self.lbl_status.setText(f"Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Indic Speech Annotation Tool v3")
    print(f"  Engines: IndicConformer={'✓' if _HF else '✗'}  "
          f"faster-whisper={'✓' if _FW else '✗'}  HF={'✓' if _HF else '✗'}")
    if _HF:
        cuda = torch.cuda.is_available()
        print(f"  CUDA: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton{padding:5px 10px;border:1px solid #d1d5db;border-radius:4px;
                    background:#f9fafb;font-size:12px}
        QPushButton:hover{background:#e5e7eb}
        QPushButton:disabled{color:#9ca3af;background:#f3f4f6}
        QComboBox,QSpinBox,QDoubleSpinBox{padding:3px 6px;font-size:12px}
    """)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())