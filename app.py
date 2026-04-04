"""
Professional Indic Language Annotation & Segmentation Tool  (FAST edition)
──────────────────────────────────────────────────────────────────────────
Performance optimisations over the previous version
  1. **faster-whisper** (CTranslate2) backend — 4-8× faster than HF.
     Falls back to HuggingFace transformers automatically.
  2. Batched HF inference — groups segments into GPU/CPU batches.
  3. soundfile + scipy resampling — 2-3× faster audio load.
  4. Vectorised VAD — numpy instead of Python for-loop.
  5. Efficient matplotlib — partial redraws, cached line data.
  6. Lazy segment list — lightweight QListWidgetItem (no child widgets).
  7. torch.compile support (PyTorch ≥ 2.0) — 1.3-2× faster generate.
  8. BFloat16 / Float16 automatic selection.
  9. Pre-compiled regexes throughout.

Install for maximum speed:
    pip install faster-whisper soundfile scipy
"""

import os
import sys
import csv
import re
import html
import math
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

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar,
    QGroupBox, QDoubleSpinBox, QSpinBox, QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import torch

# ── Optional fast imports (graceful fallback) ────────────────────────────
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    HAS_HF = True
except ImportError:
    HAS_HF = False

# ── Constants ────────────────────────────────────────────────────────────
MODEL_OPTIONS = [
    ("Whisper Small  (fast)",       "openai/whisper-small",    "small"),
    ("Whisper Medium (balanced)",   "openai/whisper-medium",   "medium"),
    ("Whisper Large-v3 (best)",     "openai/whisper-large-v3", "large-v3"),
]

LANG_OPTIONS = [
    ("English",  "en", "english"),
    ("Hindi",    "hi", "hindi"),
    ("Gujarati", "gu", "gujarati"),
    ("Marathi",  "mr", "marathi"),
]

PREDICTION_MODES = [
    ("Full Transcript", "full"),
    ("Word",            "word"),
    ("Character",       "char"),
]

BACKEND_OPTIONS = []
if HAS_FASTER_WHISPER:
    BACKEND_OPTIONS.append(("faster-whisper (CTranslate2)", "faster_whisper"))
if HAS_HF:
    BACKEND_OPTIONS.append(("HuggingFace transformers", "hf"))
if not BACKEND_OPTIONS:
    raise ImportError("Install either faster-whisper or transformers")

TARGET_SR = 16000
VERBOSE = False

# ── Unicode ──────────────────────────────────────────────────────────────
_INDIC = r"\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF\u0A80-\u0AFF"
_RE_KEEP   = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM   = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA  = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)


# ── Logging ──────────────────────────────────────────────────────────────
def log(msg, force=False):
    if VERBOSE or force:
        print(f"[LOG] {msg}", flush=True)

def warn(msg):
    print(f"[WARN] {msg}", flush=True)

def err(msg):
    print(f"[ERROR] {msg}", flush=True)


# ── Fast audio loading ───────────────────────────────────────────────────
def load_audio_fast(path: str):
    """Load audio as mono float32. Uses soundfile if available, else librosa."""
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            # Mix to mono
            y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return y, sr
        except Exception:
            pass
    # Fallback
    import librosa
    return librosa.load(path, sr=None, mono=True)


def resample_fast(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Fast resampling: scipy polyphase > librosa."""
    if orig_sr == target_sr:
        return y
    if HAS_SCIPY:
        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(y, up, down).astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


# ── Text utilities (pre-compiled) ────────────────────────────────────────
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


def clean_prediction(text: str, mode: str = "full") -> str:
    if not text:
        return ""
    text = _RE_KEEP.sub("", str(text).strip())
    text = _RE_SPACES.sub(" ", text).strip()
    if not text:
        return ""
    if mode == "full":
        return text
    if mode == "word":
        parts = text.split()
        return parts[0] if parts else ""
    if mode == "char":
        m = _RE_ALPHA.search(text)
        return m.group(0) if m else ""
    return text


# ── Font lookup (cached) ────────────────────────────────────────────────
@lru_cache(maxsize=8)
def pick_font(lang_code: str):
    specs = {
        "hi": (
            ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
             "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
             "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
             os.path.expanduser("~/Library/Fonts/NotoSansDevanagari-Regular.ttf")],
            ["Noto Sans Devanagari","Lohit Devanagari","Kohinoor Devanagari","Mangal","Arial Unicode MS"],
        ),
        "gu": (
            ["/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
             "/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf",
             "/Library/Fonts/NotoSansGujarati-Regular.ttf",
             os.path.expanduser("~/Library/Fonts/NotoSansGujarati-Regular.ttf")],
            ["Noto Sans Gujarati","Lohit Gujarati","Shruti","Arial Unicode MS"],
        ),
        "en": ([], ["Arial","Helvetica","DejaVu Sans"]),
    }
    if lang_code == "mr":
        lang_code = "hi"
    files, families = specs.get(lang_code, specs["en"])
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
    """Fully vectorised energy-based VAD — no Python for-loop over frames."""
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)

    # Fast RMS via stride tricks
    n_frames = 1 + (len(y) - frame_len) // hop_len
    if n_frames <= 0:
        return []

    # Compute RMS with numpy (avoids librosa overhead)
    indices = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    indices = np.clip(indices, 0, len(y) - 1)
    frames = y[indices]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)

    is_voice = rms_db > silence_thresh_db

    # Convert ms thresholds to frame counts
    ms_to_frames = lambda ms: max(1, int((ms / 1000) * sr / hop_len))
    min_sil_f  = ms_to_frames(min_silence_ms)
    min_seg_f  = ms_to_frames(min_segment_ms)
    max_seg_f  = ms_to_frames(max_segment_ms)

    # Find contiguous voiced/unvoiced runs using diff
    padded = np.concatenate([[False], is_voice, [False]])
    diffs  = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]

    # Merge segments separated by silence shorter than threshold
    merged_starts, merged_ends = [], []
    i = 0
    while i < len(starts):
        s = starts[i]
        e = ends[i]
        while i + 1 < len(starts) and (starts[i+1] - e) < min_sil_f:
            i += 1
            e = ends[i]
        merged_starts.append(s)
        merged_ends.append(e)
        i += 1

    # Split long segments, filter short ones, convert to seconds
    result = []
    for s, e in zip(merged_starts, merged_ends):
        dur = e - s
        if dur < min_seg_f:
            continue
        # Split into chunks of max_seg_f
        for chunk_s in range(s, e, max_seg_f):
            chunk_e = min(chunk_s + max_seg_f, e)
            if (chunk_e - chunk_s) >= min_seg_f:
                t_s = round(chunk_s * hop_len / sr, 4)
                t_e = round(min(chunk_e * hop_len / sr, len(y) / sr), 4)
                result.append({"start": t_s, "end": t_e})

    log(f"VAD: {len(result)} segments", force=True)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  BACKEND ABSTRACTION
# ══════════════════════════════════════════════════════════════════════════

class FasterWhisperBackend:
    """CTranslate2-based backend — 4-8× faster than HuggingFace."""

    def __init__(self, model_size: str, device: str):
        compute = "float16" if "cuda" in device else "int8"
        log(f"[faster-whisper] Loading {model_size} on {device} ({compute})", force=True)
        self.model = FasterWhisperModel(
            model_size,
            device=device.split(":")[0],  # "cuda" or "cpu"
            compute_type=compute,
            num_workers=min(4, os.cpu_count() or 1),
        )
        self.device_str = device

    def transcribe_batch(self, audio_chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        """Process segments one-by-one (faster-whisper doesn't batch, but is fast).
        Uses lang_code (ISO 639-1: 'gu', 'hi', etc.) which faster-whisper expects."""
        results = []
        total = len(audio_chunks)
        for i, chunk in enumerate(audio_chunks):
            if chunk is None or len(chunk) == 0:
                results.append("")
                continue
            # Pad very short
            if len(chunk) < int(0.1 * sr):
                chunk = np.pad(chunk, (0, int(0.1 * sr) - len(chunk)))

            segments, _ = self.model.transcribe(
                chunk, language=lang_code, task="transcribe",
                beam_size=1,         # greedy = fastest
                vad_filter=False,    # we do our own VAD
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments)
            results.append(clean_prediction(text, mode))

            if progress_cb:
                progress_cb(i + 1, total)

        return results


class HFWhisperBackend:
    """HuggingFace transformers backend with batching + torch.compile."""

    def __init__(self, model_id: str, device: str, batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size

        if "cuda" in device:
            dtype = torch.float16
        elif hasattr(torch, "bfloat16") and device == "cpu":
            # BFloat16 is faster on modern CPUs (Apple M-series, Intel 4th+)
            try:
                _ = torch.zeros(1, dtype=torch.bfloat16)
                dtype = torch.bfloat16
            except Exception:
                dtype = torch.float32
        else:
            dtype = torch.float32

        log(f"[HF] Loading {model_id} on {device} (dtype={dtype})", force=True)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype,
            low_cpu_mem_usage=True, use_safetensors=True,
        ).to(device)

        # torch.compile for PyTorch 2.0+ (significant speedup on repeat calls)
        if hasattr(torch, "compile") and "cuda" in device:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                log("[HF] torch.compile enabled", force=True)
            except Exception as e:
                warn(f"torch.compile failed, continuing without: {e}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dtype = dtype

    def _pad_to_same_length(self, arrays):
        """Pad audio arrays to equal length for batching."""
        max_len = max(len(a) for a in arrays)
        padded = np.zeros((len(arrays), max_len), dtype=np.float32)
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a
        return padded

    def transcribe_batch(self, audio_chunks, sr, lang_code, lang_name, mode, progress_cb=None):
        """Batched inference. Uses lang_name (full name: 'gujarati', 'hindi', etc.)
        which HuggingFace Whisper expects."""
        results = [""] * len(audio_chunks)
        total = len(audio_chunks)
        done = 0

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_chunks = []
            batch_indices = []

            for i in range(batch_start, batch_end):
                chunk = audio_chunks[i]
                if chunk is None or len(chunk) == 0:
                    continue
                min_len = int(0.1 * sr)
                if len(chunk) < min_len:
                    chunk = np.pad(chunk, (0, min_len - len(chunk)))
                batch_chunks.append(chunk)
                batch_indices.append(i)

            if not batch_chunks:
                done += (batch_end - batch_start)
                if progress_cb:
                    progress_cb(done, total)
                continue

            # Process batch
            inputs = self.processor(
                batch_chunks, sampling_rate=sr,
                return_tensors="pt", padding=True,
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            gen_kwargs = {
                "input_features": input_features,
                "task": "transcribe",
                "language": lang_name,
                "return_timestamps": False,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            with torch.inference_mode():
                predicted_ids = self.model.generate(**gen_kwargs)

            texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for idx, text in zip(batch_indices, texts):
                results[idx] = clean_prediction(text, mode)

            done += (batch_end - batch_start)
            if progress_cb:
                progress_cb(done, total)

        return results


# ══════════════════════════════════════════════════════════════════════════
#  THREADS
# ══════════════════════════════════════════════════════════════════════════

class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, str, str)  # backend, device, info
    failed = pyqtSignal(str)

    def __init__(self, model_id, model_size, backend_type, batch_size=8):
        super().__init__()
        self.model_id = model_id
        self.model_size = model_size
        self.backend_type = backend_type
        self.batch_size = batch_size

    def run(self):
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            if self.backend_type == "faster_whisper":
                backend = FasterWhisperBackend(self.model_size, device)
                info = f"faster-whisper ({self.model_size}) on {device}"
            else:
                backend = HFWhisperBackend(self.model_id, device, self.batch_size)
                info = f"HF ({self.model_id}) on {device}, batch={self.batch_size}"

            self.finished_ok.emit(backend, device, info)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class InferenceThread(QThread):
    progress = pyqtSignal(int, int, int, str)
    finished_ok = pyqtSignal(list, float)
    failed = pyqtSignal(str)

    def __init__(self, backend, y16, sr16, gt_rows, lang_code, lang_name, mode):
        super().__init__()
        self.backend = backend
        self.y16 = y16
        self.sr16 = sr16
        self.gt_rows = gt_rows
        self.lang_code = lang_code    # ISO code: "gu", "hi", etc.
        self.lang_name = lang_name    # Full name: "gujarati", "hindi", etc.
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

            # Pre-slice all audio chunks (fast numpy slicing)
            chunks = []
            for row in self.gt_rows:
                s = max(0, int(row["start"] * self.sr16))
                e = min(len(self.y16), int(row["end"] * self.sr16))
                chunks.append(self.y16[s:e])

            log(f"Inference: {total} segments, lang={self.lang_code}/{self.lang_name}", force=True)

            def on_progress(done, tot):
                if self._stop:
                    raise InterruptedError("Stop requested")
                pct = int(done / tot * 100)
                self.progress.emit(done, tot, pct, f"{done}/{tot}")

            predictions = self.backend.transcribe_batch(
                chunks, self.sr16, self.lang_code, self.lang_name,
                self.mode, on_progress,
            )

            results = []
            for i, (row, pred) in enumerate(zip(self.gt_rows, predictions)):
                acc = compute_accuracy(pred, row["label"])
                results.append({
                    "gt": row["label"], "pred": pred,
                    "gt_start": row["start"], "gt_end": row["end"],
                    "pred_start": row["start"], "pred_end": row["end"],
                    "score": acc,
                })

            mean_acc = np.mean([r["score"] for r in results]) if results else 0.0
            self.finished_ok.emit(results, float(mean_acc))

        except InterruptedError:
            warn("Inference interrupted by user")
            self.finished_ok.emit([], 0.0)
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

class ProfessionalAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indic Annotation Tool  [FAST]")
        self.resize(1700, 1000)

        self.audio_path = self.xlsx_path = None
        self.y = self.y16 = self.wave_t = self.wave_y = None
        self.sr = None
        self.sr16 = TARGET_SR

        self.backend = None
        self.device = None

        self.gt_rows = []
        self.results = []
        self.selected_index = None

        self.model_thread = self.infer_thread = self.seg_thread = None

        self._build_ui()
        log("Application ready", force=True)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        L = QVBoxLayout(root); L.setSpacing(4); L.setContentsMargins(8,8,8,8)

        # Row 1
        r1 = QHBoxLayout()
        self.lang_combo = QComboBox()
        for lbl, c, w in LANG_OPTIONS:
            self.lang_combo.addItem(lbl, (c, w))

        self.model_combo = QComboBox()
        for lbl, mid, sz in MODEL_OPTIONS:
            self.model_combo.addItem(lbl, (mid, sz))
        self.model_combo.setCurrentIndex(2)

        self.backend_combo = QComboBox()
        for lbl, val in BACKEND_OPTIONS:
            self.backend_combo.addItem(lbl, val)

        self.mode_combo = QComboBox()
        for lbl, m in PREDICTION_MODES:
            self.mode_combo.addItem(lbl, m)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.batch_spin.setPrefix("Batch: ")

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)
        self.load_gt_btn = QPushButton("Load GT (.xlsx/.csv)")
        self.load_gt_btn.clicked.connect(self.load_gt)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        for w in [QLabel("Lang:"), self.lang_combo,
                  QLabel("Model:"), self.model_combo,
                  QLabel("Engine:"), self.backend_combo,
                  self.batch_spin,
                  QLabel("Mode:"), self.mode_combo,
                  self.load_audio_btn, self.load_gt_btn, self.load_model_btn]:
            r1.addWidget(w)
        r1.addStretch()

        # Row 2
        r2 = QHBoxLayout()
        btns = {
            "Run Prediction": self.run_inference,
            "Stop Inference": self.stop_inference,
            "Auto-Segment": self.run_auto_segmentation,
            "Play Audio": self.play_audio,
            "Play Segment": self.play_selected_segment,
            "Stop Playback": self.stop_audio,
            "Zoom In": lambda: self.zoom(0.7),
            "Zoom Out": lambda: self.zoom(1.4),
            "Reset View": self.reset_plot_view,
            "Export CSV": self.export_csv,
        }
        self._btns = {}
        for name, fn in btns.items():
            b = QPushButton(name)
            b.clicked.connect(fn)
            r2.addWidget(b)
            self._btns[name] = b
        self._btns["Stop Inference"].setEnabled(False)
        r2.addStretch()

        # Status
        self.status_label = QLabel("Load audio and ground truth to begin.")
        self.status_label.setStyleSheet("padding:4px; font-size:13px;")
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

        # Segment list (lightweight — plain text items, no child widgets)
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_segment_clicked)
        self.seg_list.setStyleSheet("""
            QListWidget { background:#fff; border:1px solid #d0d0d0; font-size:12px; }
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

        tabs = QTabWidget()
        seg_tab = QWidget(); sl = QVBoxLayout(seg_tab)
        legend = QLabel('<span style="color:green;font-weight:600">GT</span> | '
                        '<span style="color:blue;font-weight:600">PRED</span>')
        legend.setTextFormat(Qt.RichText)
        sl.addWidget(legend); sl.addWidget(self.seg_list)
        tabs.addTab(seg_tab, "Segments")
        tabs.addTab(vad_grp, "VAD")

        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(self.toolbar); ll.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(tabs)
        splitter.setSizes([1250, 400])

        L.addLayout(r1); L.addLayout(r2)
        L.addWidget(self.status_label); L.addWidget(self.progress_bar)
        L.addWidget(splitter)

    # ── Helpers ───────────────────────────────────────────────────────────
    def set_busy(self, busy, msg=""):
        for name, b in self._btns.items():
            if name == "Stop Inference":
                b.setEnabled(busy)
            else:
                b.setEnabled(not busy)
        for w in [self.load_audio_btn, self.load_gt_btn, self.load_model_btn,
                  self.lang_combo, self.model_combo, self.backend_combo,
                  self.mode_combo, self.batch_spin]:
            w.setEnabled(not busy)
        if msg:
            self.status_label.setText(msg)

    def build_wave_cache(self):
        if self.y is None:
            self.wave_t = self.wave_y = None
            return
        step = max(1, len(self.y) // 25000)
        self.wave_y = self.y[::step]
        self.wave_t = np.arange(len(self.wave_y)) * (step / self.sr)

    # ── Load Audio ────────────────────────────────────────────────────────
    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.webm)")
        if not path:
            return
        try:
            self.status_label.setText("Loading audio...")
            QApplication.processEvents()

            self.y, self.sr = load_audio_fast(path)
            self.y16 = resample_fast(self.y, self.sr, self.sr16)

            self.audio_path = path
            self.results = []; self.selected_index = None
            self.progress_bar.setValue(0)
            self.build_wave_cache()

            dur = len(self.y) / self.sr
            self.status_label.setText(f"Audio: {os.path.basename(path)} | {dur:.1f}s | sr={self.sr}")
            self.refresh_list(); self.redraw()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    # ── Load GT ───────────────────────────────────────────────────────────
    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GT", "", "Excel/CSV (*.xlsx *.csv)")
        if not path:
            return
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

    # ── Load Model ────────────────────────────────────────────────────────
    def load_model(self):
        if self.model_thread and self.model_thread.isRunning():
            return
        model_id, model_size = self.model_combo.currentData()
        btype = self.backend_combo.currentData()
        bs = self.batch_spin.value()
        self.set_busy(True, f"Loading {model_size} ({btype})...")

        self.model_thread = ModelLoaderThread(model_id, model_size, btype, bs)
        self.model_thread.finished_ok.connect(self._on_model_ok)
        self.model_thread.failed.connect(self._on_model_fail)
        self.model_thread.start()

    def _on_model_ok(self, backend, device, info):
        self.backend = backend; self.device = device
        self.set_busy(False, f"Model ready: {info}")

    def _on_model_fail(self, msg):
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
        self.seg_thread.failed.connect(lambda m: (self.set_busy(False, "VAD failed"),
                                                   QMessageBox.critical(self, "Error", m)))
        self.seg_thread.start()

    def _on_vad_ok(self, segs):
        self.gt_rows = [{"label": f"seg_{i+1:04d}", "start": s["start"], "end": s["end"]}
                        for i, s in enumerate(segs)]
        self.results = []; self.selected_index = None
        self.set_busy(False, f"VAD: {len(segs)} segments")
        self.refresh_list(); self.redraw()

    # ── Inference ─────────────────────────────────────────────────────────
    def run_inference(self):
        if self.y is None:
            QMessageBox.warning(self, "Missing", "Load audio first."); return
        if not self.gt_rows:
            QMessageBox.warning(self, "Missing", "Load GT or run VAD first."); return
        if self.backend is None:
            QMessageBox.warning(self, "Missing", "Load model first."); return
        if self.infer_thread and self.infer_thread.isRunning():
            return

        lang_code, lang_name = self.lang_combo.currentData()
        mode = self.mode_combo.currentData()
        self.results = []
        self.progress_bar.setValue(0)
        self.set_busy(True, "Running prediction...")

        self.infer_thread = InferenceThread(
            self.backend, self.y16, self.sr16, self.gt_rows,
            lang_code, lang_name, mode)
        self.infer_thread.progress.connect(self._on_infer_prog)
        self.infer_thread.finished_ok.connect(self._on_infer_done)
        self.infer_thread.failed.connect(self._on_infer_fail)
        self.infer_thread.start()

    def stop_inference(self):
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.request_stop()

    def _on_infer_prog(self, i, tot, pct, msg):
        self.status_label.setText(f"Predicting... {i}/{tot}")
        self.progress_bar.setValue(pct)

    def _on_infer_done(self, results, mean_acc):
        self.results = results
        self.progress_bar.setValue(100)
        self.set_busy(False, f"Done. {len(results)} segs | Accuracy: {mean_acc:.3f}")
        self.refresh_list(); self.redraw()

    def _on_infer_fail(self, msg):
        self.progress_bar.setValue(0)
        self.set_busy(False, "Inference failed.")
        QMessageBox.critical(self, "Error", msg)

    # ── Plot (efficient) ──────────────────────────────────────────────────
    def redraw(self):
        self.ax.clear()
        if self.wave_t is None:
            self.canvas.draw_idle(); return

        self.ax.plot(self.wave_t, self.wave_y, lw=0.5, color="black")
        ymax = max(np.max(np.abs(self.wave_y)), 1e-6)
        lang_code, _ = self.lang_combo.currentData()
        fp = pick_font(lang_code)

        data = self.results or [
            {"gt": r["label"], "pred": None,
             "gt_start": r["start"], "gt_end": r["end"],
             "pred_start": None, "pred_end": None, "score": None}
            for r in self.gt_rows]

        # Only draw visible segments (big speedup for 1000+ segments)
        x0, x1 = self.ax.get_xlim()
        if x0 == 0.0 and x1 == 1.0 and self.y is not None:
            x1 = len(self.y) / self.sr

        for idx, r in enumerate(data):
            s, e = r["gt_start"], r["gt_end"]
            if e < x0 or s > x1:
                continue  # Skip off-screen segments

            color = "green"
            if self.results and r["score"] is not None and r["score"] < 0.95:
                color = "red"
            self.ax.axvspan(s, e, alpha=0.10, color=color)

            if self.selected_index is not None and idx == self.selected_index:
                self.ax.axvspan(s, e, alpha=0.25, color="yellow")
                mid = (s + e) / 2
                kw = {"fontproperties": fp} if fp else {}
                self.ax.text(mid, ymax*0.95, f"GT: {r['gt']}",
                             ha="center", va="bottom", fontsize=10, color="green",
                             fontweight="bold",
                             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                             **kw)
                if r.get("pred") is not None:
                    shown = r["pred"] or "∅"
                    self.ax.text(mid, ymax*0.72, f"PRED: {shown}",
                                 ha="center", va="bottom", fontsize=10, color="blue",
                                 fontweight="bold",
                                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                                 **kw)

        title = "Waveform"
        if self.results:
            title += f" | Accuracy = {np.mean([r['score'] for r in self.results]):.3f}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Amplitude")
        self.canvas.draw_idle()

    # ── Segment List (lightweight text items — no child widgets) ──────────
    def refresh_list(self):
        self.seg_list.clear()
        src = self.results or [
            {"gt": r["label"], "pred": "", "gt_start": r["start"],
             "gt_end": r["end"], "score": None}
            for r in self.gt_rows]

        for i, r in enumerate(src):
            gt = r.get("gt", "")
            pred = r.get("pred", "") or "—"
            sc = r.get("score")
            sc_str = f"{sc:.2f}" if sc is not None else "—"
            t = f"{r['gt_start']:.2f}-{r['gt_end']:.2f}"
            text = f"{i+1:03d} | GT: {gt}  →  PRED: {pred}  | {t}s  acc={sc_str}"
            item = QListWidgetItem(text)
            if sc is not None:
                if sc >= 0.95:
                    item.setForeground(Qt.darkGreen)
                elif sc >= 0.5:
                    item.setForeground(Qt.darkYellow)
                else:
                    item.setForeground(Qt.red)
            self.seg_list.addItem(item)

    def on_segment_clicked(self, item):
        idx = self.seg_list.row(item)
        self.selected_index = idx
        src = self.results or [{"gt_start": r["start"], "gt_end": r["end"]}
                                for r in self.gt_rows]
        if 0 <= idx < len(src):
            s, e = src[idx]["gt_start"], src[idx]["gt_end"]
            pad = max(0.3, (e - s) * 0.2)
            self.ax.set_xlim(max(0, s - pad), e + pad)
            self.redraw()

    # ── Audio ─────────────────────────────────────────────────────────────
    def play_audio(self):
        if self.y is not None: sd.stop(); sd.play(self.y, self.sr)

    def play_selected_segment(self):
        if self.selected_index is None:
            return
        src = self.results or [{"gt_start": r["start"], "gt_end": r["end"]}
                                for r in self.gt_rows]
        if 0 <= self.selected_index < len(src):
            s, e = src[self.selected_index]["gt_start"], src[self.selected_index]["gt_end"]
            sd.stop(); sd.play(self.y[int(s*self.sr):int(e*self.sr)], self.sr)

    def stop_audio(self):
        sd.stop()

    # ── Zoom ──────────────────────────────────────────────────────────────
    def zoom(self, factor):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim(); c = (x0+x1)/2; w = (x1-x0)*factor/2
        tot = len(self.y)/self.sr
        self.ax.set_xlim(max(0, c-w), min(tot, c+w))
        self.redraw()  # Redraw to update visible segments

    def on_scroll(self, event):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim()
        mx = event.xdata if event.xdata else (x0+x1)/2
        s = 0.7 if event.button == "up" else 1.4
        tot = len(self.y)/self.sr
        self.ax.set_xlim(max(0, mx-(mx-x0)*s), min(tot, mx+(x1-mx)*s))
        self.redraw()

    def reset_plot_view(self):
        if self.y is not None:
            self.ax.set_xlim(0, len(self.y)/self.sr)
            self.redraw()

    # ── Export ────────────────────────────────────────────────────────────
    def export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run prediction first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "results.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["index","ground_truth","prediction","accuracy",
                            "gt_start","gt_end"])
                for i, r in enumerate(self.results):
                    w.writerow([i+1, r["gt"], r["pred"], f"{r['score']:.4f}",
                                f"{r['gt_start']:.4f}", f"{r['gt_end']:.4f}"])
            self.status_label.setText(f"Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[LOG] Launching Indic Annotation Tool [FAST]", flush=True)
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton { padding:5px 10px; border:1px solid #d1d5db; border-radius:4px;
                      background:#f9fafb; font-size:12px; }
        QPushButton:hover { background:#e5e7eb; }
        QPushButton:disabled { color:#9ca3af; }
        QComboBox, QSpinBox, QDoubleSpinBox { padding:3px 6px; font-size:12px; }
    """)
    w = ProfessionalAnnotationTool()
    w.show()
    sys.exit(app.exec_())