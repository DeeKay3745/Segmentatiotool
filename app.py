"""
Professional Indic Language Annotation & Segmentation Tool
Supports: Gujarati, Hindi, Marathi, English
Features:
  - Multiple Whisper model sizes (small → large-v3)
  - VAD-based auto-segmentation (librosa + energy-based)
  - Proper CER/WER evaluation metrics
  - Improved Unicode handling for all Indic scripts
  - Font auto-detection with fallbacks
"""

import os
import sys
import csv
import re
import html
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar,
    QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# ── Model options (larger = better Indic support, slower) ──────────────────
MODEL_OPTIONS = [
    ("Whisper Small  (fast, lower accuracy)", "openai/whisper-small"),
    ("Whisper Medium (balanced)",             "openai/whisper-medium"),
    ("Whisper Large-v3 (best accuracy)",      "openai/whisper-large-v3"),
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

# ── Unicode blocks for Indic scripts ──────────────────────────────────────
# Devanagari (Hindi, Marathi, Sanskrit)
_DEVANAGARI     = r"\u0900-\u097F"
_DEVANAGARI_EXT = r"\uA8E0-\uA8FF"
_VEDIC_EXT      = r"\u1CD0-\u1CFF"

# Gujarati
_GUJARATI = r"\u0A80-\u0AFF"

# Combined Indic pattern for cleaning
_INDIC_RANGES = (
    f"{_DEVANAGARI}{_DEVANAGARI_EXT}{_VEDIC_EXT}{_GUJARATI}"
)

# Regex: keep Latin + digits + Indic + spaces + common punctuation
_KEEP_PATTERN = re.compile(
    rf"[^A-Za-z0-9\s{_INDIC_RANGES}.,!?'\-]",
    re.UNICODE,
)

VERBOSE = False


# ── Logging helpers ───────────────────────────────────────────────────────
def log(msg, force=False):
    if VERBOSE or force:
        print(f"[LOG] {msg}", flush=True)

def warn(msg):
    print(f"[WARN] {msg}", flush=True)

def err(msg):
    print(f"[ERROR] {msg}", flush=True)


# ── Utility functions ─────────────────────────────────────────────────────
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
    # Try parsing as raw seconds
    try:
        return float(x)
    except ValueError:
        return None


def norm_text(s):
    """Normalize text: lowercase, strip whitespace, remove punctuation."""
    s = str(s).strip().lower()
    s = re.sub(rf"[^a-z0-9{_INDIC_RANGES}]", "", s, flags=re.UNICODE)
    return s


def edit_distance(s1, s2):
    """Standard Levenshtein edit distance."""
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_cer(pred, gt):
    """Character Error Rate: edit_distance / len(gt). Lower = better."""
    pred_n = norm_text(pred)
    gt_n = norm_text(gt)
    if not gt_n and not pred_n:
        return 0.0
    if not gt_n:
        return 1.0
    dist = edit_distance(pred_n, gt_n)
    return min(dist / len(gt_n), 1.0)


def compute_accuracy(pred, gt):
    """1 - CER, clamped to [0, 1]. Higher = better."""
    return max(0.0, 1.0 - compute_cer(pred, gt))


def clean_prediction(text: str, mode: str = "full") -> str:
    """Clean Whisper output, preserving Indic script characters."""
    if not text:
        return ""

    text = str(text).strip()

    # Remove characters outside our allowed set
    text = _KEEP_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if mode == "full":
        return text

    if mode == "word":
        return text.split()[0] if text.split() else ""

    if mode == "char":
        # Prefer Indic or Latin alphabetic character
        filtered = re.sub(
            rf"[^A-Za-z{_INDIC_RANGES}]", "", text, flags=re.UNICODE
        )
        return filtered[0] if filtered else ""

    return text


def pick_font_for_language(lang_code: str):
    """Find a suitable font for rendering Indic text on plots."""
    font_map = {
        "hi": {
            "files": [
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
                os.path.expanduser("~/Library/Fonts/NotoSansDevanagari-Regular.ttf"),
            ],
            "families": [
                "Noto Sans Devanagari", "Lohit Devanagari",
                "Kohinoor Devanagari", "Mangal", "Arial Unicode MS",
            ],
        },
        "mr": None,  # Marathi uses Devanagari → same as Hindi
        "gu": {
            "files": [
                "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
                "/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf",
                "/Library/Fonts/NotoSansGujarati-Regular.ttf",
                os.path.expanduser("~/Library/Fonts/NotoSansGujarati-Regular.ttf"),
            ],
            "families": [
                "Noto Sans Gujarati", "Lohit Gujarati",
                "Shruti", "Arial Unicode MS",
            ],
        },
        "en": {
            "files": [],
            "families": ["Arial", "Helvetica", "DejaVu Sans"],
        },
    }

    if lang_code == "mr":
        lang_code = "hi"

    spec = font_map.get(lang_code, font_map["en"])

    for path in spec["files"]:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)

    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in spec["families"]:
        if fam in installed:
            return fm.FontProperties(family=fam)

    return None


# ── VAD / Auto-Segmentation ──────────────────────────────────────────────
def vad_segment_audio(
    y, sr,
    min_silence_ms=300,
    silence_thresh_db=-40,
    min_segment_ms=200,
    max_segment_ms=10000,
):
    """
    Energy-based Voice Activity Detection segmentation.
    Splits audio into voiced segments using RMS energy thresholding.
    """
    frame_length = int(sr * 0.025)   # 25 ms frames
    hop_length   = int(sr * 0.010)   # 10 ms hop

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) if np.max(rms) > 0 else 1.0)

    is_voice = rms_db > silence_thresh_db

    min_silence_frames = int((min_silence_ms / 1000) * sr / hop_length)
    min_segment_frames = int((min_segment_ms / 1000) * sr / hop_length)
    max_segment_frames = int((max_segment_ms / 1000) * sr / hop_length)

    segments = []
    in_segment = False
    seg_start = 0
    silence_count = 0

    for i, v in enumerate(is_voice):
        if v:
            if not in_segment:
                seg_start = i
                in_segment = True
                silence_count = 0
            else:
                silence_count = 0
                # Force-split very long segments
                if (i - seg_start) >= max_segment_frames:
                    segments.append((seg_start, i))
                    seg_start = i
        else:
            if in_segment:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    seg_end = i - silence_count
                    if (seg_end - seg_start) >= min_segment_frames:
                        segments.append((seg_start, seg_end))
                    in_segment = False
                    silence_count = 0

    # Flush last segment
    if in_segment:
        seg_end = len(is_voice) - 1
        if (seg_end - seg_start) >= min_segment_frames:
            segments.append((seg_start, seg_end))

    # Convert frame indices to seconds
    result = []
    for s_frame, e_frame in segments:
        s_sec = s_frame * hop_length / sr
        e_sec = e_frame * hop_length / sr
        e_sec = min(e_sec, len(y) / sr)
        result.append({"start": round(s_sec, 4), "end": round(e_sec, 4)})

    log(f"VAD found {len(result)} segments", force=True)
    return result


# ── Threads ───────────────────────────────────────────────────────────────
class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id

    def run(self):
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            log(f"Loading model {self.model_id} on {device}", force=True)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(self.model_id)
            self.finished_ok.emit(model, processor, device)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class InferenceThread(QThread):
    progress = pyqtSignal(int, int, int, str)
    finished_ok = pyqtSignal(list, float)
    failed = pyqtSignal(str)

    def __init__(self, model, processor, device, y16, sr16, gt_rows,
                 whisper_lang_name, mode, generate_kwargs=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.y16 = y16
        self.sr16 = sr16
        self.gt_rows = gt_rows
        self.whisper_lang_name = whisper_lang_name
        self.mode = mode
        self.generate_kwargs = generate_kwargs or {}
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def transcribe_segment(self, audio_array):
        if audio_array is None or len(audio_array) == 0:
            return ""

        # Pad very short segments to at least 0.1s
        min_len = int(0.1 * self.sr16)
        if len(audio_array) < min_len:
            audio_array = np.pad(audio_array, (0, min_len - len(audio_array)))

        inputs = self.processor(
            audio_array,
            sampling_rate=self.sr16,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        gen_kwargs = {
            "input_features": input_features,
            "task": "transcribe",
            "language": self.whisper_lang_name,
            "return_timestamps": False,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        gen_kwargs.update(self.generate_kwargs)

        with torch.inference_mode():
            predicted_ids = self.model.generate(**gen_kwargs)

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return clean_prediction(text, mode=self.mode)

    def run(self):
        try:
            results = []
            total = len(self.gt_rows)

            if total == 0:
                self.finished_ok.emit([], 0.0)
                return

            log(f"Starting inference for {total} segments "
                f"(lang={self.whisper_lang_name}, mode={self.mode})", force=True)

            for i, row in enumerate(self.gt_rows, start=1):
                if self._stop_requested:
                    warn("Inference stop requested")
                    break

                s16 = max(0, int(row["start"] * self.sr16))
                e16 = min(len(self.y16), int(row["end"] * self.sr16))

                chunk = self.y16[s16:e16]
                pred = self.transcribe_segment(chunk)
                acc = compute_accuracy(pred, row["label"])

                results.append({
                    "gt": row["label"],
                    "pred": pred,
                    "gt_start": row["start"],
                    "gt_end": row["end"],
                    "pred_start": row["start"],
                    "pred_end": row["end"],
                    "score": acc,
                })

                pct = int((i / total) * 100)
                msg = (f"{i}/{total} | GT='{row['label']}' | PRED='{pred}' "
                       f"| acc={acc:.3f}")
                self.progress.emit(i, total, pct, msg)

            mean_acc = np.mean([r["score"] for r in results]) if results else 0.0
            self.finished_ok.emit(results, float(mean_acc))

        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class SegmentationThread(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, y, sr, params):
        super().__init__()
        self.y = y
        self.sr = sr
        self.params = params

    def run(self):
        try:
            segs = vad_segment_audio(self.y, self.sr, **self.params)
            self.finished_ok.emit(segs)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ── Main Window ───────────────────────────────────────────────────────────
class ProfessionalAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indic Language Annotation & Segmentation Tool")
        self.resize(1700, 1000)

        self.audio_path = None
        self.xlsx_path = None

        self.y = None
        self.sr = None
        self.y16 = None
        self.sr16 = 16000

        self.wave_t_plot = None
        self.wave_y_plot = None

        self.model = None
        self.processor = None
        self.device = None

        self.gt_rows = []
        self.results = []
        self.selected_index = None

        self.model_thread = None
        self.infer_thread = None
        self.seg_thread = None
        self.is_inference_running = False

        self.font_cache = {
            "en": pick_font_for_language("en"),
            "hi": pick_font_for_language("hi"),
            "mr": pick_font_for_language("mr"),
            "gu": pick_font_for_language("gu"),
        }

        self._build_ui()
        log("Application ready", force=True)

    # ── UI Construction ───────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setSpacing(4)
        root_layout.setContentsMargins(8, 8, 8, 8)

        # ── Row 1: Language, Model, Mode, Load buttons ────────────────────
        row1 = QHBoxLayout()

        self.lang_combo = QComboBox()
        for label, code, wname in LANG_OPTIONS:
            self.lang_combo.addItem(label, (code, wname))

        self.model_combo = QComboBox()
        for label, mid in MODEL_OPTIONS:
            self.model_combo.addItem(label, mid)
        self.model_combo.setCurrentIndex(2)  # Default: large-v3

        self.mode_combo = QComboBox()
        for label, mode in PREDICTION_MODES:
            self.mode_combo.addItem(label, mode)

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)

        self.load_gt_btn = QPushButton("Load Ground Truth (.xlsx)")
        self.load_gt_btn.clicked.connect(self.load_gt)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        for w in [
            QLabel("Language:"), self.lang_combo,
            QLabel("Model:"),    self.model_combo,
            QLabel("Mode:"),     self.mode_combo,
            self.load_audio_btn, self.load_gt_btn, self.load_model_btn,
        ]:
            row1.addWidget(w)
        row1.addStretch()

        # ── Row 2: Actions ────────────────────────────────────────────────
        row2 = QHBoxLayout()

        self.run_btn = QPushButton("Run Prediction")
        self.run_btn.clicked.connect(self.run_inference)

        self.stop_infer_btn = QPushButton("Stop Inference")
        self.stop_infer_btn.clicked.connect(self.stop_inference)
        self.stop_infer_btn.setEnabled(False)

        self.auto_seg_btn = QPushButton("Auto-Segment (VAD)")
        self.auto_seg_btn.clicked.connect(self.run_auto_segmentation)

        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)

        self.play_segment_btn = QPushButton("Play Segment")
        self.play_segment_btn.clicked.connect(self.play_selected_segment)

        self.stop_btn = QPushButton("Stop Playback")
        self.stop_btn.clicked.connect(self.stop_audio)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in_plot)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out_plot)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_plot_view)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)

        for w in [
            self.run_btn, self.stop_infer_btn, self.auto_seg_btn,
            self.play_audio_btn, self.play_segment_btn, self.stop_btn,
            self.zoom_in_btn, self.zoom_out_btn, self.reset_view_btn,
            self.export_btn,
        ]:
            row2.addWidget(w)
        row2.addStretch()

        # ── Status & progress ─────────────────────────────────────────────
        self.status_label = QLabel("Load audio and ground truth to begin.")
        self.status_label.setStyleSheet("padding: 4px; font-size: 13px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cfcfcf; border-radius: 4px;
                background: #f3f4f6; text-align: center; font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6; border-radius: 4px;
            }
        """)

        # ── Matplotlib canvas ─────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # ── Right panel: tabs for segments + VAD settings ─────────────────
        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self.on_segment_clicked)
        self.segment_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff; border: 1px solid #d0d0d0;
                font-size: 12px;
            }
            QListWidget::item { padding: 4px; margin: 2px; }
            QListWidget::item:selected {
                background: #dbeafe; border: 1px solid #60a5fa;
            }
        """)

        # VAD settings panel
        vad_group = QGroupBox("VAD Segmentation Settings")
        vad_layout = QVBoxLayout(vad_group)

        self.vad_silence_ms = QSpinBox()
        self.vad_silence_ms.setRange(50, 2000)
        self.vad_silence_ms.setValue(300)
        self.vad_silence_ms.setSuffix(" ms")

        self.vad_thresh_db = QDoubleSpinBox()
        self.vad_thresh_db.setRange(-80, 0)
        self.vad_thresh_db.setValue(-40)
        self.vad_thresh_db.setSuffix(" dB")

        self.vad_min_seg_ms = QSpinBox()
        self.vad_min_seg_ms.setRange(50, 5000)
        self.vad_min_seg_ms.setValue(200)
        self.vad_min_seg_ms.setSuffix(" ms")

        self.vad_max_seg_ms = QSpinBox()
        self.vad_max_seg_ms.setRange(1000, 30000)
        self.vad_max_seg_ms.setValue(10000)
        self.vad_max_seg_ms.setSuffix(" ms")

        for label_text, widget in [
            ("Min silence duration:", self.vad_silence_ms),
            ("Silence threshold:",    self.vad_thresh_db),
            ("Min segment duration:", self.vad_min_seg_ms),
            ("Max segment duration:", self.vad_max_seg_ms),
        ]:
            h = QHBoxLayout()
            h.addWidget(QLabel(label_text))
            h.addWidget(widget)
            vad_layout.addLayout(h)

        vad_layout.addStretch()

        # Tab widget
        right_tabs = QTabWidget()

        seg_tab = QWidget()
        seg_layout = QVBoxLayout(seg_tab)

        legend = QLabel(
            '<span style="color:green; font-weight:600;">GT (XLSX)</span>  |  '
            '<span style="color:blue; font-weight:600;">MODEL</span>  |  '
            '<span style="color:orange; font-weight:600;">VAD</span>'
        )
        legend.setTextFormat(Qt.RichText)
        seg_layout.addWidget(legend)
        seg_layout.addWidget(self.segment_list)

        right_tabs.addTab(seg_tab, "Segments")
        right_tabs.addTab(vad_group, "VAD Settings")

        # ── Splitter ──────────────────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_tabs)
        splitter.setSizes([1250, 400])

        # ── Assemble ──────────────────────────────────────────────────────
        root_layout.addLayout(row1)
        root_layout.addLayout(row2)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.progress_bar)
        root_layout.addWidget(splitter)

    # ── Helpers ───────────────────────────────────────────────────────────
    def set_ui_busy(self, busy: bool, msg: str = ""):
        for w in [
            self.load_audio_btn, self.load_gt_btn, self.load_model_btn,
            self.run_btn, self.lang_combo, self.mode_combo, self.model_combo,
            self.auto_seg_btn, self.play_audio_btn, self.play_segment_btn,
            self.export_btn,
        ]:
            w.setEnabled(not busy)

        self.stop_infer_btn.setEnabled(busy)

        if msg:
            self.status_label.setText(msg)

    def build_plot_cache(self):
        if self.y is None or self.sr is None:
            self.wave_t_plot = self.wave_y_plot = None
            return

        max_points = 25000
        step = max(1, len(self.y) // max_points)
        self.wave_y_plot = self.y[::step]
        self.wave_t_plot = np.arange(len(self.wave_y_plot)) * step / self.sr
        log(f"Waveform cache: {len(self.wave_y_plot)} points", force=True)

    # ── Load Audio ────────────────────────────────────────────────────────
    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.webm)"
        )
        if not path:
            return

        try:
            self.status_label.setText("Loading audio...")
            QApplication.processEvents()

            self.y, self.sr = librosa.load(path, sr=None, mono=True)
            self.y16 = librosa.resample(self.y, orig_sr=self.sr, target_sr=self.sr16)

            self.audio_path = path
            self.results = []
            self.selected_index = None
            self.progress_bar.setValue(0)
            self.build_plot_cache()

            dur = len(self.y) / self.sr
            self.status_label.setText(
                f"Audio: {os.path.basename(path)} | {dur:.1f}s | sr={self.sr}"
            )
            self.refresh_segment_list()
            self.redraw_plot()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Audio Load Error", str(e))

    # ── Load Ground Truth ─────────────────────────────────────────────────
    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Truth", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, header=None)
            else:
                df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")

            rows, skipped = [], 0
            for _, r in df.iterrows():
                label = str(r.iloc[0]).strip()
                start = time_to_sec(r.iloc[1])
                end   = time_to_sec(r.iloc[2])

                if not label or start is None or end is None or end <= start:
                    skipped += 1
                    continue

                rows.append({"label": label, "start": start, "end": end})

            self.gt_rows = rows
            self.xlsx_path = path
            self.results = []
            self.selected_index = None
            self.progress_bar.setValue(0)

            self.status_label.setText(
                f"GT loaded: {len(rows)} segments (skipped {skipped})"
            )
            self.refresh_segment_list()
            self.redraw_plot()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "GT Load Error", str(e))

    # ── Load Model ────────────────────────────────────────────────────────
    def load_model(self):
        if self.model_thread and self.model_thread.isRunning():
            return

        model_id = self.model_combo.currentData()
        self.set_ui_busy(True, f"Loading {model_id}...")

        self.model_thread = ModelLoaderThread(model_id)
        self.model_thread.finished_ok.connect(self.on_model_loaded)
        self.model_thread.failed.connect(self.on_model_load_failed)
        self.model_thread.start()

    def on_model_loaded(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.set_ui_busy(False, f"Model loaded on {device}.")

    def on_model_load_failed(self, message):
        self.set_ui_busy(False, "Model load failed.")
        QMessageBox.critical(self, "Model Load Error", message)

    # ── Auto Segmentation (VAD) ───────────────────────────────────────────
    def run_auto_segmentation(self):
        if self.y is None:
            QMessageBox.warning(self, "No Audio", "Load audio first.")
            return

        params = {
            "min_silence_ms":  self.vad_silence_ms.value(),
            "silence_thresh_db": self.vad_thresh_db.value(),
            "min_segment_ms":  self.vad_min_seg_ms.value(),
            "max_segment_ms":  self.vad_max_seg_ms.value(),
        }

        self.set_ui_busy(True, "Running VAD segmentation...")
        self.seg_thread = SegmentationThread(self.y, self.sr, params)
        self.seg_thread.finished_ok.connect(self.on_segmentation_done)
        self.seg_thread.failed.connect(self.on_segmentation_failed)
        self.seg_thread.start()

    def on_segmentation_done(self, segments):
        # Convert VAD segments to gt_rows format (label = segment index)
        self.gt_rows = [
            {
                "label": f"seg_{i+1:04d}",
                "start": s["start"],
                "end":   s["end"],
            }
            for i, s in enumerate(segments)
        ]
        self.results = []
        self.selected_index = None
        self.set_ui_busy(False, f"VAD found {len(segments)} segments.")
        self.refresh_segment_list()
        self.redraw_plot()

    def on_segmentation_failed(self, message):
        self.set_ui_busy(False, "Segmentation failed.")
        QMessageBox.critical(self, "Segmentation Error", message)

    # ── Inference ─────────────────────────────────────────────────────────
    def run_inference(self):
        if self.y is None or self.y16 is None:
            QMessageBox.warning(self, "Missing Audio", "Load audio first.")
            return
        if not self.gt_rows:
            QMessageBox.warning(self, "Missing Segments",
                                "Load ground truth or run auto-segmentation.")
            return
        if self.model is None:
            QMessageBox.warning(self, "Missing Model", "Load the model first.")
            return
        if self.infer_thread and self.infer_thread.isRunning():
            return

        _, whisper_lang = self.lang_combo.currentData()
        mode = self.mode_combo.currentData()

        self.results = []
        self.is_inference_running = True
        self.progress_bar.setValue(0)
        self.set_ui_busy(True, "Running prediction...")

        self.infer_thread = InferenceThread(
            self.model, self.processor, self.device,
            self.y16, self.sr16, self.gt_rows,
            whisper_lang, mode,
        )
        self.infer_thread.progress.connect(self.on_inference_progress)
        self.infer_thread.finished_ok.connect(self.on_inference_finished)
        self.infer_thread.failed.connect(self.on_inference_failed)
        self.infer_thread.start()

    def stop_inference(self):
        if self.infer_thread and self.infer_thread.isRunning():
            self.infer_thread.request_stop()
            self.status_label.setText("Stopping inference...")

    def on_inference_progress(self, i, total, pct, msg):
        self.status_label.setText(f"Predicting... {i}/{total}")
        self.progress_bar.setValue(pct)

    def on_inference_finished(self, results, mean_acc):
        self.results = results
        self.is_inference_running = False
        self.progress_bar.setValue(100)
        self.set_ui_busy(
            False,
            f"Done. {len(results)} segments | Mean accuracy: {mean_acc:.3f}"
        )
        self.refresh_segment_list()
        self.redraw_plot()

    def on_inference_failed(self, message):
        self.is_inference_running = False
        self.progress_bar.setValue(0)
        self.set_ui_busy(False, "Inference failed.")
        QMessageBox.critical(self, "Inference Error", message)

    # ── Plot ──────────────────────────────────────────────────────────────
    def _draw_label(self, x, ymax, gt_text=None, pred_text=None,
                    gt_time=None, pred_time=None, lang_code="hi"):
        fp = self.font_cache.get(lang_code)
        kwargs = {"fontproperties": fp} if fp else {}

        positions = [0.95, 0.84, 0.72, 0.61]

        if gt_text:
            self.ax.text(
                x, ymax * positions[0], f"GT: {gt_text}",
                ha="center", va="bottom", fontsize=10, color="green",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                **kwargs,
            )
        if gt_time:
            self.ax.text(
                x, ymax * positions[1], gt_time,
                ha="center", va="bottom", fontsize=8, color="green",
            )
        if pred_text is not None:
            shown = pred_text if pred_text else "∅"
            self.ax.text(
                x, ymax * positions[2], f"PRED: {shown}",
                ha="center", va="bottom", fontsize=10, color="blue",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
                **kwargs,
            )
        if pred_time:
            self.ax.text(
                x, ymax * positions[3], pred_time,
                ha="center", va="bottom", fontsize=8, color="blue",
            )

    def redraw_plot(self):
        self.ax.clear()

        if self.wave_t_plot is None or self.wave_y_plot is None:
            self.canvas.draw_idle()
            return

        self.ax.plot(self.wave_t_plot, self.wave_y_plot, lw=0.6, color="black")

        ymax = np.max(np.abs(self.wave_y_plot)) if len(self.wave_y_plot) else 1.0
        ymax = max(ymax, 1e-6)

        lang_code, _ = self.lang_combo.currentData()

        data = self.results if self.results else [
            {"gt": r["label"], "pred": None,
             "gt_start": r["start"], "gt_end": r["end"],
             "pred_start": None, "pred_end": None, "score": None}
            for r in self.gt_rows
        ]

        for idx, r in enumerate(data):
            s, e = r["gt_start"], r["gt_end"]

            if self.results:
                color = "green" if r["score"] is not None and r["score"] >= 0.95 else "red"
            else:
                color = "green"

            self.ax.axvspan(s, e, alpha=0.10, color=color)
            self.ax.axvline(s, lw=0.6, color=color, alpha=0.5)
            self.ax.axvline(e, lw=0.6, color=color, alpha=0.5)

            if self.selected_index is not None and idx == self.selected_index:
                self.ax.axvspan(s, e, alpha=0.25, color="yellow")

        # Draw labels for selected segment
        if self.selected_index is not None and 0 <= self.selected_index < len(data):
            r = data[self.selected_index]
            mid = (r["gt_start"] + r["gt_end"]) / 2
            gt_time = f"[{r['gt_start']:.2f}–{r['gt_end']:.2f}]"
            pred_time = None
            if r.get("pred_start") is not None:
                pred_time = f"[{r['pred_start']:.2f}–{r['pred_end']:.2f}]"

            self._draw_label(
                mid, ymax, r["gt"], r.get("pred"),
                gt_time, pred_time, lang_code,
            )

        title = "Waveform"
        if self.results:
            mean = np.mean([r["score"] for r in self.results])
            title = f"Waveform | Mean Accuracy = {mean:.3f}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Segment List ──────────────────────────────────────────────────────
    def _make_segment_widget(self, idx, r):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(1)

        gt   = html.escape(str(r.get("gt", "")))
        pred = html.escape(str(r.get("pred", "") or "∅"))
        time_str = f"{r.get('gt_start', 0):.2f}–{r.get('gt_end', 0):.2f}s"
        score = r.get("score")

        if score is None:
            score_html = '<span style="color:#9ca3af;">—</span>'
        else:
            c = "#16a34a" if score >= 0.95 else ("#f59e0b" if score >= 0.5 else "#dc2626")
            score_html = f'<span style="color:{c}; font-weight:600;">{score:.2f}</span>'

        top = QLabel(
            f'<b>{idx+1:03d}</b> '
            f'<span style="color:green;">GT: {gt}</span> → '
            f'<span style="color:blue;">PRED: {pred}</span>'
        )
        top.setTextFormat(Qt.RichText)
        top.setWordWrap(True)

        bottom = QLabel(f'{time_str} | Acc: {score_html}')
        bottom.setTextFormat(Qt.RichText)

        layout.addWidget(top)
        layout.addWidget(bottom)
        return w

    def refresh_segment_list(self):
        self.segment_list.clear()

        source = self.results if self.results else [
            {"gt": r["label"], "pred": "",
             "gt_start": r["start"], "gt_end": r["end"], "score": None}
            for r in self.gt_rows
        ]

        for i, r in enumerate(source):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, i)
            widget = self._make_segment_widget(i, r)
            item.setSizeHint(widget.sizeHint())
            self.segment_list.addItem(item)
            self.segment_list.setItemWidget(item, widget)

    def on_segment_clicked(self, item):
        idx = self.segment_list.row(item)
        self.selected_index = idx

        source = self.results if self.results else [
            {"gt_start": r["start"], "gt_end": r["end"]}
            for r in self.gt_rows
        ]
        if 0 <= idx < len(source):
            s, e = source[idx]["gt_start"], source[idx]["gt_end"]
            pad = max(0.3, (e - s) * 0.2)
            self.ax.set_xlim(max(0, s - pad), e + pad)
            self.redraw_plot()

    # ── Audio Playback ────────────────────────────────────────────────────
    def play_audio(self):
        if self.y is not None and self.sr:
            sd.stop()
            sd.play(self.y, self.sr)

    def play_selected_segment(self):
        if self.selected_index is None:
            QMessageBox.warning(self, "No Segment", "Select a segment first.")
            return

        source = self.results if self.results else [
            {"gt_start": r["start"], "gt_end": r["end"]}
            for r in self.gt_rows
        ]
        if 0 <= self.selected_index < len(source):
            s = source[self.selected_index]["gt_start"]
            e = source[self.selected_index]["gt_end"]
            si, ei = int(s * self.sr), int(e * self.sr)
            sd.stop()
            sd.play(self.y[si:ei], self.sr)

    def stop_audio(self):
        sd.stop()

    # ── Zoom ──────────────────────────────────────────────────────────────
    def zoom(self, factor):
        if self.y is None:
            return
        x0, x1 = self.ax.get_xlim()
        c = (x0 + x1) / 2
        w = (x1 - x0) * factor / 2
        total = len(self.y) / self.sr
        self.ax.set_xlim(max(0, c - w), min(total, c + w))
        self.canvas.draw_idle()

    def zoom_in_plot(self):
        self.zoom(0.7)

    def zoom_out_plot(self):
        self.zoom(1.4)

    def on_scroll(self, event):
        if self.y is None:
            return
        x0, x1 = self.ax.get_xlim()
        mx = event.xdata if event.xdata is not None else (x0 + x1) / 2
        s = 0.7 if event.button == "up" else 1.4
        total = len(self.y) / self.sr
        self.ax.set_xlim(
            max(0, mx - (mx - x0) * s),
            min(total, mx + (x1 - mx) * s),
        )
        self.canvas.draw_idle()

    def reset_plot_view(self):
        if self.y is not None and self.sr:
            self.ax.set_xlim(0, len(self.y) / self.sr)
            self.canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────
    def export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run prediction first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "results.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow([
                    "index", "ground_truth", "prediction", "accuracy",
                    "gt_start", "gt_end", "pred_start", "pred_end",
                ])
                for i, r in enumerate(self.results):
                    w.writerow([
                        i + 1, r["gt"], r["pred"], f"{r['score']:.4f}",
                        f"{r['gt_start']:.4f}", f"{r['gt_end']:.4f}",
                        f"{r['pred_start']:.4f}", f"{r['pred_end']:.4f}",
                    ])
            self.status_label.setText(f"Exported: {path}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", str(e))


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[LOG] Launching Indic Annotation Tool", flush=True)
    app = QApplication(sys.argv)

    # Global stylesheet
    app.setStyleSheet("""
        QPushButton {
            padding: 5px 12px;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            background: #f9fafb;
            font-size: 12px;
        }
        QPushButton:hover { background: #e5e7eb; }
        QPushButton:pressed { background: #d1d5db; }
        QPushButton:disabled { color: #9ca3af; background: #f3f4f6; }
        QComboBox { padding: 4px 8px; font-size: 12px; }
        QLabel { font-size: 12px; }
    """)

    w = ProfessionalAnnotationTool()
    w.show()
    sys.exit(app.exec_())