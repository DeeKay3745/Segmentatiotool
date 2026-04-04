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
    QListWidget, QListWidgetItem, QSplitter, QProgressBar
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "openai/whisper-small"

LANG_OPTIONS = [
    ("English", "en", "english"),
    ("Hindi", "hi", "hindi"),
    ("Gujarati", "gu", "gujarati"),
    ("Marathi", "mr", "marathi"),
]

PREDICTION_MODES = [
    ("Word", "word"),
    ("Character", "char"),
]

VERBOSE = False


def log(msg, force=False):
    if VERBOSE or force:
        print(f"[LOG] {msg}", flush=True)


def warn(msg):
    print(f"[WARN] {msg}", flush=True)


def err(msg):
    print(f"[ERROR] {msg}", flush=True)


def time_to_sec(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x:
        return None
    try:
        dt = datetime.strptime(x, "%H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.strptime(x, "%H:%M:%S")
        except ValueError:
            return None
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def norm_text(s):
    return "".join(str(s).strip().split())


def simple_match(pred, gt):
    pred = norm_text(pred)
    gt = norm_text(gt)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    if pred == gt:
        return 1.0
    common = sum(min(pred.count(ch), gt.count(ch)) for ch in set(pred) | set(gt))
    return common / max(len(pred), len(gt))


def clean_prediction(text: str, mode: str = "word") -> str:
    if not text:
        return ""

    text = str(text).strip()

    # Keep English letters/numbers/underscore + spaces + Indic blocks
    text = re.sub(r"[^\w\s\u0900-\u097F\u0A80-\u0AFF]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if mode == "word":
        return text.split(" ")[0].strip()

    if mode == "char":
        # Prefer Indic or English alphabetic character only
        filtered = re.sub(r"[^A-Za-z\u0900-\u097F\u0A80-\u0AFF]", "", text)
        return filtered[0] if filtered else ""

    return text


def pick_font_for_language(lang_code: str):
    if lang_code in ("hi", "mr"):
        candidates = [
            "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
            "/System/Library/Fonts/Supplemental/NotoSansDevanagari-Regular.ttf",
            os.path.expanduser("~/Library/Fonts/NotoSansDevanagari-Regular.ttf"),
        ]
        families = [
            "Noto Sans Devanagari",
            "Kohinoor Devanagari",
            "Arial Unicode MS",
            "Mangal",
        ]
    elif lang_code == "gu":
        candidates = [
            "/Library/Fonts/NotoSansGujarati-Regular.ttf",
            "/System/Library/Fonts/Supplemental/NotoSansGujarati-Regular.ttf",
            os.path.expanduser("~/Library/Fonts/NotoSansGujarati-Regular.ttf"),
        ]
        families = [
            "Noto Sans Gujarati",
            "Shruti",
            "Arial Unicode MS",
        ]
    else:  # English and fallback
        candidates = []
        families = [
            "Arial",
            "Helvetica",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

    for path in candidates:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)

    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in families:
        if fam in installed:
            return fm.FontProperties(family=fam)

    return None


class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            log(f"Loading model {MODEL_ID} on {device}", force=True)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=torch.cuda.is_available(),
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.finished_ok.emit(model, processor, device)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class InferenceThread(QThread):
    progress = pyqtSignal(int, int, int, str)  # current, total, percent, message
    finished_ok = pyqtSignal(list, float)
    failed = pyqtSignal(str)

    def __init__(self, model, processor, device, y16, sr16, gt_rows, whisper_lang_name, mode):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.y16 = y16
        self.sr16 = sr16
        self.gt_rows = gt_rows
        self.whisper_lang_name = whisper_lang_name
        self.mode = mode
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def transcribe_segment(self, audio_array):
        if audio_array is None or len(audio_array) == 0:
            return ""

        inputs = self.processor(
            audio_array,
            sampling_rate=self.sr16,
            return_tensors="pt",
            return_attention_mask=True
        )

        input_features = inputs.input_features.to(self.device)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language=self.whisper_lang_name
            )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return clean_prediction(text, mode=self.mode)

    def run(self):
        try:
            results = []
            total = len(self.gt_rows)

            if total == 0:
                self.finished_ok.emit([], 0.0)
                return

            log(f"Starting background inference for {total} segments", force=True)

            for i, row in enumerate(self.gt_rows, start=1):
                if self._stop_requested:
                    warn("Inference stop requested")
                    break

                s16 = int(row["start"] * self.sr16)
                e16 = int(row["end"] * self.sr16)
                s16 = max(0, s16)
                e16 = min(len(self.y16), e16)

                chunk16 = self.y16[s16:e16]
                pred = self.transcribe_segment(chunk16)
                score = simple_match(pred, row["label"])

                results.append({
                    "gt": row["label"],
                    "pred": pred,
                    "gt_start": row["start"],
                    "gt_end": row["end"],
                    "pred_start": row["start"],
                    "pred_end": row["end"],
                    "score": score,
                })

                percent = int((i / total) * 100)
                msg = f"{i}/{total} | XLSX='{row['label']}' | MODEL='{pred}' | score={score:.3f}"
                self.progress.emit(i, total, percent, msg)

                if i == 1 or i % 5 == 0 or i == total:
                    log(msg, force=True)

            mean_score = np.mean([r["score"] for r in results]) if results else 0.0
            self.finished_ok.emit(results, float(mean_score))

        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class ProfessionalAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Indic Annotation Tool")
        self.resize(1650, 980)

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
        self.is_inference_running = False

        self.font_cache = {
            "en": pick_font_for_language("en"),
            "hi": pick_font_for_language("hi"),
            "mr": pick_font_for_language("mr"),
            "gu": pick_font_for_language("gu"),
        }

        self._build_ui()
        log("Application ready", force=True)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        menu_row_1 = QHBoxLayout()

        self.lang_combo = QComboBox()
        for label, lang_code, whisper_lang in LANG_OPTIONS:
            self.lang_combo.addItem(label, (lang_code, whisper_lang))

        self.mode_combo = QComboBox()
        for label, mode in PREDICTION_MODES:
            self.mode_combo.addItem(label, mode)

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)

        self.load_gt_btn = QPushButton("Load Ground Truth (.xlsx)")
        self.load_gt_btn.clicked.connect(self.load_gt)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        self.run_btn = QPushButton("Run Prediction")
        self.run_btn.clicked.connect(self.run_inference)

        for w in [
            QLabel("Language"),
            self.lang_combo,
            QLabel("Prediction"),
            self.mode_combo,
            self.load_audio_btn,
            self.load_gt_btn,
            self.load_model_btn,
            self.run_btn,
        ]:
            menu_row_1.addWidget(w)

        menu_row_1.addStretch()

        menu_row_2 = QHBoxLayout()

        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)

        self.play_segment_btn = QPushButton("Play Segment")
        self.play_segment_btn.clicked.connect(self.play_selected_segment)

        self.stop_btn = QPushButton("Stop")
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
            self.play_audio_btn,
            self.play_segment_btn,
            self.stop_btn,
            self.zoom_in_btn,
            self.zoom_out_btn,
            self.reset_view_btn,
            self.export_btn,
        ]:
            menu_row_2.addWidget(w)

        menu_row_2.addStretch()

        self.status_label = QLabel("Load audio and ground truth to begin.")
        self.status_label.setStyleSheet("padding: 6px; font-size: 13px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cfcfcf;
                border-radius: 4px;
                background: #f3f4f6;
                text-align: center;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
        """)

        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self.on_segment_clicked)
        self.segment_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                border: 1px solid #60a5fa;
            }
        """)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        seg_head = QLabel("Segments")
        seg_head.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.legend_label = QLabel(
            '<span style="color: green; font-weight: 600;">XLSX</span>  |  '
            '<span style="color: blue; font-weight: 600;">MODEL</span>'
        )
        self.legend_label.setTextFormat(Qt.RichText)

        right_layout.addWidget(seg_head)
        right_layout.addWidget(self.legend_label)
        right_layout.addWidget(self.segment_list)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([1250, 350])

        root_layout.addLayout(menu_row_1)
        root_layout.addLayout(menu_row_2)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.progress_bar)
        root_layout.addWidget(splitter)

    def set_ui_busy(self, busy: bool, msg: str = ""):
        self.load_audio_btn.setEnabled(not busy)
        self.load_gt_btn.setEnabled(not busy)
        self.load_model_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)
        self.lang_combo.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)

        self.play_audio_btn.setEnabled(not busy)
        self.play_segment_btn.setEnabled(not busy)
        self.export_btn.setEnabled(not busy)

        if msg:
            self.status_label.setText(msg)

    def build_plot_cache(self):
        if self.y is None or self.sr is None:
            self.wave_t_plot = None
            self.wave_y_plot = None
            return

        max_points = 20000
        step = max(1, len(self.y) // max_points)
        y_plot = self.y[::step]
        t_plot = np.arange(len(y_plot)) * step / self.sr

        self.wave_t_plot = t_plot
        self.wave_y_plot = y_plot

        log(f"Waveform cache built with {len(y_plot)} points", force=True)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if not path:
            return

        try:
            self.status_label.setText("Loading audio...")
            QApplication.processEvents()

            log(f"Loading audio: {path}", force=True)
            self.y, self.sr = librosa.load(path, sr=None, mono=True)
            self.y16 = librosa.resample(self.y, orig_sr=self.sr, target_sr=self.sr16)

            self.audio_path = path
            self.results = []
            self.selected_index = None
            self.progress_bar.setValue(0)

            self.build_plot_cache()

            duration = len(self.y) / self.sr if self.sr else 0.0
            log(f"Audio loaded | sr={self.sr} | duration={duration:.2f}s | sr16={self.sr16}", force=True)

            self.status_label.setText(f"Loaded audio: {os.path.basename(path)}")
            self.refresh_segment_list()
            self.redraw_plot()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Audio Load Error", str(e))

    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Ground Truth",
            "",
            "Excel Files (*.xlsx)"
        )
        if not path:
            return

        try:
            log(f"Loading ground truth: {path}", force=True)
            df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")

            rows = []
            skipped = 0

            for _, r in df.iterrows():
                label = str(r[0]).strip()
                start = time_to_sec(r[1])
                end = time_to_sec(r[2])

                if not label or start is None or end is None:
                    skipped += 1
                    continue
                if end <= start:
                    skipped += 1
                    continue

                rows.append({
                    "label": label,
                    "start": start,
                    "end": end,
                })

            self.gt_rows = rows
            self.xlsx_path = path
            self.results = []
            self.selected_index = None
            self.progress_bar.setValue(0)

            log(f"GT rows loaded: {len(rows)} | skipped={skipped}", force=True)

            self.status_label.setText(f"Loaded GT rows: {len(rows)}")
            self.refresh_segment_list()
            self.redraw_plot()

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "GT Load Error", str(e))

    def load_model(self):
        if self.model_thread is not None and self.model_thread.isRunning():
            warn("Model loading is already running")
            return

        self.set_ui_busy(True, "Loading Whisper model...")
        self.model_thread = ModelLoaderThread()
        self.model_thread.finished_ok.connect(self.on_model_loaded)
        self.model_thread.failed.connect(self.on_model_load_failed)
        self.model_thread.start()

    def on_model_loaded(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

        log(f"Model loaded successfully on {device}", force=True)
        self.set_ui_busy(False, "Model loaded successfully.")

    def on_model_load_failed(self, message):
        err(f"Model load failed: {message}")
        self.set_ui_busy(False, "Model load failed.")
        QMessageBox.critical(self, "Model Load Error", message)

    def run_inference(self):
        if self.audio_path is None or self.y is None or self.y16 is None:
            QMessageBox.warning(self, "Missing Audio", "Load audio first.")
            return

        if not self.gt_rows:
            QMessageBox.warning(self, "Missing GT", "Load ground truth first.")
            return

        if self.model is None or self.processor is None:
            QMessageBox.warning(self, "Missing Model", "Load the model first.")
            return

        if self.infer_thread is not None and self.infer_thread.isRunning():
            warn("Inference is already running")
            return

        _, whisper_lang_name = self.lang_combo.currentData()
        mode = self.mode_combo.currentData()

        self.results = []
        self.refresh_segment_list()
        self.redraw_plot()

        self.is_inference_running = True
        self.progress_bar.setValue(0)
        self.set_ui_busy(True, "Running prediction in background...")

        self.infer_thread = InferenceThread(
            model=self.model,
            processor=self.processor,
            device=self.device,
            y16=self.y16,
            sr16=self.sr16,
            gt_rows=self.gt_rows,
            whisper_lang_name=whisper_lang_name,
            mode=mode
        )
        self.infer_thread.progress.connect(self.on_inference_progress)
        self.infer_thread.finished_ok.connect(self.on_inference_finished)
        self.infer_thread.failed.connect(self.on_inference_failed)
        self.infer_thread.start()

    def on_inference_progress(self, i, total, percent, msg):
        self.status_label.setText(f"Running prediction... {i}/{total}")
        self.progress_bar.setValue(percent)
        log(msg, force=True)

    def on_inference_finished(self, results, mean_score):
        self.results = results
        self.is_inference_running = False

        log(f"Inference finished | mean_score={mean_score:.3f}", force=True)

        self.progress_bar.setValue(100)
        self.set_ui_busy(False, f"Done. Segments={len(results)} | Mean match={mean_score:.3f}")
        self.refresh_segment_list()
        self.redraw_plot()

    def on_inference_failed(self, message):
        self.is_inference_running = False
        err(f"Inference failed: {message}")
        self.progress_bar.setValue(0)
        self.set_ui_busy(False, "Inference failed.")
        QMessageBox.critical(self, "Inference Error", message)

    def _draw_label(self, x, ymax, gt_text=None, pred_text=None, gt_time=None, pred_time=None, lang_code="hi"):
        font_prop = self.font_cache.get(lang_code, None)

        y1 = ymax * 0.95
        y2 = ymax * 0.84
        y3 = ymax * 0.72
        y4 = ymax * 0.61

        if gt_text:
            self.ax.text(
                x, y1, f"XLSX: {gt_text}",
                ha="center", va="bottom",
                fontsize=10, color="green",
                fontproperties=font_prop, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.18, edgecolor="none", pad=2)
            )

        if gt_time:
            self.ax.text(
                x, y2, f"XLSX Time: {gt_time}",
                ha="center", va="bottom",
                fontsize=8, color="green"
            )

        if pred_text is not None:
            shown_pred = pred_text if pred_text != "" else "∅"
            self.ax.text(
                x, y3, f"MODEL: {shown_pred}",
                ha="center", va="bottom",
                fontsize=10, color="blue",
                fontproperties=font_prop, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.18, edgecolor="none", pad=2)
            )

        if pred_time:
            self.ax.text(
                x, y4, f"MODEL Time: {pred_time}",
                ha="center", va="bottom",
                fontsize=8, color="blue"
            )

    def redraw_plot(self):
        self.ax.clear()

        if self.wave_t_plot is None or self.wave_y_plot is None:
            self.canvas.draw_idle()
            return

        self.ax.plot(self.wave_t_plot, self.wave_y_plot, linewidth=0.7, color="black")

        ymax = np.max(np.abs(self.wave_y_plot)) if len(self.wave_y_plot) else 1.0
        if ymax == 0:
            ymax = 1.0

        lang_code, _ = self.lang_combo.currentData()

        data = self.results if self.results else [
            {
                "gt": r["label"],
                "pred": None,
                "gt_start": r["start"],
                "gt_end": r["end"],
                "pred_start": None,
                "pred_end": None,
                "score": None,
            }
            for r in self.gt_rows
        ]

        for idx, r in enumerate(data):
            s, e = r["gt_start"], r["gt_end"]

            if self.results:
                color = "green" if r["score"] is not None and r["score"] >= 0.99 else "red"
            else:
                color = "green"

            self.ax.axvspan(s, e, alpha=0.12, color=color)
            self.ax.axvline(s, linewidth=0.8, color=color)
            self.ax.axvline(e, linewidth=0.8, color=color)

            if self.selected_index is not None and idx == self.selected_index:
                self.ax.axvspan(s, e, alpha=0.20, color="yellow")

        if self.selected_index is not None and 0 <= self.selected_index < len(data):
            r = data[self.selected_index]
            mid = (r["gt_start"] + r["gt_end"]) / 2
            gt_time = f"[{r['gt_start']:.2f}-{r['gt_end']:.2f}]"
            pred_time = None
            if r["pred_start"] is not None and r["pred_end"] is not None:
                pred_time = f"[{r['pred_start']:.2f}-{r['pred_end']:.2f}]"

            self._draw_label(
                x=mid,
                ymax=ymax,
                gt_text=r["gt"],
                pred_text=r["pred"],
                gt_time=gt_time,
                pred_time=pred_time,
                lang_code=lang_code,
            )

        title = "Waveform"
        if self.results:
            mean_score = np.mean([r["score"] for r in self.results]) if self.results else 0.0
            title = f"Waveform | mean match={mean_score:.3f}"

        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw_idle()

    def _make_segment_widget(self, idx, r):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        gt = html.escape(str(r.get("gt", "")))
        pred_val = r.get("pred", "")
        pred = html.escape(str(pred_val if pred_val != "" else "∅"))

        time_line = f"{r.get('gt_start', 0):.2f}-{r.get('gt_end', 0):.2f}"
        score = r.get("score", None)

        if score is None:
            score_html = '<span style="color:#6b7280;">score=NA</span>'
        else:
            score_color = "#16a34a" if score >= 0.99 else "#dc2626"
            score_html = f'<span style="color:{score_color};">score={score:.2f}</span>'

        top = QLabel(
            f'<span style="color:#111827; font-weight:600;">{idx+1:02d}</span> | '
            f'<span style="color:green; font-weight:600;">XLSX:</span> '
            f'<span style="color:green;">{gt}</span>'
        )
        top.setTextFormat(Qt.RichText)
        top.setWordWrap(True)

        mid = QLabel(
            f'<span style="color:blue; font-weight:600;">MODEL:</span> '
            f'<span style="color:blue;">{pred}</span>'
        )
        mid.setTextFormat(Qt.RichText)
        mid.setWordWrap(True)

        bottom = QLabel(
            f'<span style="color:#6b7280;">{time_line}</span> | {score_html}'
        )
        bottom.setTextFormat(Qt.RichText)
        bottom.setWordWrap(True)

        layout.addWidget(top)
        layout.addWidget(mid)
        layout.addWidget(bottom)
        return widget

    def refresh_segment_list(self):
        self.segment_list.clear()

        source = self.results if self.results else [
            {
                "gt": r["label"],
                "pred": "",
                "gt_start": r["start"],
                "gt_end": r["end"],
                "score": None,
            }
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
            {
                "gt_start": r["start"],
                "gt_end": r["end"],
            }
            for r in self.gt_rows
        ]

        if idx < 0 or idx >= len(source):
            return

        s = source[idx]["gt_start"]
        e = source[idx]["gt_end"]

        pad = 0.3
        self.ax.set_xlim(max(0, s - pad), e + pad)
        self.redraw_plot()

    def play_audio(self):
        if self.y is None or self.sr is None:
            return
        log("Playing full audio", force=True)
        sd.stop()
        sd.play(self.y, self.sr)

    def play_selected_segment(self):
        if self.selected_index is None:
            QMessageBox.warning(self, "No Segment", "Select a segment from the side panel.")
            return

        source = self.results if self.results else [
            {
                "gt_start": r["start"],
                "gt_end": r["end"],
            }
            for r in self.gt_rows
        ]

        if self.selected_index < 0 or self.selected_index >= len(source):
            return

        s = source[self.selected_index]["gt_start"]
        e = source[self.selected_index]["gt_end"]
        self.play_time_range(s, e)

    def play_time_range(self, start_sec, end_sec):
        if self.y is None or self.sr is None:
            return
        s = int(start_sec * self.sr)
        e = int(end_sec * self.sr)
        log(f"Playing segment {start_sec:.2f}-{end_sec:.2f}s", force=True)
        sd.stop()
        sd.play(self.y[s:e], self.sr)

    def stop_audio(self):
        log("Stopping audio playback", force=True)
        sd.stop()

    def zoom(self, factor):
        if self.y is None or self.sr is None:
            return

        x0, x1 = self.ax.get_xlim()
        center = (x0 + x1) / 2
        width = (x1 - x0) * factor / 2

        new_x0 = center - width
        new_x1 = center + width

        total = len(self.y) / self.sr
        new_x0 = max(0, new_x0)
        new_x1 = min(total, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.canvas.draw_idle()

    def zoom_in_plot(self):
        self.zoom(0.7)

    def zoom_out_plot(self):
        self.zoom(1.4)

    def on_scroll(self, event):
        if self.y is None or self.sr is None:
            return

        x0, x1 = self.ax.get_xlim()
        mouse_x = event.xdata if event.xdata is not None else (x0 + x1) / 2
        scale = 0.7 if event.button == "up" else 1.4

        new_x0 = mouse_x - (mouse_x - x0) * scale
        new_x1 = mouse_x + (x1 - mouse_x) * scale

        total = len(self.y) / self.sr
        new_x0 = max(0, new_x0)
        new_x1 = min(total, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.canvas.draw_idle()

    def reset_plot_view(self):
        if self.y is None or self.sr is None:
            return
        total = len(self.y) / self.sr
        self.ax.set_xlim(0, total)
        self.canvas.draw_idle()

    def export_csv(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run prediction first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "results.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "gt_from_xlsx", "pred_from_model", "score",
                    "gt_start", "gt_end",
                    "pred_start", "pred_end"
                ])
                for r in self.results:
                    writer.writerow([
                        r["gt"], r["pred"], r["score"],
                        r["gt_start"], r["gt_end"],
                        r["pred_start"], r["pred_end"],
                    ])
            self.status_label.setText(f"Exported CSV: {path}")
            log(f"CSV exported: {path}", force=True)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", str(e))


if __name__ == "__main__":
    print("[LOG] Launching application", flush=True)
    app = QApplication(sys.argv)
    w = ProfessionalAnnotationTool()
    w.show()
    sys.exit(app.exec_())