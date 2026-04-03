import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import nemo.collections.asr as nemo_asr


MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
LANG_OPTIONS = [("Hindi", "hi"), ("Gujarati", "gu"), ("Marathi", "mr")]


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


class OneWaveformAI4Bharat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI4Bharat One-Waveform ASR Comparison")
        self.resize(1400, 800)

        self.audio_path = None
        self.xlsx_path = None
        self.y = None
        self.sr = None
        self.gt_rows = []
        self.results = []
        self.model = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        controls = QHBoxLayout()

        self.lang_combo = QComboBox()
        for label, code in LANG_OPTIONS:
            self.lang_combo.addItem(label, code)

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)

        self.load_gt_btn = QPushButton("Load Ground Truth (.xlsx)")
        self.load_gt_btn.clicked.connect(self.load_gt)

        self.load_model_btn = QPushButton("Load AI4Bharat Model")
        self.load_model_btn.clicked.connect(self.load_model)

        self.run_btn = QPushButton("Run One Result")
        self.run_btn.clicked.connect(self.run_inference)

        controls.addWidget(QLabel("Language"))
        controls.addWidget(self.lang_combo)
        controls.addWidget(self.load_audio_btn)
        controls.addWidget(self.load_gt_btn)
        controls.addWidget(self.load_model_btn)
        controls.addWidget(self.run_btn)

        self.status_label = QLabel("Choose language, then load audio and ground truth.")
        layout.addLayout(controls)
        layout.addWidget(self.status_label)

        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        if not path:
            return
        try:
            self.y, self.sr = librosa.load(path, sr=None, mono=True)
            self.audio_path = path
            self.status_label.setText(f"Loaded audio: {os.path.basename(path)}")
            self.plot_waveform_only()
        except Exception as e:
            QMessageBox.critical(self, "Audio Load Error", str(e))

    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Truth", "", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
            rows = []
            for _, r in df.iterrows():
                label = str(r[0]).strip()
                start = time_to_sec(r[1])
                end = time_to_sec(r[2])
                if not label or start is None or end is None or end <= start:
                    continue
                rows.append({"label": label, "start": start, "end": end})
            self.gt_rows = rows
            self.xlsx_path = path
            self.status_label.setText(f"Loaded GT rows: {len(rows)}")
            self.plot_waveform_only()
        except Exception as e:
            QMessageBox.critical(self, "GT Load Error", str(e))

    def load_model(self):
        try:
            self.status_label.setText("Loading AI4Bharat model...")
            QApplication.processEvents()
            self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
            self.model.eval()
            self.status_label.setText("Model loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    def transcribe_segment(self, wav_path, lang_code):
        if hasattr(self.model, "cur_decoder"):
            self.model.cur_decoder = "ctc"
        out = self.model.transcribe(
            [wav_path],
            batch_size=1,
            logprobs=False,
            language_id=lang_code
        )
        if isinstance(out, list) and out:
            return str(out[0]).strip()
        return str(out).strip()

    def run_inference(self):
        if self.audio_path is None or self.y is None:
            QMessageBox.warning(self, "Missing Audio", "Load audio first.")
            return
        if not self.gt_rows:
            QMessageBox.warning(self, "Missing GT", "Load ground truth first.")
            return
        if self.model is None:
            QMessageBox.warning(self, "Missing Model", "Load the AI4Bharat model first.")
            return

        lang_code = self.lang_combo.currentData()
        results = []

        try:
            self.status_label.setText(f"Running ASR for language: {lang_code}")
            QApplication.processEvents()

            with tempfile.TemporaryDirectory() as td:
                for i, row in enumerate(self.gt_rows):
                    s = int(row["start"] * self.sr)
                    e = int(row["end"] * self.sr)
                    chunk = self.y[s:e]

                    if len(chunk) == 0:
                        pred = ""
                    else:
                        seg_path = os.path.join(td, f"seg_{i}.wav")
                        chunk16 = librosa.resample(chunk, orig_sr=self.sr, target_sr=16000)
                        sf.write(seg_path, chunk16, 16000)
                        try:
                            pred = self.transcribe_segment(seg_path, lang_code)
                        except Exception:
                            pred = ""

                    results.append({
                        "gt": row["label"],
                        "pred": pred,
                        "start": row["start"],
                        "end": row["end"],
                        "score": simple_match(pred, row["label"])
                    })

            self.results = results
            mean_score = np.mean([r["score"] for r in results]) if results else 0.0
            self.status_label.setText(
                f"Done. Language={lang_code} | Segments={len(results)} | Mean match={mean_score:.3f}"
            )
            self.plot_result()

        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))

    def plot_waveform_only(self):
        self.ax.clear()
        if self.y is None or self.sr is None:
            self.canvas.draw()
            return

        t = np.arange(len(self.y)) / self.sr
        self.ax.plot(t, self.y, linewidth=0.6)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        ymax = np.max(np.abs(self.y)) if len(self.y) else 1.0
        if ymax == 0:
            ymax = 1.0

        for row in self.gt_rows:
            s, e = row["start"], row["end"]
            self.ax.axvspan(s, e, alpha=0.16, color="green")
            self.ax.axvline(s, linewidth=1.0, color="green")
            self.ax.axvline(e, linewidth=1.0, color="green")
            mid = (s + e) / 2
            self.ax.text(mid, ymax * 0.85, f"GT: {row['label']}", ha="center", va="bottom", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_result(self):
        self.ax.clear()

        t = np.arange(len(self.y)) / self.sr
        self.ax.plot(t, self.y, linewidth=0.6)

        ymax = np.max(np.abs(self.y)) if len(self.y) else 1.0
        if ymax == 0:
            ymax = 1.0

        for r in self.results:
            s, e = r["start"], r["end"]
            self.ax.axvspan(s, e, alpha=0.18, color="green")
            self.ax.axvline(s, linewidth=1.0, color="green")
            self.ax.axvline(e, linewidth=1.0, color="green")
            mid = (s + e) / 2
            txt = f"GT: {r['gt']}\nPred: {r['pred']}\n[{s:.2f}-{e:.2f}]"
            self.ax.text(mid, ymax * 0.82, txt, ha="center", va="bottom", fontsize=8)

        lang_code = self.lang_combo.currentData()
        mean_score = np.mean([r["score"] for r in self.results]) if self.results else 0.0
        self.ax.set_title(f"One waveform result | language={lang_code} | mean match={mean_score:.3f}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OneWaveformAI4Bharat()
    w.show()
    sys.exit(app.exec_())