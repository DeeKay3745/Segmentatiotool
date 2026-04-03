import sys
import json
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import pyqtgraph as pg
import torch

from datetime import datetime
from transformers import AutoProcessor, AutoModelForCTC
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QListWidget, QMessageBox
)
from PyQt5.QtGui import QTransform


MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


class AudioAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vowel / Character Annotation Tool (XLSR eSpeak)")
        self.setGeometry(100, 100, 1750, 1020)

        self.audio_path = None
        self.y = None
        self.sr = None

        self.regions = []
        self.gt_regions = []
        self.current_selected_region = None

        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.COLORS = {
            "auto": (80, 140, 255, 70),       # blue
            "manual": (255, 140, 40, 90),     # orange
            "ground_truth": (0, 255, 0, 60)   # green
        }

        self.vowels = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ"]

        self.characters = [
            "क", "ख", "ग", "घ", "ङ",
            "च", "छ", "ज", "झ",
            "ट", "ठ", "ड", "ढ", "ण",
            "त", "थ", "द", "ध", "न",
            "प", "फ", "ब", "भ", "म",
            "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"
        ]

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        top_bar = QHBoxLayout()

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self.load_audio)

        self.load_model_btn = QPushButton("Load XLSR Model")
        self.load_model_btn.clicked.connect(self.load_xlsr_model)

        self.auto_segment_btn = QPushButton("Auto Segment")
        self.auto_segment_btn.clicked.connect(self.run_segmentation)

        self.auto_label_btn = QPushButton("Auto Label (XLSR)")
        self.auto_label_btn.clicked.connect(self.run_xlsr_autolabel)

        self.load_gt_btn = QPushButton("Load Ground Truth (.xlsx)")
        self.load_gt_btn.clicked.connect(self.load_ground_truth_from_excel)

        self.play_audio_btn = QPushButton("Play Full Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)

        self.play_segment_btn = QPushButton("Play Selected Segment")
        self.play_segment_btn.clicked.connect(self.play_selected_segment)

        self.stop_audio_btn = QPushButton("Stop Audio")
        self.stop_audio_btn.clicked.connect(self.stop_audio)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)

        self.save_btn = QPushButton("Save Labels")
        self.save_btn.clicked.connect(self.save_labels)

        self.tier_selector = QComboBox()
        self.tier_selector.addItems(["Vowel", "Character"])
        self.tier_selector.currentIndexChanged.connect(self.update_label_options)

        self.label_selector = QComboBox()
        self.update_label_options()

        self.assign_label_btn = QPushButton("Assign Label")
        self.assign_label_btn.clicked.connect(self.assign_label_to_selected)

        self.metrics_label = QLabel("Metrics: Not computed")
        self.info_label = QLabel("Load audio to start.")
        self.legend_label = QLabel(
            "Legend: Auto = Blue | Manual = Orange | Ground Truth = Green | Guide = Yellow"
        )
        self.current_prediction_label = QLabel(
            "Current spoken label: None | Raw model output: None | Manual label: None"
        )

        widgets = [
            self.load_audio_btn,
            self.load_model_btn,
            self.auto_segment_btn,
            self.auto_label_btn,
            self.load_gt_btn,
            self.play_audio_btn,
            self.play_segment_btn,
            self.stop_audio_btn,
            self.zoom_in_btn,
            self.zoom_out_btn,
            self.reset_zoom_btn,
            self.save_btn,
            self.tier_selector,
            self.label_selector,
            self.assign_label_btn
        ]

        for w in widgets:
            top_bar.addWidget(w)

        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_plot.setMouseEnabled(x=True, y=False)
        self.waveform_plot.showGrid(x=True, y=True)
        self.waveform_plot.setBackground("#161b22")

        self.spec_plot = pg.PlotWidget(title="Spectrogram")
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_plot.showGrid(x=True, y=True)
        self.spec_plot.setBackground("#161b22")

        self.wave_cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('y', width=2))
        self.spec_cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.waveform_plot.addItem(self.wave_cursor)
        self.spec_plot.addItem(self.spec_cursor)
        self.wave_cursor.sigPositionChanged.connect(self.sync_cursor)

        self.segment_list = QListWidget()
        self.segment_list.itemClicked.connect(self.select_segment_from_list)

        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(self.metrics_label)
        main_layout.addWidget(self.legend_label)
        main_layout.addWidget(self.current_prediction_label)
        main_layout.addWidget(self.waveform_plot, stretch=2)
        main_layout.addWidget(self.spec_plot, stretch=2)
        main_layout.addWidget(QLabel("Segments"))
        main_layout.addWidget(self.segment_list, stretch=1)

        central.setLayout(main_layout)

    def sync_cursor(self):
        x = self.wave_cursor.value()
        self.spec_cursor.setValue(x)

    def update_label_options(self):
        tier = self.tier_selector.currentText()
        self.label_selector.clear()

        if tier == "Vowel":
            self.label_selector.addItems(self.vowels)
        else:
            self.label_selector.addItems(self.characters)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )

        if not file_path:
            self.info_label.setText("No audio selected.")
            return

        try:
            self.audio_path = file_path
            self.y, self.sr = librosa.load(file_path, sr=None, mono=True)

            self.clear_predictions()
            self.clear_ground_truth()
            self.segment_list.clear()

            self.show_waveform()
            self.show_spectrogram()
            self.restore_cursor()
            self.reset_zoom()

            duration = len(self.y) / self.sr
            self.info_label.setText(
                f"Loaded audio: {file_path} | SR: {self.sr} Hz | Duration: {duration:.3f}s"
            )
            self.metrics_label.setText("Metrics: Not computed")
            self.current_prediction_label.setText(
                "Current spoken label: None | Raw model output: None | Manual label: None"
            )

        except Exception as e:
            QMessageBox.critical(self, "Audio Load Error", str(e))

    def load_xlsr_model(self):
        if not MODEL_ID:
            QMessageBox.warning(self, "Model Error", "MODEL_ID is empty.")
            return

        try:
            self.info_label.setText(f"Loading model on {self.device}: {MODEL_ID}")
            QApplication.processEvents()

            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self.device)
            self.model.eval()

            self.info_label.setText(f"Loaded model: {MODEL_ID}")

        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    def restore_cursor(self):
        try:
            self.waveform_plot.addItem(self.wave_cursor)
        except Exception:
            pass
        try:
            self.spec_plot.addItem(self.spec_cursor)
        except Exception:
            pass

        self.wave_cursor.setValue(0.0)
        self.spec_cursor.setValue(0.0)

    def show_waveform(self):
        self.waveform_plot.clear()
        t = np.arange(len(self.y)) / self.sr
        self.waveform_plot.plot(t, self.y, pen='c')

    def show_spectrogram(self):
        self.spec_plot.clear()

        n_fft = 1024
        hop_length = 256

        S = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        img = pg.ImageItem(S_db)
        lut = pg.colormap.get("inferno").getLookupTable()
        img.setLookupTable(lut)
        img.setLevels((S_db.min(), S_db.max()))

        transform = QTransform()
        transform.scale(hop_length / self.sr, 1)
        img.setTransform(transform)

        self.spec_plot.addItem(img)
        self.spec_plot.invertY(True)

    def zoom_in(self):
        self.waveform_plot.getViewBox().scaleBy((0.8, 1.0))
        self.spec_plot.getViewBox().scaleBy((0.8, 1.0))

    def zoom_out(self):
        self.waveform_plot.getViewBox().scaleBy((1.25, 1.0))
        self.spec_plot.getViewBox().scaleBy((1.25, 1.0))

    def reset_zoom(self):
        if self.y is None or self.sr is None:
            return

        duration = len(self.y) / self.sr
        self.waveform_plot.setXRange(0, duration, padding=0)
        self.spec_plot.setXRange(0, duration, padding=0)

    def time_to_seconds(self, t):
        if pd.isna(t):
            return None

        t = str(t).strip()

        try:
            dt = datetime.strptime(t, "%H:%M:%S.%f")
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            try:
                dt = datetime.strptime(t, "%H:%M:%S")
                return dt.hour * 3600 + dt.minute * 60 + dt.second
            except ValueError:
                return None

    def segment_audio_dummy(self):
        duration = len(self.y) / self.sr

        if self.tier_selector.currentText() == "Vowel":
            n = len(self.vowels)
        else:
            n = 10

        step = duration / max(n, 1)
        segments = []
        for i in range(n):
            start = i * step
            end = (i + 1) * step
            segments.append((start, end))
        return segments

    def run_segmentation(self):
        if self.y is None:
            QMessageBox.warning(self, "No audio", "Please load audio first.")
            return

        try:
            self.clear_predictions()
            self.segment_list.clear()

            segments = self.segment_audio_dummy()
            tier = self.tier_selector.currentText()

            for idx, (start, end) in enumerate(segments):
                default_label = self.vowels[idx] if (tier == "Vowel" and idx < len(self.vowels)) else f"seg_{idx+1}"
                self.add_prediction_region(
                    start=start,
                    end=end,
                    mapped_label=default_label,
                    tier=tier,
                    source="auto",
                    raw_pred="",
                    manual_label=""
                )

            self.refresh_segment_list()
            self.info_label.setText("Auto segmentation completed.")
            self.update_metrics_display()

        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", str(e))

    def add_prediction_region(
        self,
        start,
        end,
        mapped_label="",
        tier="Vowel",
        source="manual",
        raw_pred="",
        manual_label=""
    ):
        region = pg.LinearRegionItem([start, end], movable=True)
        region.setZValue(10)

        spec_region = pg.LinearRegionItem([start, end], movable=False)
        spec_region.setZValue(10)

        brush = self.COLORS["auto"] if source == "auto" else self.COLORS["manual"]
        region.setBrush(brush)
        spec_region.setBrush(brush)

        self.waveform_plot.addItem(region)
        self.spec_plot.addItem(spec_region)

        segment_data = {
            "region": region,
            "spec_region": spec_region,
            "label": mapped_label,
            "raw_pred": raw_pred,
            "manual_label": manual_label,
            "tier": tier,
            "source": source
        }

        self.regions.append(segment_data)
        region.sigRegionChanged.connect(lambda: self.sync_region(segment_data))

    def sync_region(self, segment_data):
        start, end = segment_data["region"].getRegion()
        segment_data["spec_region"].setRegion((start, end))
        self.refresh_segment_list()
        self.update_metrics_display()

    def refresh_segment_list(self):
        self.segment_list.clear()

        for seg in self.regions:
            start, end = seg["region"].getRegion()
            raw_pred = seg.get("raw_pred", "")
            mapped = seg.get("label", "")
            manual = seg.get("manual_label", "")

            self.segment_list.addItem(
                f"Raw: {raw_pred} | Mapped: {mapped} | Manual: {manual} | "
                f"[{seg['tier']}] [{seg['source']}] : {start:.3f}s - {end:.3f}s"
            )

    def select_segment_from_list(self, item):
        row = self.segment_list.row(item)
        if 0 <= row < len(self.regions):
            self.current_selected_region = self.regions[row]
            start, end = self.current_selected_region["region"].getRegion()

            spoken_label = self.current_selected_region.get("label", "")
            raw_pred = self.current_selected_region.get("raw_pred", "")
            manual_label = self.current_selected_region.get("manual_label", "")

            self.current_prediction_label.setText(
                f"Current spoken label: {spoken_label} | "
                f"Raw model output: {raw_pred} | "
                f"Manual label: {manual_label}"
            )

            self.waveform_plot.setXRange(max(0, start - 0.5), end + 0.5)
            self.spec_plot.setXRange(max(0, start - 0.5), end + 0.5)
            self.wave_cursor.setValue(start)
            self.spec_cursor.setValue(start)

    def assign_label_to_selected(self):
        if self.current_selected_region is None:
            QMessageBox.warning(self, "No selection", "Select a segment first.")
            return

        chosen = self.label_selector.currentText()

        self.current_selected_region["manual_label"] = chosen
        self.current_selected_region["label"] = chosen
        self.current_selected_region["tier"] = self.tier_selector.currentText()
        self.current_selected_region["source"] = "manual"

        self.current_selected_region["region"].setBrush(self.COLORS["manual"])
        self.current_selected_region["spec_region"].setBrush(self.COLORS["manual"])

        raw_pred = self.current_selected_region.get("raw_pred", "")
        self.current_prediction_label.setText(
            f"Current spoken label: {chosen} | Raw model output: {raw_pred} | Manual label: {chosen}"
        )

        self.refresh_segment_list()
        self.update_metrics_display()

    def play_audio(self):
        if self.y is None:
            return
        sd.stop()
        sd.play(self.y, self.sr)

    def play_selected_segment(self):
        if self.current_selected_region is None or self.y is None:
            QMessageBox.warning(self, "No segment", "Select a segment first.")
            return

        start, end = self.current_selected_region["region"].getRegion()
        s = int(start * self.sr)
        e = int(end * self.sr)

        self.wave_cursor.setValue(start)
        self.spec_cursor.setValue(start)

        sd.stop()
        sd.play(self.y[s:e], self.sr)

    def stop_audio(self):
        sd.stop()

    def predict_segment_text(self, audio_chunk):
        if self.processor is None or self.model is None:
            return ""

        if audio_chunk is None or len(audio_chunk) == 0:
            return ""

        wav16 = librosa.resample(audio_chunk, orig_sr=self.sr, target_sr=16000)

        if wav16 is None or len(wav16) == 0:
            return ""

        inputs = self.processor(
            wav16,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values=input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].strip()
        return text

    def normalize_prediction_to_label(self, raw_text, tier):
        raw_text = raw_text.strip().lower()
        merged = raw_text.replace(" ", "")

        if tier == "Vowel":
            mapping = {
                "aa": "आ",
                "ii": "ई",
                "uu": "ऊ",
                "ai": "ऐ",
                "au": "औ",
                "ri": "ऋ",
                "a": "अ",
                "i": "इ",
                "u": "उ",
                "e": "ए",
                "o": "ओ",
            }

            for k, v in mapping.items():
                if k in merged:
                    return v

            return raw_text if raw_text else "?"

        char_map = {
            "ksha": "क्ष",
            "chha": "छ",
            "kha": "ख",
            "gha": "घ",
            "jha": "झ",
            "tha": "थ",
            "dha": "ध",
            "pha": "फ",
            "bha": "भ",
            "sha": "श",
            "tra": "त्र",
            "gna": "ज्ञ",
            "ka": "क",
            "ga": "ग",
            "cha": "च",
            "ja": "ज",
            "ta": "त",
            "da": "द",
            "na": "न",
            "pa": "प",
            "ba": "ब",
            "ma": "म",
            "ya": "य",
            "ra": "र",
            "la": "ल",
            "va": "व",
            "sa": "स",
            "ha": "ह"
        }

        for k, v in char_map.items():
            if k in merged:
                return v

        return raw_text if raw_text else "?"

    def run_xlsr_autolabel(self):
        if self.y is None or self.sr is None:
            QMessageBox.warning(self, "No audio", "Load audio first.")
            return

        if self.processor is None or self.model is None:
            QMessageBox.warning(self, "No model", "Load XLSR model first.")
            return

        if not self.regions:
            QMessageBox.warning(self, "No segments", "Run Auto Segment first.")
            return

        try:
            tier = self.tier_selector.currentText()
            self.info_label.setText("Running XLSR auto-labeling...")
            QApplication.processEvents()

            for seg in self.regions:
                start, end = seg["region"].getRegion()
                s = int(start * self.sr)
                e = int(end * self.sr)

                if e <= s:
                    seg["raw_pred"] = ""
                    seg["label"] = "?"
                    seg["manual_label"] = ""
                    continue

                chunk = self.y[s:e]
                raw_pred = self.predict_segment_text(chunk)
                mapped_pred = self.normalize_prediction_to_label(raw_pred, tier)

                seg["raw_pred"] = raw_pred
                seg["label"] = mapped_pred
                seg["manual_label"] = ""
                seg["tier"] = tier
                seg["source"] = "auto"

                seg["region"].setBrush(self.COLORS["auto"])
                seg["spec_region"].setBrush(self.COLORS["auto"])

            self.refresh_segment_list()
            self.update_metrics_display()
            self.info_label.setText("XLSR auto-labeling completed.")

        except Exception as e:
            QMessageBox.critical(self, "Auto Label Error", str(e))

    def load_ground_truth_from_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Ground Truth Excel",
            "",
            "Excel Files (*.xlsx)"
        )

        if not file_path:
            return

        if self.y is None:
            QMessageBox.warning(self, "No audio", "Load audio first.")
            return

        try:
            self.clear_ground_truth()

            df = pd.read_excel(
                file_path,
                sheet_name=0,
                header=None,
                engine="openpyxl"
            )

            loaded_count = 0

            for _, row in df.iterrows():
                try:
                    label = str(row[0]).strip()
                    start = self.time_to_seconds(row[1])
                    end = self.time_to_seconds(row[2])
                except Exception:
                    continue

                if not label or start is None or end is None or end <= start:
                    continue

                gt_region = pg.LinearRegionItem([start, end], movable=False)
                gt_region.setBrush(self.COLORS["ground_truth"])
                gt_region.setZValue(5)
                self.waveform_plot.addItem(gt_region)

                gt_spec_region = pg.LinearRegionItem([start, end], movable=False)
                gt_spec_region.setBrush(self.COLORS["ground_truth"])
                gt_spec_region.setZValue(5)
                self.spec_plot.addItem(gt_spec_region)

                self.gt_regions.append({
                    "region": gt_region,
                    "spec_region": gt_spec_region,
                    "label": label,
                    "tier": "GroundTruth",
                    "source": "ground_truth"
                })

                loaded_count += 1

            self.info_label.setText(f"Loaded {loaded_count} GT segments from first sheet")
            self.update_metrics_display()

        except Exception as e:
            QMessageBox.critical(self, "Excel Load Error", str(e))

    def compute_metrics(self):
        if not self.regions or not self.gt_regions:
            return None

        n = min(len(self.regions), len(self.gt_regions))
        if n == 0:
            return None

        ious = []
        boundary_errors = []
        label_matches = 0

        for i in range(n):
            ps, pe = self.regions[i]["region"].getRegion()
            gs, ge = self.gt_regions[i]["region"].getRegion()

            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union > 0 else 0.0
            ious.append(iou)

            boundary_error = (abs(ps - gs) + abs(pe - ge)) / 2.0
            boundary_errors.append(boundary_error)

            pred_label = str(self.regions[i]["label"]).strip()
            gt_label = str(self.gt_regions[i]["label"]).strip()
            if pred_label == gt_label:
                label_matches += 1

        return {
            "num_compared_segments": n,
            "avg_iou": float(np.mean(ious)),
            "avg_boundary_error_sec": float(np.mean(boundary_errors)),
            "label_accuracy": float(label_matches / n)
        }

    def update_metrics_display(self):
        metrics = self.compute_metrics()
        if metrics is None:
            self.metrics_label.setText("Metrics: Not computed")
            return

        self.metrics_label.setText(
            f"Metrics | Compared: {metrics['num_compared_segments']} | "
            f"IoU: {metrics['avg_iou']:.4f} | "
            f"Boundary Error: {metrics['avg_boundary_error_sec']:.4f}s | "
            f"Label Accuracy: {metrics['label_accuracy']:.4f}"
        )

    def save_labels(self):
        if not self.regions:
            QMessageBox.warning(self, "No labels", "No predicted segments to save.")
            return

        try:
            output = []
            for seg in self.regions:
                start, end = seg["region"].getRegion()
                output.append({
                    "raw_prediction": seg.get("raw_pred", ""),
                    "mapped_label": seg.get("label", ""),
                    "manual_label": seg.get("manual_label", ""),
                    "tier": seg.get("tier", ""),
                    "source": seg.get("source", ""),
                    "start": round(start, 4),
                    "end": round(end, 4)
                })

            metrics = self.compute_metrics()
            save_obj = {
                "audio_file": self.audio_path,
                "annotations": output,
                "metrics": metrics
            }

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save JSON",
                "output.json",
                "JSON Files (*.json)"
            )

            if not save_path:
                return

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_obj, f, ensure_ascii=False, indent=2)

            self.info_label.setText(f"Saved labels to {save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def clear_predictions(self):
        for seg in self.regions:
            self.waveform_plot.removeItem(seg["region"])
            self.spec_plot.removeItem(seg["spec_region"])
        self.regions = []
        self.current_selected_region = None

    def clear_ground_truth(self):
        for seg in self.gt_regions:
            self.waveform_plot.removeItem(seg["region"])
            self.spec_plot.removeItem(seg["spec_region"])
        self.gt_regions = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AudioAnnotator()
    win.show()
    sys.exit(app.exec_())