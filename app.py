#!/usr/bin/env python3
"""
Indic Speech Annotation & Segmentation Tool — Auto-Mode Edition
================================================================
Auto-detects segment types and applies the correct prediction mode:
  • Single character (અ, आ, க) → First Character mode
  • Single word (ગુજરાત, भारत)  → First Word mode
  • S1-S5 / Paragraph labels     → Full Transcript + compare against txt file
  • Multi-word text              → Full Transcript mode

Transcript files: Place .txt files in 'transcripts/' folder next to app.py.
  transcripts/gu.txt, transcripts/hi.txt, transcripts/mr.txt, etc.
  Format:  [S1]\\nExpected sentence text\\n[S2]\\n...\\n[Paragraph]\\n...

Install:
    pip install transformers torchaudio torch soundfile scipy \\
                numpy pandas matplotlib PyQt5 sounddevice openpyxl
"""

import os, sys, csv, re, time, traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

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

# ── Optional imports ─────────────────────────────────────────────────────
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
    import torch, torchaudio
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
APP_DIR = Path(__file__).parent.resolve()
TRANSCRIPTS_DIR = APP_DIR / "transcripts"

# ── Models ───────────────────────────────────────────────────────────────
MODELS = []
if _HF:
    MODELS += [
        ("★ IndicConformer-600M (BEST for Indic)",
         "indicconformer", "ai4bharat/indic-conformer-600m-multilingual",
         "AI4Bharat – all 22 Indian languages, correct script output"),
    ]
if _FW:
    MODELS += [
        ("faster-whisper tiny",    "fw", "tiny",     ""),
        ("faster-whisper base",    "fw", "base",     ""),
        ("faster-whisper small",   "fw", "small",    ""),
        ("faster-whisper medium",  "fw", "medium",   ""),
        ("faster-whisper large-v3","fw", "large-v3", ""),
    ]
if _HF:
    MODELS += [
        ("HF Whisper small",    "hf_whisper", "openai/whisper-small",    ""),
        ("HF Whisper medium",   "hf_whisper", "openai/whisper-medium",   ""),
        ("HF Whisper large-v3", "hf_whisper", "openai/whisper-large-v3", ""),
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

# ── Unicode ──────────────────────────────────────────────────────────────
_INDIC = (
    r"\u0900-\u097F\uA8E0-\uA8FF"  # Devanagari
    r"\u0A80-\u0AFF"                 # Gujarati
    r"\u0980-\u09FF"                 # Bengali
    r"\u0B00-\u0B7F"                 # Odia
    r"\u0B80-\u0BFF"                 # Tamil
    r"\u0C00-\u0C7F"                 # Telugu
    r"\u0C80-\u0CFF"                 # Kannada
    r"\u0D00-\u0D7F"                 # Malayalam
    r"\u0A00-\u0A7F"                 # Gurmukhi
)
_RE_CLEAN  = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM   = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA  = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)

# Sequence label pattern
_RE_SEQ = re.compile(r"^(?:s|p|sent|para|sentence|paragraph|section|seg)\s*\d*$", re.I)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  TRANSCRIPT STORE — loads S1-S5 / Paragraph text from .txt files
# ══════════════════════════════════════════════════════════════════════════
class TranscriptStore:
    """
    Loads and manages transcript reference files.
    File format (transcripts/gu.txt):

        [S1]
        Expected sentence text for S1
        [S2]
        Expected sentence text for S2
        ...
        [Paragraph]
        Full paragraph text here
    """

    def __init__(self):
        self.data = {}       # {"s1": "text...", "s2": "text...", "paragraph": "text..."}
        self.lang_code = None
        self.file_path = None

    def load(self, lang_iso: str) -> tuple:
        """Load transcript file for given language. Returns (success, message)."""
        self.data = {}
        self.lang_code = lang_iso
        self.file_path = TRANSCRIPTS_DIR / f"{lang_iso}.txt"

        if not self.file_path.exists():
            return False, f"No transcript file: {self.file_path}"

        try:
            text = self.file_path.read_text(encoding="utf-8")
            current_key = None
            current_lines = []

            for line in text.splitlines():
                line_stripped = line.strip()

                # Skip comments and empty lines (when not inside a section)
                if line_stripped.startswith("#"):
                    continue

                # Check for section header: [S1], [S2], [Paragraph], etc.
                m = re.match(r"^\[(.+?)\]$", line_stripped)
                if m:
                    # Save previous section
                    if current_key is not None:
                        self.data[current_key] = "\n".join(current_lines).strip()
                    current_key = m.group(1).strip().lower()
                    current_lines = []
                elif current_key is not None:
                    current_lines.append(line.rstrip())

            # Save last section
            if current_key is not None:
                self.data[current_key] = "\n".join(current_lines).strip()

            n = len(self.data)
            keys = ", ".join(sorted(self.data.keys()))
            log(f"Transcript loaded: {self.file_path.name} → {n} sections: {keys}")
            return True, f"Loaded {n} sections from {self.file_path.name}"

        except Exception as e:
            return False, f"Error reading {self.file_path}: {e}"

    def get_reference(self, label: str) -> str:
        """Get expected transcript text for a sequence label like S1, Paragraph, etc."""
        key = label.strip().lower()
        return self.data.get(key, "")

    @property
    def is_loaded(self):
        return bool(self.data)

    @property
    def sections(self):
        return list(self.data.keys())


TRANSCRIPTS = TranscriptStore()


# ══════════════════════════════════════════════════════════════════════════
#  SEGMENT CLASSIFIER — auto-detects char / word / sentence
# ══════════════════════════════════════════════════════════════════════════

def is_sequence_label(label: str) -> bool:
    """Check if label is a sequence marker (S1, Paragraph, etc.)."""
    label = str(label).strip()
    if not label:
        return False
    if _RE_SEQ.match(label):
        return True
    if label.lower() in {"paragraph", "para", "passage"}:
        return True
    return False


def classify_segment(label: str) -> dict:
    """
    Auto-classify a segment label and return processing instructions.

    Returns dict with:
        type:       "char" | "word" | "sentence" | "paragraph"
        mode:       "char" | "word" | "full"  (passed to clean_text)
        compare_to: what to compare prediction against (label itself or transcript text)
        is_seq:     True if this is a sequence label (S1, Paragraph, etc.)

    Classification logic:
        1. Label matches S1-S5/Paragraph → "sentence"/"paragraph", compare to txt file
        2. Label is 1 character → "char"
        3. Label is 1 word (no spaces) → "word"
        4. Label has multiple words → "sentence" (full transcript)
    """
    label = str(label).strip()
    label_lower = label.lower()

    # ── 1. Sequence labels: S1, S2, ..., Paragraph ──────────────────────
    if is_sequence_label(label):
        ref = TRANSCRIPTS.get_reference(label)
        seg_type = "paragraph" if "para" in label_lower or "passage" in label_lower else "sentence"
        return {
            "type": seg_type,
            "mode": "full",
            "compare_to": ref if ref else None,  # None = no accuracy comparison
            "is_seq": True,
            "ref_key": label,
        }

    # ── 2. Single character (vowel or consonant) ────────────────────────
    # Strip spaces and check if what remains is a single character
    chars_only = re.sub(r"\s+", "", label)
    if len(chars_only) == 1:
        return {
            "type": "char",
            "mode": "char",
            "compare_to": label,
            "is_seq": False,
        }

    # ── 3. Single word (no spaces) ──────────────────────────────────────
    words = label.split()
    if len(words) == 1:
        return {
            "type": "word",
            "mode": "word",
            "compare_to": label,
            "is_seq": False,
        }

    # ── 4. Multi-word → sentence ────────────────────────────────────────
    return {
        "type": "sentence",
        "mode": "full",
        "compare_to": label,
        "is_seq": False,
    }


# ══════════════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
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
        except Exception: pass
    import librosa
    return librosa.load(path, sr=None, mono=True)[0].astype(np.float32), librosa.load(path, sr=None, mono=True)[1]

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
        except ValueError: continue
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
                if fn in files: return fm.FontProperties(fname=os.path.join(root, fn))
    installed = {f.name for f in fm.fontManager.ttflist}
    for fam in families:
        if fam in installed: return fm.FontProperties(family=fam)
    return None

def vad_segment(y, sr, silence_ms=300, thresh_db=-40, min_ms=200, max_ms=10000):
    frame_len = int(sr * 0.025); hop = int(sr * 0.010)
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
        self.backend = None
        self.engine_key = None
        self.processor = None
        self.device = None
        self._cache_key = None

    @property
    def is_loaded(self): return self.backend is not None

    def load(self, engine_key, model_id):
        ck = (engine_key, model_id)
        if self._cache_key == ck and self.backend is not None:
            return True, f"Already loaded: {model_id}", 0.0
        t0 = time.time()
        self.backend = self.processor = None
        self.engine_key = engine_key
        try:
            if engine_key == "indicconformer": self._load_ic(model_id)
            elif engine_key == "fw": self._load_fw(model_id)
            elif engine_key == "hf_whisper": self._load_hf(model_id)
            else: return False, f"Unknown engine: {engine_key}", 0.0
            self._cache_key = ck
            return True, f"Loaded {model_id} in {time.time()-t0:.1f}s", time.time()-t0
        except Exception as e:
            traceback.print_exc(); self.backend = None
            return False, str(e), time.time()-t0

    def _load_ic(self, mid):
        log(f"Loading IndicConformer: {mid}")
        self.device = "cuda" if (_HF and torch.cuda.is_available()) else "cpu"
        self.backend = AutoModel.from_pretrained(mid, trust_remote_code=True)
        if hasattr(self.backend, 'to'):
            try: self.backend = self.backend.to(self.device)
            except: self.device = "cpu"

    def _load_fw(self, size):
        has_cuda = _HF and torch.cuda.is_available() if _HF else False
        self.backend = _FWModel(size,
            device="cuda" if has_cuda else "cpu",
            compute_type="float16" if has_cuda else "int8",
            cpu_threads=CPU_THR)

    def _load_hf(self, mid):
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if torch.cuda.is_available() else torch.float32
        self.backend = AutoModelForSpeechSeq2Seq.from_pretrained(
            mid, torch_dtype=dt, low_cpu_mem_usage=True, use_safetensors=True).to(dev)
        self.processor = AutoProcessor.from_pretrained(mid)
        self.device = dev

    def transcribe(self, audio_16k, lang_iso, lang_name, mode="full"):
        if self.backend is None or audio_16k is None or len(audio_16k) == 0: return ""
        min_s = int(0.1 * SR16)
        if len(audio_16k) < min_s:
            audio_16k = np.pad(audio_16k, (0, min_s - len(audio_16k)))
        audio_16k = audio_16k.astype(np.float32)
        if self.engine_key == "indicconformer": return self._tx_ic(audio_16k, lang_iso, mode)
        elif self.engine_key == "fw": return self._tx_fw(audio_16k, lang_iso, mode)
        elif self.engine_key == "hf_whisper": return self._tx_hf(audio_16k, lang_name, mode)
        return ""

    def _tx_ic(self, audio, lang_iso, mode):
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if self.device and self.device != "cpu": wav = wav.to(self.device)
        with torch.inference_mode():
            text = self.backend(wav, lang_iso, "ctc")
        if isinstance(text, (list, tuple)): text = text[0] if text else ""
        return clean_text(str(text), mode)

    def _tx_fw(self, audio, lang_iso, mode):
        seg_gen, _ = self.backend.transcribe(audio, language=lang_iso, task="transcribe",
            beam_size=1, best_of=1, vad_filter=False, without_timestamps=True,
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in list(seg_gen) if s.text.strip())
        return clean_text(text, mode)

    def _tx_hf(self, audio, lang_name, mode):
        inp = self.processor(audio, sampling_rate=SR16, return_tensors="pt", return_attention_mask=True)
        feat = inp.input_features.to(self.device)
        mask = getattr(inp, "attention_mask", None)
        if mask is not None: mask = mask.to(self.device)
        gk = {"input_features": feat, "task": "transcribe", "language": lang_name, "return_timestamps": False}
        if mask is not None: gk["attention_mask"] = mask
        with torch.inference_mode(): ids = self.backend.generate(**gk)
        return clean_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0], mode)

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
        except Exception as e: traceback.print_exc(); self.error.emit(str(e))

class ModelLoadThread(QThread):
    done = pyqtSignal(bool, str, float)
    error = pyqtSignal(str)
    def __init__(self, ek, mid): super().__init__(); self.ek, self.mid = ek, mid
    def run(self):
        try:
            ok, msg, el = ENGINE.load(self.ek, self.mid)
            self.done.emit(ok, msg, el)
        except Exception as e: traceback.print_exc(); self.error.emit(str(e))


class PredictionThread(QThread):
    """
    Auto-mode prediction: classifies each segment and applies correct mode.
    """
    seg_result = pyqtSignal(int, str, float, str)  # idx, pred, acc, seg_type
    progress = pyqtSignal(int, int)
    done = pyqtSignal(list, float, float)
    error = pyqtSignal(str)

    def __init__(self, y16, segments, lang_iso, lang_name):
        super().__init__()
        self.y16, self.segments = y16, segments
        self.lang_iso, self.lang_name = lang_iso, lang_name
        self._stop = False

    def request_stop(self): self._stop = True

    def run(self):
        try:
            t0 = time.time()
            total = len(self.segments)
            results = []
            log(f"Auto-mode prediction: {total} segs, lang={self.lang_iso}")

            for i, seg in enumerate(self.segments):
                if self._stop: log("Stopped"); break

                # ── Auto-classify this segment ────────────────────────────
                info = classify_segment(seg["label"])
                seg_type = info["type"]       # "char" / "word" / "sentence" / "paragraph"
                mode = info["mode"]           # "char" / "word" / "full"
                compare_to = info["compare_to"]
                is_seq = info["is_seq"]

                # ── Transcribe ────────────────────────────────────────────
                s = max(0, int(seg["start"] * SR16))
                e = min(len(self.y16), int(seg["end"] * SR16))
                chunk = self.y16[s:e]

                pred = ENGINE.transcribe(chunk, self.lang_iso, self.lang_name, mode)

                # ── Compute accuracy ──────────────────────────────────────
                if is_seq and compare_to:
                    # Sequence label WITH reference text → compare against txt file
                    acc = accuracy(pred, compare_to)
                elif is_seq and not compare_to:
                    # Sequence label WITHOUT reference → no accuracy (show transcript only)
                    acc = -1.0
                else:
                    # Regular label → compare against the label itself
                    acc = accuracy(pred, seg["label"])

                results.append({
                    "i": i, "gt": seg["label"], "pred": pred,
                    "start": seg["start"], "end": seg["end"],
                    "acc": acc, "type": seg_type, "mode": mode,
                    "is_seq": is_seq, "compare_to": compare_to or seg["label"],
                })

                self.seg_result.emit(i, pred, acc, seg_type)
                self.progress.emit(i + 1, total)

                acc_str = f"{acc:.3f}" if acc >= 0 else "N/A"
                if (i + 1) % 5 == 0 or (i + 1) == total:
                    log(f"  [{i+1}/{total}] [{seg_type:>9}] GT='{seg['label']}' "
                        f"→ PRED='{pred[:40]}' acc={acc_str}")

            elapsed = time.time() - t0
            scored = [r["acc"] for r in results if r["acc"] >= 0]
            mean_acc = np.mean(scored) if scored else 0.0
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
        self.setWindowTitle("Indic Speech Annotation Tool — Auto Mode")
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
        self.cb_lang.currentIndexChanged.connect(self._on_lang_changed)

        self.cb_model = QComboBox()
        for d, ek, mid, note in MODELS:
            self.cb_model.addItem(d, (ek, mid))

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
            "QPushButton:disabled{color:#9ca3af;background:#f3f4f6;border:1px solid #d1d5db}")

        # Auto-mode indicator (replaces the old mode dropdown)
        self.lbl_auto = QLabel("🤖 Auto-Mode")
        self.lbl_auto.setStyleSheet(
            "padding:4px 10px;background:#dbeafe;border:1px solid #93c5fd;"
            "border-radius:4px;font-weight:bold;color:#1e40af;font-size:12px")
        self.lbl_auto.setToolTip(
            "Auto-Mode: The tool automatically detects segment types:\n"
            "• Single character (અ, आ) → First Character\n"
            "• Single word (ગુજરાત) → First Word\n"
            "• S1-S5 / Paragraph → Full Transcript (compared to txt file)\n"
            "• Multi-word text → Full Transcript")

        self.lbl_transcript = QLabel("")
        self.lbl_transcript.setStyleSheet("color:#6b7280;font-size:11px")

        for w in [QLabel("Language:"), self.cb_lang,
                  QLabel("Model:"), self.cb_model,
                  self.lbl_auto,
                  self.btn_audio, self.btn_gt, self.btn_model]:
            r1.addWidget(w)
        r1.addWidget(self.lbl_transcript)
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
            b = QPushButton(name); b.clicked.connect(fn)
            if style: b.setStyleSheet(f"QPushButton{{{style}}}")
            r2.addWidget(b); self._btns[name] = b
        self._btns["⏹ Stop"].setEnabled(False)
        r2.addStretch()

        # ── Status ────────────────────────────────────────────────────────
        sr_row = QHBoxLayout()
        self.lbl_status = QLabel("Step 1: Select language → Step 2: Load Model")
        self.lbl_status.setStyleSheet("padding:4px;font-size:13px")
        self.lbl_speed = QLabel("")
        self.lbl_speed.setStyleSheet("color:#6b7280;font-size:12px")
        sr_row.addWidget(self.lbl_status, 1); sr_row.addWidget(self.lbl_speed)

        self.pbar = QProgressBar()
        self.pbar.setRange(0,100); self.pbar.setValue(0); self.pbar.setFixedHeight(18)
        self.pbar.setStyleSheet(
            "QProgressBar{border:1px solid #ccc;border-radius:4px;background:#f3f4f6;"
            "text-align:center;font-size:11px}"
            "QProgressBar::chunk{background:#3b82f6;border-radius:4px}")

        # ── Canvas ────────────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # ── Right panel ───────────────────────────────────────────────────
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_seg_clicked)
        self.seg_list.setStyleSheet(
            "QListWidget{background:#fff;border:1px solid #d0d0d0;"
            "font-family:Consolas,Courier New,monospace;font-size:11px}"
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

        # Transcript info tab
        tx_box = QGroupBox("Transcript Files")
        tl = QVBoxLayout(tx_box)
        self.lbl_tx_info = QLabel(
            f"<b>Transcript folder:</b><br><code>{TRANSCRIPTS_DIR}</code><br><br>"
            "<b>How to use:</b><br>"
            "1. Create a .txt file for each language:<br>"
            "   <code>transcripts/gu.txt</code>, <code>hi.txt</code>, <code>mr.txt</code>, etc.<br><br>"
            "2. Format:<br>"
            "<code>[S1]<br>Expected sentence text<br>[S2]<br>Another sentence<br>"
            "[Paragraph]<br>Full paragraph text</code><br><br>"
            "3. When you select a language, the tool auto-loads the matching file.<br><br>"
            "<b>Auto-Mode classification:</b><br>"
            "• <span style='color:#7c3aed'>Ⓒ</span> Single char (અ, आ) → character mode<br>"
            "• <span style='color:#0284c7'>Ⓦ</span> Single word (ગુજરાત) → word mode<br>"
            "• <span style='color:#16a34a'>Ⓢ</span> S1-S5 → full transcript, compared to txt<br>"
            "• <span style='color:#ea580c'>Ⓟ</span> Paragraph → full transcript, compared to txt<br>"
            "• <span style='color:#374151'>Ⓕ</span> Multi-word → full transcript mode"
        )
        self.lbl_tx_info.setTextFormat(Qt.RichText)
        self.lbl_tx_info.setWordWrap(True)
        self.lbl_tx_info.setStyleSheet("font-size:11px")
        tl.addWidget(self.lbl_tx_info); tl.addStretch()

        tabs = QTabWidget()
        seg_tab = QWidget(); sl = QVBoxLayout(seg_tab)
        legend = QLabel(
            '<span style="color:#7c3aed;font-weight:bold">Ⓒ Char</span>  '
            '<span style="color:#0284c7;font-weight:bold">Ⓦ Word</span>  '
            '<span style="color:#16a34a;font-weight:bold">Ⓢ Sentence</span>  '
            '<span style="color:#ea580c;font-weight:bold">Ⓟ Paragraph</span>  |  '
            '<span style="color:#16a34a">✓≥95%</span> '
            '<span style="color:#d97706">~50-95%</span> '
            '<span style="color:#dc2626">✗&lt;50%</span> '
            '<span style="color:#6366f1">📝no-ref</span>')
        legend.setTextFormat(Qt.RichText)
        sl.addWidget(legend); sl.addWidget(self.seg_list)
        tabs.addTab(seg_tab, "Segments")
        tabs.addTab(vad_box, "VAD")
        tabs.addTab(tx_box, "Transcripts")

        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(self.toolbar); ll.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(tabs)
        splitter.setSizes([1250, 400])

        L.addLayout(r1); L.addLayout(r2)
        L.addLayout(sr_row); L.addWidget(self.pbar); L.addWidget(splitter)

        # Load transcript for default language
        self._on_lang_changed()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _set_busy(self, busy, msg=""):
        en = not busy
        for n, b in self._btns.items():
            b.setEnabled(busy if n == "⏹ Stop" else en)
        for w in [self.btn_audio, self.btn_gt, self.btn_model, self.cb_lang, self.cb_model]:
            w.setEnabled(en)
        if msg: self.lbl_status.setText(msg)

    def _build_wave(self):
        if self.y is None: self.wave_t = self.wave_y = None; return
        step = max(1, len(self.y) // 30000)
        self.wave_y = self.y[::step]
        self.wave_t = np.arange(len(self.wave_y)) * (step / self.sr)

    def _play(self, y, sr):
        if y is not None and sr: sd.stop(); sd.play(y, sr)

    # ── Language changed → load transcript ────────────────────────────────
    def _on_lang_changed(self):
        iso, name = self.cb_lang.currentData()
        ok, msg = TRANSCRIPTS.load(iso)
        if ok:
            secs = ", ".join(TRANSCRIPTS.sections)
            self.lbl_transcript.setText(f"📄 {iso}.txt: {secs}")
            self.lbl_transcript.setStyleSheet("color:#16a34a;font-size:11px;font-weight:bold")
        else:
            self.lbl_transcript.setText(f"⚠ No {iso}.txt (S1-S5 won't have reference)")
            self.lbl_transcript.setStyleSheet("color:#d97706;font-size:11px")
        log(f"Language: {name} | Transcript: {msg}")

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
        self.lbl_status.setText(f"Audio: {os.path.basename(path)} ({len(y)/sr:.1f}s)")
        self.btn_audio.setEnabled(True)
        self._refresh_list(); self._redraw()

    def _audio_err(self, msg):
        self.btn_audio.setEnabled(True); QMessageBox.critical(self, "Error", msg)

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

            # Count segment types
            types = {"char": 0, "word": 0, "sentence": 0, "paragraph": 0}
            for seg in segs:
                info = classify_segment(seg["label"])
                types[info["type"]] += 1

            type_str = "  |  ".join(f"{k}: {v}" for k, v in types.items() if v > 0)
            self.lbl_status.setText(
                f"GT: {len(segs)} segments ({type_str})"
                + (f"  |  skipped {skip}" if skip else ""))
            self._refresh_list(); self._redraw()
        except Exception as e:
            traceback.print_exc(); QMessageBox.critical(self, "Error", str(e))

    # ── Load Model ────────────────────────────────────────────────────────
    def on_load_model(self):
        t = self._threads.get("model")
        if t and t.isRunning():
            QMessageBox.information(self, "Loading", "Model is still downloading."); return
        ek, mid = self.cb_model.currentData()
        self._set_busy(True, f"Loading {mid}...")
        self.btn_model.setText("Loading..."); self.btn_model.setEnabled(False)
        self._load_t0 = time.time()
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: self.lbl_status.setText(
            f"Loading model... {int(time.time()-self._load_t0)}s"))
        self._timer.start(1000)
        t = ModelLoadThread(ek, mid)
        t.done.connect(self._model_ok); t.error.connect(self._model_err)
        self._threads["model"] = t; t.start()

    def _stop_timer(self):
        if self._timer: self._timer.stop(); self._timer = None

    def _model_ok(self, ok, msg, el):
        self._stop_timer(); self._set_busy(False); self.btn_model.setEnabled(True)
        if ok:
            self.btn_model.setText("✓ Model Loaded")
            self.btn_model.setStyleSheet(
                "QPushButton{font-weight:bold;color:white;background:#16a34a;"
                "border:1px solid #15803d;padding:5px 14px;border-radius:4px}")
            self.lbl_status.setText(f"✓ {msg}"); self.lbl_speed.setText(f"Engine: {ENGINE.engine_key}")
        else:
            self.btn_model.setText("⬇ Load Model")
            QMessageBox.critical(self, "Error", f"Failed:\n{msg}")

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

    # ── Run Prediction (AUTO MODE) ────────────────────────────────────────
    def on_run(self):
        if self.y16 is None: QMessageBox.warning(self, "", "Load audio."); return
        if not self.segments: QMessageBox.warning(self, "", "Load GT or VAD."); return
        if not ENGINE.is_loaded: QMessageBox.warning(self, "", "Load model first."); return
        t = self._threads.get("pred")
        if t and t.isRunning(): return

        iso, name = self.cb_lang.currentData()
        self.results = []; self.pbar.setValue(0); self._pred_t0 = time.time()
        self._set_busy(True, "Running auto-mode prediction...")
        self._refresh_list()

        # NO manual mode — PredictionThread auto-classifies each segment
        t = PredictionThread(self.y16, self.segments, iso, name)
        t.seg_result.connect(self._on_seg_result)
        t.progress.connect(self._on_pred_prog)
        t.done.connect(self._on_pred_done)
        t.error.connect(self._on_pred_err)
        self._threads["pred"] = t; t.start()

    def _on_seg_result(self, idx, pred, acc, seg_type):
        if idx >= self.seg_list.count(): return
        seg = self.segments[idx]
        item = self.seg_list.item(idx)

        # Type icon
        icons = {"char": "Ⓒ", "word": "Ⓦ", "sentence": "Ⓢ", "paragraph": "Ⓟ"}
        icon = icons.get(seg_type, "?")

        if acc < 0:
            # No reference text → just show transcript
            ps = (pred[:45] + "…") if len(pred) > 45 else pred
            item.setText(f"{idx+1:03d} {icon} {seg['label']}  →  {ps}  │ {seg['start']:.2f}-{seg['end']:.2f}s  [no-ref]")
            from PyQt5.QtGui import QColor
            item.setForeground(QColor("#6366f1"))  # indigo
        else:
            ps = (pred[:35] + "…") if len(pred) > 35 else pred
            item.setText(f"{idx+1:03d} {icon} {seg['label']}  →  {ps}  │ {seg['start']:.2f}-{seg['end']:.2f}s  acc={acc:.2f}")
            if acc >= 0.95: item.setForeground(Qt.darkGreen)
            elif acc >= 0.5: item.setForeground(Qt.darkYellow)
            else: item.setForeground(Qt.red)

    def _on_pred_prog(self, done, total):
        pct = int(done / total * 100) if total else 0
        self.pbar.setValue(pct)
        el = time.time() - self._pred_t0
        spd = done / el if el > 0 else 0
        eta = (total - done) / spd if spd > 0 else 0
        self.lbl_status.setText(f"Predicting {done}/{total} ({spd:.1f} seg/s, ETA {eta:.0f}s)")

    def _on_pred_done(self, results, mean_acc, elapsed):
        self.results = results; self.pbar.setValue(100)
        n = len(results)
        n_scored = sum(1 for r in results if r["acc"] >= 0)
        n_good = sum(1 for r in results if r["acc"] >= 0.95)
        n_noref = sum(1 for r in results if r["acc"] < 0)
        spd = n / elapsed if elapsed > 0 else 0

        # Count by type
        types = {}
        for r in results:
            types[r["type"]] = types.get(r["type"], 0) + 1

        type_str = "  ".join(f"{k}:{v}" for k, v in types.items())
        parts = [f"✓ {n} segs in {elapsed:.1f}s"]
        if n_scored: parts.append(f"Acc: {mean_acc:.3f} ({n_good}/{n_scored} ≥95%)")
        if n_noref: parts.append(f"📝 {n_noref} no-ref")
        parts.append(type_str)
        parts.append(f"{spd:.1f} seg/s")

        self._set_busy(False, " | ".join(parts))
        self.lbl_speed.setText(f"{spd:.1f} seg/s")
        self._refresh_list(); self._redraw()

    def _on_pred_err(self, msg):
        self.pbar.setValue(0); self._set_busy(False, f"✗ {msg}")
        QMessageBox.critical(self, "Error", msg)

    def on_stop(self):
        t = self._threads.get("pred")
        if t and t.isRunning(): t.request_stop()

    # ── Plot ──────────────────────────────────────────────────────────────
    def _redraw(self, xlim=None):
        self.ax.clear()
        if self.wave_t is None: self.canvas.draw_idle(); return
        self.ax.plot(self.wave_t, self.wave_y, lw=0.5, color="#1e293b")
        ymax = max(np.max(np.abs(self.wave_y)), 1e-6)
        iso, _ = self.cb_lang.currentData()
        fp = get_font(iso)

        if xlim is None:
            if self.segments:
                s_min = min(s["start"] for s in self.segments)
                s_max = max(s["end"] for s in self.segments)
                pad = max(1.0, (s_max - s_min) * 0.05)
                xlim = (max(0, s_min - pad), s_max + pad)
            elif self.y is not None:
                xlim = (0, len(self.y) / self.sr)

        if xlim: self.ax.set_xlim(*xlim)
        x0, x1 = self.ax.get_xlim()

        # Type colors
        type_colors = {
            "char":      ("#ede9fe", "#7c3aed"),  # purple
            "word":      ("#e0f2fe", "#0284c7"),  # blue
            "sentence":  ("#dcfce7", "#16a34a"),  # green
            "paragraph": ("#fff7ed", "#ea580c"),  # orange
        }

        for i, seg in enumerate(self.segments):
            s, e = seg["start"], seg["end"]
            if e < x0 or s > x1: continue

            # Determine type and colors
            if self.results and i < len(self.results):
                r = self.results[i]
                seg_type = r.get("type", "word")
                acc = r["acc"]
                if acc < 0:
                    clr, bar = "#e0e7ff", "#6366f1"  # indigo for no-ref
                elif acc >= 0.95:
                    clr, bar = type_colors.get(seg_type, ("#e0f2fe", "#0284c7"))
                elif acc >= 0.5:
                    clr, bar = "#fef3c7", "#d97706"
                else:
                    clr, bar = "#fecaca", "#dc2626"
            else:
                info = classify_segment(seg["label"])
                clr, bar = type_colors.get(info["type"], ("#e0f2fe", "#0284c7"))

            self.ax.axvspan(s, e, alpha=0.2, color=clr, zorder=1)
            self.ax.plot([s, e], [-ymax*0.92, -ymax*0.92], lw=3, color=bar,
                        solid_capstyle="butt", zorder=3)

            if self.selected_idx is not None and i == self.selected_idx:
                self.ax.axvspan(s, e, alpha=0.15, color="#facc15", zorder=2)
                mid = (s + e) / 2
                fkw = {"fontproperties": fp} if fp else {}
                bbox = dict(facecolor="white", alpha=0.9, edgecolor="#d1d5db",
                           boxstyle="round,pad=0.3")
                self.ax.text(mid, ymax*0.9, f"GT: {seg['label']}", ha="center",
                    va="bottom", fontsize=10, color="#15803d", fontweight="bold",
                    bbox=bbox, zorder=5, **fkw)
                if self.results and i < len(self.results):
                    r = self.results[i]
                    p = (r['pred'][:50] + "…") if len(r['pred']) > 50 else r['pred']
                    acc_s = f"acc={r['acc']:.2f}" if r['acc'] >= 0 else "no-ref"
                    self.ax.text(mid, ymax*0.65, f"PRED: {p or '∅'} ({acc_s})",
                        ha="center", va="bottom", fontsize=9, color="#1d4ed8",
                        fontweight="bold", bbox=bbox, zorder=5, **fkw)

        title = "Waveform"
        if self.results:
            scored = [r["acc"] for r in self.results if r["acc"] >= 0]
            if scored:
                title += f"  │  Accuracy: {np.mean(scored):.3f}"
                n_good = sum(1 for a in scored if a >= 0.95)
                title += f"  │  {n_good}/{len(scored)} correct"
        self.ax.set_title(title, fontsize=11, fontweight="bold")
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-ymax*1.05, ymax*1.05)
        self.ax.grid(axis="x", alpha=0.15, linestyle="--")
        if xlim: self.ax.set_xlim(*xlim)
        self.canvas.draw_idle()

    # ── Segment List ──────────────────────────────────────────────────────
    def _refresh_list(self):
        self.seg_list.clear()
        icons = {"char": "Ⓒ", "word": "Ⓦ", "sentence": "Ⓢ", "paragraph": "Ⓟ"}

        for i, seg in enumerate(self.segments):
            info = classify_segment(seg["label"])
            icon = icons.get(info["type"], "?")

            if self.results and i < len(self.results):
                r = self.results[i]
                acc = r["acc"]
                ps = (r['pred'][:35] + "…") if len(r['pred']) > 35 else r['pred']

                if acc < 0:
                    text = f"{i+1:03d} {icon} {seg['label']}  →  {ps}  │ {seg['start']:.2f}-{seg['end']:.2f}s  [no-ref]"
                    item = QListWidgetItem(text)
                    from PyQt5.QtGui import QColor
                    item.setForeground(QColor("#6366f1"))
                else:
                    text = f"{i+1:03d} {icon} {seg['label']}  →  {ps}  │ {seg['start']:.2f}-{seg['end']:.2f}s  acc={acc:.2f}"
                    item = QListWidgetItem(text)
                    if acc >= 0.95: item.setForeground(Qt.darkGreen)
                    elif acc >= 0.5: item.setForeground(Qt.darkYellow)
                    else: item.setForeground(Qt.red)
            else:
                text = f"{i+1:03d} {icon} {seg['label']}  →  ...  │ {seg['start']:.2f}-{seg['end']:.2f}s"
                item = QListWidgetItem(text)
            self.seg_list.addItem(item)

    def on_seg_clicked(self, item):
        self.selected_idx = self.seg_list.row(item)
        if 0 <= self.selected_idx < len(self.segments):
            seg = self.segments[self.selected_idx]
            pad = max(0.3, (seg["end"] - seg["start"]) * 0.3)
            self._redraw(xlim=(max(0, seg["start"] - pad), seg["end"] + pad))

    def on_play_seg(self):
        if self.selected_idx is not None and self.y is not None:
            seg = self.segments[self.selected_idx]
            sd.stop(); sd.play(self.y[int(seg["start"]*self.sr):int(seg["end"]*self.sr)], self.sr)

    # ── Zoom ──────────────────────────────────────────────────────────────
    def _zoom(self, f):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim(); c = (x0+x1)/2; w = (x1-x0)*f/2
        self._redraw(xlim=(max(0, c-w), min(len(self.y)/self.sr, c+w)))

    def _on_scroll(self, ev):
        if self.y is None: return
        x0, x1 = self.ax.get_xlim()
        mx = ev.xdata if ev.xdata else (x0+x1)/2
        f = 0.5 if ev.button == "up" else 2.0
        self._redraw(xlim=(max(0, mx-(mx-x0)*f), min(len(self.y)/self.sr, mx+(x1-mx)*f)))

    def _fit_segments(self):
        if self.segments: self._redraw()  # default xlim auto-fits
        elif self.y is not None: self._redraw(xlim=(0, len(self.y)/self.sr))

    def _reset_view(self):
        if self.y is not None: self._redraw(xlim=(0, len(self.y)/self.sr))

    # ── Export ────────────────────────────────────────────────────────────
    def on_export(self):
        if not self.results: QMessageBox.warning(self, "", "Run prediction first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "results.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["index", "type", "mode", "ground_truth", "prediction",
                            "compare_to", "accuracy", "start", "end"])
                for r in self.results:
                    acc_s = f"{r['acc']:.4f}" if r["acc"] >= 0 else "N/A"
                    w.writerow([r["i"]+1, r["type"], r["mode"], r["gt"],
                                r["pred"], r.get("compare_to", ""),
                                acc_s, f"{r['start']:.4f}", f"{r['end']:.4f}"])
            self.lbl_status.setText(f"Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Create transcripts folder if missing
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  Indic Speech Annotation Tool — Auto Mode")
    print(f"  Transcripts folder: {TRANSCRIPTS_DIR}")
    if _HF:
        cuda = torch.cuda.is_available()
        print(f"  CUDA: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    print("=" * 65)

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