import os
import io
import re
import csv
import time
import tempfile
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from fastapi import FastAPI

try:
    import soundfile as sf
    _SF = True
except Exception:
    _SF = False

try:
    from scipy.signal import resample_poly
    _SCIPY = True
except Exception:
    _SCIPY = False

try:
    import torch
    import torchaudio  # noqa: F401
    from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor
    _HF = True
except Exception:
    _HF = False

try:
    from faster_whisper import WhisperModel as _FWModel
    _FW = True
except Exception:
    _FW = False

if not _HF and not _FW:
    raise RuntimeError("Install transformers+torch+torchaudio or faster-whisper first.")

SR16 = 16000
CPU_THR = max(1, min(os.cpu_count() or 4, 8))
APP_DIR = Path(__file__).parent.resolve()
TRANSCRIPTS_DIR = APP_DIR / "transcripts"
LANGUAGES = [
    ("Gujarati", "gu", "gujarati"),
    ("Hindi", "hi", "hindi"),
    ("Marathi", "mr", "marathi"),
    ("English", "en", "english"),
]
LANGUAGE_MAP = {d: (iso, name) for d, iso, name in LANGUAGES}
INDIC_MODEL = ("indicconformer", "ai4bharat/indic-conformer-600m-multilingual") if _HF else None
ENGLISH_MODEL = ("fw", "large-v3") if _FW else (("hf_whisper", "openai/whisper-large-v3") if _HF else None)

_INDIC = r"\u0900-\u097F\uA8E0-\uA8FF\u0A80-\u0AFF"
_RE_CLEAN = re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_NORM = re.compile(rf"[^a-z0-9{_INDIC}]", re.UNICODE)
_RE_ALPHA = re.compile(rf"[A-Za-z{_INDIC}]", re.UNICODE)
_RE_SEQ = re.compile(r"^(?:s|p|sent|para|sentence|paragraph|section|seg)\s*\d*$", re.I)

_PLACEHOLDER = [
    "write your", "લખો", "लिखें", "लिहा", "your first", "your second",
    "your third", "your fourth", "your fifth", "your paragraph", "અહીં તમારું",
    "अपना", "तुमचे", "येथे"
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TranscriptStore:
    def __init__(self):
        self.data = {}
        self.lang_code = None
        self.file_path = None

    def load(self, lang_iso: str):
        self.data = {}
        self.lang_code = lang_iso
        self.file_path = TRANSCRIPTS_DIR / f"{lang_iso}.txt"
        if not self.file_path.exists():
            return False, f"transcripts/{lang_iso}.txt not found"
        try:
            text = self.file_path.read_text(encoding="utf-8")
            current_key = None
            current_lines = []
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("#"):
                    continue
                match = re.match(r"^\[(.+?)\]$", ls)
                if match:
                    if current_key is not None:
                        self.data[current_key] = "\n".join(current_lines).strip()
                    current_key = match.group(1).strip().lower()
                    current_lines = []
                elif current_key is not None:
                    current_lines.append(line.rstrip())
            if current_key is not None:
                self.data[current_key] = "\n".join(current_lines).strip()
            return True, f"Loaded {len(self.data)} transcript sections"
        except Exception as e:
            return False, str(e)

    def get_reference(self, label: str) -> str:
        return self.data.get(label.strip().lower(), "")

    def is_placeholder(self, key: str) -> bool:
        text = self.data.get(key.lower(), "")
        if not text:
            return True
        return any(p in text.lower() for p in _PLACEHOLDER)


TRANSCRIPTS = TranscriptStore()


def is_sequence_label(label: str) -> bool:
    label = str(label).strip()
    if not label:
        return False
    if _RE_SEQ.match(label):
        return True
    return label.lower() in {"paragraph", "para", "passage"}


def classify_segment(label: str):
    label = str(label).strip()
    ll = label.lower()
    if is_sequence_label(label):
        ref = TRANSCRIPTS.get_reference(label)
        is_ph = TRANSCRIPTS.is_placeholder(label) if ref else True
        seg_type = "paragraph" if "para" in ll or "passage" in ll else "sentence"
        return {
            "type": seg_type,
            "mode": "full",
            "compare_to": ref if (ref and not is_ph) else None,
            "is_seq": True,
            "ref_key": label,
            "ref_status": "ok" if (ref and not is_ph) else ("placeholder" if ref else "missing"),
        }
    chars = re.sub(r"\s+", "", label)
    if len(chars) == 1:
        return {"type": "char", "mode": "char", "compare_to": label, "is_seq": False}
    if len(label.split()) == 1:
        return {"type": "word", "mode": "word", "compare_to": label, "is_seq": False}
    return {"type": "sentence", "mode": "full", "compare_to": label, "is_seq": False}


def clean_text(text: str, mode: str = "full") -> str:
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


def norm_text(s: str) -> str:
    return _RE_NORM.sub("", str(s).strip().lower())


def edit_distance(a: str, b: str) -> int:
    na, nb = len(a), len(b)
    if na == 0:
        return nb
    if nb == 0:
        return na
    dp = list(range(nb + 1))
    for i in range(1, na + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, nb + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[nb]


def accuracy(pred: str, gt: str) -> float:
    p, g = norm_text(pred), norm_text(gt)
    if not g and not p:
        return 1.0
    if not g:
        return 0.0
    return max(0.0, 1.0 - edit_distance(p, g) / len(g))


def load_audio(path: str):
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


def resample_audio(y: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return y.astype(np.float32)
    if _SCIPY:
        g = gcd(int(original_sr), int(target_sr))
        return resample_poly(y, target_sr // g, original_sr // g).astype(np.float32)
    import librosa
    return librosa.resample(y, orig_sr=original_sr, target_sr=target_sr).astype(np.float32)


def parse_time(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if not x:
        return None
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"):
        try:
            dt = datetime.strptime(x, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except Exception:
            continue
    try:
        return float(x)
    except Exception:
        return None


def vad_segment(y: np.ndarray, sr: int, silence_ms=300, thresh_db=-40, min_ms=200, max_ms=10000):
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    n_frames = 1 + (len(y) - frame_len) // hop
    if n_frames <= 0:
        return []
    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop
    np.clip(idx, 0, len(y) - 1, out=idx)
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = rms_db > thresh_db
    padded = np.concatenate([[False], voiced, [False]])
    d = np.diff(padded.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    silence_frames = max(1, int(silence_ms / 1000 * sr / hop))
    min_frames = max(1, int(min_ms / 1000 * sr / hop))
    max_frames = max(1, int(max_ms / 1000 * sr / hop))

    merged = []
    i = 0
    while i < len(starts):
        s, e = starts[i], ends[i]
        while i + 1 < len(starts) and (starts[i + 1] - e) < silence_frames:
            i += 1
            e = ends[i]
        merged.append((s, e))
        i += 1

    result = []
    for s, e in merged:
        if (e - s) < min_frames:
            continue
        for cs in range(s, e, max_frames):
            ce = min(cs + max_frames, e)
            if (ce - cs) >= min_frames:
                result.append({
                    "start": round(cs * hop / sr, 4),
                    "end": round(min(ce * hop / sr, len(y) / sr), 4),
                })
    return result


class Engine:
    def __init__(self):
        self.backend = None
        self.processor = None
        self.device = None
        self.engine_key = None
        self._ck = None

    @property
    def is_loaded(self):
        return self.backend is not None

    def load(self, engine_key: str, model_id: str):
        ck = (engine_key, model_id)
        if self._ck == ck and self.backend is not None:
            return True, f"Already loaded: {model_id}"
        self.backend = None
        self.processor = None
        self.engine_key = engine_key
        t0 = time.time()
        try:
            if engine_key == "indicconformer":
                self._load_indic(model_id)
            elif engine_key == "fw":
                self._load_fw(model_id)
            elif engine_key == "hf_whisper":
                self._load_hf_whisper(model_id)
            else:
                return False, f"Unknown engine: {engine_key}"
            self._ck = ck
            return True, f"Loaded {model_id} in {time.time() - t0:.1f}s"
        except Exception as e:
            traceback.print_exc()
            self.backend = None
            return False, str(e)

    def _load_indic(self, model_id: str):
        self.device = "cuda" if (_HF and torch.cuda.is_available()) else "cpu"
        log(f"Loading IndicConformer on {self.device}")
        self.backend = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(self.backend, "to"):
            try:
                self.backend = self.backend.to(self.device)
            except Exception:
                self.device = "cpu"

    def _load_fw(self, size: str):
        has_cuda = _HF and torch.cuda.is_available() if _HF else False
        log(f"Loading faster-whisper {size}")
        self.backend = _FWModel(
            size,
            device="cuda" if has_cuda else "cpu",
            compute_type="float16" if has_cuda else "int8",
            cpu_threads=CPU_THR,
        )

    def _load_hf_whisper(self, model_id: str):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log(f"Loading HF Whisper {model_id}")
        self.backend = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def transcribe(self, audio16: np.ndarray, lang_iso: str, lang_name: str, mode: str = "full") -> str:
        if self.backend is None or audio16 is None or len(audio16) == 0:
            return ""
        if len(audio16) < int(0.1 * SR16):
            audio16 = np.pad(audio16, (0, int(0.1 * SR16) - len(audio16)))
        audio16 = audio16.astype(np.float32)
        if self.engine_key == "indicconformer":
            return self._transcribe_indic(audio16, lang_iso, mode)
        if self.engine_key == "fw":
            return self._transcribe_fw(audio16, lang_iso, mode)
        if self.engine_key == "hf_whisper":
            return self._transcribe_hf(audio16, lang_name, mode)
        return ""

    def _transcribe_indic(self, audio16: np.ndarray, lang_iso: str, mode: str):
        waveform = torch.tensor(audio16, dtype=torch.float32).unsqueeze(0)
        if self.device and self.device != "cpu":
            waveform = waveform.to(self.device)
        with torch.inference_mode():
            text = self.backend(waveform, lang_iso, "ctc")
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return clean_text(str(text), mode)

    def _transcribe_fw(self, audio16: np.ndarray, lang_iso: str, mode: str):
        segments, _ = self.backend.transcribe(
            audio16,
            language=lang_iso,
            task="transcribe",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in list(segments) if s.text.strip())
        return clean_text(text, mode)

    def _transcribe_hf(self, audio16: np.ndarray, lang_name: str, mode: str):
        inputs = self.processor(audio16, sampling_rate=SR16, return_tensors="pt", return_attention_mask=True)
        features = inputs.input_features.to(self.device)
        mask = getattr(inputs, "attention_mask", None)
        if mask is not None:
            mask = mask.to(self.device)
        gen_kwargs = {
            "input_features": features,
            "task": "transcribe",
            "language": lang_name,
            "return_timestamps": False,
        }
        if mask is not None:
            gen_kwargs["attention_mask"] = mask
        with torch.inference_mode():
            ids = self.backend.generate(**gen_kwargs)
        return clean_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0], mode)


ENGINE = Engine()


def default_state():
    return {
        "audio_path": None,
        "y": None,
        "sr": None,
        "y16": None,
        "segments": [],
        "results": [],
        "language_display": "Gujarati",
        "model_info": "Not loaded",
        "status": "Select language → Load Model → Load Audio → Load GT/VAD → Run",
    }


def model_for_language(lang_display: str):
    iso, name = LANGUAGE_MAP[lang_display]
    model = INDIC_MODEL if iso in ("gu", "hi", "mr") else ENGLISH_MODEL
    return iso, name, model


def make_waveform_figure(y, sr, segments=None, results=None):
    fig = go.Figure()
    if y is None or sr is None:
        fig.update_layout(title="Waveform", template="plotly_white", height=420)
        return fig
    step = max(1, len(y) // 30000)
    y_plot = y[::step]
    x_plot = np.arange(len(y_plot)) * (step / sr)
    fig.add_trace(go.Scattergl(x=x_plot, y=y_plot, mode="lines", name="waveform"))

    color_map = {
        "char": ("rgba(124,58,237,0.18)", "rgba(124,58,237,0.85)"),
        "word": ("rgba(2,132,199,0.18)", "rgba(2,132,199,0.85)"),
        "sentence": ("rgba(22,163,74,0.18)", "rgba(22,163,74,0.85)"),
        "paragraph": ("rgba(234,88,12,0.18)", "rgba(234,88,12,0.85)"),
    }

    if segments:
        ymin = float(np.min(y_plot)) if len(y_plot) else -1.0
        ymax = float(np.max(y_plot)) if len(y_plot) else 1.0
        for i, seg in enumerate(segments):
            info = classify_segment(seg["label"])
            fill, line = color_map.get(info["type"], color_map["word"])
            if results and i < len(results):
                r = results[i]
                if r["acc"] < 0:
                    fill, line = ("rgba(99,102,241,0.15)", "rgba(99,102,241,0.85)")
                elif r["acc"] < 0.5:
                    fill, line = ("rgba(220,38,38,0.15)", "rgba(220,38,38,0.9)")
                elif r["acc"] < 0.95:
                    fill, line = ("rgba(217,119,6,0.15)", "rgba(217,119,6,0.9)")
            fig.add_vrect(x0=seg["start"], x1=seg["end"], fillcolor=fill, line_width=0)
            fig.add_trace(go.Scatter(
                x=[seg["start"], seg["end"]],
                y=[ymin * 0.92, ymin * 0.92],
                mode="lines",
                line=dict(color=line, width=4),
                showlegend=False,
                hovertemplate=f"{seg['label']}<br>{seg['start']:.2f}-{seg['end']:.2f}s<extra></extra>",
            ))
    title = "Waveform"
    if results:
        scores = [r["acc"] for r in results if r["acc"] >= 0]
        if scores:
            title += f" | Mean accuracy: {np.mean(scores):.3f}"
    fig.update_layout(template="plotly_white", title=title, height=420, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    return fig


def results_to_df(results):
    rows = []
    for r in results:
        rows.append({
            "index": r["i"] + 1,
            "type": r["type"],
            "mode": r["mode"],
            "ground_truth": r["gt"],
            "prediction": r["pred"],
            "compared_against": r.get("compare_to", ""),
            "accuracy": None if r["acc"] < 0 else round(float(r["acc"]), 4),
            "start": round(float(r["start"]), 4),
            "end": round(float(r["end"]), 4),
            "ref_status": r.get("ref_status", ""),
        })
    return pd.DataFrame(rows)


def segments_to_df(segments):
    rows = []
    for i, seg in enumerate(segments, start=1):
        info = classify_segment(seg["label"])
        rows.append({
            "index": i,
            "label": seg["label"],
            "type": info["type"],
            "start": round(float(seg["start"]), 4),
            "end": round(float(seg["end"]), 4),
            "duration": round(float(seg["end"] - seg["start"]), 4),
        })
    return pd.DataFrame(rows)


def export_results_csv(results):
    if not results:
        return None
    fd, path = tempfile.mkstemp(suffix="_results.csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "type", "mode", "ground_truth", "prediction", "compared_against", "accuracy", "start", "end", "ref_status"])
        for r in results:
            writer.writerow([
                r["i"] + 1,
                r["type"],
                r["mode"],
                r["gt"],
                r["pred"],
                r.get("compare_to", ""),
                "N/A" if r["acc"] < 0 else f"{r['acc']:.4f}",
                f"{r['start']:.4f}",
                f"{r['end']:.4f}",
                r.get("ref_status", ""),
            ])
    return path


def on_language_change(lang_display, state):
    state = state or default_state()
    state["language_display"] = lang_display
    iso, _, _ = LANGUAGE_MAP[lang_display][0], LANGUAGE_MAP[lang_display][1], model_for_language(lang_display)[2]
    ok, message = TRANSCRIPTS.load(iso)
    transcript_msg = message if ok else f"Transcript warning: {message}"
    _, _, model = model_for_language(lang_display)
    model_name = "None"
    if model:
        engine_key, model_id = model
        model_name = "IndicConformer-600M" if engine_key == "indicconformer" else f"Whisper {model_id}"
    state["status"] = f"Language set to {lang_display}. {transcript_msg}"
    return state, model_name, state["status"]


def load_model(lang_display, state):
    state = state or default_state()
    iso, name, model = model_for_language(lang_display)
    TRANSCRIPTS.load(iso)
    if not model:
        state["status"] = "No model available for selected language"
        return state, state["status"], state["model_info"]
    engine_key, model_id = model
    ok, msg = ENGINE.load(engine_key, model_id)
    if ok:
        state["model_info"] = f"Loaded: {model_id} ({ENGINE.engine_key})"
    state["status"] = msg
    return state, state["status"], state["model_info"]


def handle_audio_upload(audio_file, state):
    state = state or default_state()
    if audio_file is None:
        return state, "No audio selected", None, None
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", None)
    if not path:
        return state, "Could not read audio path", None, None
    y, sr = load_audio(path)
    y16 = resample_audio(y, sr, SR16)
    state["audio_path"] = path
    state["y"] = y
    state["sr"] = sr
    state["y16"] = y16
    fig = make_waveform_figure(y, sr, state.get("segments", []), state.get("results", []))
    audio_info = pd.DataFrame([{
        "file": os.path.basename(path),
        "sample_rate": sr,
        "duration_sec": round(len(y) / sr, 3),
        "samples": len(y),
    }])
    state["status"] = f"Loaded audio: {os.path.basename(path)} ({len(y) / sr:.1f}s)"
    return state, state["status"], fig, audio_info


def parse_ground_truth(file_path: str):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, header=None)
    elif file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, header=None, sep="\t")
    else:
        df = pd.read_excel(file_path, sheet_name=0, header=None, engine="openpyxl")
    segments = []
    skipped = 0
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        start = parse_time(row.iloc[1])
        end = parse_time(row.iloc[2])
        if not label or start is None or end is None or end <= start:
            skipped += 1
            continue
        segments.append({"label": label, "start": start, "end": end})
    return segments, skipped


def load_ground_truth(gt_file, state):
    state = state or default_state()
    if gt_file is None:
        return state, "No ground-truth file selected", None, None
    path = gt_file if isinstance(gt_file, str) else getattr(gt_file, "name", None)
    if not path:
        return state, "Could not read ground-truth path", None, None
    segments, skipped = parse_ground_truth(path)
    state["segments"] = segments
    state["results"] = []
    fig = make_waveform_figure(state.get("y"), state.get("sr"), segments, [])
    seg_df = segments_to_df(segments)
    counts = seg_df["type"].value_counts().to_dict() if not seg_df.empty else {}
    count_text = " ".join(f"{k}:{v}" for k, v in counts.items())
    state["status"] = f"Loaded GT: {len(segments)} segments | {count_text}" + (f" | skipped:{skipped}" if skipped else "")
    return state, state["status"], seg_df, fig


def run_vad(state, silence_ms, thresh_db, min_ms, max_ms):
    state = state or default_state()
    if state.get("y") is None:
        return state, "Load audio first", None, None
    segments = vad_segment(
        state["y"],
        state["sr"],
        silence_ms=int(silence_ms),
        thresh_db=float(thresh_db),
        min_ms=int(min_ms),
        max_ms=int(max_ms),
    )
    state["segments"] = [
        {"label": f"seg_{i+1:04d}", "start": s["start"], "end": s["end"]}
        for i, s in enumerate(segments)
    ]
    state["results"] = []
    seg_df = segments_to_df(state["segments"])
    fig = make_waveform_figure(state["y"], state["sr"], state["segments"], [])
    state["status"] = f"VAD created {len(state['segments'])} segments"
    return state, state["status"], seg_df, fig


def run_prediction(lang_display, state, progress=gr.Progress()):
    state = state or default_state()
    if state.get("y16") is None:
        return state, "Load audio first", None, None, None
    if not state.get("segments"):
        return state, "Load GT or run VAD first", None, None, None
    if not ENGINE.is_loaded:
        return state, "Load model first", None, None, None

    iso, lang_name, model = model_for_language(lang_display)
    if model and ENGINE._ck != model:
        return state, "Wrong model loaded for selected language. Click Load Model.", None, None, None

    TRANSCRIPTS.load(iso)
    results = []
    total = len(state["segments"])
    t0 = time.time()
    for i, seg in enumerate(state["segments"]):
        progress((i, total), desc=f"Processing segment {i+1}/{total}")
        info = classify_segment(seg["label"])
        s = max(0, int(seg["start"] * SR16))
        e = min(len(state["y16"]), int(seg["end"] * SR16))
        pred = ENGINE.transcribe(state["y16"][s:e], iso, lang_name, info["mode"])
        if info["is_seq"] and info["compare_to"]:
            acc = accuracy(pred, info["compare_to"])
        elif info["is_seq"]:
            acc = -1.0
        else:
            acc = accuracy(pred, seg["label"])
        results.append({
            "i": i,
            "gt": seg["label"],
            "pred": pred,
            "start": seg["start"],
            "end": seg["end"],
            "acc": acc,
            "type": info["type"],
            "mode": info["mode"],
            "is_seq": info["is_seq"],
            "compare_to": info.get("compare_to") or "",
            "ref_status": info.get("ref_status", ""),
        })
    elapsed = time.time() - t0
    state["results"] = results
    fig = make_waveform_figure(state["y"], state["sr"], state["segments"], results)
    df = results_to_df(results)
    csv_path = export_results_csv(results)
    valid_scores = [r["acc"] for r in results if r["acc"] >= 0]
    mean_acc = np.mean(valid_scores) if valid_scores else 0.0
    summary = pd.DataFrame([{
        "segments": len(results),
        "mean_accuracy": round(float(mean_acc), 4),
        "elapsed_sec": round(float(elapsed), 2),
        "engine": ENGINE.engine_key,
    }])
    state["status"] = f"Done: {len(results)} segments in {elapsed:.2f}s | mean accuracy={mean_acc:.4f}"
    return state, state["status"], fig, df, summary, csv_path


def play_selected_segment(state, segment_index):
    state = state or default_state()
    if state.get("y") is None or not state.get("segments"):
        return None, "Load audio and segments first"
    try:
        idx = int(segment_index) - 1
    except Exception:
        return None, "Enter a valid segment index"
    if idx < 0 or idx >= len(state["segments"]):
        return None, "Segment index out of range"
    seg = state["segments"][idx]
    s = max(0, int(seg["start"] * state["sr"]))
    e = min(len(state["y"]), int(seg["end"] * state["sr"]))
    clip = state["y"][s:e]
    status = f"Playing segment {idx+1}: {seg['label']} ({seg['start']:.2f}-{seg['end']:.2f}s)"
    return (state["sr"], clip.astype(np.float32)), status


with gr.Blocks(title="Indic Speech Annotation Tool (Web)") as demo:
    gr.Markdown("# Indic Speech Annotation Tool — Gradio + FastAPI")
    gr.Markdown(
        "Browser version of your PyQt tool: language/model selection, audio upload, GT spreadsheet parsing, VAD auto segmentation, per-segment transcription, accuracy scoring, waveform display, segment playback, and CSV export."
    )

    state = gr.State(default_state())

    with gr.Row():
        language = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="Gujarati", label="Language")
        model_name = gr.Textbox(label="Required Model", value="IndicConformer-600M", interactive=False)
        model_info = gr.Textbox(label="Loaded Model", value="Not loaded", interactive=False)
        load_model_btn = gr.Button("Load Model", variant="primary")

    status = gr.Textbox(label="Status", value="Select language → Load Model → Load Audio → Load GT/VAD → Run", interactive=False)

    with gr.Row():
        audio_input = gr.File(label="Audio File", file_count="single", file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"])
        gt_input = gr.File(label="Ground Truth File", file_count="single", file_types=[".csv", ".tsv", ".xlsx"])

    with gr.Row():
        load_audio_btn = gr.Button("Load Audio")
        load_gt_btn = gr.Button("Load Ground Truth")
        run_btn = gr.Button("Run Prediction", variant="primary")

    with gr.Accordion("VAD Settings", open=False):
        with gr.Row():
            silence_ms = gr.Slider(50, 2000, value=300, step=10, label="Silence (ms)")
            thresh_db = gr.Slider(-80, 0, value=-40, step=1, label="Threshold (dB)")
            min_ms = gr.Slider(50, 5000, value=200, step=10, label="Min Segment (ms)")
            max_ms = gr.Slider(1000, 30000, value=10000, step=100, label="Max Segment (ms)")
        vad_btn = gr.Button("Run VAD")

    waveform = gr.Plot(label="Waveform + Segments")
    audio_meta = gr.Dataframe(label="Audio Info", interactive=False)
    segments_df = gr.Dataframe(label="Segments", interactive=False)
    results_df = gr.Dataframe(label="Prediction Results", interactive=False)
    summary_df = gr.Dataframe(label="Run Summary", interactive=False)
    csv_file = gr.File(label="Download CSV Results")

    with gr.Row():
        segment_index = gr.Number(label="Segment Index to Play", precision=0, value=1)
        play_btn = gr.Button("Play Segment")
    segment_audio = gr.Audio(label="Selected Segment")

    language.change(on_language_change, inputs=[language, state], outputs=[state, model_name, status])
    load_model_btn.click(load_model, inputs=[language, state], outputs=[state, status, model_info])
    load_audio_btn.click(handle_audio_upload, inputs=[audio_input, state], outputs=[state, status, waveform, audio_meta])
    load_gt_btn.click(load_ground_truth, inputs=[gt_input, state], outputs=[state, status, segments_df, waveform])
    vad_btn.click(run_vad, inputs=[state, silence_ms, thresh_db, min_ms, max_ms], outputs=[state, status, segments_df, waveform])
    run_btn.click(run_prediction, inputs=[language, state], outputs=[state, status, waveform, results_df, summary_df, csv_file])
    play_btn.click(play_selected_segment, inputs=[state, segment_index], outputs=[segment_audio, status])


app = FastAPI(title="Indic Speech Annotation Tool API")
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
