#!/usr/bin/env python3
"""
Indic Speech Annotation Tool
==============================
Languages: Gujarati, Hindi, Marathi, English
Models auto-selected: IndicConformer (Indic) | Whisper large-v3 (English)
Auto-detects: char → word → sentence/paragraph per segment
"""

import os, sys, csv, re, time, traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd, sounddevice as sd
import matplotlib; matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QGroupBox, QDoubleSpinBox, QSpinBox, QTabWidget)

try: import soundfile as sf; _SF=True
except: _SF=False
try: from scipy.signal import resample_poly; from math import gcd; _SCIPY=True
except: _SCIPY=False
try: import torch,torchaudio; from transformers import AutoModel,AutoModelForSpeechSeq2Seq,AutoProcessor; _HF=True
except: _HF=False
try: from faster_whisper import WhisperModel as _FWModel; _FW=True
except: _FW=False
if not _HF and not _FW: print("pip install transformers torch torchaudio (or) faster-whisper"); sys.exit(1)

SR16=16000; CPU_THR=max(1,min(os.cpu_count() or 4,8))
APP_DIR=Path(__file__).parent.resolve(); TRANSCRIPTS_DIR=APP_DIR/"transcripts"
LANGUAGES=[("Gujarati","gu","gujarati"),("Hindi","hi","hindi"),("Marathi","mr","marathi"),("English","en","english")]
INDIC_MODEL=("indicconformer","ai4bharat/indic-conformer-600m-multilingual") if _HF else None
ENGLISH_MODEL=("fw","large-v3") if _FW else (("hf_whisper","openai/whisper-large-v3") if _HF else None)

_INDIC=r"\u0900-\u097F\uA8E0-\uA8FF\u0A80-\u0AFF"
_RE_CLEAN=re.compile(rf"[^A-Za-z0-9\s{_INDIC}.,!?'\-]",re.UNICODE)
_RE_SPACES=re.compile(r"\s+"); _RE_NORM=re.compile(rf"[^a-z0-9{_INDIC}]",re.UNICODE)
_RE_ALPHA=re.compile(rf"[A-Za-z{_INDIC}]",re.UNICODE)
_RE_SEQ=re.compile(r"^(?:s|p|sent|para|sentence|paragraph|section|seg)\s*\d*$",re.I)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

_PLACEHOLDER=["write your","લખો","लिखें","लिहा","your first","your second","your third",
    "your fourth","your fifth","your paragraph","અહીં તમારું","अपना","तुमचे","येथे"]

class TranscriptStore:
    def __init__(self): self.data={}; self.lang_code=None; self.file_path=None
    def load(self,lang_iso):
        self.data={}; self.lang_code=lang_iso; self.file_path=TRANSCRIPTS_DIR/f"{lang_iso}.txt"
        if not self.file_path.exists(): return False,f"transcripts/{lang_iso}.txt not found"
        try:
            text=self.file_path.read_text(encoding="utf-8"); ck=None; cl=[]
            for line in text.splitlines():
                ls=line.strip()
                if ls.startswith("#"): continue
                m=re.match(r"^\[(.+?)\]$",ls)
                if m:
                    if ck is not None: self.data[ck]="\n".join(cl).strip()
                    ck=m.group(1).strip().lower(); cl=[]
                elif ck is not None: cl.append(line.rstrip())
            if ck is not None: self.data[ck]="\n".join(cl).strip()
            return True,f"{len(self.data)} sections"
        except Exception as e: return False,str(e)
    def get_reference(self,label): return self.data.get(label.strip().lower(),"")
    def is_placeholder(self,key):
        t=self.data.get(key.lower(),"")
        if not t: return True
        return any(p in t.lower() for p in _PLACEHOLDER)
    def get_completeness(self):
        r={}
        for k in ["s1","s2","s3","s4","s5","paragraph"]:
            if k not in self.data: r[k]="missing"
            elif self.is_placeholder(k): r[k]="placeholder"
            else: r[k]="ok"
        return r
    @property
    def is_loaded(self): return bool(self.data)
TRANSCRIPTS=TranscriptStore()

def is_sequence_label(label):
    label=str(label).strip()
    if not label: return False
    if _RE_SEQ.match(label): return True
    return label.lower() in {"paragraph","para","passage"}

def classify_segment(label):
    label=str(label).strip(); ll=label.lower()
    if is_sequence_label(label):
        ref=TRANSCRIPTS.get_reference(label)
        is_ph=TRANSCRIPTS.is_placeholder(label) if ref else True
        st="paragraph" if "para" in ll or "passage" in ll else "sentence"
        return {"type":st,"mode":"full","compare_to":ref if(ref and not is_ph)else None,
                "is_seq":True,"ref_key":label,"ref_status":"ok" if(ref and not is_ph)else("placeholder" if ref else "missing")}
    chars=re.sub(r"\s+","",label)
    if len(chars)==1: return {"type":"char","mode":"char","compare_to":label,"is_seq":False}
    if len(label.split())==1: return {"type":"word","mode":"word","compare_to":label,"is_seq":False}
    return {"type":"sentence","mode":"full","compare_to":label,"is_seq":False}

def clean_text(text,mode="full"):
    if not text: return ""
    text=_RE_CLEAN.sub("",str(text).strip()); text=_RE_SPACES.sub(" ",text).strip()
    if not text: return ""
    if mode=="word": return text.split()[0] if text.split() else ""
    if mode=="char":
        m=_RE_ALPHA.search(text); return m.group(0) if m else ""
    return text
def norm_text(s): return _RE_NORM.sub("",str(s).strip().lower())
def edit_distance(a,b):
    na,nb=len(a),len(b)
    if na==0: return nb
    if nb==0: return na
    dp=list(range(nb+1))
    for i in range(1,na+1):
        prev,dp[0]=dp[0],i
        for j in range(1,nb+1):
            tmp=dp[j]; dp[j]=prev if a[i-1]==b[j-1] else 1+min(prev,dp[j],dp[j-1]); prev=tmp
    return dp[nb]
def accuracy(pred,gt):
    p,g=norm_text(pred),norm_text(gt)
    if not g and not p: return 1.0
    if not g: return 0.0
    return max(0.0,1.0-edit_distance(p,g)/len(g))
def load_audio(path):
    if _SF:
        try:
            data,sr=sf.read(path,dtype="float32",always_2d=True)
            y=data.mean(axis=1) if data.shape[1]>1 else data[:,0]; return y.astype(np.float32),sr
        except: pass
    import librosa; y,sr=librosa.load(path,sr=None,mono=True); return y.astype(np.float32),sr
def resample_audio(y,osr,tsr):
    if osr==tsr: return y
    if _SCIPY: g=gcd(int(osr),int(tsr)); return resample_poly(y,tsr//g,osr//g).astype(np.float32)
    import librosa; return librosa.resample(y,orig_sr=osr,target_sr=tsr).astype(np.float32)
def parse_time(x):
    if pd.isna(x): return None
    x=str(x).strip()
    if not x: return None
    for fmt in ("%H:%M:%S.%f","%H:%M:%S","%M:%S.%f","%M:%S"):
        try: dt=datetime.strptime(x,fmt); return dt.hour*3600+dt.minute*60+dt.second+dt.microsecond/1e6
        except: continue
    try: return float(x)
    except: return None
@lru_cache(maxsize=8)
def get_font(lc):
    m={"hi":(["NotoSansDevanagari-Regular.ttf","Lohit-Devanagari.ttf"],["Noto Sans Devanagari","Lohit Devanagari","Mangal","Arial Unicode MS"]),
       "gu":(["NotoSansGujarati-Regular.ttf","Lohit-Gujarati.ttf"],["Noto Sans Gujarati","Lohit Gujarati","Shruti","Arial Unicode MS"]),
       "en":([],["Arial","Helvetica","DejaVu Sans"])}
    lc="hi" if lc=="mr" else lc; fns,fams=m.get(lc,m["en"])
    for d in ["/usr/share/fonts","/Library/Fonts",os.path.expanduser("~/Library/Fonts"),os.path.expanduser("~/.local/share/fonts")]:
        if not os.path.isdir(d): continue
        for root,_,files in os.walk(d):
            for fn in fns:
                if fn in files: return fm.FontProperties(fname=os.path.join(root,fn))
    inst={f.name for f in fm.fontManager.ttflist}
    for f in fams:
        if f in inst: return fm.FontProperties(family=f)
    return None
def vad_segment(y,sr,silence_ms=300,thresh_db=-40,min_ms=200,max_ms=10000):
    fl=int(sr*0.025);hp=int(sr*0.010);nf=1+(len(y)-fl)//hp
    if nf<=0: return []
    idx=np.arange(fl)[None,:]+np.arange(nf)[:,None]*hp; np.clip(idx,0,len(y)-1,out=idx)
    rms=np.sqrt(np.mean(y[idx]**2,axis=1)+1e-12)
    rms_db=20*np.log10(rms/(rms.max()+1e-12)+1e-12); v=rms_db>thresh_db
    pad=np.concatenate([[False],v,[False]]); d=np.diff(pad.astype(np.int8))
    ss,ee=np.where(d==1)[0],np.where(d==-1)[0]
    sf2=max(1,int(silence_ms/1000*sr/hp));mf=max(1,int(min_ms/1000*sr/hp));xf=max(1,int(max_ms/1000*sr/hp))
    mg=[];i=0
    while i<len(ss):
        s,e=ss[i],ee[i]
        while i+1<len(ss) and(ss[i+1]-e)<sf2: i+=1;e=ee[i]
        mg.append((s,e));i+=1
    r=[]
    for s,e in mg:
        if(e-s)<mf: continue
        for cs in range(s,e,xf):
            ce=min(cs+xf,e)
            if(ce-cs)>=mf: r.append({"start":round(cs*hp/sr,4),"end":round(min(ce*hp/sr,len(y)/sr),4)})
    return r

class Engine:
    def __init__(self): self.backend=self.processor=self.device=self.engine_key=None; self._ck=None
    @property
    def is_loaded(self): return self.backend is not None
    def load(self,ek,mid):
        ck=(ek,mid)
        if self._ck==ck and self.backend: return True,f"Already loaded: {mid}",0.0
        t0=time.time(); self.backend=self.processor=None; self.engine_key=ek
        try:
            if ek=="indicconformer": self._lic(mid)
            elif ek=="fw": self._lfw(mid)
            elif ek=="hf_whisper": self._lhf(mid)
            else: return False,f"Unknown: {ek}",0.0
            self._ck=ck; return True,f"Loaded {mid} in {time.time()-t0:.1f}s",time.time()-t0
        except Exception as e: traceback.print_exc();self.backend=None;return False,str(e),time.time()-t0
    def _lic(self,mid):
        self.device="cuda" if(_HF and torch.cuda.is_available())else "cpu"
        log(f"Loading IndicConformer on {self.device}...")
        self.backend=AutoModel.from_pretrained(mid,trust_remote_code=True)
        if hasattr(self.backend,'to'):
            try: self.backend=self.backend.to(self.device)
            except: self.device="cpu"
    def _lfw(self,sz):
        hc=_HF and torch.cuda.is_available() if _HF else False
        log(f"Loading faster-whisper {sz}...")
        self.backend=_FWModel(sz,device="cuda" if hc else "cpu",compute_type="float16" if hc else "int8",cpu_threads=CPU_THR)
    def _lhf(self,mid):
        dv="cuda:0" if torch.cuda.is_available() else "cpu"
        dt=torch.float16 if torch.cuda.is_available() else torch.float32
        log(f"Loading HF Whisper {mid}...")
        self.backend=AutoModelForSpeechSeq2Seq.from_pretrained(mid,torch_dtype=dt,low_cpu_mem_usage=True,use_safetensors=True).to(dv)
        self.processor=AutoProcessor.from_pretrained(mid);self.device=dv
    def transcribe(self,a16,liso,lname,mode="full"):
        if self.backend is None or a16 is None or len(a16)==0: return ""
        if len(a16)<int(0.1*SR16): a16=np.pad(a16,(0,int(0.1*SR16)-len(a16)))
        a16=a16.astype(np.float32)
        if self.engine_key=="indicconformer": return self._tic(a16,liso,mode)
        elif self.engine_key=="fw": return self._tfw(a16,liso,mode)
        elif self.engine_key=="hf_whisper": return self._thf(a16,lname,mode)
        return ""
    def _tic(self,a,li,m):
        w=torch.tensor(a,dtype=torch.float32).unsqueeze(0)
        if self.device and self.device!="cpu": w=w.to(self.device)
        with torch.inference_mode(): t=self.backend(w,li,"ctc")
        if isinstance(t,(list,tuple)): t=t[0] if t else ""
        return clean_text(str(t),m)
    def _tfw(self,a,li,m):
        sg,_=self.backend.transcribe(a,language=li,task="transcribe",beam_size=1,best_of=1,vad_filter=False,without_timestamps=True,condition_on_previous_text=False)
        t=" ".join(s.text.strip() for s in list(sg) if s.text.strip()); return clean_text(t,m)
    def _thf(self,a,ln,m):
        inp=self.processor(a,sampling_rate=SR16,return_tensors="pt",return_attention_mask=True)
        ft=inp.input_features.to(self.device);mk=getattr(inp,"attention_mask",None)
        if mk is not None: mk=mk.to(self.device)
        gk={"input_features":ft,"task":"transcribe","language":ln,"return_timestamps":False}
        if mk is not None: gk["attention_mask"]=mk
        with torch.inference_mode(): ids=self.backend.generate(**gk)
        return clean_text(self.processor.batch_decode(ids,skip_special_tokens=True)[0],m)
ENGINE=Engine()

class AudioLoadThread(QThread):
    done=pyqtSignal(np.ndarray,int,np.ndarray,str); error=pyqtSignal(str)
    def __init__(self,p): super().__init__();self.path=p
    def run(self):
        try: y,sr=load_audio(self.path);y16=resample_audio(y,sr,SR16);self.done.emit(y,sr,y16,self.path)
        except Exception as e: traceback.print_exc();self.error.emit(str(e))
class ModelLoadThread(QThread):
    done=pyqtSignal(bool,str,float); error=pyqtSignal(str)
    def __init__(self,ek,mid): super().__init__();self.ek,self.mid=ek,mid
    def run(self):
        try: ok,msg,el=ENGINE.load(self.ek,self.mid);self.done.emit(ok,msg,el)
        except Exception as e: traceback.print_exc();self.error.emit(str(e))
class PredictionThread(QThread):
    seg_result=pyqtSignal(int,str,float,str); progress=pyqtSignal(int,int)
    done=pyqtSignal(list,float,float); error=pyqtSignal(str)
    def __init__(self,y16,segs,liso,lname):
        super().__init__();self.y16,self.segs=y16,segs;self.liso,self.lname=liso,lname;self._stop=False
    def request_stop(self): self._stop=True
    def run(self):
        try:
            t0=time.time();results=[];total=len(self.segs)
            for i,seg in enumerate(self.segs):
                if self._stop: break
                info=classify_segment(seg["label"])
                s=max(0,int(seg["start"]*SR16));e=min(len(self.y16),int(seg["end"]*SR16))
                pred=ENGINE.transcribe(self.y16[s:e],self.liso,self.lname,info["mode"])
                if info["is_seq"] and info["compare_to"]: acc=accuracy(pred,info["compare_to"])
                elif info["is_seq"]: acc=-1.0
                else: acc=accuracy(pred,seg["label"])
                results.append({"i":i,"gt":seg["label"],"pred":pred,"start":seg["start"],"end":seg["end"],
                    "acc":acc,"type":info["type"],"mode":info["mode"],"is_seq":info["is_seq"],
                    "compare_to":info["compare_to"] or "","ref_status":info.get("ref_status","")})
                self.seg_result.emit(i,pred,acc,info["type"]); self.progress.emit(i+1,total)
            el=time.time()-t0;sc=[r["acc"] for r in results if r["acc"]>=0]
            self.done.emit(results,float(np.mean(sc)) if sc else 0.0,el)
        except Exception as e: traceback.print_exc();self.error.emit(str(e))
class VADThread(QThread):
    done=pyqtSignal(list);error=pyqtSignal(str)
    def __init__(self,y,sr,p): super().__init__();self.y,self.sr,self.p=y,sr,p
    def run(self):
        try: self.done.emit(vad_segment(self.y,self.sr,**self.p))
        except Exception as e: traceback.print_exc();self.error.emit(str(e))

TI={"char":"Ⓒ","word":"Ⓦ","sentence":"Ⓢ","paragraph":"Ⓟ"}
TC={"char":QColor("#7c3aed"),"word":QColor("#0284c7"),"sentence":QColor("#16a34a"),"paragraph":QColor("#ea580c")}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Indic Speech Annotation Tool"); self.resize(1700,1000)
        self.audio_path=None;self.y=self.y16=self.wave_t=self.wave_y=None;self.sr=None
        self.segments,self.results=[],[];self.selected_idx=None;self._threads={};self._pred_t0=None;self._timer=None
        self._build_ui()

    def _build_ui(self):
        root=QWidget();self.setCentralWidget(root);L=QVBoxLayout(root);L.setSpacing(4);L.setContentsMargins(8,8,8,8)
        r1=QHBoxLayout()
        self.cb_lang=QComboBox()
        for d,iso,name in LANGUAGES: self.cb_lang.addItem(d,(iso,name))
        self.cb_lang.currentIndexChanged.connect(self._on_lang_changed)
        self.lbl_mn=QLabel("");self.lbl_mn.setStyleSheet("color:#374151;font-size:11px;padding:2px 6px")
        self.btn_audio=QPushButton("Load Audio");self.btn_audio.clicked.connect(self.on_load_audio)
        self.btn_gt=QPushButton("Load Ground Truth");self.btn_gt.clicked.connect(self.on_load_gt)
        self.btn_model=QPushButton("⬇ Load Model");self.btn_model.clicked.connect(self.on_load_model)
        self.btn_model.setStyleSheet("QPushButton{font-weight:bold;color:white;background:#2563eb;border:1px solid #1d4ed8;padding:5px 14px;border-radius:4px}QPushButton:hover{background:#1d4ed8}QPushButton:disabled{color:#9ca3af;background:#f3f4f6}")
        for w in [QLabel("Language:"),self.cb_lang,self.lbl_mn,self.btn_audio,self.btn_gt,self.btn_model]: r1.addWidget(w)
        r1.addStretch()

        r2=QHBoxLayout()
        actions=[("▶ Run",self.on_run,"font-weight:bold;color:#16a34a"),("⏹ Stop",self.on_stop,""),
            ("VAD Segment",self.on_vad,""),("♫ Play Seg",self.on_play_seg,""),("⏸ Stop",sd.stop,""),
            ("🔍+",lambda:self._zoom(0.5),""),("🔍−",lambda:self._zoom(2.0),""),
            ("Fit",self._fit,""),("Reset",self._reset,""),("Export CSV",self.on_export,"")]
        self._btns={}
        for n,fn,st in actions:
            b=QPushButton(n);b.clicked.connect(fn)
            if st: b.setStyleSheet(f"QPushButton{{{st}}}")
            r2.addWidget(b);self._btns[n]=b
        self._btns["⏹ Stop"].setEnabled(False);r2.addStretch()

        sr=QHBoxLayout()
        self.lbl_status=QLabel("Select language → Load Model → Load Audio → Load GT → Run")
        self.lbl_status.setStyleSheet("padding:4px;font-size:13px")
        self.lbl_speed=QLabel("");self.lbl_speed.setStyleSheet("color:#6b7280;font-size:12px")
        sr.addWidget(self.lbl_status,1);sr.addWidget(self.lbl_speed)
        self.pbar=QProgressBar();self.pbar.setRange(0,100);self.pbar.setValue(0);self.pbar.setFixedHeight(18)
        self.pbar.setStyleSheet("QProgressBar{border:1px solid #ccc;border-radius:4px;background:#f3f4f6;text-align:center;font-size:11px}QProgressBar::chunk{background:#3b82f6;border-radius:4px}")

        self.fig,self.ax=plt.subplots(figsize=(14,5));self.fig.set_tight_layout(True)
        self.canvas=FigureCanvas(self.fig);self.toolbar=NavigationToolbar(self.canvas,self)
        self.canvas.mpl_connect("scroll_event",self._on_scroll)

        self.seg_list=QListWidget();self.seg_list.itemClicked.connect(self.on_seg_clicked)
        self.seg_list.setStyleSheet("QListWidget{background:#fff;border:1px solid #d0d0d0;font-family:Consolas,Courier New,monospace;font-size:11px}QListWidget::item{padding:3px 6px}QListWidget::item:selected{background:#dbeafe}")
        vad_box=QGroupBox("VAD");vl=QVBoxLayout(vad_box)
        self.sp_sil=QSpinBox();self.sp_sil.setRange(50,2000);self.sp_sil.setValue(300);self.sp_sil.setSuffix(" ms")
        self.sp_db=QDoubleSpinBox();self.sp_db.setRange(-80,0);self.sp_db.setValue(-40);self.sp_db.setSuffix(" dB")
        self.sp_min=QSpinBox();self.sp_min.setRange(50,5000);self.sp_min.setValue(200);self.sp_min.setSuffix(" ms")
        self.sp_max=QSpinBox();self.sp_max.setRange(1000,30000);self.sp_max.setValue(10000);self.sp_max.setSuffix(" ms")
        for lb,w in [("Silence:",self.sp_sil),("Thresh:",self.sp_db),("Min:",self.sp_min),("Max:",self.sp_max)]:
            h=QHBoxLayout();h.addWidget(QLabel(lb));h.addWidget(w);vl.addLayout(h)
        vl.addStretch()
        tabs=QTabWidget()
        seg_tab=QWidget();sl=QVBoxLayout(seg_tab)
        legend=QLabel('<span style="color:#7c3aed;font-weight:bold">Ⓒ Char</span>  <span style="color:#0284c7;font-weight:bold">Ⓦ Word</span>  <span style="color:#16a34a;font-weight:bold">Ⓢ Sentence</span>  <span style="color:#ea580c;font-weight:bold">Ⓟ Para</span>  │  <span style="color:#16a34a">✓≥95%</span> <span style="color:#d97706">~50%</span> <span style="color:#dc2626">✗&lt;50%</span> <span style="color:#6366f1">📝no-ref</span>')
        legend.setTextFormat(Qt.RichText);sl.addWidget(legend);sl.addWidget(self.seg_list)
        tabs.addTab(seg_tab,"Segments");tabs.addTab(vad_box,"VAD")
        left=QWidget();ll=QVBoxLayout(left);ll.addWidget(self.toolbar);ll.addWidget(self.canvas)
        spl=QSplitter(Qt.Horizontal);spl.addWidget(left);spl.addWidget(tabs);spl.setSizes([1250,400])
        L.addLayout(r1);L.addLayout(r2);L.addLayout(sr);L.addWidget(self.pbar);L.addWidget(spl)
        self._on_lang_changed()

    def _set_busy(self,busy,msg=""):
        en=not busy
        for n,b in self._btns.items(): b.setEnabled(busy if n=="⏹ Stop" else en)
        for w in [self.btn_audio,self.btn_gt,self.btn_model,self.cb_lang]: w.setEnabled(en)
        if msg: self.lbl_status.setText(msg)
    def _build_wave(self):
        if self.y is None: self.wave_t=self.wave_y=None;return
        st=max(1,len(self.y)//30000);self.wave_y=self.y[::st];self.wave_t=np.arange(len(self.wave_y))*(st/self.sr)
    def _get_model(self,iso): return INDIC_MODEL if iso in("gu","hi","mr") else ENGLISH_MODEL

    def _on_lang_changed(self):
        iso,name=self.cb_lang.currentData()
        mdl=self._get_model(iso)
        if mdl:
            ek,mid=mdl;mn="IndicConformer-600M" if ek=="indicconformer" else f"Whisper {mid.split('/')[-1]}"
            self.lbl_mn.setText(f"Model: {mn}")
            # Check if loaded model matches what this language needs
            if ENGINE.is_loaded and ENGINE._ck==(ek,mid):
                self.btn_model.setText("✓ Loaded")
                self.btn_model.setStyleSheet("QPushButton{font-weight:bold;color:white;background:#16a34a;border:1px solid #15803d;padding:5px 14px;border-radius:4px}")
            else:
                self.btn_model.setText("⬇ Load Model")
                self.btn_model.setStyleSheet("QPushButton{font-weight:bold;color:white;background:#2563eb;border:1px solid #1d4ed8;padding:5px 14px;border-radius:4px}QPushButton:hover{background:#1d4ed8}QPushButton:disabled{color:#9ca3af;background:#f3f4f6}")
                if ENGINE.is_loaded:
                    self.lbl_status.setText(f"⚠ Switch to {name} requires loading {mn} — click Load Model")
        else: self.lbl_mn.setText("⚠ No model")
        TRANSCRIPTS.load(iso)

    def on_load_audio(self):
        path,_=QFileDialog.getOpenFileName(self,"Audio","","Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not path: return
        self.lbl_status.setText(f"Loading...");self.btn_audio.setEnabled(False)
        t=AudioLoadThread(path);t.done.connect(self._aok);t.error.connect(self._aerr);self._threads["audio"]=t;t.start()
    def _aok(self,y,sr,y16,path):
        self.y,self.sr,self.y16,self.audio_path=y,sr,y16,path;self.results=[];self.selected_idx=None;self.pbar.setValue(0)
        self._build_wave();self.lbl_status.setText(f"Audio: {os.path.basename(path)} ({len(y)/sr:.1f}s)");self.btn_audio.setEnabled(True)
        self._refresh_list();self._redraw()
    def _aerr(self,m): self.btn_audio.setEnabled(True);QMessageBox.critical(self,"Error",m)

    def on_load_gt(self):
        path,_=QFileDialog.getOpenFileName(self,"GT","","Spreadsheets (*.xlsx *.csv *.tsv)")
        if not path: return
        try:
            if path.endswith(".csv"): df=pd.read_csv(path,header=None)
            elif path.endswith(".tsv"): df=pd.read_csv(path,header=None,sep="\t")
            else: df=pd.read_excel(path,sheet_name=0,header=None,engine="openpyxl")
            segs,skip=[],0
            for _,row in df.iterrows():
                lab=str(row.iloc[0]).strip();s,e=parse_time(row.iloc[1]),parse_time(row.iloc[2])
                if not lab or s is None or e is None or e<=s: skip+=1;continue
                segs.append({"label":lab,"start":s,"end":e})
            self.segments=segs;self.results=[];self.selected_idx=None
            types={}
            for seg in segs: t=classify_segment(seg["label"])["type"];types[t]=types.get(t,0)+1
            ts=" ".join(f"{k}:{v}" for k,v in types.items())
            self.lbl_status.setText(f"GT: {len(segs)} segs │ {ts}"+(f" │ skip:{skip}" if skip else ""))
            self._refresh_list();self._redraw()
        except Exception as e: traceback.print_exc();QMessageBox.critical(self,"Error",str(e))

    def on_load_model(self):
        t=self._threads.get("model")
        if t and t.isRunning(): QMessageBox.information(self,"","Still loading...");return
        iso,_=self.cb_lang.currentData();mdl=self._get_model(iso)
        if not mdl: QMessageBox.critical(self,"","No model available.");return
        ek,mid=mdl;self._set_busy(True,f"Loading {mid}...")
        self.btn_model.setText("Loading...");self.btn_model.setEnabled(False)
        self._lt0=time.time();self._timer=QTimer()
        self._timer.timeout.connect(lambda:self.lbl_status.setText(f"Loading... {int(time.time()-self._lt0)}s"))
        self._timer.start(1000);t=ModelLoadThread(ek,mid);t.done.connect(self._mok);t.error.connect(self._merr)
        self._threads["model"]=t;t.start()
    def _st(self):
        if self._timer: self._timer.stop();self._timer=None
    def _mok(self,ok,msg,el):
        self._st();self._set_busy(False);self.btn_model.setEnabled(True)
        if ok:
            self.btn_model.setText("✓ Loaded");self.btn_model.setStyleSheet("QPushButton{font-weight:bold;color:white;background:#16a34a;border:1px solid #15803d;padding:5px 14px;border-radius:4px}")
            self.lbl_status.setText(f"✓ {msg}");self.lbl_speed.setText(f"Engine: {ENGINE.engine_key}")
        else: self.btn_model.setText("⬇ Load Model");QMessageBox.critical(self,"Error",f"Failed:\n{msg}")
    def _merr(self,m): self._st();self._set_busy(False);self.btn_model.setText("⬇ Load Model");self.btn_model.setEnabled(True);QMessageBox.critical(self,"Error",m)

    def on_vad(self):
        if self.y is None: QMessageBox.warning(self,"","Load audio.");return
        p={"silence_ms":self.sp_sil.value(),"thresh_db":self.sp_db.value(),"min_ms":self.sp_min.value(),"max_ms":self.sp_max.value()}
        self._set_busy(True,"VAD...");t=VADThread(self.y,self.sr,p);t.done.connect(self._vok)
        t.error.connect(lambda m:(self._set_busy(False),QMessageBox.critical(self,"Error",m)));self._threads["vad"]=t;t.start()
    def _vok(self,segs):
        self.segments=[{"label":f"seg_{i+1:04d}","start":s["start"],"end":s["end"]} for i,s in enumerate(segs)]
        self.results=[];self.selected_idx=None;self._set_busy(False,f"VAD: {len(segs)} segs");self._refresh_list();self._redraw()

    def on_run(self):
        if self.y16 is None: QMessageBox.warning(self,"","Load audio.");return
        if not self.segments: QMessageBox.warning(self,"","Load GT or VAD.");return
        if not ENGINE.is_loaded: QMessageBox.warning(self,"","Load model first.");return
        # Check model matches language
        iso,name=self.cb_lang.currentData()
        mdl=self._get_model(iso)
        if mdl:
            ek,mid=mdl
            if ENGINE._ck!=(ek,mid):
                mn="IndicConformer-600M" if ek=="indicconformer" else f"Whisper {mid.split('/')[-1]}"
                QMessageBox.warning(self,"Wrong Model",
                    f"Language '{name}' requires {mn},\n"
                    f"but a different model is loaded.\n\n"
                    f"Click 'Load Model' first.")
                return
        t=self._threads.get("pred")
        if t and t.isRunning(): return
        self.results=[];self.pbar.setValue(0);self._pred_t0=time.time()
        self._set_busy(True,"Running...");self._refresh_list()
        t=PredictionThread(self.y16,self.segments,iso,name)
        t.seg_result.connect(self._osr);t.progress.connect(self._opp);t.done.connect(self._opd);t.error.connect(self._ope)
        self._threads["pred"]=t;t.start()

    def _osr(self,idx,pred,acc,stype):
        if idx>=self.seg_list.count(): return
        seg=self.segments[idx];item=self.seg_list.item(idx);icon=TI.get(stype,"?")
        info=classify_segment(seg["label"])
        if info["is_seq"]:
            ref=info.get("compare_to")
            gd=(ref[:25]+"…") if ref and len(ref)>25 else (ref if ref else f"[{seg['label']}]")
        else: gd=seg["label"]
        ps=(pred[:30]+"…") if len(pred)>30 else pred;ts=f"{seg['start']:.2f}-{seg['end']:.2f}s"
        if acc<0:
            item.setText(f"{idx+1:03d} {icon} {gd}  →  {ps}  │ {ts}  📝no-ref");item.setForeground(QColor("#6366f1"))
        else:
            item.setText(f"{idx+1:03d} {icon} {gd}  →  {ps}  │ {ts}  acc={acc:.2f}")
            if acc>=0.95: item.setForeground(Qt.darkGreen)
            elif acc>=0.5: item.setForeground(Qt.darkYellow)
            else: item.setForeground(Qt.red)

    def _opp(self,done,total):
        self.pbar.setValue(int(done/total*100) if total else 0)
        el=time.time()-self._pred_t0;spd=done/el if el>0 else 0;eta=(total-done)/spd if spd>0 else 0
        self.lbl_status.setText(f"Predicting {done}/{total} ({spd:.1f}/s ETA {eta:.0f}s)")

    def _opd(self,results,mean_acc,elapsed):
        self.results=results;self.pbar.setValue(100);n=len(results)
        ns=sum(1 for r in results if r["acc"]>=0);ng=sum(1 for r in results if r["acc"]>=0.95)
        nnr=n-ns;spd=n/elapsed if elapsed>0 else 0
        types={};
        for r in results: types[r["type"]]=types.get(r["type"],0)+1
        ts=" ".join(f"{k}:{v}" for k,v in types.items())
        p=[f"✓ {n} segs in {elapsed:.1f}s"]
        if ns: p.append(f"Acc:{mean_acc:.3f} ({ng}/{ns}≥95%)")
        if nnr: p.append(f"📝{nnr} no-ref")
        p.append(ts);p.append(f"{spd:.1f}/s")
        self._set_busy(False," │ ".join(p));self.lbl_speed.setText(f"{spd:.1f}/s");self._refresh_list();self._redraw()

    def _ope(self,m): self.pbar.setValue(0);self._set_busy(False,f"✗ {m}");QMessageBox.critical(self,"Error",m)
    def on_stop(self):
        t=self._threads.get("pred")
        if t and t.isRunning(): t.request_stop()

    def _redraw(self,xlim=None):
        self.ax.clear()
        if self.wave_t is None: self.canvas.draw_idle();return
        self.ax.plot(self.wave_t,self.wave_y,lw=0.5,color="#1e293b")
        ymax=max(np.max(np.abs(self.wave_y)),1e-6);iso,_=self.cb_lang.currentData();fp=get_font(iso)
        if xlim is None:
            if self.segments:
                s0=min(s["start"] for s in self.segments);s1=max(s["end"] for s in self.segments)
                pad=max(1,(s1-s0)*0.05);xlim=(max(0,s0-pad),s1+pad)
            elif self.y is not None: xlim=(0,len(self.y)/self.sr)
        if xlim: self.ax.set_xlim(*xlim)
        x0,x1=self.ax.get_xlim()
        pc={"char":("#ede9fe","#7c3aed"),"word":("#e0f2fe","#0284c7"),"sentence":("#dcfce7","#16a34a"),"paragraph":("#fff7ed","#ea580c")}
        for i,seg in enumerate(self.segments):
            s,e=seg["start"],seg["end"]
            if e<x0 or s>x1: continue
            if self.results and i<len(self.results):
                r=self.results[i]
                if r["acc"]<0: cl,br="#e0e7ff","#6366f1"
                elif r["acc"]>=0.95: cl,br=pc.get(r["type"],("#e0f2fe","#0284c7"))
                elif r["acc"]>=0.5: cl,br="#fef3c7","#d97706"
                else: cl,br="#fecaca","#dc2626"
            else:
                inf=classify_segment(seg["label"]);cl,br=pc.get(inf["type"],("#e0f2fe","#0284c7"))
            self.ax.axvspan(s,e,alpha=0.2,color=cl,zorder=1)
            self.ax.plot([s,e],[-ymax*0.92,-ymax*0.92],lw=3,color=br,solid_capstyle="butt",zorder=3)
            if self.selected_idx is not None and i==self.selected_idx:
                self.ax.axvspan(s,e,alpha=0.15,color="#facc15",zorder=2);mid=(s+e)/2
                fkw={"fontproperties":fp} if fp else {};bb=dict(facecolor="white",alpha=0.9,edgecolor="#d1d5db",boxstyle="round,pad=0.3")
                self.ax.text(mid,ymax*0.9,f"GT: {seg['label']}",ha="center",va="bottom",fontsize=10,color="#15803d",fontweight="bold",bbox=bb,zorder=5,**fkw)
                if self.results and i<len(self.results):
                    r=self.results[i];p=(r['pred'][:50]+"…") if len(r['pred'])>50 else r['pred']
                    a=f"acc={r['acc']:.2f}" if r['acc']>=0 else "no-ref"
                    self.ax.text(mid,ymax*0.65,f"PRED: {p or '∅'} ({a})",ha="center",va="bottom",fontsize=9,color="#1d4ed8",fontweight="bold",bbox=bb,zorder=5,**fkw)
        title="Waveform"
        if self.results:
            sc=[r["acc"] for r in self.results if r["acc"]>=0]
            if sc: title+=f" │ Acc:{np.mean(sc):.3f} │ {sum(1 for a in sc if a>=0.95)}/{len(sc)} correct"
        self.ax.set_title(title,fontsize=11,fontweight="bold");self.ax.set_xlabel("Time (s)");self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-ymax*1.05,ymax*1.05);self.ax.grid(axis="x",alpha=0.15,linestyle="--")
        if xlim: self.ax.set_xlim(*xlim)
        self.canvas.draw_idle()

    def _refresh_list(self):
        self.seg_list.clear()
        for i,seg in enumerate(self.segments):
            info=classify_segment(seg["label"]);icon=TI.get(info["type"],"?")
            if info["is_seq"]:
                ref=info.get("compare_to")
                gd=(ref[:25]+"…") if ref and len(ref)>25 else (ref if ref else f"[{seg['label']}]")
            else: gd=seg["label"]
            ts=f"{seg['start']:.2f}-{seg['end']:.2f}s"
            if self.results and i<len(self.results):
                r=self.results[i];ps=(r['pred'][:30]+"…") if len(r['pred'])>30 else r['pred']
                if r["acc"]<0:
                    text=f"{i+1:03d} {icon} {gd}  →  {ps}  │ {ts}  📝no-ref";item=QListWidgetItem(text);item.setForeground(QColor("#6366f1"))
                else:
                    text=f"{i+1:03d} {icon} {gd}  →  {ps}  │ {ts}  acc={r['acc']:.2f}";item=QListWidgetItem(text)
                    if r["acc"]>=0.95: item.setForeground(Qt.darkGreen)
                    elif r["acc"]>=0.5: item.setForeground(Qt.darkYellow)
                    else: item.setForeground(Qt.red)
            else: text=f"{i+1:03d} {icon} {gd}  →  ...  │ {ts}";item=QListWidgetItem(text)
            self.seg_list.addItem(item)

    def on_seg_clicked(self,item):
        self.selected_idx=self.seg_list.row(item)
        if 0<=self.selected_idx<len(self.segments):
            s=self.segments[self.selected_idx];pd2=max(0.3,(s["end"]-s["start"])*0.3)
            self._redraw(xlim=(max(0,s["start"]-pd2),s["end"]+pd2))
    def on_play_seg(self):
        if self.selected_idx is not None and self.y is not None:
            s=self.segments[self.selected_idx];sd.stop();sd.play(self.y[int(s["start"]*self.sr):int(s["end"]*self.sr)],self.sr)
    def _zoom(self,f):
        if self.y is None: return
        x0,x1=self.ax.get_xlim();c=(x0+x1)/2;w=(x1-x0)*f/2
        self._redraw(xlim=(max(0,c-w),min(len(self.y)/self.sr,c+w)))
    def _on_scroll(self,ev):
        if self.y is None: return
        x0,x1=self.ax.get_xlim();mx=ev.xdata if ev.xdata else(x0+x1)/2
        f=0.5 if ev.button=="up" else 2.0
        self._redraw(xlim=(max(0,mx-(mx-x0)*f),min(len(self.y)/self.sr,mx+(x1-mx)*f)))
    def _fit(self):
        if self.segments: self._redraw()
        elif self.y is not None: self._redraw(xlim=(0,len(self.y)/self.sr))
    def _reset(self):
        if self.y is not None: self._redraw(xlim=(0,len(self.y)/self.sr))
    def on_export(self):
        if not self.results: QMessageBox.warning(self,"","Run prediction first.");return
        path,_=QFileDialog.getSaveFileName(self,"Export","results.csv","CSV (*.csv)")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["index","type","mode","ground_truth","prediction","compared_against","accuracy","start","end"])
                for r in self.results:
                    a=f"{r['acc']:.4f}" if r["acc"]>=0 else "N/A"
                    w.writerow([r["i"]+1,r["type"],r["mode"],r["gt"],r["pred"],r.get("compare_to",""),a,f"{r['start']:.4f}",f"{r['end']:.4f}"])
            self.lbl_status.setText(f"Exported: {path}")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

if __name__=="__main__":
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    print("="*60);print("  Indic Speech Annotation Tool")
    print(f"  Transcripts: {TRANSCRIPTS_DIR}")
    if _HF: print(f"  CUDA: {torch.cuda.is_available()}")
    print("="*60)
    app=QApplication(sys.argv)
    app.setStyleSheet("QPushButton{padding:5px 10px;border:1px solid #d1d5db;border-radius:4px;background:#f9fafb;font-size:12px}QPushButton:hover{background:#e5e7eb}QPushButton:disabled{color:#9ca3af;background:#f3f4f6}QComboBox,QSpinBox,QDoubleSpinBox{padding:3px 6px;font-size:12px}")
    w=MainWindow();w.show();sys.exit(app.exec_())