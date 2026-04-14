#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Build Script — Indic Speech Annotation Tool
#  Creates a standalone .app (Mac) or .exe (Windows)
# ══════════════════════════════════════════════════════════════════
#
#  STEP 1: Set up a clean virtual environment
#     python3 -m venv build_env
#     source build_env/bin/activate        # Mac/Linux
#     # build_env\Scripts\activate         # Windows
#
#  STEP 2: Install dependencies
#     pip install pyinstaller
#     pip install transformers torch torchaudio soundfile scipy
#     pip install numpy pandas matplotlib PyQt5 sounddevice openpyxl
#     pip install faster-whisper   # optional, for English
#
#  STEP 3: Run this script
#     bash build.sh
#
#  STEP 4: Find your app in dist/ folder
#     Mac:     dist/IndicAnnotationTool.app
#     Windows: dist/IndicAnnotationTool/IndicAnnotationTool.exe
#
# ══════════════════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════════"
echo "  Building Indic Speech Annotation Tool"
echo "════════════════════════════════════════════"

# Create transcripts folder if needed
mkdir -p transcripts

# Run PyInstaller
pyinstaller \
    --name "IndicAnnotationTool" \
    --windowed \
    --noconfirm \
    --clean \
    --add-data "transcripts:transcripts" \
    --hidden-import "sklearn.utils._cython_blas" \
    --hidden-import "scipy.signal" \
    --hidden-import "soundfile" \
    --hidden-import "torchaudio" \
    --hidden-import "transformers" \
    --hidden-import "huggingface_hub" \
    --hidden-import "tokenizers" \
    --hidden-import "safetensors" \
    --hidden-import "ctranslate2" \
    --hidden-import "faster_whisper" \
    --hidden-import "matplotlib.backends.backend_qtagg" \
    --collect-all "transformers" \
    --collect-all "torchaudio" \
    --collect-all "tokenizers" \
    --collect-data "huggingface_hub" \
    --exclude-module "tkinter" \
    --exclude-module "test" \
    --exclude-module "unittest" \
    app.py

echo ""
echo "════════════════════════════════════════════"
echo "  BUILD COMPLETE!"
echo ""
echo "  Your app is in: dist/IndicAnnotationTool/"
echo ""
echo "  To distribute:"
echo "    1. Copy the dist/IndicAnnotationTool/ folder"
echo "    2. Include the transcripts/ folder"
echo "    3. The app auto-downloads models on first run"
echo "════════════════════════════════════════════"
