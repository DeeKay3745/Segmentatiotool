#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Indic Speech Annotation Tool — Setup & Run
#  Just double-click this file or run: bash setup_and_run.sh
# ══════════════════════════════════════════════════════════

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "   Install from: https://www.python.org/downloads/"
    echo "   Or: brew install python3"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "══════════════════════════════════════════"
echo "  Indic Speech Annotation Tool"
echo "══════════════════════════════════════════"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies (first time may take a few minutes)..."
pip install -q -r requirements.txt

# Create transcripts folder
mkdir -p transcripts

# Launch
echo "🚀 Launching app..."
python app.py
