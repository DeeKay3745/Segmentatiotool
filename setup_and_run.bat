@echo off
REM ══════════════════════════════════════════════════════════
REM  Indic Speech Annotation Tool — Setup & Run (Windows)
REM  Just double-click this file
REM ══════════════════════════════════════════════════════════

cd /d "%~dp0"

echo ══════════════════════════════════════════
echo   Indic Speech Annotation Tool
echo ══════════════════════════════════════════

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo    Download from: https://www.python.org/downloads/
    echo    IMPORTANT: Check "Add Python to PATH" during install
    pause
    exit /b 1
)

REM Create venv if not exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate
call venv\Scripts\activate

REM Install dependencies
echo 📦 Installing dependencies...
pip install -q -r requirements.txt

REM Create transcripts folder
if not exist "transcripts" mkdir transcripts

REM Launch
echo 🚀 Launching app...
python app.py

pause
