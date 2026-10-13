@echo off
chcp 65001 > nul
echo ============================================
echo  AI Teddy - Tech Buddy Setup
echo ============================================
echo.

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] Installing packages...
python -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & pause & exit /b 1 )

echo.
echo [2/3] Building vector database (this may take 3-5 minutes)...
python build_index.py
if errorlevel 1 ( echo ERROR: build_db.py failed & pause & exit /b 1 )

echo.
echo [3/3] Starting Tech Buddy server...
echo Open your browser at: http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause
