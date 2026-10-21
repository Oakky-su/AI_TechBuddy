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

echo [1/2] Installing packages...
python -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & pause & exit /b 1 )


echo.
echo [2/2] Starting Tech Buddy server...
echo Open your browser at: http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause
