@echo off
REM Smart Focus System Launcher for Windows

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo Installing/Checking dependencies...
pip install opencv-python ultralytics torch numpy

echo Starting Smart Focus System...
python smart_focus.py %*

pause