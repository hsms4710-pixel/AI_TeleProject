@echo off
chcp 65001 >nul
SETLOCAL EnableDelayedExpansion

echo.
echo ==========================================
echo   BERT4MIMO WebUI Launcher
echo ==========================================
echo.

REM Check if virtual environment exists
if exist "..\\.venv\\Scripts\\python.exe" (
    echo [*] Using virtual environment Python...
    set PYTHON_CMD="..\\.venv\\Scripts\\python.exe"
) else if exist ".venv\\Scripts\\python.exe" (
    echo [*] Using virtual environment Python...
    set PYTHON_CMD=".venv\\Scripts\\python.exe"
) else (
    echo [*] Using system Python...
    set PYTHON_CMD=python
)

REM Check Python availability
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [*] Python found
%PYTHON_CMD% --version

REM Check if webui exists
if not exist "webui\\app.py" (
    echo [ERROR] webui\\app.py not found!
    pause
    exit /b 1
)

echo.
echo [*] Starting WebUI...
echo [*] Access: http://127.0.0.1:7861
echo [*] Press Ctrl+C to stop
echo ==========================================
echo.

REM Start WebUI
%PYTHON_CMD% webui\\app.py

if errorlevel 1 (
    echo.
    echo [ERROR] WebUI failed to start
    echo.
    echo Possible causes:
    echo   1. Missing dependencies - run: pip install -r requirements.txt
    echo   2. Port 7861 is already in use
    echo   3. Python version too old (need 3.8+)
    echo.
    pause
    exit /b 1
)

echo.
echo [*] WebUI closed
pause
