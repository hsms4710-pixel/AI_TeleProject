@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo        BERT4MIMO 项目启动
echo ============================================================
echo.

REM 检查虚拟环境
echo 第1步/4: 检查虚拟环境...

if exist "..\\.venv\\Scripts\\python.exe" (
    echo    虚拟环境已存在
    set "PYTHON_EXE=..\\.venv\\Scripts\\python.exe"
) else if exist ".venv\\Scripts\\python.exe" (
    echo    虚拟环境已存在
    set "PYTHON_EXE=.venv\\Scripts\\python.exe"
) else (
    echo    虚拟环境不存在，正在创建...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    set "PYTHON_EXE=.venv\\Scripts\\python.exe"
)

REM 检查依赖
echo.
echo 第2步/4: 检查依赖...

"%PYTHON_EXE%" -c "import torch, transformers, gradio" >nul 2>&1
if errorlevel 1 (
    echo    依赖不完整，正在安装...
    echo    这可能需要几分钟，请耐心等待...
    
    "%PYTHON_EXE%" -m pip install --quiet --upgrade pip
    "%PYTHON_EXE%" -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    "%PYTHON_EXE%" -m pip install --quiet -r requirements.txt
    
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
    echo    依赖安装完成
) else (
    echo    依赖已安装
)

REM 验证安装
echo.
echo 第3步/4: 验证安装...
"%PYTHON_EXE%" -c "import torch; import transformers; import gradio; print('   所有关键依赖已安装')" 2>nul

REM 启动WebUI
echo.
echo 第4步/4: 启动WebUI...
echo.
echo ============================================================
echo 访问地址: http://127.0.0.1:7861
echo 按 Ctrl+C 停止服务器
echo ============================================================
echo.

"%PYTHON_EXE%" webui\app.py

if errorlevel 1 (
    echo.
    echo [错误] WebUI启动失败
    pause
    exit /b 1
)
