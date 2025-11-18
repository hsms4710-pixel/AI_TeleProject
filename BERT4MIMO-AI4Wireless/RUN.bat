@echo off
REM CSIBERT 命令行训练脚本
REM 适合高级用户和自动化脚本调用
REM 推荐使用 START.bat 启动 WebUI 进行可视化训练
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo        BERT4MIMO-AI4Wireless 训练脚本
echo ============================================================
echo.

REM 检测Python环境
echo [1/3] 检测Python环境...

if exist "..\\.venv\\Scripts\\python.exe" (
    echo    ✓ 使用项目虚拟环境: ..\\.venv
    set "PYTHON_EXE=..\\.venv\\Scripts\\python.exe"
) else if exist ".venv\\Scripts\\python.exe" (
    echo    ✓ 使用本地虚拟环境: .venv
    set "PYTHON_EXE=.venv\\Scripts\\python.exe"
) else (
    echo    ✓ 使用系统Python
    set "PYTHON_EXE=python"
)

REM 检查依赖
echo.
echo [2/3] 检查依赖...
"%PYTHON_EXE%" -c "import torch, transformers, sklearn" >nul 2>&1
if errorlevel 1 (
    echo    ⚠ 依赖不完整，请先运行: pip install -r requirements.txt
    echo    ⚠ 如果需要安装PyTorch，请访问: https://pytorch.org/
    pause
    exit /b 1
) else (
    echo    ✓ 依赖检查通过
)

REM 运行训练脚本
echo.
echo [3/3] 启动训练...
echo ============================================================
echo.

"%PYTHON_EXE%" train.py ^
    --hidden_size 256 ^
    --num_hidden_layers 4 ^
    --num_attention_heads 4 ^
    --learning_rate 1e-4 ^
    --batch_size 16 ^
    --num_epochs 50

echo.
echo ============================================================
if errorlevel 1 (
    echo ❌ 训练过程中出现错误
) else (
    echo ✅ 训练完成
    echo.
    echo 生成的文件:
    echo   - checkpoints/best_model.pt       (最佳模型)
    echo   - training_validation_loss.png    (训练曲线)
    echo   - validation_data/test_data.npy   (测试数据)
    echo.
    echo 下一步:
    echo   运行 model_validation.py 进行模型验证
)
echo ============================================================

pause
