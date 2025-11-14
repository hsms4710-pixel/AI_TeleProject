#!/bin/bash
# BERT4MIMO 完整启动脚本 - 包括环境初始化和 WebUI 启动
# 适合首次使用或环境缺失的情况

echo ""
echo "============================================================"
echo "        BERT4MIMO 项目启动"
echo "============================================================"
echo ""

# 获取当前目录
CURRENT_DIR=$(pwd)

# 第1步：检查虚拟环境
echo "[1/4] 检查虚拟环境..."

if [ -d ".venv" ]; then
    echo "[✓] 虚拟环境已存在"
else
    echo "[*] 虚拟环境不存在，正在创建..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo "[错误] 找不到 Python3！"
        echo "请确保 Python3 已安装"
        echo "Ubuntu/Debian: sudo apt-get install python3 python3-venv"
        echo "macOS: brew install python3"
        echo ""
        exit 1
    fi
    
    echo "正在创建虚拟环境（需要几秒钟）..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败！"
        exit 1
    fi
    echo "[✓] 虚拟环境创建完成"
fi

# 第2步：安装依赖
echo ""
echo "[2/4] 检查依赖..."

PYTHON_EXE="./.venv/bin/python3"

# 尝试导入关键模块
"$PYTHON_EXE" -c "import torch, transformers, gradio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[*] 依赖不完整，正在安装..."
    echo "这可能需要 5-10 分钟，请耐心等待..."
    echo ""
    
    # 升级 pip
    "$PYTHON_EXE" -m pip install --upgrade pip >/dev/null 2>&1
    
    # 安装 PyTorch
    echo "正在安装 PyTorch（最耗时的步骤）..."
    "$PYTHON_EXE" -m pip install torch torchvision torchaudio >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[警告] PyTorch 安装可能失败，继续安装其他依赖..."
    fi
    
    # 安装 requirements
    echo "正在安装其他依赖..."
    "$PYTHON_EXE" -m pip install -r requirements.txt >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败！"
        echo ""
        exit 1
    fi
    
    echo "[✓] 依赖安装完成"
else
    echo "[✓] 依赖已安装"
fi

# 第3步：验证安装
echo ""
echo "[3/4] 验证安装..."

"$PYTHON_EXE" -c "import torch; import transformers; import gradio; print('[✓] 所有关键依赖已安装')"
if [ $? -ne 0 ]; then
    echo "[警告] 验证失败，但继续启动..."
fi

# 第4步：启动 WebUI
echo ""
echo "[4/4] 启动 WebUI..."
echo ""
echo "🌐 CSIBERT WebUI - 4大功能"
echo "============================================================"
echo "📍 地址: http://127.0.0.1:7861"
echo ""
echo "功能菜单："
echo "  1. ⚡ 一键训练 - 使用最优预设参数快速启动"
echo "  2. 📂 导入数据训练 - 自定义参数和数据进行训练"
echo "  3. 🔧 生成数据 - 生成合成CSI数据集"
echo "  4. 🔬 进行实验 - 运行8种高级实验和验证"
echo ""
echo "ℹ️ 关于 - 查看项目信息和文档"
echo ""
echo "按 Ctrl+C 停止 WebUI"
echo "============================================================"
echo ""

cd "$CURRENT_DIR"
"$PYTHON_EXE" webui/app.py 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] WebUI 启动失败！"
    echo "请检查上面的错误信息"
    echo ""
    exit 1
fi
