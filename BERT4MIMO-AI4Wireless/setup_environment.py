#!/usr/bin/env python3
"""
项目初始化脚本 - 自动创建虚拟环境和安装依赖

使用方法：
    python setup_environment.py

功能：
    1. 检测 Python 版本
    2. 创建虚拟环境 (.venv)
    3. 自动安装所有依赖
    4. 提示用户后续步骤
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class EnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / ".venv"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.is_windows = platform.system() == "Windows"
        self.venv_bin = self.venv_path / ("Scripts" if self.is_windows else "bin")
        self.python_exe = self.venv_bin / ("python.exe" if self.is_windows else "python")
        
    def print_header(self, message):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"  {message}")
        print(f"{'='*60}\n")
        
    def print_step(self, step_num, message):
        """打印步骤"""
        print(f"[{step_num}/5] {message}")
        
    def check_python_version(self):
        """检查 Python 版本"""
        self.print_step(1, "检查 Python 版本...")
        print(f"   当前版本: Python {self.python_version}")
        
        if sys.version_info < (3, 8):
            print("   ❌ 错误: 需要 Python 3.8 或更高版本")
            sys.exit(1)
        else:
            print("   ✅ Python 版本满足要求")
            
    def create_venv(self):
        """创建虚拟环境"""
        self.print_step(2, "创建虚拟环境...")
        
        if self.venv_path.exists():
            print(f"   ℹ️  虚拟环境已存在: {self.venv_path}")
            return True
            
        try:
            print(f"   创建虚拟环境: {self.venv_path}")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True,
                capture_output=True
            )
            print("   ✅ 虚拟环境创建成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 虚拟环境创建失败: {e}")
            return False
            
    def upgrade_pip(self):
        """升级 pip"""
        self.print_step(3, "升级 pip...")
        try:
            subprocess.run(
                [str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                timeout=60
            )
            print("   ✅ pip 升级成功")
            return True
        except Exception as e:
            print(f"   ⚠️  pip 升级失败（非关键）: {e}")
            return True  # 继续进行
            
    def install_torch(self):
        """安装 PyTorch"""
        self.print_step(4, "安装 PyTorch...")
        
        print("   正在安装 PyTorch（可能需要几分钟）...")
        
        # 检测 CUDA 支持
        torch_cmd = [
            str(self.python_exe), "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        try:
            subprocess.run(torch_cmd, check=True, timeout=600)
            print("   ✅ PyTorch 安装成功")
            return True
        except subprocess.TimeoutExpired:
            print("   ⚠️  安装超时，请检查网络连接")
            return False
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  PyTorch 安装失败: {e}")
            print("   提示: 可以手动访问 https://pytorch.org 下载合适版本")
            return False
            
    def install_requirements(self):
        """安装项目依赖"""
        self.print_step(5, "安装项目依赖...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print(f"   ❌ 找不到 requirements.txt")
            return False
            
        try:
            print("   正在安装依赖（可能需要几分钟）...")
            subprocess.run(
                [str(self.python_exe), "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                timeout=300
            )
            print("   ✅ 依赖安装成功")
            return True
        except Exception as e:
            print(f"   ❌ 依赖安装失败: {e}")
            return False
            
    def verify_installation(self):
        """验证安装"""
        print("\n验证安装...")
        try:
            result = subprocess.run(
                [str(self.python_exe), "-c", 
                 "import torch; import transformers; import gradio; print('✅ 所有依赖已安装')"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"   {result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
            return False
            
    def print_next_steps(self):
        """打印后续步骤"""
        self.print_header("✅ 环境设置完成！")
        
        if self.is_windows:
            print("后续步骤：\n")
            print("1️⃣  启动 WebUI 进行训练:")
            print(f"   .\\start_webui.bat\n")
            print("2️⃣  或运行验证脚本:")
            print(f"   .\\run_validation.bat\n")
        else:
            print("后续步骤：\n")
            print("1️⃣  启动 WebUI 进行训练:")
            print(f"   bash start_webui.sh\n")
            print("2️⃣  或运行验证脚本:")
            print(f"   bash run_validation.sh\n")
            
        print("📄 详细说明请查看: README.md\n")
        
    def run(self):
        """运行完整的设置流程"""
        self.print_header("🚀 BERT4MIMO 项目初始化")
        
        print(f"项目目录: {self.project_root}")
        print(f"Python: {sys.executable}")
        print(f"操作系统: {platform.system()}\n")
        
        # 执行各步骤
        self.check_python_version()
        
        if not self.create_venv():
            sys.exit(1)
            
        if not self.upgrade_pip():
            pass  # 非关键，继续
            
        if not self.install_torch():
            print("\n⚠️  PyTorch 安装失败，请手动安装后重试")
            print("   访问: https://pytorch.org\n")
            
        if not self.install_requirements():
            sys.exit(1)
            
        if not self.verify_installation():
            print("\n⚠️  安装验证失败，请检查日志")
            
        self.print_next_steps()


if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run()
