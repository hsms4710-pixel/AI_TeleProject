#!/usr/bin/env python3
import sys
import os

print("=" * 60)
print("WebUI 启动测试")
print("=" * 60)

# 测试导入
try:
    print("📦 检查依赖...")
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    import gradio as gr
    print(f"✓ Gradio: {gr.__version__}")
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 导入项目模块
print("\n🔍 加载项目模块...")
try:
    from webui.app import create_interface
    print("✓ WebUI 模块加载成功")
except Exception as e:
    print(f"❌ WebUI 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建界面
print("\n⚙️  创建 Gradio 界面...")
try:
    app = create_interface()
    print("✓ 界面创建成功")
except Exception as e:
    print(f"❌ 界面创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 启动服务
print("\n" + "=" * 60)
print("🌐 CSIBERT WebUI 启动中...")
print("=" * 60)
print("📍 访问地址: http://127.0.0.1:7861")
print("⏹️  按 Ctrl+C 停止服务器")
print("=" * 60 + "\n")

try:
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_api=False
    )
except KeyboardInterrupt:
    print("\n\n已停止服务器")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
