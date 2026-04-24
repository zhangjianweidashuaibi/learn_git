# -*- coding: utf-8 -*-
"""
多模态处理使用示例
演示如何运行多模态处理工具
"""

import os
import subprocess
import sys

def run_integration():
    """运行数据整合"""
    print("="*50)
    print("运行多模态数据整合...")
    print("="*50)

    try:
        # 运行整合脚本
        result = subprocess.run([sys.executable, "multimodal_integrator.py"],
                              capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"返回码: {result.returncode}")

    except Exception as e:
        print(f"运行整合脚本时出错: {e}")

def show_guide():
    """显示使用指南"""
    print("\n" + "="*50)
    print("多模态处理使用指南")
    print("="*50)

    guide_content = """
1. 首先运行数据整合（处理已有数据）：
   python multimodal_integrator.py

2. 检查输出结果：
   - Data/integrated_multimodal/ 包含整合后的数据

3. 对于新项目，使用完整处理流程：
   python multimodal_processor.py

4. 参考 MULTIMODAL_GUIDE.md 获取详细说明
    """

    print(guide_content)

def check_dependencies():
    """检查依赖"""
    print("\n" + "="*50)
    print("检查依赖...")
    print("="*50)

    required_packages = ['torch', 'clip', 'PIL', 'numpy', 'tiktoken']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'clip':
                import clip
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install torch torchvision torchaudio clip-by-openai pillow numpy tiktoken")
    else:
        print("\n✓ 所有依赖都已安装")

    return len(missing_packages) == 0

def main():
    print("多模态文档处理工具集")
    print("包含以下组件：")
    print("- multimodal_integrator.py: 整合现有文本分块和图片向量")
    print("- multimodal_processor.py: 一体化多模态处理")
    print("- MULTIMODAL_GUIDE.md: 详细使用指南")

    # 检查依赖
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n请先安装缺失的依赖包，然后重新运行。")
        return

    # 显示指南
    show_guide()

    # 询问是否运行整合
    response = input("\n是否现在运行数据整合? (y/n): ").lower().strip()

    if response in ['y', 'yes', '是']:
        run_integration()
    else:
        print("\n您可以随时运行以下命令：")
        print("python multimodal_integrator.py  # 整合现有数据")
        print("python multimodal_processor.py  # 新的多模态处理")

if __name__ == "__main__":
    main()