"""
重新处理所有数据的完整流程脚本
1. 预处理图像
2. 重新OCR
3. 重新生成chunks
4. 重新构建向量库
"""
import os
import subprocess
import sys

def run_command(command, description):
    """运行命令并显示描述"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        print("✓ 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("开始完整的数据重新处理流程...")

    # 1. 预处理图像并重新OCR
    if not run_command("python ocr_image.py", "预处理图像并执行OCR"):
        return

    # 2. 重新生成chunks（需要确保chunk_manuals.py使用新的OCR结果）
    if not run_command("python chunk_manuals.py", "重新生成chunks"):
        return


    print("\n✅ 所有步骤完成！现在可以测试改进后的系统了。")

if __name__ == "__main__":
    main()