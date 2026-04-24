import os
import json
import re
from tqdm import tqdm
import concurrent.futures
from threading import Semaphore
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

def extract_from_text_response(text):
    """
    从非JSON格式的文本响应中提取OCR文字和描述

    Args:
        text (str): 模型返回的文本

    Returns:
        tuple: (ocr_text_list, visual_desc)
    """
    ocr_text = []
    visual_desc = ""

    try:
        # 尝试查找OCR相关的文字
        # 匹配可能的文字列表格式
        lines = text.split('\n')
        ocr_section = False
        desc_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测OCR部分
            if any(keyword in line.lower() for keyword in ['ocr', '文字', 'text', '提取']):
                ocr_section = True
                desc_section = False
                continue
            elif any(keyword in line.lower() for keyword in ['描述', 'description', 'visual', '图片']):
                ocr_section = False
                desc_section = True
                continue

            # 提取OCR文字
            if ocr_section:
                # 移除序号、引号等
                clean_line = re.sub(r'^[\d\.\-\*\s"]+|["\s]+$', '', line)
                if clean_line and len(clean_line) > 2:
                    ocr_text.append(clean_line)

            # 提取描述
            if desc_section and not visual_desc:
                visual_desc = line

        # 如果没找到明确的分段，尝试其他方法
        if not ocr_text and not visual_desc:
            # 假设前几行是OCR，最后一行是描述
            words = re.findall(r'\b\w{3,}\b', text)
            if words:
                # 简单启发式：取所有单词作为OCR候选
                ocr_text = list(set(words))
                visual_desc = text[:200] if len(text) > 50 else text

        if not visual_desc:
            visual_desc = "无法生成描述"

    except Exception as e:
        print(f"文本提取失败: {e}")
        visual_desc = "无法生成描述"

    return ocr_text, visual_desc

# 配置DashScope API密钥（请替换为你的实际API密钥）
# 你可以通过环境 variable 设置：export DASHSCOPE_API_KEY='your-api-key'

# 配置路径
image_dir = r"D:\PyCharm\clean\Data\Images"  # 输入图片文件夹
ocr_result_dir = r"D:\PyCharm\clean\ocr_results"  # OCR结果保存文件夹

# 并发控制配置
MAX_CONCURRENT = 3  # 根据DashScope API限制调整，建议2-5
semaphore = Semaphore(MAX_CONCURRENT)

# 创建输出目录
os.makedirs(ocr_result_dir, exist_ok=True)

def call_qwen_vl_plus(image_path):
    """
    调用Qwen VL Plus模型进行OCR和图片描述生成

    Args:
        image_path (str): 图片文件路径

    Returns:
        tuple: (ocr_text_list, visual_desc) 或 (None, None) 如果失败
    """
    try:
        # 构建多模态对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": f"file://{image_path}"
                    },
                    {
                        "text": "提取图片中的所有文字内容，并用一句话描述图片。只返回纯JSON格式，不要任何其他文字：{\"ocr_text\": [\"文字1\", \"文字2\"], \"visual_desc\": \"描述\"}"
                    }
                ]
            }
        ]

        # 调用Qwen VL Plus模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            # 解析响应
            result_text = response.output.choices[0].message.content[0]['text']

            # 尝试解析JSON
            try:
                # 先清理可能的Markdown代码块包围
                clean_result_text = result_text.strip()
                if clean_result_text.startswith('```'):
                    # 移除Markdown代码块标记
                    lines = clean_result_text.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('```'):
                            if not in_json:
                                in_json = True
                            else:
                                break
                        elif in_json:
                            json_lines.append(line)
                    clean_result_text = '\n'.join(json_lines)

                result_data = json.loads(clean_result_text)
                ocr_text = result_data.get('ocr_text', [])
                visual_desc = result_data.get('visual_desc', '')
                return ocr_text, visual_desc
            except json.JSONDecodeError:
                # 如果JSON解析失败，保存原始响应用于调试，并尝试提取
                print(f"JSON解析失败，原始响应: {result_text}")
                # 保存原始响应到调试文件
                debug_file = os.path.join(ocr_result_dir, "debug_response.txt")
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write(f"=== 响应时间: {os.path.basename(image_path)} ===\n")
                    f.write(result_text + "\n\n")

                return extract_from_text_response(result_text)
        else:
            print(f"API调用失败: {response.code} - {response.message}")
            return None, None

    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {e}")
        return None, None

def process_single_image(img_name):
    """处理单张图片"""
    with semaphore:  # 控制并发数
        img_path = os.path.join(image_dir, img_name)
        output_file = os.path.join(ocr_result_dir, f"{os.path.splitext(img_name)[0]}.json")

        # 跳过已处理的文件
        if os.path.exists(output_file):
            return f"跳过: {img_name}"

        # 调用Qwen VL Plus模型
        ocr_text, visual_desc = call_qwen_vl_plus(img_path)

        if ocr_text is not None and visual_desc is not None:
            # 精简OCR结果：去重、去空、只保留长度>2的关键字
            clean_ocr = []
            for text in ocr_text:
                if isinstance(text, str):
                    text_clean = text.strip()
                    if len(text_clean) > 2 and (text_clean.replace(' ', '').isalnum() or any(c.isalpha() for c in text_clean)):
                        clean_ocr.append(text_clean)

            clean_ocr = list(set(clean_ocr))  # 去重

            # 保存结果
            result_data = {
                "ocr_text": clean_ocr,
                "visual_desc": visual_desc
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            return f"成功: {img_name}"
        else:
            # 处理失败时保存空结果
            result_data = {
                "ocr_text": [],
                "visual_desc": "处理失败"
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            return f"失败: {img_name}"

def process_images_parallel():
    """并行批量处理图片"""
    # 获取所有支持的图片文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    valid_imgs = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(valid_extensions)]

    if not valid_imgs:
        print(f"在目录 {image_dir} 中未找到支持的图片文件")
        return

    print(f"找到 {len(valid_imgs)} 张图片，使用 {MAX_CONCURRENT} 个并发线程处理...")

    # 使用线程池并行处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # 提交所有任务
        future_to_img = {executor.submit(process_single_image, img_name): img_name
                        for img_name in valid_imgs}

        # 显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_img),
                          total=len(valid_imgs), desc="处理进度"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                img_name = future_to_img[future]
                print(f"处理 {img_name} 时发生异常: {e}")
                results.append(f"异常: {img_name}")

    # 打印总结
    success_count = sum(1 for r in results if r.startswith("成功"))
    skip_count = sum(1 for r in results if r.startswith("跳过"))
    fail_count = sum(1 for r in results if r.startswith("失败") or r.startswith("异常"))

    print(f"\n处理完成！成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")

if __name__ == "__main__":
    # 检查API密钥是否设置
    if not dashscope.api_key:
        print("请先设置DASHSCOPE_API_KEY环境变量")
        print("获取地址: https://dashscope.console.aliyun.com/apiKey")
    else:
        process_images_parallel()
        print("OCR批量处理完成！")