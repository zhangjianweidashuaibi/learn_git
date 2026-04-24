import os
import json
from tqdm import tqdm
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation

# 配置DashScope API密钥（请替换为你的实际API密钥）
# 你可以通过环境变量设置：export DASHSCOPE_API_KEY='your-api-key'
# 或者直接在这里设置：dashscope.api_key = 'your-api-key'

# 配置路径
image_dir = r"D:\PyCharm\clean\Data\Images"  # 输入图片文件夹
ocr_result_dir = r"D:\PyCharm\clean\ocr_results"  # OCR结果保存文件夹

# 创建输出目录
os.makedirs(ocr_result_dir, exist_ok=True)

def call_qwen_vl_plus(image_path):
    """
    调用Qwen3.5-VL-Plus模型进行OCR和图片描述生成

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
                        "text": "请执行以下任务：\n1. 提取图片中的所有文字内容\n2. 生成一段简洁的图片描述\n\n请按以下JSON格式返回结果：\n{\"ocr_text\": [\"提取的文字1\", \"提取的文字2\", ...], \"visual_desc\": \"图片描述\"}"
                    }
                ]
            }
        ]

        # 调用Qwen3.5-VL-Plus模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',  # Qwen3.5-VL-Plus对应的模型名称
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            # 解析响应
            result_text = response.output.choices[0].message.content[0]['text']

            # 尝试解析JSON
            try:
                import json
                result_data = json.loads(result_text)
                ocr_text = result_data.get('ocr_text', [])
                visual_desc = result_data.get('visual_desc', '')
                return ocr_text, visual_desc
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试从文本中提取信息
                print(f"JSON解析失败，原始响应: {result_text}")
                return [], "无法生成描述"
        else:
            print(f"API调用失败，状态码: {response.status_code}, 错误信息: {response.code}")
            return None, None

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None, None

def process_images():
    """批量处理图片"""
    # 获取所有支持的图片文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    valid_imgs = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(valid_extensions)]

    if not valid_imgs:
        print(f"在目录 {image_dir} 中未找到支持的图片文件")
        return

    print(f"找到 {len(valid_imgs)} 张图片，开始处理...")

    # 批量处理
    for img_name in tqdm(valid_imgs, desc="处理图片"):
        img_path = os.path.join(image_dir, img_name)
        output_file = os.path.join(ocr_result_dir, f"{os.path.splitext(img_name)[0]}.json")

        # 跳过已处理的文件
        if os.path.exists(output_file):
            print(f"跳过已存在的文件: {output_file}")
            continue

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

            print(f"成功处理: {img_name}")
        else:
            # 处理失败时保存空结果
            result_data = {
                "ocr_text": [],
                "visual_desc": "处理失败"
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"处理失败: {img_name}")

if __name__ == "__main__":
    # 检查API密钥是否设置
    if not dashscope.api_key:
        print("请先设置DASHSCOPE_API_KEY环境变量或在代码中设置dashscope.api_key")
        print("你可以从 https://dashscope.console.aliyun.com/apiKey 获取API密钥")
    else:
        process_images()
        print("OCR批量处理完成！")