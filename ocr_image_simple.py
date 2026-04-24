"""
简化版OCR处理 - 直接使用原始图像，但改进文本过滤
"""
import easyocr
import os
import json
from tqdm import tqdm

# 配置OCR - 添加更多语言支持和参数优化
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)  # 支持中英文，启用GPU

standard_img_dir = "./images_standard"
ocr_result_dir = "./ocr_results"
os.makedirs(ocr_result_dir, exist_ok=True)

# 获取图片列表
valid_imgs = [f for f in os.listdir(standard_img_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 批量处理
for img_name in tqdm(valid_imgs):
    img_path = os.path.join(standard_img_dir, img_name)

    try:
        # 使用更宽松的参数进行OCR
        ocr_results = reader.readtext(
            img_path,
            detail=1,  # 获取详细信息包括置信度
            min_size=10,  # 最小文字尺寸
            text_threshold=0.5,  # 文字检测阈值
            low_text=0.3,  # 低置信度文字阈值
            contrast_ths=0.1,  # 对比度阈值
            adjust_contrast=0.5  # 自动对比度调整
        )

        # 过滤高置信度的结果
        clean_ocr = []
        for bbox, text, confidence in ocr_results:
            text_clean = text.strip()
            # 只保留置信度>0.6且长度>2的文字
            if confidence > 0.6 and len(text_clean) > 2:
                # 进一步过滤：只保留包含字母或数字的文字
                if any(c.isalnum() for c in text_clean):
                    clean_ocr.append(text_clean)

        clean_ocr = list(set(clean_ocr))  # 去重

    except Exception as e:
        print(f"处理图片 {img_name} 时出错: {e}")
        clean_ocr = []

    # 保存结果
    with open(os.path.join(ocr_result_dir, img_name.split(".")[0] + ".json"), "w", encoding="utf-8") as f:
        json.dump({
            "ocr_text": clean_ocr,
            "visual_desc": f"Technical manual diagram: {img_name.split('_')[0]} section"
        }, f, ensure_ascii=False, indent=2)

print("简化版OCR批量处理完成！")