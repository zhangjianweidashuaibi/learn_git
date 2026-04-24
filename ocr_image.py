import easyocr
import os
import json
from tqdm import tqdm
from image_preprocessor import DocumentImagePreprocessor

# 配置OCR
reader = easyocr.Reader(['en'])  # 只识别英文，提高准确率
standard_img_dir = "./images_standard"  # 标准化图片文件夹
preprocessed_img_dir = "./images_preprocessed"  # 预处理后的图片文件夹
ocr_result_dir = "./ocr_results"  # OCR结果保存文件夹

os.makedirs(preprocessed_img_dir, exist_ok=True)
os.makedirs(ocr_result_dir, exist_ok=True)

# 创建预处理器
preprocessor = DocumentImagePreprocessor()

# 获取图片列表
valid_imgs = [f for f in os.listdir(standard_img_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 批量处理
for img_name in tqdm(valid_imgs):
    img_path = os.path.join(standard_img_dir, img_name)
    preprocessed_path = os.path.join(preprocessed_img_dir, f"preprocessed_{img_name}")

    try:
        # 先进行图像预处理
        preprocessor.preprocess_image(img_path, preprocessed_path)

        # 使用预处理后的图像进行OCR
        ocr_text_list = reader.readtext(preprocessed_path, detail=0)

        # 精简OCR：去重、去空、只保留长度>2的关键词（过滤单个字母/符号）
        clean_ocr = []
        for text in ocr_text_list:
            text_clean = text.strip()
            if len(text_clean) > 2 and (text_clean.replace(' ', '').isalnum() or any(c.isalpha() for c in text_clean)):
                clean_ocr.append(text_clean)

        clean_ocr = list(set(clean_ocr))  # 去重

    except Exception as e:
        print(f"处理图片 {img_name} 时出错: {e}")
        # 如果预处理失败，回退到原始图像
        try:
            ocr_text_list = reader.readtext(img_path, detail=0)
            clean_ocr = []
            for text in ocr_text_list:
                text_clean = text.strip()
                if len(text_clean) > 2 and (text_clean.replace(' ', '').isalnum() or any(c.isalpha() for c in text_clean)):
                    clean_ocr.append(text_clean)
            clean_ocr = list(set(clean_ocr))
        except Exception as e2:
            print(f"回退处理也失败: {e2}")
            clean_ocr = []

    # 保存为JSON，方便后续读取
    with open(os.path.join(ocr_result_dir, img_name.split(".")[0] + ".json"), "w", encoding="utf-8") as f:
        json.dump({"ocr_text": clean_ocr, "visual_desc": f"Technical manual diagram: {img_name.split('_')[0]} section"}, f)

print("OCR批量处理完成！")