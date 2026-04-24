# -*- coding: utf-8 -*-
"""
CLIP 多模态向量生成脚本（最终修正版）
功能：批量生成图片视觉向量 + OCR文本特征向量，保存为JSON格式
"""
import os
import json
import torch
import clip
from PIL import Image
import numpy as np

# ===================== 1. 基础配置（请修改为你的本地路径）=====================
IMAGE_DIR = "./images_standard"       # 标准图片存放目录
OCR_DIR = "./ocr_results"    # OCR结果JSON存放目录（文件名与图片一一对应）
SAVE_DIR = "./multimodal_vectors"    # 向量文件保存目录
SUPPORT_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")  # 支持的图片格式

# ===================== 2. 环境初始化 =====================
# 自动检测GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载CLIP模型（修正为OpenAI原生模型名ViT-B/32）
model, preprocess = clip.load("ViT-B/32", device=device)

# 自动创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 3. 获取有效图片列表 =====================
def get_valid_images(image_dir):
    valid_imgs = []
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(SUPPORT_FORMATS):
            valid_imgs.append(filename)
    return valid_imgs

valid_imgs = get_valid_images(IMAGE_DIR)
print(f"共找到 {len(valid_imgs)} 张有效图片")

# ===================== 4. 批量生成多模态向量 =====================
def generate_embeddings():
    for img_name in valid_imgs:
        try:
            img_label = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMAGE_DIR, img_name)

            # 生成图片视觉向量
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                img_emb = model.encode_image(image)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # 归一化
            img_emb_np = img_emb.cpu().numpy().tolist()[0]

            # 生成OCR文本向量
            ocr_emb_np, ocr_text = [], ""
            ocr_json_path = os.path.join(OCR_DIR, f"{img_label}.json")
            
            if os.path.exists(ocr_json_path):
                with open(ocr_json_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                # 适配通用OCR JSON格式
                ocr_text = ocr_data.get("text", "") or "".join(ocr_data.get("words", []))
                
                if ocr_text.strip():
                    text = clip.tokenize([ocr_text]).to(device)
                    with torch.no_grad():
                        text_emb = model.encode_text(text)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    ocr_emb_np = text_emb.cpu().numpy().tolist()[0]

            # 保存结果
            result = {
                "image_label": img_label,
                "image_embedding": img_emb_np,
                "ocr_text": ocr_text,
                "ocr_embedding": ocr_emb_np,
                "embedding_dim": len(img_emb_np)
            }
            save_path = os.path.join(SAVE_DIR, f"{img_label}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"✅ 处理完成: {img_name}")
        except Exception as e:
            print(f"❌ 处理失败 {img_name}: {str(e)}")

    print(f"\n🎉 所有图片处理完毕！向量文件保存至: {SAVE_DIR}")

# ===================== 5. 执行主函数 =====================
if __name__ == "__main__":
    generate_embeddings()