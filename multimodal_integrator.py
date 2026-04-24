# -*- coding: utf-8 -*-
"""
多模态数据整合补救脚本
功能：将已有的文本分块和图片向量整合为统一的多模态表示
"""

import os
import json
import re
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken

@dataclass
class IntegratedMultimodalChunk:
    """整合后的多模态分块"""
    id: str
    text: str
    image_paths: List[str]
    image_embeddings: List[List[float]]
    ocr_texts: List[str]
    ocr_embeddings: List[List[float]]
    combined_embedding: List[float]
    manual_path: str
    matched_pic_labels: List[str]  # 记录匹配的 PIC 标签

class MultimodalIntegrator:
    def __init__(self, txt_chunks_dir: str, image_vectors_dir: str, max_tokens: int = 800):
        self.txt_chunks_dir = txt_chunks_dir
        self.image_vectors_dir = image_vectors_dir
        self.max_tokens = max_tokens

        # 初始化CLIP模型用于可能的额外处理
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"使用设备: {device}")

        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        # 初始化tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-2")
        except:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # 简单的字符计数作为近似
            return len(text) // 4

    def load_txt_chunks(self) -> Dict[str, List[Dict]]:
        """加载已有的文本分块数据"""
        txt_chunks_data = {}

        for filename in os.listdir(self.txt_chunks_dir):
            if filename.endswith('_chunks.json'):
                filepath = os.path.join(self.txt_chunks_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    txt_chunks_data[filename.replace('_chunks.json', '')] = chunks

        print(f"加载了 {len(txt_chunks_data)} 个文本分块文件")
        return txt_chunks_data

    def load_image_vectors(self) -> Dict[str, Dict]:
        """加载已有的图片向量数据"""
        image_vectors_data = {}

        for filename in os.listdir(self.image_vectors_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.image_vectors_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 使用文件名（不含扩展名）作为键
                    key = os.path.splitext(filename)[0]
                    image_vectors_data[key] = data

        print(f"加载了 {len(image_vectors_data)} 个图片向量文件")
        return image_vectors_data

    def find_matching_images_for_manual(self, manual_basename: str, image_vectors_data: Dict) -> List[Dict]:
        """为手册找到对应的图片数据"""
        matching_images = []

        # 根据手册文件名匹配图片（支持多种命名约定）
        for img_key, img_data in image_vectors_data.items():
            # 检查图片是否与当前手册相关
            # 可能的匹配策略：
            # 1. 图片键包含手册名
            # 2. 图片键以手册名开头
            # 3. 更复杂的匹配逻辑
            if manual_basename in img_key or img_key.startswith(manual_basename.replace("已清洗", "")):
                matching_images.append({
                    'key': img_key,
                    'data': img_data
                })

        return matching_images

    def find_matching_images_for_chunk(self, txt_chunk: Dict, image_vectors_data: Dict) -> List[Dict]:
        """根据文本 chunk 中的<PIC:XXX>标签找到匹配的图片数据"""
        matching_images = []
        text = txt_chunk.get('text', '')

        # 提取所有 PIC 标签
        pic_labels = re.findall(r'<PIC:([^>]+)>', text)

        for label in pic_labels:
            if label in image_vectors_data:
                matching_images.append({
                    'key': label,
                    'data': image_vectors_data[label],
                    'pic_label': label
                })

        return matching_images

    def create_integrated_chunk(self, txt_chunk: Dict, matching_images: List[Dict], chunk_idx: int) -> IntegratedMultimodalChunk:
        """将文本块与匹配的图片数据整合"""
        # 提取文本块信息
        text = txt_chunk.get('text', '')
        manual_path = txt_chunk.get('manual_path', '')

        # 准备图片相关数据
        image_paths = []
        image_embeddings = []
        ocr_texts = []
        ocr_embeddings = []
        matched_pic_labels = []

        # 添加匹配的图片数据
        for img_info in matching_images:
            img_data = img_info['data']
            if 'image_embedding' in img_data:
                image_embeddings.append(img_data['image_embedding'])
            if 'ocr_text' in img_data:
                # ocr_text 可能是列表或字符串
                ocr_text = img_data['ocr_text']
                if isinstance(ocr_text, list):
                    ocr_texts.append(' '.join(ocr_text))
                else:
                    ocr_texts.append(ocr_text)
            if 'ocr_embedding' in img_data:
                ocr_embeddings.append(img_data['ocr_embedding'])
            # 图片路径使用匹配的 PIC 标签
            image_paths.append(img_info['key'])
            if 'pic_label' in img_info:
                matched_pic_labels.append(img_info['pic_label'])

        # 创建组合嵌入向量
        combined_embedding = self.create_combined_embedding(text, image_embeddings, ocr_embeddings)

        integrated_chunk = IntegratedMultimodalChunk(
            id=f"{txt_chunk.get('id', f'chunk_{chunk_idx}')}_integrated",
            text=text,
            image_paths=image_paths,
            image_embeddings=image_embeddings,
            ocr_texts=ocr_texts,
            ocr_embeddings=ocr_embeddings,
            combined_embedding=combined_embedding,
            manual_path=manual_path,
            matched_pic_labels=matched_pic_labels
        )

        return integrated_chunk

    def create_combined_embedding(self, text: str, image_embeddings: List[List[float]],
                               ocr_embeddings: List[List[float]]) -> List[float]:
        """创建组合嵌入向量"""
        # 这里提供几种可能的组合策略：
        # 1. 如果有文本，优先使用文本
        # 2. 如果有图片嵌入，将其平均
        # 3. 如果有OCR嵌入，将其平均
        # 4. 最后综合这些

        all_embeddings = []

        # 添加文本嵌入（如果有文本）
        if text and len(text.strip()) > 0:
            text_emb = self.get_text_embedding(text)
            if text_emb is not None:
                all_embeddings.append(text_emb)

        # 添加图片嵌入（如果存在）
        if image_embeddings:
            # 平均所有图片嵌入
            avg_img_emb = np.mean(image_embeddings, axis=0).tolist()
            all_embeddings.append(avg_img_emb)

        # 添加OCR嵌入（如果存在）
        if ocr_embeddings:
            # 平均所有OCR嵌入
            avg_ocr_emb = np.mean(ocr_embeddings, axis=0).tolist()
            all_embeddings.append(avg_ocr_emb)

        if all_embeddings:
            # 最终平均所有可用的嵌入
            combined = np.mean(all_embeddings, axis=0).tolist()
        else:
            # 默认嵌入（全零向量，维度与CLIP一致）
            combined = [0.0] * 512  # CLIP ViT-B/32 的输出维度

        return combined

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的 CLIP 嵌入向量"""
        try:
            if not text or not text.strip():
                return None

            # CLIP 对文本长度有限制，超过 77 tokens 会出错
            # 如果文本过长，截取前一部分
            if len(text) > 500:
                # 截取前 500 字符并尝试保持句子完整
                truncated = text[:450]
                last_space = truncated.rfind(' ')
                if last_space > 400:
                    truncated = truncated[:last_space]
                text = truncated

            text_input = clip.tokenize([text], truncate=True).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(text_input)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            return text_emb.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"处理文本嵌入时出错：{str(e)}")
            return None

    def integrate_manual(self, manual_name: str, txt_chunks: List[Dict], image_vectors_data: Dict) -> List[IntegratedMultimodalChunk]:
        """整合单个手册的文本和图片数据"""
        print(f"正在整合手册：{manual_name}")

        integrated_chunks = []

        # 为每个 chunk 单独匹配图片 (基于<PIC:XXX>标签)
        total_matched = 0
        for idx, txt_chunk in enumerate(txt_chunks):
            # 使用基于 chunk 的匹配方法
            matching_images = self.find_matching_images_for_chunk(txt_chunk, image_vectors_data)
            total_matched += len(matching_images)
            integrated_chunk = self.create_integrated_chunk(txt_chunk, matching_images, idx)
            integrated_chunks.append(integrated_chunk)

        print(f"  为手册 {manual_name} 创建了 {len(integrated_chunks)} 个整合块，匹配了 {total_matched} 个图片")
        return integrated_chunks


    def process_all(self) -> Dict[str, List[IntegratedMultimodalChunk]]:
        """处理所有数据"""
        # 加载数据
        txt_chunks_data = self.load_txt_chunks()
        image_vectors_data = self.load_image_vectors()

        # 整合所有手册
        integrated_results = {}

        for manual_name, txt_chunks in txt_chunks_data.items():
            integrated_chunks = self.integrate_manual(manual_name, txt_chunks, image_vectors_data)
            integrated_results[manual_name] = integrated_chunks

        return integrated_results

    def save_integrated_results(self, results: Dict[str, List[IntegratedMultimodalChunk]], output_dir: str):
        """保存整合结果"""
        os.makedirs(output_dir, exist_ok=True)

        total_chunks = 0

        for manual_name, chunks in results.items():
            output_file = os.path.join(output_dir, f"{manual_name}_integrated_multimodal.json")

            # 转换为可序列化的格式
            serializable_chunks = []
            for chunk in chunks:
                serializable_chunk = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "image_paths": chunk.image_paths,
                    "image_embeddings_count": len(chunk.image_embeddings),
                    "ocr_texts_count": len(chunk.ocr_texts),
                    "ocr_embeddings_count": len(chunk.ocr_embeddings),
                    "combined_embedding": chunk.combined_embedding,
                    "manual_path": chunk.manual_path,
                    "text_length": len(chunk.text)
                }
                serializable_chunks.append(serializable_chunk)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

            total_chunks += len(chunks)
            print(f"已保存: {output_file} ({len(chunks)} 个块)")

        print(f"\n整合完成! 总共处理了 {len(results)} 个手册，生成 {total_chunks} 个整合块")

        # 保存统计信息
        stats = {
            "manual_count": len(results),
            "total_chunks": total_chunks,
            "average_chunks_per_manual": total_chunks / len(results) if results else 0
        }

        stats_file = os.path.join(output_dir, "integration_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"统计信息已保存至: {stats_file}")


def main():
    # 配置路径
    txt_chunks_dir = r"D:\PyCharm\clean\Data\processed_chunks"  # 你已有的文本分块目录
    image_vectors_dir = r"D:\PyCharm\clean\ocrtext_image_embeddings"  # 你已有的图片向量目录
    output_dir = r"D:\PyCharm\clean\Data\integrated_multimodal"

    # 创建整合器
    integrator = MultimodalIntegrator(txt_chunks_dir, image_vectors_dir)

    # 执行整合
    results = integrator.process_all()

    # 保存结果
    integrator.save_integrated_results(results, output_dir)


if __name__ == "__main__":
    main()