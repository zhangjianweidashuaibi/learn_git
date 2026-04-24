# -*- coding: utf-8 -*-
"""
多模态文档处理器
功能：将文本和图片统一进行分块和向量化处理
"""
import os
import re
import json
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken

@dataclass
class MultimodalChunk:
    """多模态分块数据结构"""
    id: str
    text: str
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None
    ocr_text: Optional[str] = None
    ocr_embedding: Optional[List[float]] = None
    combined_embedding: Optional[List[float]] = None
    manual_path: Optional[str] = None

class MultimodalProcessor:
    def __init__(self, data_dir: str, image_dir: str, max_tokens: int = 800):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.max_tokens = max_tokens

        # 初始化CLIP模型
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

    def get_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """获取图片的CLIP视觉嵌入向量"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_emb = self.model.encode_image(image)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # 归一化
            return img_emb.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            return None

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的CLIP嵌入向量"""
        try:
            if not text.strip():
                return None
            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(text_input)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            return text_emb.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return None

    def combine_embeddings(self, text_emb: Optional[List[float]],
                         img_emb: Optional[List[float]]) -> Optional[List[float]]:
        """结合文本和图像嵌入向量"""
        if text_emb is not None and img_emb is not None:
            # 简单平均结合
            return [(t + i) / 2.0 for t, i in zip(text_emb, img_emb)]
        elif text_emb is not None:
            return text_emb
        elif img_emb is not None:
            return img_emb
        else:
            return None

    def process_pic_tags(self, content: str, manual_file_path: str) -> List[MultimodalChunk]:
        """处理包含<PIC>标签的文本内容，生成多模态分块"""
        # 按<PIC>标签分割内容
        parts = re.split(r'(<PIC:.+?>)', content)

        chunks = []
        current_text = ""
        chunk_id = 0

        for part in parts:
            if part.startswith('<PIC:') and part.endswith('>'):
                # 这是一个图片标签，先保存之前的文本
                if current_text.strip():
                    chunk = self.create_multimodal_chunk(
                        current_text.strip(),
                        None,
                        manual_file_path,
                        chunk_id
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                # 处理图片标签
                image_name = re.search(r'<PIC:(.+?)>', part).group(1)
                image_path = self.find_image_path(manual_file_path, image_name)

                # 创建包含图片的空文本块
                chunk = self.create_multimodal_chunk(
                    "",  # 图片块通常只包含图片信息
                    image_path,
                    manual_file_path,
                    chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1

                current_text = ""  # 重置当前文本
            else:
                # 这是一段普通文本
                current_text += part

        # 处理最后一部分文本
        if current_text.strip():
            chunk = self.create_multimodal_chunk(
                current_text.strip(),
                None,
                manual_file_path,
                chunk_id
            )
            chunks.append(chunk)

        return chunks

    def find_image_path(self, manual_file_path: str, image_name: str) -> Optional[str]:
        """根据手册路径和图片名称查找实际图片路径"""
        # 构造可能的图片路径
        base_name = os.path.splitext(os.path.basename(manual_file_path))[0]

        # 在images目录下查找对应的图片
        potential_paths = [
            os.path.join(self.image_dir, base_name, image_name),
            os.path.join(self.image_dir, image_name),
            os.path.join(os.path.dirname(manual_file_path), image_name),
        ]

        for path in potential_paths:
            if os.path.exists(path):
                return path

        print(f"警告: 未找到图片 {image_name} 对应的文件")
        return None

    def create_multimodal_chunk(self, text: str, image_path: Optional[str],
                              manual_path: str, chunk_id: int) -> MultimodalChunk:
        """创建多模态分块对象"""
        # 获取图片嵌入
        image_embedding = None
        ocr_text = ""
        ocr_embedding = None

        if image_path:
            image_embedding = self.get_image_embedding(image_path)

            # 尝试获取OCR文本
            ocr_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(ocr_path):
                try:
                    with open(ocr_path, 'r', encoding='utf-8') as f:
                        ocr_text = f.read().strip()
                    ocr_embedding = self.get_text_embedding(ocr_text)
                except:
                    pass  # OCR文本读取失败

        # 获取文本嵌入
        text_embedding = self.get_text_embedding(text) if text.strip() else None

        # 结合文本和图片嵌入
        combined_embedding = self.combine_embeddings(text_embedding, image_embedding)

        chunk = MultimodalChunk(
            id=f"{os.path.basename(manual_path)}_chunk_{chunk_id}",
            text=text,
            image_path=image_path,
            image_embedding=image_embedding,
            ocr_text=ocr_text,
            ocr_embedding=ocr_embedding,
            combined_embedding=combined_embedding,
            manual_path=manual_path
        )

        return chunk

    def merge_small_chunks(self, chunks: List[MultimodalChunk]) -> List[MultimodalChunk]:
        """合并小的分块以提高效率"""
        if not chunks:
            return []

        merged_chunks = [chunks[0]]

        for chunk in chunks[1:]:
            last_chunk = merged_chunks[-1]

            # 计算合并后是否仍小于最大token数
            combined_text = last_chunk.text + "\n\n" + chunk.text
            combined_tokens = self.count_tokens(combined_text)

            # 只有在合并后仍然合理大小且两个都是纯文本块时才合并
            if (combined_tokens <= self.max_tokens * 0.8 and  # 设置略低于上限以留余地
                last_chunk.image_path is None and
                chunk.image_path is None):

                # 合并文本块
                merged_chunk = MultimodalChunk(
                    id=f"{last_chunk.id}_merged",
                    text=combined_text,
                    image_path=None,
                    image_embedding=None,
                    ocr_text="",
                    ocr_embedding=None,
                    combined_embedding=self.get_text_embedding(combined_text),
                    manual_path=last_chunk.manual_path
                )
                merged_chunks[-1] = merged_chunk
            else:
                # 不合并，直接添加
                merged_chunks.append(chunk)

        return merged_chunks

    def semantic_split(self, text: str) -> List[str]:
        """按语义层次分割文本"""
        # 定义标题层级模式
        heading_patterns = [
            (r'^#{1}\s+(.+)', 1),  # 一级标题 #
            (r'^#{2}\s+(.+)', 2),  # 二级标题 ##
            (r'^#{3}\s+(.+)', 3),  # 三级标题 ###
            (r'^\d+\.\d+\.\d+\s+.+', 4),  # 1.1.1 格式
            (r'^\d+\.\d+\s+.+', 5),       # 1.1 格式
            (r'^\d+\.\s+.+', 6),          # 1. 格式
            (r'^[一二三四五六七八九十]+、', 7),  # 中文编号
            (r'^[（\(][一二三四五六七八九十]+[）\)]', 8),  # （一）格式
        ]

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()

            if not para:
                i += 1
                continue

            # 检查是否是标题
            is_heading = False
            heading_level = float('inf')
            for pattern, level in heading_patterns:
                if re.match(pattern, para.strip(), re.MULTILINE):
                    is_heading = True
                    heading_level = level
                    break

            # 如果当前段落是标题，且当前块不为空，则开始新块
            if is_heading and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            # 如果当前块会超过最大token数，则开启新块
            elif self.count_tokens(current_chunk + '\n\n' + para) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            # 否则添加到当前块
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para

            i += 1

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_document_with_pic_tags(self, file_path: str) -> List[MultimodalChunk]:
        """处理包含<PIC>标签的文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 先按照语义层级进行预分割
        semantic_chunks = self.semantic_split(content)

        # 将每个语义块处理成多模态块
        multimodal_chunks = []
        for i, chunk_text in enumerate(semantic_chunks):
            # 对每个语义块再处理其中的<PIC>标签
            chunk_parts = self.process_pic_tags(chunk_text, file_path)

            # 更新ID以反映层级结构
            for j, part in enumerate(chunk_parts):
                part.id = f"{part.id}_semantic_{i}_part_{j}"

            multimodal_chunks.extend(chunk_parts)

        # 合并过小的块
        multimodal_chunks = self.merge_small_chunks(multimodal_chunks)

        return multimodal_chunks

    def process_single_file(self, file_path: str) -> List[MultimodalChunk]:
        """处理单个文档文件"""
        print(f"正在处理文件: {file_path}")

        # 根据是否存在<PIC>标签选择处理方法
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if '<PIC:' in content:
            # 包含图片标签，使用专门的方法处理
            chunks = self.process_document_with_pic_tags(file_path)
        else:
            # 不包含图片标签，仅处理纯文本
            chunks = self.process_document_with_pic_tags(file_path)

        print(f"文件 {file_path} 处理完成，生成 {len(chunks)} 个多模态分块")
        return chunks

    def process_directory(self) -> Dict[str, List[MultimodalChunk]]:
        """处理整个目录中的所有txt文件"""
        results = {}

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.txt') and 'clean_vr_Manuals' in root:
                    file_path = os.path.join(root, file)
                    print(f"开始处理: {file_path}")

                    try:
                        chunks = self.process_single_file(file_path)
                        results[file_path] = chunks
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()

        return results

def save_multimodal_results(results: Dict[str, List[MultimodalChunk]], output_dir: str):
    """保存多模态处理结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 将每个文件的结果保存为单独的JSON文件
    for file_path, chunks in results.items():
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(file_path)}_multimodal_chunks.json"
        )

        # 转换为可序列化的字典（排除嵌入向量以减少文件大小）
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = {
                "id": chunk.id,
                "text": chunk.text,
                "image_path": chunk.image_path,
                "has_image": chunk.image_path is not None,
                "has_image_embedding": chunk.image_embedding is not None,
                "has_ocr": bool(chunk.ocr_text),
                "has_combined_embedding": chunk.combined_embedding is not None,
                "text_length": len(chunk.text),
                "ocr_text": chunk.ocr_text if chunk.ocr_text else "",
                "manual_path": chunk.manual_path
            }
            serializable_chunks.append(serializable_chunk)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

        print(f"结果已保存至: {output_file}")

def save_embeddings_separately(results: Dict[str, List[MultimodalChunk]], embedding_dir: str):
    """单独保存嵌入向量以节省空间"""
    os.makedirs(embedding_dir, exist_ok=True)

    for file_path, chunks in results.items():
        embedding_file = os.path.join(
            embedding_dir,
            f"{os.path.basename(file_path)}_embeddings.json"
        )

        embeddings_data = []
        for chunk in chunks:
            embedding_info = {
                "id": chunk.id,
                "image_embedding": chunk.image_embedding,
                "ocr_embedding": chunk.ocr_embedding,
                "combined_embedding": chunk.combined_embedding
            }
            embeddings_data.append(embedding_info)

        with open(embedding_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

        print(f"嵌入向量已保存至: {embedding_file}")

def main():
    # 配置路径
    data_dir = r"D:\PyCharm\clean\Data\clean_vr_Manuals"  # 你的清理后的文本目录
    image_dir = r"D:\PyCharm\clean\images"  # 你的图片目录
    output_dir = r"D:\PyCharm\clean\Data\multimodal_processed"
    embedding_dir = r"D:\PyCharm\clean\Data\multimodal_embeddings"

    # 创建处理器
    processor = MultimodalProcessor(data_dir, image_dir)

    # 处理所有文档
    results = processor.process_directory()

    # 保存结果
    save_multimodal_results(results, output_dir)
    save_embeddings_separately(results, embedding_dir)

    # 统计信息
    total_chunks = sum(len(chunks) for chunks in results.values())
    print(f"\n处理完成!")
    print(f"共处理 {len(results)} 个文档")
    print(f"生成 {total_chunks} 个多模态分块")

if __name__ == "__main__":
    main()