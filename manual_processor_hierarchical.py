import os
import re
import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import tiktoken
import glob


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    text: str
    image_names: List[str]
    image_paths: List[str]
    ocr_texts: List[str]
    visual_descriptions: List[str]
    manual_path: str


class HierarchicalManualProcessor:
    """
    基于层级语义的手册分块处理器

    分块策略：
    1. 首先按一级标题 (#) 分割文档
    2. 合并语义相近的短 # 片段，避免碎片化
    3. 对过长的 # 片段按自然段进一步拆分（空行、列表、图表分组）
    4. 保证单个块尽量在 300-800 字范围内
    5. # 章节内的多图保持完整，不需拆分
    """

    def __init__(self, data_dir: str, min_tokens: int = 300, max_tokens: int = 800,
                 images_dir: str = r"D:\PyCharm\clean\images_standard",
                 embeddings_dir: str = r"D:\PyCharm\clean\ocrtext_image_embeddings"):
        self.data_dir = data_dir
        self.min_tokens = min_tokens  # 最小字数阈值
        self.max_tokens = max_tokens  # 最大字数阈值
        self.images_dir = images_dir
        self.embeddings_dir = embeddings_dir

        # 使用 GPT-2 tokenizer 作为近似
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-2")
        except:
            # 如果 tiktoken 不可用，则使用简单的字符计数作为近似
            self.tokenizer = None

        # 预加载所有 OCR 和 description 数据到内存
        self.embedding_cache = self._load_all_embeddings()

    def _load_all_embeddings(self) -> Dict[str, dict]:
        """预加载所有 OCR 和 visual_description JSON 文件到内存"""
        cache = {}
        if os.path.exists(self.embeddings_dir):
            for json_file in glob.glob(os.path.join(self.embeddings_dir, "*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        file_name = os.path.splitext(os.path.basename(json_file))[0]
                        cache[file_name] = data
                except Exception as e:
                    print(f"加载 embedding 文件 {json_file} 时出错：{str(e)}")
        print(f"已加载 {len(cache)} 个 embedding 文件到缓存")
        return cache

    def get_image_embedding_data(self, image_name: str) -> dict:
        """根据图片名称获取对应的 OCR 和 visual_description 数据"""
        base_name = os.path.splitext(image_name)[0]

        if base_name in self.embedding_cache:
            return self.embedding_cache[base_name]

        if image_name in self.embedding_cache:
            return self.embedding_cache[image_name]

        return {
            "ocr_text": [],
            "visual_desc": ""
        }

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4

    def find_associated_image_data(self, manual_file_path: str, pic_tag: str) -> Dict:
        """查找与<PIC>标签关联的图片数据 - 支持一个文档包含多张图片的情况"""
        pic_match = re.search(r'<PIC:(.+?)>', pic_tag)
        if pic_match:
            image_name_raw = pic_match.group(1)

            base_name = os.path.splitext(image_name_raw)[0]
            image_name_final = base_name + '.png'

            image_path = os.path.join(self.images_dir, image_name_final)

            embedding_data = self.get_image_embedding_data(base_name)
            ocr_text_list = embedding_data.get("ocr_text", [])
            visual_desc = embedding_data.get("visual_desc", "")

            return {
                "image_names": [image_name_final],
                "image_paths": [image_path],
                "ocr_texts": [ocr_text_list] if ocr_text_list else [[]],
                "visual_descriptions": [visual_desc] if visual_desc else [""]
            }

        return {
            "image_names": [],
            "image_paths": [],
            "ocr_texts": [],
            "visual_descriptions": []
        }

    def extract_all_pic_tags(self, text: str) -> List[str]:
        """提取文本中所有的<PIC>标签"""
        return re.findall(r'<PIC:[^>]+>', text)

    def split_by_level_one_headings(self, text: str) -> List[Dict]:
        """
        按一级标题 (#) 分割文档，返回带标题信息的片段列表

        返回格式：[{"title": "标题", "content": "内容", "has_heading": True/False}]
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_title = ""
        has_heading = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 检测一级标题 (# 开头，后面跟空格)
            if re.match(r'^#\s+(.+)', line):
                # 保存之前的章节
                if current_section:
                    sections.append({
                        "title": current_title,
                        "content": '\n'.join(current_section).strip(),
                        "has_heading": has_heading
                    })
                # 开始新章节
                current_title = line
                current_section = []
                has_heading = True
            else:
                current_section.append(lines[i])

            i += 1

        # 添加最后一个章节
        if current_section:
            sections.append({
                "title": current_title,
                "content": '\n'.join(current_section).strip(),
                "has_heading": has_heading
            })

        return sections

    def merge_short_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        合并语义相近的短 # 片段，避免碎片化

        合并策略：
        1. 连续的几个短 # 片段（低于 min_tokens）尝试合并
        2. 合并后不超过 max_tokens 则执行合并
        3. 有标题的片段优先保持独立（可能是重要章节）
        """
        if len(sections) <= 1:
            return sections

        merged_sections = []
        current_group = []
        current_group_tokens = 0

        for section in sections:
            section_tokens = self.count_tokens(section["content"])

            # 如果当前片段本身已经足够长（>= min_tokens），且有标题，直接作为独立块
            if section_tokens >= self.min_tokens and section["has_heading"]:
                # 先处理之前累积的短片段
                if current_group:
                    merged = self._merge_group(current_group)
                    if merged:
                        merged_sections.append(merged)
                    current_group = []
                    current_group_tokens = 0
                merged_sections.append(section)
                continue

            # 如果当前片段很短，加入待合并组
            if current_group_tokens + section_tokens <= self.max_tokens:
                current_group.append(section)
                current_group_tokens += section_tokens
            else:
                # 超出限制，先保存当前组，重新开始
                if current_group:
                    merged = self._merge_group(current_group)
                    if merged:
                        merged_sections.append(merged)
                current_group = [section]
                current_group_tokens = section_tokens

        # 处理最后剩余的组
        if current_group:
            merged = self._merge_group(current_group)
            if merged:
                merged_sections.append(merged)

        return merged_sections

    def _merge_group(self, group: List[Dict]) -> Dict:
        """合并一组片段"""
        if len(group) == 1:
            return group[0]

        # 合并内容
        merged_content = '\n\n'.join(s["content"] for s in group if s["content"])
        # 使用第一个有标题的片段作为标题
        merged_title = ""
        for s in group:
            if s["has_heading"] and s["title"]:
                merged_title = s["title"]
                break

        return {
            "title": merged_title,
            "content": merged_content.strip(),
            "has_heading": any(s["has_heading"] for s in group)
        }

    def split_long_section_by_paragraphs(self, section: Dict) -> List[Dict]:
        """
        对过长的 # 片段按自然段进一步拆分

        拆分优先级：
        1. 按空行（段落）拆分
        2. 按列表项（1. 2. 3. 或 - ）拆分
        3. 按图表分组（<PIC>标签周围的完整语义块）

        保证单个块尽量在 300-800 字范围内
        """
        content = section["content"]
        content_tokens = self.count_tokens(content)

        # 如果已经在目标范围内，不需要拆分
        if self.min_tokens <= content_tokens <= self.max_tokens:
            return [section]

        # 如果太短，直接返回
        if content_tokens < self.min_tokens:
            return [section]

        # 需要拆分过长的内容
        sub_sections = []

        # 第一步：尝试按段落（空行）拆分
        paragraphs = re.split(r'\n\s*\n', content)

        # 第二步：对每个段落，如果有列表则进一步拆分
        refined_paragraphs = []
        for para in paragraphs:
            if not para.strip():
                continue
            # 检测是否包含列表
            if re.search(r'(^\d+\.\s+|^-|\s+[-•]\s+)', para, re.MULTILINE):
                # 按列表项拆分，保留列表标记
                list_items = re.findall(r'(?m)^(?:\d+\.\s+|-\s+|\s+[-•]\s+).+$', para)
                if list_items:
                    for item in list_items:
                        if item.strip():
                            refined_paragraphs.append(item.strip())
                else:
                    refined_paragraphs.append(para.strip())
            else:
                refined_paragraphs.append(para.strip())

        # 第三步：按目标范围组合段落
        current_block = ""
        current_tokens = 0
        current_pic_tags = []

        for para in refined_paragraphs:
            para_tokens = self.count_tokens(para)
            para_pic_tags = self.extract_all_pic_tags(para)

            # 如果当前段落包含图片，尽量保持完整
            if para_pic_tags:
                # 如果当前块已有内容且加上这个段落会超限，先保存当前块
                if current_tokens > 0 and current_tokens + para_tokens > self.max_tokens:
                    sub_sections.append({
                        "title": section["title"],
                        "content": current_block.strip(),
                        "has_heading": section["has_heading"]
                    })
                    current_block = ""
                    current_tokens = 0
                    current_pic_tags = []

                # 如果单个带图片的段落就超限，尝试进一步拆分（但不拆分图片所在的部分）
                if para_tokens > self.max_tokens:
                    # 将带图片的段落按句子进一步拆分
                    sub_paragraphs = self._split_paragraph_preserving_images(para)
                    for sub_para in sub_paragraphs:
                        if sub_para.strip():
                            sub_sections.append({
                                "title": section["title"],
                                "content": sub_para.strip(),
                                "has_heading": False  # 子块不再保留标题标记
                            })
                else:
                    # 添加到当前块
                    if current_block:
                        current_block += '\n\n' + para
                    else:
                        current_block = para
                    current_tokens += para_tokens
                    current_pic_tags.extend(para_pic_tags)

                    # 如果当前块达到最小阈值，可以考虑保存
                    if current_tokens >= self.min_tokens and current_tokens <= self.max_tokens:
                        sub_sections.append({
                            "title": section["title"],
                            "content": current_block.strip(),
                            "has_heading": section["has_heading"] if len(sub_sections) == 0 else False
                        })
                        current_block = ""
                        current_tokens = 0
                        current_pic_tags = []
            else:
                # 不带图片的段落
                if current_tokens + para_tokens <= self.max_tokens:
                    if current_block:
                        current_block += '\n\n' + para
                    else:
                        current_block = para
                    current_tokens += para_tokens

                    # 达到目标范围就保存
                    if current_tokens >= self.min_tokens:
                        sub_sections.append({
                            "title": section["title"],
                            "content": current_block.strip(),
                            "has_heading": section["has_heading"] if len(sub_sections) == 0 else False
                        })
                        current_block = ""
                        current_tokens = 0
                else:
                    # 保存当前块，开始新块
                    if current_block:
                        sub_sections.append({
                            "title": section["title"],
                            "content": current_block.strip(),
                            "has_heading": section["has_heading"] if len(sub_sections) == 0 else False
                        })
                    current_block = para
                    current_tokens = para_tokens

        # 保存最后剩余的内容
        if current_block:
            sub_sections.append({
                "title": section["title"],
                "content": current_block.strip(),
                "has_heading": section["has_heading"] if len(sub_sections) == 0 else False
            })

        # 如果拆分后只有一个块且过长，进一步强制拆分
        if len(sub_sections) == 1 and self.count_tokens(sub_sections[0]["content"]) > self.max_tokens:
            return self._force_split_section(sub_sections[0])

        return sub_sections if sub_sections else [section]

    def _split_paragraph_preserving_images(self, paragraph: str) -> List[str]:
        """
        在保持图片完整性的前提下拆分段落
        """
        # 提取段落中的所有图片标签
        pic_tags = self.extract_all_pic_tags(paragraph)

        if len(pic_tags) <= 1:
            # 只有一个或没有图片，按句子拆分
            return self._split_by_sentences(paragraph)

        # 多个图片，尝试按图片分组
        parts = re.split(r'(<PIC:[^>]+>)', paragraph)
        groups = []
        current_group = ""

        for part in parts:
            if part.startswith('<PIC:'):
                current_group += part
            else:
                if current_group and self.count_tokens(current_group) > self.max_tokens:
                    groups.append(current_group.strip())
                    current_group = part
                else:
                    current_group += part

        if current_group:
            groups.append(current_group.strip())

        # 如果分组后仍有过长的，进一步按句子拆分
        result = []
        for group in groups:
            if self.count_tokens(group) > self.max_tokens:
                result.extend(self._split_by_sentences(group))
            else:
                result.append(group)

        return result

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子拆分文本"""
        # 按中文和英文句子结束符拆分
        sentences = re.split(r'([。.！？!?])', text)

        chunks = []
        current_chunk = ""

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # 添加标点符号（如果有）
            if i + 1 < len(sentences) and re.match(r'[。.！？!？]', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1

            if not sentence.strip():
                continue

            if self.count_tokens(current_chunk + sentence) <= self.max_tokens:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split_section(self, section: Dict) -> List[Dict]:
        """强制拆分过长的片段（最后手段）"""
        content = section["content"]
        paragraphs = re.split(r'\n\n', content)

        result = []
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue

            para_tokens = self.count_tokens(para)
            if para_tokens <= self.max_tokens:
                result.append({
                    "title": section["title"],
                    "content": para.strip(),
                    "has_heading": section["has_heading"] if i == 0 else False
                })
            else:
                # 进一步按句子拆分
                sub_sentences = self._split_by_sentences(para)
                for j, sub in enumerate(sub_sentences):
                    result.append({
                        "title": section["title"],
                        "content": sub.strip(),
                        "has_heading": section["has_heading"] if (i == 0 and j == 0) else False
                    })

        return result if result else [section]

    def process_sections_with_images(self, sections: List[Dict], manual_file_path: str) -> List[Chunk]:
        """处理包含图片信息的分段"""
        processed_chunks = []

        for i, section in enumerate(sections):
            content = section["content"]
            pic_tags = self.extract_all_pic_tags(content)

            if pic_tags:
                all_image_names = []
                all_image_paths = []
                all_ocr_texts = []
                all_visual_descriptions = []

                for pic_tag in pic_tags:
                    associated_data = self.find_associated_image_data(manual_file_path, pic_tag)
                    all_image_names.extend(associated_data["image_names"])
                    all_image_paths.extend(associated_data["image_paths"])
                    all_ocr_texts.extend(associated_data["ocr_texts"])
                    all_visual_descriptions.extend(associated_data["visual_descriptions"])

                # 保留<PIC>标签在文本中，不再移除
                text_with_pic_tags = content.strip()

                chunk_obj = Chunk(
                    id=f"{os.path.basename(manual_file_path)}_chunk_{i}",
                    text=text_with_pic_tags,
                    image_names=all_image_names,
                    image_paths=all_image_paths,
                    ocr_texts=all_ocr_texts,
                    visual_descriptions=all_visual_descriptions,
                    manual_path=manual_file_path
                )
                processed_chunks.append(chunk_obj)
            else:
                chunk_obj = Chunk(
                    id=f"{os.path.basename(manual_file_path)}_chunk_{i}",
                    text=content,
                    image_names=[],
                    image_paths=[],
                    ocr_texts=[],
                    visual_descriptions=[],
                    manual_path=manual_file_path
                )
                processed_chunks.append(chunk_obj)

        return processed_chunks

    def handle_edge_cases(self, chunks: List[Chunk]) -> List[Chunk]:
        """处理边缘情况 - 简化逻辑，避免文本丢失"""
        processed_chunks = []

        for chunk in chunks:
            # 最终检查 token 限制
            if self.count_tokens(chunk.text) <= self.max_tokens:
                processed_chunks.append(chunk)
            else:
                # 进一步分割
                sub_texts = self.further_split_chunk(chunk.text)
                for idx, sub_text in enumerate(sub_texts):
                    if sub_text.strip():  # 确保不为空
                        sub_chunk = Chunk(
                            id=f"{chunk.id}_sub_{idx}",
                            text=sub_text.strip(),
                            image_names=chunk.image_names if idx == 0 else [],
                            image_paths=chunk.image_paths if idx == 0 else [],
                            ocr_texts=chunk.ocr_texts if idx == 0 else [],
                            visual_descriptions=chunk.visual_descriptions if idx == 0 else [],
                            manual_path=chunk.manual_path
                        )
                        processed_chunks.append(sub_chunk)

        return processed_chunks

    def verify_text_integrity(self, original_text: str, chunks: List[Chunk]) -> bool:
        """
        验证分块后的文本完整性
        检查所有分块的 text 加起来是否等于原文
        """
        # 移除原文中的所有<PIC>标签
        original_without_pic = re.sub(r'<PIC:[^>]+>', '', original_text).strip()

        # 合并所有分块的文本（也移除<PIC>标签以便比较）
        merged_text = ''.join(chunk.text for chunk in chunks)
        merged_without_pic = re.sub(r'<PIC:[^>]+>', '', merged_text).strip()

        # 比较
        is_complete = original_without_pic == merged_without_pic

        if not is_complete:
            print(f"  [WARNING] 文本完整性检查失败!")
            print(f"     原文长度：{len(original_without_pic)}")
            print(f"     合并后长度：{len(merged_without_pic)}")
            print(f"     差异：{len(original_without_pic) - len(merged_without_pic)} 字符")
        else:
            print(f"  [OK] 文本完整性检查通过")

        return is_complete

    def verify_sections_integrity(self, original_sections: List[Dict], final_chunks: List[Chunk]) -> bool:
        """
        验证从原始分段到最终分块的文本完整性
        """
        # 合并原始分段的所有内容
        original_merged = ''.join(s["content"] for s in original_sections)

        # 合并所有分块的文本
        chunks_merged = ''.join(c.text for c in final_chunks)

        # 移除<PIC>标签后比较
        original_clean = re.sub(r'<PIC:[^>]+>', '', original_merged).strip()
        chunks_clean = re.sub(r'<PIC:[^>]+>', '', chunks_merged).strip()

        is_complete = original_clean == chunks_clean

        if not is_complete:
            print(f"  [WARNING] 分段到分块的完整性检查失败!")
            print(f"     原始分段总长度：{len(original_clean)}")
            print(f"     最终分块总长度：{len(chunks_clean)}")
            print(f"     差异：{len(original_clean) - len(chunks_clean)} 字符")
        else:
            print(f"  [OK] 分段到分块的完整性检查通过")

        return is_complete

    def further_split_chunk(self, text: str) -> List[str]:
        """当块超过 token 限制时进一步分割"""
        sentences = re.split(r'[。.！？.!?]', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if self.count_tokens(current_chunk + ' ' + sentence) <= self.max_tokens:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_single_file(self, file_path: str) -> List[Chunk]:
        """处理单个文档文件 - 使用新的层级分块策略，带文本完整性验证"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 步骤 1: 按一级标题 (#) 分割文档
        sections = self.split_by_level_one_headings(content)
        print(f"  - 按一级标题分割为 {len(sections)} 个章节")

        # 步骤 2: 合并语义相近的短片段
        merged_sections = self.merge_short_sections(sections)
        print(f"  - 合并短片段后：{len(merged_sections)} 个章节")

        # 步骤 3: 对过长的片段按自然段进一步拆分
        refined_sections = []
        for section in merged_sections:
            refined = self.split_long_section_by_paragraphs(section)
            refined_sections.extend(refined)
        print(f"  - 拆分长片段后：{len(refined_sections)} 个分段")

        # 步骤 4: 处理图片信息并转换为 Chunk 对象
        chunks = self.process_sections_with_images(refined_sections, file_path)

        # 步骤 5: 处理边缘情况
        final_chunks = self.handle_edge_cases(chunks)

        # 步骤 6: 验证文本完整性
        self.verify_sections_integrity(refined_sections, final_chunks)

        return final_chunks

    def process_directory(self) -> Dict[str, List[Chunk]]:
        """处理整个目录中的所有 txt 文件"""
        results = {}

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.txt') and 'clean_vr_Manuals' in root:
                    file_path = os.path.join(root, file)
                    print(f"\n正在处理：{file_path}")

                    try:
                        chunks = self.process_single_file(file_path)
                        results[file_path] = chunks
                        print(f"完成处理 {file_path}，生成 {len(chunks)} 个分块")

                        total_chunks_with_images = sum(1 for c in chunks if c.image_names)
                        print(f"  - 其中 {total_chunks_with_images} 个分块包含图片")

                        # 统计字数分布
                        token_counts = [self.count_tokens(c.text) for c in chunks]
                        if token_counts:
                            avg_tokens = sum(token_counts) / len(token_counts)
                            min_tokens = min(token_counts)
                            max_tokens = max(token_counts)
                            print(f"  - 分块大小：平均={avg_tokens:.0f} tokens, 最小={min_tokens}, 最大={max_tokens}")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错：{str(e)}")

        return results


def save_results(results: Dict[str, List[Chunk]], output_dir: str):
    """将结果保存到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)

    for file_path, chunks in results.items():
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(file_path)}_chunks.json"
        )

        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = {
                "id": chunk.id,
                "text": chunk.text,
                "image_names": chunk.image_names,
                "image_paths": chunk.image_paths,
                "ocr_texts": chunk.ocr_texts,
                "visual_descriptions": chunk.visual_descriptions,
                "manual_path": chunk.manual_path
            }
            serializable_chunks.append(serializable_chunk)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

        print(f"结果已保存至：{output_file}")


def main():
    data_dir = r"D:\PyCharm\clean\Data\clean_vr_Manuals"
    output_dir = r"D:\PyCharm\clean\Data\processed_chunks_hierarchical"
    images_dir = r"D:\PyCharm\clean\images_standard"
    embeddings_dir = r"D:\PyCharm\clean\ocrtext_image_embeddings"

    processor = HierarchicalManualProcessor(
        data_dir,
        min_tokens=300,
        max_tokens=800,
        images_dir=images_dir,
        embeddings_dir=embeddings_dir
    )
    results = processor.process_directory()
    save_results(results, output_dir)

    total_chunks = sum(len(chunks) for chunks in results.values())
    chunks_with_images = sum(sum(1 for c in chunks if c.image_names) for chunks in results.values())

    print(f"\n处理完成!")
    print(f"共处理 {len(results)} 个文档")
    print(f"生成 {total_chunks} 个分块")
    print(f"其中 {chunks_with_images} 个分块包含图片信息")


if __name__ == "__main__":
    main()
