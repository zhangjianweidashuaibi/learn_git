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


class ManualProcessor:
    def __init__(self, data_dir: str, max_tokens: int = 800,
                 images_dir: str = r"D:\PyCharm\clean\images_standard",
                 embeddings_dir: str = r"D:\PyCharm\clean\ocrtext_image_embeddings"):
        self.data_dir = data_dir
        self.max_tokens = max_tokens
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

    def split_by_semantic_hierarchy(self, text: str) -> List[str]:
        """根据层级语义进行分割"""
        heading_patterns = [
            (r'^#{1}\s+(.+)', 1),
            (r'^#{2}\s+(.+)', 2),
            (r'^#{3}\s+(.+)', 3),
            (r'^\d+\.\d+\.\d+\s+.+', 4),
            (r'^\d+\.\d+\s+.+', 5),
            (r'^\d+\.\s+.+', 6),
            (r'^[一二三四五六七八九十]+、', 7),
            (r'^[（\(][一二三四五六七八九十]+\)]', 8),
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

            is_heading = False
            heading_level = float('inf')
            for pattern, level in heading_patterns:
                if re.match(pattern, para.strip(), re.MULTILINE):
                    is_heading = True
                    heading_level = level
                    break

            if is_heading and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            elif self.count_tokens(current_chunk + '\n\n' + para) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para

            i += 1

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_chunks_with_images(self, chunks: List[str], manual_file_path: str) -> List[Chunk]:
        """处理包含图片信息的分块"""
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            pic_tags = self.extract_all_pic_tags(chunk)

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

                clean_text = re.sub(r'<PIC:[^>]+>', '', chunk).strip()

                chunk_obj = Chunk(
                    id=f"{os.path.basename(manual_file_path)}_chunk_{i}",
                    text=clean_text,
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
                    text=chunk,
                    image_names=[],
                    image_paths=[],
                    ocr_texts=[],
                    visual_descriptions=[],
                    manual_path=manual_file_path
                )
                processed_chunks.append(chunk_obj)

        return processed_chunks

    def handle_edge_cases(self, chunks: List[Chunk]) -> List[Chunk]:
        """处理边缘情况"""
        processed_chunks = []

        for chunk in chunks:
            pic_count_in_text = len(re.findall(r'<PIC:', chunk.text))
            if pic_count_in_text > len(chunk.image_names):
                remaining_text_parts = re.split(r'(<PIC:.+?>)', chunk.text)
                new_chunks = []
                current_part = ""

                for part in remaining_text_parts:
                    if part.startswith('<PIC:'):
                        associated_data = self.find_associated_image_data(chunk.manual_path, part)

                        if associated_data["image_names"]:
                            if current_part.strip():
                                new_chunk = Chunk(
                                    id=f"{chunk.id}_split_{len(new_chunks)}",
                                    text=current_part.strip(),
                                    image_names=associated_data["image_names"],
                                    image_paths=associated_data["image_paths"],
                                    ocr_texts=associated_data["ocr_texts"],
                                    visual_descriptions=associated_data["visual_descriptions"],
                                    manual_path=chunk.manual_path
                                )
                                new_chunks.append(new_chunk)
                            current_part = ""
                        else:
                            current_part += part
                    else:
                        current_part += part

                if current_part.strip():
                    new_chunk = Chunk(
                        id=f"{chunk.id}_split_final",
                        text=current_part.strip(),
                        image_names=chunk.image_names,
                        image_paths=chunk.image_paths,
                        ocr_texts=chunk.ocr_texts,
                        visual_descriptions=chunk.visual_descriptions,
                        manual_path=chunk.manual_path
                    )
                    new_chunks.append(new_chunk)

                processed_chunks.extend(new_chunks)
            else:
                processed_chunks.append(chunk)

        final_chunks = []
        for chunk in processed_chunks:
            if self.count_tokens(chunk.text) <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                sub_texts = self.further_split_chunk(chunk.text)
                for idx, sub_text in enumerate(sub_texts):
                    sub_chunk = Chunk(
                        id=f"{chunk.id}_sub_{idx}",
                        text=sub_text,
                        image_names=chunk.image_names if idx == 0 else [],
                        image_paths=chunk.image_paths if idx == 0 else [],
                        ocr_texts=chunk.ocr_texts if idx == 0 else [],
                        visual_descriptions=chunk.visual_descriptions if idx == 0 else [],
                        manual_path=chunk.manual_path
                    )
                    final_chunks.append(sub_chunk)

        return final_chunks

    def further_split_chunk(self, text: str) -> List[str]:
        """当块超过 token 限制时进一步分割"""
        sentences = re.split(r'[。！？.!?]', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip() + "."
            if not sentence.strip():
                continue

            if self.count_tokens(current_chunk + sentence) <= self.max_tokens:
                if current_chunk:
                    current_chunk += sentence
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
        """处理单个文档文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        initial_chunks = self.split_by_semantic_hierarchy(content)
        chunks_with_pics = self.process_chunks_with_images(initial_chunks, file_path)
        final_chunks = self.handle_edge_cases(chunks_with_pics)

        return final_chunks

    def process_directory(self) -> Dict[str, List[Chunk]]:
        """处理整个目录中的所有 txt 文件"""
        results = {}

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.txt') and 'clean_vr_Manuals' in root:
                    file_path = os.path.join(root, file)
                    print(f"正在处理：{file_path}")

                    try:
                        chunks = self.process_single_file(file_path)
                        results[file_path] = chunks
                        print(f"完成处理 {file_path}，生成 {len(chunks)} 个分块")

                        total_chunks_with_images = sum(1 for c in chunks if c.image_names)
                        print(f"  - 其中 {total_chunks_with_images} 个分块包含图片")
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
    output_dir = r"D:\PyCharm\clean\Data\processed_chunks"
    images_dir = r"D:\PyCharm\clean\images_standard"
    embeddings_dir = r"D:\PyCharm\clean\ocrtext_image_embeddings"

    processor = ManualProcessor(data_dir, images_dir=images_dir, embeddings_dir=embeddings_dir)
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
