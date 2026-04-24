import os
import json
import re

# 配置路径
MANUALS_DIR = r"D:\PyCharm\clean\Data\clean_vr_Manuals"
IMAGES_DIR = r"D:\PyCharm\clean\Data\Images_standard"
OCR_DIR = r"D:\PyCharm\clean\ocr_results"
OUTPUT_DIR = r"D:\PyCharm\clean\chunk_results"

# 分块最大字数
MAX_CHUNK_LENGTH = 800

# 最小块合并阈值
MIN_CHUNK_LENGTH = 200


def chunk_text(text):
    """
    根据用户要求的分块规则处理文本

    规则：
    1. 从txt文档开头内容到遇到第一个#之间的内容分为第一块
    2. 某个文档开头是#，则从这个#开始到下一个#前分为第一块
    3. 中间的内容按顺序按照遇到#到下一个#之前的内容分为一块
    4. 如果两个#之间的内容太少，可以与后面的块合并为一块
    5. 分块最大不要超过800字
    6. 最后一个#到文档结尾的内容分为最后一块
    """
    if not text.strip():
        return []

    # 查找所有#的位置
    hash_positions = []
    for i, char in enumerate(text):
        if char == '#':
            hash_positions.append(i)

    # 如果没有#号，整个文本作为一块
    if not hash_positions:
        # 检查长度，如果超过800字需要分割
        if len(text) > MAX_CHUNK_LENGTH:
            return split_long_text(text, MAX_CHUNK_LENGTH)
        return [text.strip()]

    chunks = []

    # 判断文档开头是否是#
    first_hash_pos = hash_positions[0]

    if first_hash_pos == 0:
        # 开头是#，分块逻辑：
        # - 从第一个#到下一个#前为第一块
        # - 从第二个#到第三个#前为第二块
        # - ...
        # - 最后一个#到结尾为最后一块
        for i in range(len(hash_positions)):
            start_pos = hash_positions[i]
            if i + 1 < len(hash_positions):
                end_pos = hash_positions[i + 1]
                chunk = text[start_pos:end_pos].strip()
            else:
                # 最后一个#到结尾
                chunk = text[start_pos:].strip()

            if chunk:
                chunks.append(chunk)
    else:
        # 开头不是#，分块逻辑：
        # - 从开头到第一个#之前为第一块
        # - 从第一个#到第二个#之前为第二块
        # - 从第二个#到第三个#之前为第三块
        # - ...
        # - 最后一个#到结尾为最后一块

        # 第一块：开头到第一个#之前
        chunk = text[:first_hash_pos].strip()
        if chunk:
            chunks.append(chunk)

        # 中间和最后一块：按#分割
        for i in range(len(hash_positions)):
            start_pos = hash_positions[i]
            if i + 1 < len(hash_positions):
                end_pos = hash_positions[i + 1]
                chunk = text[start_pos:end_pos].strip()
            else:
                # 最后一个#到结尾
                chunk = text[start_pos:].strip()

            if chunk:
                chunks.append(chunk)

    # 合并小块
    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]

        # 如果当前块太小且不是最后一块，尝试与后面的块合并
        while len(current_chunk) < MIN_CHUNK_LENGTH and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            # 检查合并后是否超过最大长度
            if len(current_chunk) + len(next_chunk) <= MAX_CHUNK_LENGTH:
                current_chunk = current_chunk + "\n\n" + next_chunk
                i += 1
            else:
                break

        # 检查合并后是否超过最大长度
        if len(current_chunk) > MAX_CHUNK_LENGTH:
            # 分割过长的块
            sub_chunks = split_long_text(current_chunk, MAX_CHUNK_LENGTH)
            merged_chunks.extend(sub_chunks)
        else:
            merged_chunks.append(current_chunk)

        i += 1

    return merged_chunks


def split_long_text(text, max_length):
    """
    将过长的文本分割为多个块
    """
    if len(text) <= max_length:
        return [text.strip()]

    chunks = []
    lines = text.split('\n')
    current_chunk = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if len(current_chunk) + len(line) + 1 <= max_length:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
        else:
            # 当前行超过最大长度，按字符分割
            if len(line) > max_length:
                # 如果当前块不为空，先保存
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 分割长行
                while len(line) > max_length:
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                if line:
                    current_chunk = line
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_pic_references(text):
    """
    从文本中提取<PIC:xxx>引用，返回图片名称列表
    """
    pattern = r'<PIC:([^>]+)>'
    matches = re.findall(pattern, text)
    return matches


def get_image_info(pic_name, images_dir, ocr_dir):
    """
    获取图片信息和OCR结果

    返回: (image_name, image_path, ocr_text, visual_desc)
    """
    image_name = f"{pic_name}.png"
    image_path = os.path.join(images_dir, image_name)

    ocr_json_path = os.path.join(ocr_dir, f"{pic_name}.json")

    ocr_text = ""
    visual_desc = ""

    if os.path.exists(ocr_json_path):
        try:
            with open(ocr_json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                ocr_text = ocr_data.get('ocr_text', '')
                visual_desc = ocr_data.get('visual_desc', '')
        except Exception as e:
            print(f"警告: 读取OCR文件错误 {ocr_json_path}: {e}")

    return image_name, image_path, ocr_text, visual_desc


def process_manual_file(txt_file, output_dir):
    """
    处理单个文档文件，生成对应的JSON分块文件
    """
    filename = os.path.basename(txt_file)
    manual_name = os.path.splitext(filename)[0]

    # 读取txt文档
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 分块
    chunks = chunk_text(text)

    # 为每个块生成数据
    block_data = []
    for idx, chunk_content in enumerate(chunks, start=1):
        # 提取<PIC:xxx>引用
        pic_names = extract_pic_references(chunk_content)

        # 获取图片信息
        image_names = []
        image_paths = []
        ocr_texts = []
        visual_descriptions = []

        for pic_name in pic_names:
            img_name, img_path, ocr_txt, vis_desc = get_image_info(
                pic_name, IMAGES_DIR, OCR_DIR
            )

            # 检查图片是否存在
            if os.path.exists(img_path):
                image_names.append(img_name)
                image_paths.append(img_path)
            else:
                print(f"警告: 图片不存在 - {img_path}")

            # 无论图片是否存在，都添加OCR结果
            ocr_texts.append(ocr_txt)
            visual_descriptions.append(vis_desc)

        block = {
            "id": f"{manual_name}_block_{idx}",
            "text": chunk_content,
            "image_names": image_names,
            "image_paths": image_paths,
            "ocr_texts": ocr_texts,
            "visual_descriptions": visual_descriptions,
            "manual_path": txt_file
        }
        block_data.append(block)

    # 保存JSON文件
    output_json_path = os.path.join(output_dir, f"{manual_name}.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(block_data, f, ensure_ascii=False, indent=2)

    print(f"已生成: {output_json_path}")
    print(f"  - 总块数: {len(block_data)}")
    print(f"  - 总图片数: {sum(len(b['image_names']) for b in block_data)}")

    return block_data


def main():
    """
    主函数：处理所有文档
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有txt文件
    txt_files = []
    for file in os.listdir(MANUALS_DIR):
        if file.endswith('.txt'):
            txt_files.append(os.path.join(MANUALS_DIR, file))

    # 按文件名排序
    txt_files.sort()

    print("=" * 60)
    print("开始处理手册文档...")
    print("=" * 60)
    print(f"手册目录: {MANUALS_DIR}")
    print(f"图片目录: {IMAGES_DIR}")
    print(f"OCR结果目录: {OCR_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"找到 {len(txt_files)} 个txt文档")
    print("=" * 60)

    # 处理每个文件
    total_blocks = 0
    total_images = 0
    total_ocr = 0

    for idx, txt_file in enumerate(txt_files, start=1):
        print(f"\n[{idx}/{len(txt_files)}] 处理: {os.path.basename(txt_file)}")
        block_data = process_manual_file(txt_file, OUTPUT_DIR)

        total_blocks += len(block_data)
        total_images += sum(len(b['image_names']) for b in block_data)
        total_ocr += sum(len(b['ocr_texts']) for b in block_data)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"处理文档数: {len(txt_files)}")
    print(f"生成JSON文件数: {len(txt_files)}")
    print(f"总块数: {total_blocks}")
    print(f"总图片数: {total_images}")
    print(f"总OCR条目数: {total_ocr}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()