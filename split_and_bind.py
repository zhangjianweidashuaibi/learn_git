import os
import json
from typing import List, Dict, Any, Set, Tuple


# 配置路径
MANUALS_DIR = r"D:\PyCharm\clean\Data\clean_vr_Manuals"
OCR_RESULTS_DIR = r"D:\PyCharm\clean\ocr_results"
IMAGES_DIR = r"D:\PyCharm\clean\images_standard"
OUTPUT_PATH = r"D:\PyCharm\clean\splitted_results"


def read_manual_content(file_path: str) -> str:
    """读取文档内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_document_into_blocks(content: str) -> List[str]:
    """
    将文档按 # 符号分割成多个块
    - 开头到第一个 #（不包含 #）为第一块
    - 最后一个 # 到结尾（包含 # 及其后内容）为最后一块
    - 中间按 # 之间的内容分块
    - 如果两个 # 之间内容太少，与后面合并
    - 每块不超过 800 字
    """
    blocks = []

    # 找到所有的 # 位置
    hashes_positions = [i for i, char in enumerate(content) if char == '#']

    if not hashes_positions:
        # 没有 #，整个文档为一块
        if content.strip():
            blocks.append(content.strip())
        return blocks

    # 第一部分：开头到第一个 #（不包含 #）
    first_hash_pos = hashes_positions[0]
    first_block = content[:first_hash_pos].strip()
    if first_block:
        blocks.append(first_block)

    # 最后一部分：最后一个 # 到结尾（包含 # 及其后内容）
    last_hash_pos = hashes_positions[-1]
    last_block = content[last_hash_pos:].strip()
    if last_block:
        blocks.append(last_block)

    # 中间部分：两个 # 之间的内容
    middle_hashes = hashes_positions[1:-1]
    for i in range(len(middle_hashes)):
        start_pos = middle_hashes[i] + 1  # 从 # 后面的内容开始
        if i < len(middle_hashes) - 1:
            end_pos = middle_hashes[i + 1]
        else:
            end_pos = last_hash_pos

        block_content = content[start_pos:end_pos].strip()

        if block_content:
            blocks.append(block_content)

    # 检查并合并太小的块（小于 50 字）
    merged_blocks = merge_small_blocks(blocks)

    # 确保没有块超过 800 字
    final_blocks = ensure_max_length(merged_blocks, max_length=800)

    return final_blocks


def merge_small_blocks(blocks: List[str], min_length: int = 50) -> List[str]:
    """合并过小的块"""
    if not blocks:
        return blocks

    merged = []
    current_block = blocks[0] if blocks else ""

    for i in range(1, len(blocks)):
        next_block = blocks[i]

        # 如果当前块太小，尝试与下一个合并
        if len(current_block) < min_length and len(current_block) > 0:
            current_block = current_block + "\n\n" + next_block
        else:
            # 当前块长度合适，保存并开始下一块
            if current_block.strip():
                merged.append(current_block.strip())
            current_block = next_block

    # 添加最后一个块
    if current_block.strip():
        merged.append(current_block.strip())

    return merged


def ensure_max_length(blocks: List[str], max_length: int = 800) -> List[str]:
    """确保每个块不超过最大长度，超出则拆分"""
    result = []

    for block in blocks:
        if len(block) <= max_length:
            result.append(block)
        else:
            # 超长则按字符数切分
            for i in range(0, len(block), max_length):
                chunk = block[i:i + max_length]
                result.append(chunk)

    return result


def extract_pic_tags(content: str) -> List[str]:
    """从内容中提取所有<PIC:标签名>格式的标签（按出现顺序）"""
    import re
    pattern = r'<PIC:(.+?)>'
    matches = re.findall(pattern, content)
    return matches


def count_all_pic_tags_in_document(content: str) -> int:
    """统计文档中所有<PIC:xxx>的总数（包括重复）"""
    import re
    pattern = r'<PIC:(.+?)>'
    matches = re.findall(pattern, content)
    return len(matches)


def get_json_file_by_tag(tag_name: str) -> Dict[str, Any]:
    """根据标签名查找对应的 JSON 文件并读取内容"""
    # 文件名格式：tag_name.json
    json_filename = f"{tag_name}.json"
    json_filepath = os.path.join(OCR_RESULTS_DIR, json_filename)

    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ocr_text = data.get('ocr_text', '')
                visual_desc = data.get('visual_desc', '')
                return {
                    'tag_name': tag_name,
                    'ocr_text': ocr_text,
                    'visual_desc': visual_desc,
                    'found': True
                }
        except Exception as e:
            print(f"读取 JSON 文件失败：{json_filename}, 错误：{e}")
            return {'tag_name': tag_name, 'found': False}
    else:
        return {'tag_name': tag_name, 'found': False}


def get_image_info_by_tag(tag_name: str) -> Dict[str, Any]:
    """根据标签名查找对应的图片"""
    image_filename = f"{tag_name}.png"
    image_filepath = os.path.join(IMAGES_DIR, image_filename)

    if os.path.exists(image_filepath):
        return {
            'image_name': image_filename,
            'image_path': image_filepath,
            'found': True
        }
    else:
        return {'image_name': image_filename, 'image_path': '', 'found': False}


def enrich_block_with_ocr_data(block_content: str, processed_tags: Set[str]) -> Tuple[str, int, Set[str]]:
    """
    将 OCR 数据添加到块的末尾
    返回：(enriched_content, ocr_tags_added_count, tags_added_set)
    """
    pic_tags = extract_pic_tags(block_content)

    enriched_content = block_content
    ocr_tags_added = set()

    for tag_name in pic_tags:
        if tag_name not in processed_tags:
            ocr_data = get_json_file_by_tag(tag_name)
            if ocr_data['found']:
                enrichment = f"\n\n【OCR 识别内容】:\n文本：{ocr_data['ocr_text']}\n视觉描述：{ocr_data['visual_desc']}"
                enriched_content += enrichment
                ocr_tags_added.add(tag_name)
            processed_tags.add(tag_name)

    return enriched_content, len(ocr_tags_added), ocr_tags_added


def enrich_block_with_image_data(block_content: str, processed_images: Set[str]) -> Tuple[str, int, Set[str]]:
    """
    将图片信息添加到块的末尾
    返回：(enriched_content, image_tags_added_count, unique_images_added_set)
    """
    pic_tags = extract_pic_tags(block_content)

    enriched_content = block_content
    image_tags_added = set()

    for tag_name in pic_tags:
        if tag_name not in processed_images:
            img_info = get_image_info_by_tag(tag_name)
            if img_info['found']:
                image_info_text = f"\n\n【关联图片】:\n图片名称：{img_info['image_name']}\n图片路径：{img_info['image_path']}"
                enriched_content += image_info_text
                image_tags_added.add(tag_name)

            processed_images.add(tag_name)

    return enriched_content, len(image_tags_added), image_tags_added


def count_pic_names_in_content(content: str) -> int:
    """统计内容中出现的<PIC:标签名>数量（包括重复）"""
    import re
    pattern = r'<PIC:(.+?)>'
    matches = re.findall(pattern, content)
    return len(matches)


def process_document(file_name: str) -> Dict[str, Any]:
    """处理单个文档"""
    file_path = os.path.join(MANUALS_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"警告：文件不存在 - {file_path}")
        return None

    print(f"\n处理文档：{file_name}")

    # 读取文档内容
    content = read_manual_content(file_path)

    # 统计原始文档中的<PIC:xxx>总数
    original_pic_count = count_all_pic_tags_in_document(content)
    print(f"  - 原始文档中有 {original_pic_count} 个<PIC:xxx>标签")

    # 分块
    blocks = split_document_into_blocks(content)
    total_blocks = len(blocks)
    print(f"  - 共分 {total_blocks} 块")

    # 验证：检查是否所有块的总 PIC 标签数等于原始文档中的数量
    total_pic_in_blocks = 0
    for block in blocks:
        total_pic_in_blocks += count_pic_names_in_content(block)

    if total_pic_in_blocks != original_pic_count:
        print(f"  ⚠️ 警告：分块后 PIC 标签数不匹配！原始={original_pic_count}, 分块后={total_pic_in_blocks}")
        print(f"     丢失的标签数：{original_pic_count - total_pic_in_blocks}")

    # 统计信息
    blocks_with_pic_tags = 0
    total_unique_tags_found = set()  # 文档中出现的所有唯一标签
    total_unique_images_bound = set()  # 成功绑定到的唯一图片
    total_ocr_tags_processed = 0  # 成功读取的 OCR 次数
    total_image_bindings = 0  # 成功绑定图片的次数

    enriched_blocks = []

    # 全局去重集合（跨所有块）
    ocr_tags_processed = set()
    image_tags_processed = set()

    for i, block in enumerate(blocks):
        pic_tags = extract_pic_tags(block)
        block_unique_tags = set(pic_tags)

        if block_unique_tags:
            blocks_with_pic_tags += 1
            total_unique_tags_found.update(block_unique_tags)

            # 添加 OCR 数据
            enriched_content, ocr_count, ocr_added = enrich_block_with_ocr_data(block, ocr_tags_processed)
            total_ocr_tags_processed += len(ocr_added)

            # 添加图片数据
            enriched_content, image_count, image_added = enrich_block_with_image_data(enriched_content, image_tags_processed)
            total_image_bindings += len(image_added)
            total_unique_images_bound.update(image_added)

            enriched_blocks.append({
                'block_index': i + 1,
                'content': enriched_content,
                'pic_tags': pic_tags,
                'unique_tags_in_block': list(block_unique_tags),
                'ocr_data_added': bool(ocr_added),
                'ocr_tags_added': list(ocr_added),
                'image_count': len(image_added),
                'image_tags_added': list(image_added)
            })
        else:
            enriched_blocks.append({
                'block_index': i + 1,
                'content': block,
                'pic_tags': [],
                'unique_tags_in_block': [],
                'ocr_data_added': False,
                'ocr_tags_added': [],
                'image_count': 0,
                'image_tags_added': []
            })

    blocks_without_pic = total_blocks - blocks_with_pic_tags

    summary = {
        'document_name': file_name,
        'original_pic_count': original_pic_count,
        'total_blocks': total_blocks,
        'blocks_with_pic_tags': blocks_with_pic_tags,
        'blocks_without_pic_tags': blocks_without_pic,
        'total_unique_tags_found': len(total_unique_tags_found),
        'total_unique_images_bound': len(total_unique_images_bound),
        'total_ocr_reads': total_ocr_tags_processed,
        'total_image_bindings': total_image_bindings,
        'all_unique_tags': list(total_unique_tags_found),
        'all_unique_images': list(total_unique_images_bound)
    }

    return {
        'summary': summary,
        'total_image_names_count': count_pic_names_in_content('\n'.join([b['content'] for b in enriched_blocks])),
        'blocks': enriched_blocks
    }


def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 获取所有文档文件
    if not os.path.exists(MANUALS_DIR):
        print(f"错误：文档目录不存在 - {MANUALS_DIR}")
        return

    manual_files = sorted([f for f in os.listdir(MANUALS_DIR)
                          if os.path.isfile(os.path.join(MANUALS_DIR, f))])

    print(f"找到 {len(manual_files)} 个文档待处理")
    print("=" * 60)

    # 全局统计
    grand_original_pic_count = 0
    grand_total_blocks = 0
    grand_blocks_with_pic = 0
    grand_blocks_without_pic = 0
    grand_unique_tags = set()  # 所有文档中出现的唯一标签
    grand_unique_images = set()  # 所有文档中成功绑定的唯一图片
    grand_total_ocr_reads = 0  # 总共成功读取的 OCR 次数
    grand_total_image_bindings = 0  # 总共成功绑定图片的次数（等于<PIC:xxx>数应该为 2606）
    all_summary_results = []
    all_blocks_results = []

    # 处理每个文档
    for file_name in manual_files:
        result = process_document(file_name)

        if result:
            all_summary_results.append(result['summary'])
            all_blocks_results.append(result['blocks'])

            summary = result['summary']
            grand_original_pic_count += summary['original_pic_count']
            grand_total_blocks += summary['total_blocks']
            grand_blocks_with_pic += summary['blocks_with_pic_tags']
            grand_blocks_without_pic += summary['blocks_without_pic_tags']
            grand_unique_tags.update(summary['all_unique_tags'])
            grand_unique_images.update(summary['all_unique_images'])
            grand_total_ocr_reads += summary['total_ocr_reads']
            grand_total_image_bindings += summary['total_image_bindings']

    # 输出统计信息
    print("\n" + "=" * 60)
    print("最终统计结果:")
    print("-" * 40)
    print(f"总文档数：{len(manual_files)}")
    print(f"原始文档中<PIC:xxx>总数：{grand_original_pic_count}")
    print(f"总块数：{grand_total_blocks}")
    print(f"有<PIC:xxx>的块数：{grand_blocks_with_pic}")
    print(f"无<PIC:xxx>的块数：{grand_blocks_without_pic}")
    print(f"所有文档中出现的唯一标签数：{len(grand_unique_tags)}")
    print(f"成功读取 OCR 次数：{grand_total_ocr_reads}")
    print(f"成功绑定图片次数（插入图片次数）：{grand_total_image_bindings}")
    print("-" * 40)

    # 验证检查
    print("\n" + "验证检查:")
    print("-" * 40)

    # 检查 1：<PIC:xxx>总数是否为 2606
    pic_count_check = "✓ 符合预期" if grand_original_pic_count == 2606 else "✗ 不符合预期 (应为 2606)"
    print(f"[检查 1] <PIC:xxx>总数: {grand_original_pic_count} {pic_count_check}")

    # 检查 2:<PIC:xxx>数是否等于读取插入图片次数
    binding_match = "✓ 相等" if grand_original_pic_count == grand_total_image_bindings else "✗ 不相等"
    diff = grand_original_pic_count - grand_total_image_bindings
    print(f"[检查 2]<PIC:xxx>数 ({grand_original_pic_count}) vs 图片绑定次数 ({grand_total_image_bindings}): {binding_match}")
    if diff != 0:
        print(f"       差值：{diff}")

    # 检查 3：分块后是否保留所有内容
    # （在 process_document 中已经做了验证，如果有丢失会打印警告）

    print("-" * 40)

    # 输出每个文档的分块情况
    print("\n每个文档的详细分块情况:")
    print("-" * 40)
    for summary in all_summary_results:
        print(f"文档：{summary['document_name']}")
        print(f"  原始<PIC:xxx>数：{summary['original_pic_count']}")
        print(f"  总块数：{summary['total_blocks']}")
        print(f"  有<PIC:xxx>的块数：{summary['blocks_with_pic_tags']}")
        print(f"  无<PIC:xxx>的块数：{summary['blocks_without_pic_tags']}")
        print(f"  文档中出现的唯一标签数：{summary['total_unique_tags_found']}")
        print(f"  成功读取 OCR 次数：{summary['total_ocr_reads']}")
        print(f"  成功绑定图片次数：{summary['total_image_bindings']}")
        print()

    # 保存分块结果
    output_json_path = os.path.join(OUTPUT_PATH, 'all_blocks.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_blocks_results, f, ensure_ascii=False, indent=2)
    print(f"\n分块结果已保存到：{output_json_path}")

    # 保存总结报告
    report_path = os.path.join(OUTPUT_PATH, 'processing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("文档分块与图片绑定报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("总体统计:\n")
        f.write(f"  总文档数：{len(manual_files)}\n")
        f.write(f"  原始文档中<PIC:xxx>总数：{grand_original_pic_count}\n")
        f.write(f"  总块数：{grand_total_blocks}\n")
        f.write(f"  有<PIC:xxx>的块数：{grand_blocks_with_pic}\n")
        f.write(f"  无<PIC:xxx>的块数：{grand_blocks_without_pic}\n")
        f.write(f"  所有文档中出现的唯一标签数：{len(grand_unique_tags)}\n")
        f.write(f"  成功读取 OCR 次数：{grand_total_ocr_reads}\n")
        f.write(f"  成功绑定图片次数（插入图片次数）：{grand_total_image_bindings}\n")
        f.write("\n")

        f.write("验证检查:\n")
        f.write(f"  [检查 1]<PIC:xxx>总数：{grand_original_pic_count} (预期：2606)\n")
        f.write(f"  [检查 2]<PIC:xxx>数 vs 图片绑定次数：{'相等' if grand_original_pic_count == grand_total_image_bindings else '不相等'}\n")
        f.write("\n" + "-" * 40 + "\n\n")

        for summary in all_summary_results:
            f.write(f"文档：{summary['document_name']}\n")
            f.write(f"  原始<PIC:xxx>数：{summary['original_pic_count']}\n")
            f.write(f"  总块数：{summary['total_blocks']}\n")
            f.write(f"  有<PIC:xxx>的块数：{summary['blocks_with_pic_tags']}\n")
            f.write(f"  无<PIC:xxx>的块数：{summary['blocks_without_pic_tags']}\n")
            f.write(f"  文档中出现的唯一标签数：{summary['total_unique_tags_found']}\n")
            f.write(f"  成功读取 OCR 次数：{summary['total_ocr_reads']}\n")
            f.write(f"  成功绑定图片次数：{summary['total_image_bindings']}\n\n")

    print(f"处理报告已保存到：{report_path}")


if __name__ == "__main__":
    main()
