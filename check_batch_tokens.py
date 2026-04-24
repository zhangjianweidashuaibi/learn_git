"""
检查批量操作时的token累积情况
"""
import json
from pathlib import Path

def check_batch_token_accumulation(folder_path: str, batch_size: int = 50):
    """检查批量处理时的token累积"""
    folder = Path(folder_path)
    total_chunks = []
    batches_with_issues = []

    # 收集所有分块
    for json_file in folder.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            for chunk in chunks:
                # 计算实际会传给embedding的文本
                text = chunk.get("text", "")
                ocr_texts = chunk.get("ocr_texts", [])
                visual_descriptions = chunk.get("visual_descriptions", [])

                enhanced_text = text
                if ocr_texts and any(ocr_texts):
                    ocr_content = " ".join([" ".join(ocr) for ocr in ocr_texts])
                    enhanced_text = f"{enhanced_text}\n\nOCR内容: {ocr_content}"

                if visual_descriptions:
                    visual_content = "\n".join(visual_descriptions)
                    enhanced_text = f"{enhanced_text}\n\n图片描述: {visual_content}"

                total_chunks.append({
                    "file": json_file.name,
                    "id": chunk.get("id", ""),
                    "text_len": len(enhanced_text),
                    "enhanced_text": enhanced_text[:500]  # 只保存前500字符用于调试
                })

    # 按批次检查
    print(f"总共有 {len(total_chunks)} 个分块")
    print(f"批次大小: {batch_size}\n")

    for i in range(0, len(total_chunks), batch_size):
        batch = total_chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        # 计算该批次的总字符数
        total_chars = sum(c["text_len"] for c in batch)

        # 估算token（粗略：字符数）
        total_tokens = total_chars

        if total_tokens > 169984:
            batches_with_issues.append({
                "batch_num": batch_num,
                "start_idx": i,
                "end_idx": min(i + batch_size, len(total_chunks)),
                "total_chars": total_chars,
                "total_tokens": total_tokens,
                "chunk_count": len(batch)
            })

            print(f"⚠️  批次 {batch_num} (分块 {i}-{min(i + batch_size, len(total_chunks))}): {total_chars:,} 字符")
            print(f"   这批次的分块：")
            for c in batch:
                print(f"     - {c['file']}: {c['text_len']} 字符")
                if c['text_len'] > 10000:
                    print(f"       内容预览: {c['enhanced_text']}...")
            print()
        elif total_tokens > 150000:
            print(f"⚠️  批次 {batch_num} 接近限制: {total_chars:,} 字符\n")

    # 检查单个超大分块
    print("\n" + "="*80)
    print("检查单个超大分块（超过100K字符）")
    print("="*80)

    huge_chunks = [c for c in total_chunks if c["text_len"] > 100000]
    if huge_chunks:
        print(f"发现 {len(huge_chunks)} 个超大分块:")
        for c in huge_chunks[:10]:  # 只显示前10个
            print(f"  - {c['file']}: {c['text_len']:,} 字符")
    else:
        print("没有发现单个超大分块")

    # 检查可能超过100K的分块
    print("\n" + "="*80)
    print("检查大分块（超过50K字符）")
    print("="*80)

    large_chunks = [c for c in total_chunks if c["text_len"] > 50000]
    if large_chunks:
        print(f"发现 {len(large_chunks)} 个大分块:")
        for c in large_chunks[:10]:
            print(f"  - {c['file']}: {c['text_len']:,} 字符")
    else:
        print("没有发现大分块")

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)

    if batches_with_issues:
        print(f"⚠️  发现 {len(batches_with_issues)} 个批次超过API限制!")
        print("建议：")
        print("  1. 减小批次大小（例如从50改为10或20）")
        print("  2. 检查是否有异常大的分块")
        print("  3. 考虑对超大的批次进行拆分处理")
    else:
        print("✓ 所有批次都在安全范围内")
        print(f"最大批次字符数: {max(sum(c['text_len'] for c in total_chunks[i:i+batch_size]) for i in range(0, len(total_chunks), batch_size)):,}")

if __name__ == "__main__":
    check_batch_token_accumulation(
        folder_path="d:/PyCharm/clean/chunk_results",
        batch_size=50
    )