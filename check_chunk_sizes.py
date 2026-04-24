"""
检查分块大小，找出异常大的分块
"""
import json
from pathlib import Path
from typing import List, Dict

def check_chunk_sizes(folder_path: str) -> List[Dict]:
    """检查所有分块的大小"""
    folder = Path(folder_path)
    results = []

    for json_file in folder.glob("*.json"):
        print(f"\n检查文件: {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        for idx, chunk in enumerate(chunks):
            # 计算各部分的长度
            text = chunk.get("text", "")
            ocr_texts = chunk.get("ocr_texts", [])
            visual_descriptions = chunk.get("visual_descriptions", [])

            text_len = len(text) if text else 0
            ocr_len = sum(len(" ".join(ocr)) for ocr in ocr_texts if ocr)
            visual_len = sum(len(v) for v in visual_descriptions)

            # 构建增强文本（这是实际会传给embedding的内容）
            enhanced_text = text
            if ocr_texts and any(ocr_texts):
                ocr_content = " ".join([" ".join(ocr) for ocr in ocr_texts])
                enhanced_text = f"{enhanced_text}\n\nOCR内容: {ocr_content}"

            if visual_descriptions:
                visual_content = "\n".join(visual_descriptions)
                enhanced_text = f"{enhanced_text}\n\n图片描述: {visual_content}"

            total_len = len(enhanced_text)

            # 粗略估算token数（英文约1 token=4字符，中文约1 token=1.5字符）
            # 使用保守估算
            token_estimate = len(enhanced_text)  # 最坏情况

            chunk_info = {
                "file": json_file.name,
                "chunk_id": chunk.get("id", f"chunk_{idx}"),
                "chunk_index": idx,
                "text_len": text_len,
                "ocr_len": ocr_len,
                "visual_len": visual_len,
                "total_len": total_len,
                "token_estimate": token_estimate,
                "has_images": len(chunk.get("image_paths", [])) > 0,
                "image_count": len(chunk.get("image_paths", []))
            }

            results.append(chunk_info)

            # 如果超过阈值，立即报告
            if total_len > 100000:  # 100K字符肯定有问题
                print(f"  ⚠️  警告: 分块 {idx} 特别大!")
                print(f"     文本长度: {text_len}")
                print(f"     OCR长度: {ocr_len}")
                print(f"     视觉描述: {visual_len}")
                print(f"     总长度: {total_len}")
                print(f"     估算token: {token_estimate}")
            elif total_len > 50000:
                print(f"  ⚠️  注意: 分块 {idx} 较大 ({total_len} 字符)")
            elif idx % 10 == 0:
                print(f"  已检查 {idx+1}/{len(chunks)} 个分块...")

    return results

def analyze_results(results: List[Dict]):
    """分析结果"""
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)

    # 按总长度排序
    sorted_results = sorted(results, key=lambda x: x["total_len"], reverse=True)

    # 找出前20个最大的
    print("\n【最大的20个分块】")
    print(f"{'排名':<5} {'文件':<25} {'分块索引':<10} {'总长度':<12} {'估算Token':<12}")
    print("-"*80)

    for i, r in enumerate(sorted_results[:20], 1):
        print(f"{i:<5} {r['file'][:25]:<25} {r['chunk_index']:<10} {r['total_len']:<12,} {r['token_estimate']:<12,}")

    # 统计各阈值范围内的数量
    print("\n【按大小统计】")
    ranges = [
        (100000, ">100K (严重问题)"),
        (50000, "50K-100K (较大)"),
        (10000, "10K-50K (正常)"),
        (0, "<10K (小)")
    ]

    for threshold, label in ranges:
        if threshold == 0:
            count = sum(1 for r in results if r["total_len"] < 10000)
        else:
            count = sum(1 for r in results if r["total_len"] >= threshold)
        print(f"  {label}: {count} 个分块")

    # 找出可能超过API限制的
    over_limit = [r for r in results if r["token_estimate"] > 169984]
    if over_limit:
        print(f"\n⚠️  发现 {len(over_limit)} 个分块可能超过API限制!")
        print("\n详细列表:")
        for r in over_limit:
            print(f"  - {r['file']}, 分块 {r['chunk_index']}: {r['total_len']:,} 字符")

    return sorted_results

if __name__ == "__main__":
    folder = "d:/PyCharm/clean/chunk_results"

    print("开始检查分块大小...")
    print(f"文件夹: {folder}\n")

    results = check_chunk_sizes(folder)
    analyze_results(results)