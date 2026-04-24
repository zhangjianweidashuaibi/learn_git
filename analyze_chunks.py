import json
import os
from pathlib import Path

# 设定阈值
TOTAL_SIZE_LIMIT = 169984  # API最大限制
WARNING_THRESHOLD = 150000  # 警告阈值
ALERT_THRESHOLD = 100000    # 提醒阈值

chunk_results_dir = Path(r"D:\PyCharm\clean\chunk_results")

# 统计结果
results = {
    "files": {},
    "all_chunks": []
}

print(f"开始分析 chunk_results 文件夹中的JSON文件...")
print(f"API限制: {TOTAL_SIZE_LIMIT:,} 字符\n")

# 遍历所有JSON文件（排除checkpoint文件）
json_files = sorted([f for f in chunk_results_dir.glob("*.json") if "checkpoint" not in f.name])

for json_file in json_files:
    print(f"处理: {json_file.name}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # data 直接是一个 list
        chunks = data if isinstance(data, list) else data.get("chunks", [])

        file_result = {
            "chunks": [],
            "max_text_size": 0,
            "max_ocr_total_size": 0,
            "max_total_size": 0,
            "total_chunks": 0
        }

        for idx, chunk in enumerate(chunks):
            # 计算各字段大小
            text = chunk.get("text", "")
            text_size = len(text)

            ocr_texts = chunk.get("ocr_texts", [])
            # ocr_texts 是一个数组，计算每个OCR文本的大小和总大小
            ocr_total_size = 0
            ocr_sizes = []
            for ocr_array in ocr_texts:
                if isinstance(ocr_array, list):
                    for ocr_item in ocr_array:
                        if isinstance(ocr_item, str):
                            ocr_total_size += len(ocr_item)
                            ocr_sizes.append(len(ocr_item))
                elif isinstance(ocr_array, str):
                    ocr_total_size += len(ocr_array)
                    ocr_sizes.append(len(ocr_array))

            visual_descriptions = chunk.get("visual_descriptions", [])
            visual_size = sum(len(desc) for desc in visual_descriptions if isinstance(desc, str))

            total_size = text_size + ocr_total_size + visual_size

            chunk_info = {
                "index": idx,
                "id": chunk.get("id", ""),
                "text_size": text_size,
                "ocr_total_size": ocr_total_size,
                "ocr_count": len(ocr_sizes),
                "max_ocr_item_size": max(ocr_sizes) if ocr_sizes else 0,
                "visual_size": visual_size,
                "total_size": total_size,
                "exceeds_limit": total_size > TOTAL_SIZE_LIMIT
            }

            file_result["chunks"].append(chunk_info)
            file_result["total_chunks"] += 1

            if text_size > file_result["max_text_size"]:
                file_result["max_text_size"] = text_size

            if ocr_total_size > file_result["max_ocr_total_size"]:
                file_result["max_ocr_total_size"] = ocr_total_size

            if total_size > file_result["max_total_size"]:
                file_result["max_total_size"] = total_size

            results["all_chunks"].append({
                "file": json_file.name,
                "index": idx,
                "id": chunk.get("id", ""),
                "text_size": text_size,
                "ocr_total_size": ocr_total_size,
                "ocr_count": len(ocr_sizes),
                "max_ocr_item_size": max(ocr_sizes) if ocr_sizes else 0,
                "visual_size": visual_size,
                "total_size": total_size
            })

        results["files"][json_file.name] = file_result

    except Exception as e:
        print(f"  错误: {e}")

# 汇总分析
print("\n" + "="*80)
print("汇总分析结果")
print("="*80)

# 1. 找出超出API限制的分块
exceeding_chunks = [c for c in results["all_chunks"] if c["total_size"] > TOTAL_SIZE_LIMIT]
if exceeding_chunks:
    print(f"\n[警告] 超出API限制的分块 ({TOTAL_SIZE_LIMIT:,} 字符): {len(exceeding_chunks)} 个\n")
    for chunk in sorted(exceeding_chunks, key=lambda x: x["total_size"], reverse=True):
        print(f"  文件: {chunk['file']}")
        print(f"  分块ID: {chunk['id']}")
        print(f"  索引: {chunk['index']}")
        print(f"  总大小: {chunk['total_size']:,} 字符")
        print(f"    - 文本(text): {chunk['text_size']:,} 字符")
        print(f"    - OCR总计: {chunk['ocr_total_size']:,} 字符 (共{chunk['ocr_count']}项)")
        print(f"    - 最大OCR项: {chunk['max_ocr_item_size']:,} 字符")
        print(f"    - 视觉描述: {chunk['visual_size']:,} 字符")
        print(f"  超出限制: {chunk['total_size'] - TOTAL_SIZE_LIMIT:,} 字符\n")
else:
    print("\n[OK] 没有超出API限制的分块")

# 2. 找出接近限制的分块（警告）
warning_chunks = [c for c in results["all_chunks"]
                 if ALERT_THRESHOLD < c["total_size"] <= TOTAL_SIZE_LIMIT]
if warning_chunks:
    print(f"\n[警告] 接近API限制的分块 ({ALERT_THRESHOLD:,} - {TOTAL_SIZE_LIMIT:,} 字符): {len(warning_chunks)} 个\n")
    for chunk in sorted(warning_chunks, key=lambda x: x["total_size"], reverse=True)[:10]:
        print(f"  文件: {chunk['file']}")
        print(f"  分块ID: {chunk['id']}")
        print(f"  总大小: {chunk['total_size']:,} 字符")
        print(f"    - 文本: {chunk['text_size']:,}, OCR: {chunk['ocr_total_size']:,}\n")

# 3. OCR文本最大的分块
large_ocr_chunks = sorted([c for c in results["all_chunks"] if c["ocr_total_size"] > 0],
                         key=lambda x: x["ocr_total_size"], reverse=True)[:15]
if large_ocr_chunks:
    print(f"\n[OCR] OCR文本最大的15个分块:\n")
    for i, chunk in enumerate(large_ocr_chunks, 1):
        print(f"  {i}. 文件: {chunk['file']}")
        print(f"     分块ID: {chunk['id']}")
        print(f"     OCR总计: {chunk['ocr_total_size']:,} 字符 (共{chunk['ocr_count']}项)")
        print(f"     最大OCR项: {chunk['max_ocr_item_size']:,} 字符")
        print(f"     总大小: {chunk['total_size']:,} 字符\n")

# 4. 各文件统计摘要
print("\n" + "="*80)
print("各文件统计摘要")
print("="*80)

summary_data = []
for filename, file_data in sorted(results["files"].items()):
    status = "[OK]" if file_data['max_total_size'] < ALERT_THRESHOLD else "[WARN]"
    summary_data.append({
        "name": filename,
        "chunks": file_data['total_chunks'],
        "max_text": file_data['max_text_size'],
        "max_ocr": file_data['max_ocr_total_size'],
        "max_total": file_data['max_total_size'],
        "status": status
    })

# 按最大总大小排序显示
for data in sorted(summary_data, key=lambda x: x['max_total'], reverse=True):
    print(f"\n{data['status']} {data['name']}")
    print(f"   分块数: {data['chunks']}, 最大总大小: {data['max_total']:,}")
    print(f"   最大文本: {data['max_text']:,}, 最大OCR: {data['max_ocr']:,}")

print("\n" + "="*80)
print("分析完成!")
print("="*80)

# 输出详细报告到文件
output_file = chunk_results_dir / "size_analysis_report.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("分块大小分析详细报告\n")
    f.write("="*80 + "\n\n")

    f.write(f"API限制: {TOTAL_SIZE_LIMIT:,} 字符\n\n")

    # 超出限制的
    if exceeding_chunks:
        f.write("="*80 + "\n")
        f.write(f"[警告] 超出API限制的分块 ({len(exceeding_chunks)} 个)\n")
        f.write("="*80 + "\n\n")
        for chunk in sorted(exceeding_chunks, key=lambda x: x["total_size"], reverse=True):
            f.write(f"文件: {chunk['file']}\n")
            f.write(f"分块ID: {chunk['id']}\n")
            f.write(f"总大小: {chunk['total_size']:,} 字符\n")
            f.write(f"  - 文本: {chunk['text_size']:,}\n")
            f.write(f"  - OCR总计: {chunk['ocr_total_size']:,} (共{chunk['ocr_count']}项)\n")
            f.write(f"  - 最大OCR项: {chunk['max_ocr_item_size']:,}\n")
            f.write(f"  - 视觉描述: {chunk['visual_size']:,}\n\n")

    # 警告区间的
    if warning_chunks:
        f.write("="*80 + "\n")
        f.write(f"[警告] 接近API限制的分块 ({len(warning_chunks)} 个)\n")
        f.write("="*80 + "\n\n")
        for chunk in sorted(warning_chunks, key=lambda x: x["total_size"], reverse=True):
            f.write(f"{chunk['file']}: 分块{chunk['index']}, 总大小: {chunk['total_size']:,}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("各文件详细统计\n")
    f.write("="*80 + "\n\n")

    for filename, file_data in sorted(results["files"].items()):
        f.write(f"\n文件: {filename}\n")
        f.write(f"  分块总数: {file_data['total_chunks']}\n")
        f.write(f"  最大文本大小: {file_data['max_text_size']:,}\n")
        f.write(f"  最大OCR总大小: {file_data['max_ocr_total_size']:,}\n")
        f.write(f"  最大总大小: {file_data['max_total_size']:,}\n")

        # 列出该文件中超过50k的分块
        large_chunks_in_file = [c for c in file_data['chunks'] if c['total_size'] > 50000]
        if large_chunks_in_file:
            f.write(f"  较大分块 (>50k):\n")
            for c in sorted(large_chunks_in_file, key=lambda x: x['total_size'], reverse=True):
                f.write(f"    分块{c['index']} ({c['id']}): {c['total_size']:,} 字符\n")
                f.write(f"      文本: {c['text_size']:,}, OCR: {c['ocr_total_size']:,}\n")

print(f"\n详细报告已保存到: {output_file}")