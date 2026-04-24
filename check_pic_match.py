# -*- coding: utf-8 -*-
"""检查 PIC 标签和图片文件的匹配情况"""

import os
import json
import re

def check_pic_matching(txt_chunks_dir, image_vectors_dir):
    # 读取所有文本 chunks
    all_pic_labels = set()
    manual_to_pics = {}

    for filename in os.listdir(txt_chunks_dir):
        if filename.endswith('_chunks.json'):
            filepath = os.path.join(txt_chunks_dir, filename)
            manual_name = filename.replace('_chunks.json', '')

            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # 提取 PIC 标签
            manual_pics = set()
            for chunk in chunks:
                text = chunk.get('text', '')
                matches = re.findall(r'<PIC:([^>]+)>', text)
                for m in matches:
                    manual_pics.add(m)
                    all_pic_labels.add(m)

            manual_to_pics[manual_name] = manual_pics
            print(f"{manual_name}: {len(manual_pics)} 个 PIC 标签")

    # 获取所有图片向量文件
    image_files = set()
    for filename in os.listdir(image_vectors_dir):
        if filename.endswith('.json'):
            # 去掉 .json 后缀
            key = os.path.splitext(filename)[0]
            image_files.add(key)

    print(f"\n总共：{len(all_pic_labels)} 个唯一的 PIC 标签")
    print(f"图片向量文件：{len(image_files)} 个")

    # 检查匹配
    print("\n=== 匹配检查结果 ===")
    total_pic_labels = 0
    matched_labels = 0
    unmatched_labels = []

    for manual_name, pic_labels in manual_to_pics.items():
        print(f"\n{manual_name}:")
        matched = []
        unmatched = []

        for label in pic_labels:
            total_pic_labels += 1
            # 检查图片文件是否存在
            if label in image_files:
                matched.append(label)
                matched_labels += 1
            else:
                unmatched.append(label)
                unmatched_labels.append((manual_name, label))

        print(f"  匹配：{len(matched)} 个")
        if unmatched:
            print(f"  未匹配：{len(unmatched)} 个")
            if len(unmatched) <= 10:
                for u in unmatched:
                    print(f"    - {u}")

    print(f"\n=== 总结 ===")
    print(f"总 PIC 标签引用数：{total_pic_labels}")
    print(f"匹配成功：{matched_labels}")
    print(f"匹配失败：{total_pic_labels - matched_labels}")
    print(f"匹配率：{matched_labels/total_pic_labels*100:.1f}%" if total_pic_labels > 0 else "N/A")

    if unmatched_labels:
        print(f"\n未匹配的标签示例 (前 20 个):")
        for manual, label in unmatched_labels[:20]:
            print(f"  {manual}: {label}")

if __name__ == "__main__":
    txt_chunks_dir = r"D:\PyCharm\clean\Data\processed_chunks"
    image_vectors_dir = r"D:\PyCharm\clean\ocrtext_image_embeddings"

    check_pic_matching(txt_chunks_dir, image_vectors_dir)
