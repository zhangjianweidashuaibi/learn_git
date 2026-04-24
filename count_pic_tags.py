#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计clean_vr_Manuals目录下21个txt文档中<PIC:xxx>标签的使用情况
"""

import os
import re
from collections import defaultdict

def count_pic_tags(directory):
    # 存储所有标签及其出现次数
    tag_counts = defaultdict(int)
    # 存储每个标签出现在哪些文件的哪些行
    tag_locations = defaultdict(list)
    # 存储每个文件的标签统计
    file_stats = {}

    # 遍历目录下的所有txt文件
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(directory, filename)
        file_tag_count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                # 匹配<PIC:xxx>标签
                matches = re.findall(r'<PIC:[^>]+>', line)
                for tag in matches:
                    tag_counts[tag] += 1
                    tag_locations[tag].append(f"{filename}:{line_num}")
                    file_tag_count += 1

        file_stats[filename] = file_tag_count

    return tag_counts, tag_locations, file_stats

def main():
    directory = r'D:\PyCharm\clean\Data\clean_vr_Manuals'

    tag_counts, tag_locations, file_stats = count_pic_tags(directory)

    print("=" * 80)
    print(f"{'统计结果':^80}")
    print("=" * 80)
    print(f"总文件数: {len(file_stats)}")
    print(f"总标签数: {sum(file_stats.values())}")
    print(f"唯一标签数: {len(tag_counts)}")
    print()

    # 按文件统计
    print("=" * 80)
    print("各文件标签数量:")
    print("=" * 80)
    for filename, count in sorted(file_stats.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {filename}")
    print()

    # 重复标签（出现次数 > 1）
    print("=" * 80)
    print("重复标签统计:")
    print("=" * 80)
    repeated_tags = {k: v for k, v in tag_counts.items() if v > 1}

    if repeated_tags:
        for tag, count in sorted(repeated_tags.items(), key=lambda x: -x[1]):
            print(f"\n  标签: {tag}")
            print(f"  出现次数: {count}")
            print("  位置:")
            for location in tag_locations[tag]:
                print(f"    - {location}")
    else:
        print("  没有找到重复标签")

    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
