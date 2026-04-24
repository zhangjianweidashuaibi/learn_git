import os
import re


def clean_single_submanual(content, sub_index, global_stats):
    """
    处理单个子手册的内容：将<PIC>按顺序替换为<PIC:标签>
    规则：
    - 如果<PIC>数量 <= 标签数量：按顺序一一对应
    - 如果<PIC>数量 > 标签数量：多余的<PIC>用最后一个标签重复替换

    Args:
        content: 子手册正文内容（不包含图像列表）
        sub_index: 子手册序号（用于调试输出）
        global_stats: 全局统计字典，用于记录重复使用情况
    Returns:
        处理后的内容
    """
    try:
        # 1. 去掉最外层的 [" 包裹
        if content.startswith('["'):
            content = content[2:]

        # 2. 把转义的 \n 变成真实换行
        content = content.replace('\\n', '\n')

        # 3. 查找并提取图像列表（位于子手册末尾的 [...]] 格式）
        # 匹配类似 ["Manual10_0", "Manual10_1", ...] 的数组
        pic_list_pattern = r'\[\s*"([^"]+)"(?:,\s*"([^"]+)")*\]\s*]'
        pic_list_match = re.search(pic_list_pattern, content)

        pic_list = []
        if pic_list_match:
            # 提取所有标签
            list_content = pic_list_match.group(0)
            pic_list = re.findall(r'"([^"]+)"', list_content)
            # 移除图像列表部分
            content = content[:pic_list_match.start()].strip() + content[pic_list_match.end():].strip()
            # 清理可能的多余括号
            content = content.rstrip(']').strip()

        print(f"  子手册 {sub_index}: 找到 {len(pic_list)} 个图像标签")

        # 4. 统计<PIC>数量
        pic_count = content.count('<PIC>')
        print(f"  子手册 {sub_index}: 正文中有 {pic_count} 个<PIC>")

        # 统计本手册的标签使用情况
        tag_usage = {}

        # 5. 替换<PIC>为具体的标签
        if pic_list and pic_count > 0:
            # 找到所有<PIC>的位置
            pic_positions = []
            start = 0
            while True:
                pos = content.find('<PIC>', start)
                if pos == -1:
                    break
                pic_positions.append(pos)
                start = pos + 5

            # 从后往前替换，避免位置偏移问题
            for i, pos in enumerate(reversed(pic_positions)):
                actual_index = len(pic_positions) - 1 - i

                if actual_index < len(pic_list):
                    # 正常对应
                    pic_tag = f"<PIC:{pic_list[actual_index]}>"
                    used_tag = pic_list[actual_index]
                else:
                    # PIC 多于标签，使用最后一个标签
                    pic_tag = f"<PIC:{pic_list[-1]}>"
                    used_tag = pic_list[-1]

                content = content[:pos] + pic_tag + content[pos+5:]

                # 记录标签使用次数
                tag_usage[used_tag] = tag_usage.get(used_tag, 0) + 1

            # 如果标签比 PIC 多，忽略多余的标签（不需要额外处理）
        elif not pic_list and '<PIC>' in content:
            # 没有图像列表但有<PIC>，保留原样
            print(f"  子手册 {sub_index}: 警告 - 没有找到图像列表，保留<PIC>不变")

        # 收集重复使用的标签
        repeated_tags = {tag: count for tag, count in tag_usage.items() if count > 1}
        if repeated_tags:
            global_stats['repeated'].setdefault(sub_index, []).extend(
                [(tag, count) for tag, count in repeated_tags.items()]
            )

        return content
    except Exception as e:
        print(f"  子手册 {sub_index} 处理出错：{str(e)}")
        return content


def split_into_submanuals(content):
    """
    将整个文档按子手册拆分
    每个子手册以 ], 或 ]]\n 作为分隔符
    """
    submanuals = []

    # 使用正则表达式找到所有 ]], 的位置（这是子手册之间的分隔点）
    pattern = r'\]\](?=\s*\[")'

    matches = list(re.finditer(pattern, content))

    if matches:
        # 第一个子手册
        submanuals.append(content[:matches[0].end()])

        # 中间的子手册
        for i in range(len(matches) - 1):
            submanuals.append(content[matches[i].end():matches[i+1].end()])

        # 最后一个子手册
        submanuals.append(content[matches[-1].end():])
    else:
        # 如果没有找到分隔符，整个文档作为一个子手册
        submanuals.append(content)

    return submanuals


def clean_merged_manual(content):
    """
    处理合并的多个子手册文档
    步骤：
    1. 拆分为多个子手册
    2. 对每个子手册单独处理<PIC>替换
    3. 合并结果
    """
    try:
        # 统计信息
        stats = {
            'total_pics': 0,       # 总 PIC 数
            'replaced_count': 0,   # 替换的 PIC 数
            'repeated': {}         # 重复使用的标签 {子手册序号：[(标签，次数), ...]}
        }

        # 统计原始<PIC>总数
        stats['total_pics'] = content.count('<PIC>')
        print(f"文档中总共有 {stats['total_pics']} 个<PIC>")

        # 1. 首先尝试去掉开头的 ["
        if content.startswith('["'):
            content = content[2:]

        # 2. 拆分为子手册
        submanuals = split_into_submanuals(content)
        print(f"检测到 {len(submanuals)} 个子手册\n")

        # 3. 分别处理每个子手册
        processed_submanuals = []
        for i, sub in enumerate(submanuals, 1):
            print(f"\n处理第 {i} 个子手册...")
            processed = clean_single_submanual(sub, i, stats)
            processed_submanuals.append(processed)

        # 4. 合并所有子手册
        result = ''.join(processed_submanuals)

        # 5. 规范空行（不连续空行）
        lines = result.split('\n')
        cleaned_lines = []
        last_empty = False

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if not last_empty:
                    cleaned_lines.append('')
                    last_empty = True
                continue
            cleaned_lines.append(line_stripped)
            last_empty = False

        result = '\n'.join(cleaned_lines)

        # 统计替换后的数量
        stats['replaced_count'] = result.count('<PIC:')

        # 打印统计信息
        print("\n" + "=" * 60)
        print("统计信息")
        print("=" * 60)
        print(f"  总<PIC>数量：{stats['total_pics']}")
        print(f"  已替换为<PIC:标签>数量：{stats['replaced_count']}")

        # 计算未替换的<PIC>（即单纯的<PIC>而非<PIC:xxx>）
        remaining_plain = stats['total_pics'] - stats['replaced_count']
        print(f"  未替换的<PIC>数量：{remaining_plain}")

        # 打印重复使用的标签
        if stats['repeated']:
            print("\n  重复使用的标签 (因为<PIC>多于标签，使用最后一个标签填充):")
            all_repeated = {}
            for sub_idx, repeated_list in stats['repeated'].items():
                for tag, count in repeated_list:
                    if tag not in all_repeated:
                        all_repeated[tag] = {'count': 0, 'submanuals': set()}
                    all_repeated[tag]['count'] += count
                    all_repeated[tag]['submanuals'].add(sub_idx)

            # 按重复次数排序输出
            sorted_repeated = sorted(all_repeated.items(), key=lambda x: x[1]['count'], reverse=True)
            for tag, info in sorted_repeated[:20]:  # 只显示前 20 个
                print(f"    - {tag}: 在子手册{list(info['submanuals'])}中使用了 {info['count']} 次")
            if len(sorted_repeated) > 20:
                print(f"    ... 还有 {len(sorted_repeated) - 20} 个重复标签")
        else:
            print("  没有重复使用的标签")

        print("=" * 60)

        return result
    except Exception as e:
        print(f"处理汇总文档出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return content


def process_merged_manual_file(input_file, output_file):
    """
    处理汇总英文手册文件
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"[ERROR] 输入文件不存在：{input_file}")
        return

    # 读取文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[OK] 成功读取文件：{input_file}")
        print(f"  文件大小：{len(content)} 字符\n")
    except Exception as e:
        print(f"[ERROR] 读取文件失败：{str(e)}")
        return

    # 处理内容
    cleaned_content = clean_merged_manual(content)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f"\n[OK] 处理完成！")
        print(f"  输出文件：{output_file}")
    except Exception as e:
        print(f"[ERROR] 写入文件失败：{str(e)}")


# 一键运行
if __name__ == "__main__":
    # 输入文件路径
    INPUT_FILE = r"D:\PyCharm\clean\Data\Manuals\汇总英文手册.txt"
    # 输出文件路径
    OUTPUT_FILE = r"D:\PyCharm\clean\Data\clean_vr_Manuals\汇总英文手册_已清洁.txt"

    print("=" * 60)
    print("开始处理汇总英文手册")
    print("=" * 60)

    process_merged_manual_file(INPUT_FILE, OUTPUT_FILE)
