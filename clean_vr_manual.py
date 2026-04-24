import os
import re


def clean_vr_manual(content, filename="未知文件"):
    """
    单个文本清洁逻辑（保留<PIC>，处理<PIC:xxxx>标签，规范排版）
    规则：
    1. <PIC>与前文保留在同一行
    2. 若<PIC>后紧跟#，则#及后文强制换行
    3. <PIC>按顺序用末尾的标签数组一一替换
    4. 标签多了忽略，标签少了则多出的<PIC>用最后一个标签重复
    
    返回：(cleaned_content, pic_count, replaced_count, repeated_tags_info)
    """
    try:
        # 1. 去掉最外层的 [" 包裹
        if content.startswith('["'):
            content = content[2:]

        # 2. 把转义的 \n 变成真实换行
        content = content.replace('\\n', '\n')

        # ========== 提取图片标签列表 ==========
        pic_list = []
        pic_list_start = -1
        
        # 查找所有可能的图片列表开头
        manual_start = content.rfind('["Manual')
        if manual_start != -1:
            pic_list_start = manual_start
        else:
            last_bracket = content.rfind('["')
            if last_bracket != -1:
                remaining_content = content[last_bracket:]
                if re.search(r'"\w+",', remaining_content):
                    pic_list_start = last_bracket

        if pic_list_start != -1:
            list_part = content[pic_list_start:].strip()
            pic_matches = re.findall(r'"(\w+)"', list_part)
            pic_list = [match for match in pic_matches if match]
            content = content[:pic_list_start].rstrip(']').strip()

        # ========== 统计并替换<PIC> ==========
        # 将所有<PIC:xxx>统一转换为<PIC>
        content = re.sub(r'<PIC:\w*>', '<PIC>', content)
        
        # 找到所有的<PIC>位置
        pic_markers = list(re.finditer(r'<PIC>', content))
        pic_count = len(pic_markers)
        
        repeated_tags_info = []  # 记录重复使用的标签信息
        
        # 将<PIC>按顺序替换为具体的图片标签
        if pic_list and pic_count > 0:
            tags_to_use = []
            last_tag_for_repeat = None
            
            if len(pic_list) >= pic_count:
                # 标签足够或多余，取前 pic_count 个
                tags_to_use = pic_list[:pic_count]
                ignored_count = len(pic_list) - pic_count
                if ignored_count > 0:
                    repeated_tags_info.append(f"忽略多余标签 {ignored_count} 个")
            else:
                # 标签不够，需要重复最后一个
                tags_to_use = pic_list[:]
                last_tag_for_repeat = pic_list[-1]
                repeat_needed = pic_count - len(pic_list)
                tags_to_use.extend([last_tag_for_repeat] * repeat_needed)
                repeated_tags_info.append(f"标签不足，最后 '{last_tag_for_repeat}' 被重复使用 {repeat_needed} 次")

            # 从前向后依次替换
            offset = 0
            replaced_count = 0
            for i, marker in enumerate(pic_markers):
                original_pos = marker.start()
                actual_pos = original_pos + offset
                tag_name = tags_to_use[i]
                new_tag = f"<PIC:{tag_name}>"
                
                content = content[:actual_pos] + new_tag + content[actual_pos+5:]
                offset += len(new_tag) - 5
                replaced_count += 1
        else:
            replaced_count = 0
            if not pic_list:
                repeated_tags_info.append("未找到图片标签列表")

        # ========== 处理<PIC:xxxx>后接#的换行逻辑 ==========
        lines = content.split('\n')
        processed_lines = []
        for line in lines:
            line_stripped = line.strip()
            if re.search(r'<PIC:\w+>', line_stripped):
                pic_match = re.search(r'<PIC:\w+>', line_stripped)
                pic_end = pic_match.end()
                front_part = line_stripped[:pic_end].rstrip()
                back_part = line_stripped[pic_end:].lstrip()

                if back_part.startswith('#'):
                    processed_lines.append(front_part)
                    processed_lines.append(back_part)
                else:
                    processed_lines.append(line_stripped)
            else:
                processed_lines.append(line_stripped)
        content = '\n'.join(processed_lines)

        # ========== 规范空行 ==========
        lines = content.split('\n')
        cleaned_lines = []
        last_empty = False

        for line in lines:
            line = line.strip()
            if not line:
                if not last_empty:
                    cleaned_lines.append('')
                    last_empty = True
                continue
            cleaned_lines.append(line)
            last_empty = False

        return '\n'.join(cleaned_lines), pic_count, replaced_count, repeated_tags_info

    except Exception as e:
        print(f"{filename} 清洗过程中出现错误：{str(e)}")
        import traceback
        traceback.print_exc()
        return content, 0, 0, [f"错误：{str(e)}"]


def batch_clean_all_txt(input_dir, output_dir):
    """
    批量清洗指定文件夹下所有 TXT 文件
    """
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.txt') and not f.endswith('_已清洁.txt')
    ]

    if not files:
        print(f"❌ 在 {input_dir} 中未找到需要清洗的 txt 文件")
        return

    print(f"✓ 找到 {len(files)} 个 txt 文件，开始批量清洗...\n")
    print("=" * 70)

    total_pics = 0
    total_replaced = 0
    all_repeated_info = {}

    for idx, filename in enumerate(sorted(files), 1):
        try:
            input_file_path = os.path.join(input_dir, filename)
            with open(input_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            cleaned_content, pic_count, replaced_count, repeated_info = clean_vr_manual(content, filename)

            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_已清洁{ext}"
            output_file_path = os.path.join(output_dir, output_filename)

            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            total_pics += pic_count
            total_replaced += replaced_count
            
            # 记录有重复信息的文件
            if repeated_info and "忽略" in str(repeated_info) or "重复" in str(repeated_info):
                all_repeated_info[filename] = repeated_info

            # 每个文件的详细输出
            print(f"[{idx}/{len(files)}] {filename}")
            print(f"       文档内<PIC>数量：{pic_count}")
            print(f"       成功替换为<PIC:标签>: {replaced_count}")
            if repeated_info:
                for info in repeated_info:
                    if "重复" in info or "忽略" in info:
                        print(f"       ⚠️  {info}")
            print()

        except Exception as e:
            print(f"[{idx}/{len(files)}] ❌ 清洗失败：{filename}，错误：{str(e)}\n")

    print("=" * 70)
    print(f"\n🎉 共 {len(files)} 个文件批量清洗完成！")
    print(f"\n📊 统计摘要:")
    print(f"   总<PIC>数量：     {total_pics}")
    print(f"   总替换数量：      {total_replaced}")
    
    if all_repeated_info:
        print(f"\n⚠️  标签重复/忽略情况:")
        for fname, info_list in all_repeated_info.items():
            for info in info_list:
                if "重复" in info or "忽略" in info:
                    print(f"   • {fname}: {info}")
    
    print(f"\n✓ 原始文件完全保留（在输入目录）")
    print(f"✓ 新文件保存至：{output_dir}（文件名_已清洁.txt）")


if __name__ == "__main__":
    INPUT_FILE = r"D:\PyCharm\clean\Data\Manuals"
    OUTPUT_FILE = r"D:\PyCharm\clean\Data\clean_vr_Manuals"
    batch_clean_all_txt(INPUT_FILE, OUTPUT_FILE)
