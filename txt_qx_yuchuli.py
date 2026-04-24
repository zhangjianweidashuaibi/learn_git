import os


def clean_vr_manual(content):
    """
    单个文本清洗逻辑（保留<PIC>，保留图片列表，规范排版）
    优化规则：
    1. <PIC>与前文保留在同一行
    2. 所有#开头的正文强制换行（无论是否紧跟<PIC>）
    3. 规范空行，保留图片列表
    """
    # 1. 去掉最外层的 [" 包裹
    if content.startswith('["'):
        content = content[2:]

    # 2. 把转义的 \n 变成真实换行
    content = content.replace('\\n', '\n')

    # 核心优化：处理所有#的换行逻辑（确保#正文单独成行）
    lines = content.split('\n')
    processed_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:  # 空行先保留（后续统一规范）
            processed_lines.append('')
            continue

        # 第一步：处理<PIC>与前文同行的基础逻辑
        if '<PIC>' in line_stripped:
            pic_index = line_stripped.find('<PIC>')
            pic_end_index = pic_index + len('<PIC>')
            front_part = line_stripped[:pic_end_index].rstrip()  # <PIC>及前文
            back_part = line_stripped[pic_end_index:].lstrip()   # <PIC>之后的内容
            current_line = front_part
            remaining_part = back_part
        else:
            current_line = ''
            remaining_part = line_stripped

        # 第二步：处理所有#的强制换行（核心优化点）
        if remaining_part:
            # 拆分剩余内容中所有#开头的片段
            parts = []
            temp = remaining_part
            while '#' in temp:
                hash_idx = temp.find('#')
                # #之前的内容（非空则加入）
                if hash_idx > 0:
                    parts.append(temp[:hash_idx].rstrip())
                # #及之后的内容（单独拆分）
                hash_part = temp[hash_idx:].lstrip()
                if hash_part:
                    parts.append(hash_part)
                # 继续处理剩余部分
                temp = temp[hash_idx + len(hash_part):]
            # 处理最后一段无#的内容
            if temp:
                parts.append(temp.rstrip())

            # 拼接当前行 + 拆分后的片段（#片段强制换行）
            if current_line:  # 有<PIC>前缀的情况
                if parts:
                    # 第一个片段跟<PIC>同行，其余#片段单独换行
                    current_line += ' ' + parts[0].lstrip()
                    processed_lines.append(current_line)
                    # 剩余#片段逐个换行
                    for part in parts[1:]:
                        if part:
                            processed_lines.append(part)
                else:
                    processed_lines.append(current_line)
            else:  # 无<PIC>的情况，所有#片段单独换行
                for part in parts:
                    if part:
                        processed_lines.append(part)
        else:
            # 无剩余内容，仅保留<PIC>行
            if current_line:
                processed_lines.append(current_line)

    # 合并后重新拆分（处理可能的空行）
    content = '\n'.join(processed_lines)

    # 3. 定位并保留末尾的图片编号列表（["Manual开头）
    pic_list_start = content.rfind('["Manual')
    if pic_list_start != -1:
        text_part = content[:pic_list_start].strip()
        list_part = content[pic_list_start:].strip()
        text_part = text_part.rstrip(']')  # 删掉文本末尾多余的 ]
        content = text_part + '\n\n' + list_part

    # 4. 规范空行（不连续空行）
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

    return '\n'.join(cleaned_lines)


def batch_clean_all_txt(input_dir, output_dir):
    """
    批量清洗指定文件夹下所有 TXT 文件
    :param input_dir: 原始txt文件所在目录
    :param output_dir: 清洗后文件保存目录
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 筛选输入目录下需要清洗的txt文件
    files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.txt') and not f.endswith('_已清洗.txt')
    ]

    if not files:
        print(f"❌ 在 {input_dir} 中未找到需要清洗的 txt 文件")
        return

    print(f"✅ 找到 {len(files)} 个 txt 文件，开始批量清洗...\n")

    for filename in files:
        try:
            # 拼接原始文件的完整路径
            input_file_path = os.path.join(input_dir, filename)
            # 读取原始文件
            with open(input_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 清洗
            cleaned_content = clean_vr_manual(content)

            # 生成新文件名 + 输出完整路径
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_已清洗{ext}"
            output_file_path = os.path.join(output_dir, output_filename)

            # 写入新文件
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f"✅ 已清洗：{filename} → {output_filename}")

        except Exception as e:
            print(f"❌ 清洗失败：{filename}，错误：{str(e)}")

    print(f"\n🎉 共 {len(files)} 个文件批量清洗完成！")
    print("✅ 原始文件完全保留（在输入目录）")
    print(f"✅ 新文件保存至：{output_dir}（文件名_已清洗.txt）")
    print("✅ 所有 <PIC> 和图片列表均已保留")
    print("✅ 所有 # 开头的正文已强制换行到新行")


# 一键运行
if __name__ == "__main__":
    # 输入你的原始文件路径（请确认该路径下有txt文件）
    INPUT_FILE = r"D:\PyCharm\clean\Data\Manuals"
    # 输出清洗后的文件路径
    OUTPUT_FILE = r"D:\PyCharm\clean\Data\clean_vr_Manuals"
    # 调用函数时传入指定的输入/输出目录
    batch_clean_all_txt(INPUT_FILE, OUTPUT_FILE)