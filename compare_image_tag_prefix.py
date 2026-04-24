import os
import re
from collections import defaultdict


BASE_DIR = r"D:\PyCharm\clean\Data"
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
TAGS_FILE = os.path.join(BASE_DIR, "Manuals", "标签统计结果.txt")
OUT_FILE = os.path.join(BASE_DIR, "Manuals", "标签图片前缀比对结果.txt")


def to_prefix(name: str) -> str:
    m = re.match(r"^(.*)_\d+$", name)
    return m.group(1) if m else name


def load_images(images_dir: str):
    image_files = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            image_files.append(os.path.join(root, fn))

    prefixes = set()
    image_name_prefix_pairs = []
    for fp in image_files:
        image_name = os.path.basename(fp)
        stem = os.path.splitext(image_name)[0]
        prefix = to_prefix(stem)
        prefixes.add(prefix)
        image_name_prefix_pairs.append((image_name, prefix))
    return image_files, prefixes, image_name_prefix_pairs


def load_tags_from_report(report_path: str):
    file_to_tags = defaultdict(list)
    current_file = None

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("文件:"):
                current_file = line.split(":", 1)[1].strip()
            elif line.startswith("标签清单:") and current_file is not None:
                tag_str = line.split(":", 1)[1].strip()
                if tag_str and tag_str != "无":
                    file_to_tags[current_file].extend([t for t in tag_str.split("|") if t])
    return file_to_tags


def main():
    image_files, image_prefixes, image_name_prefix_pairs = load_images(IMAGES_DIR)
    file_to_tags = load_tags_from_report(TAGS_FILE)

    tag_mismatches = []
    total_tags = 0
    tag_prefixes = set()

    for manual_file, tags in file_to_tags.items():
        for tag in tags:
            total_tags += 1
            tag_prefix = to_prefix(tag)
            tag_prefixes.add(tag_prefix)
            if tag_prefix not in image_prefixes:
                tag_mismatches.append((manual_file, tag, tag_prefix))

    image_mismatches = [
        (image_name, image_prefix)
        for image_name, image_prefix in image_name_prefix_pairs
        if image_prefix not in tag_prefixes
    ]

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("标签前缀与图片名称前缀比对结果\n")
        f.write(f"图片总数: {len(image_files)}\n")
        f.write(f"标签总数: {total_tags}\n")
        f.write(f"标签前缀不匹配数: {len(tag_mismatches)}\n")
        f.write(f"图片名前缀不匹配数: {len(image_mismatches)}\n")
        f.write("-" * 80 + "\n")

        if tag_mismatches:
            f.write("【标签不匹配图片前缀】明细（文件 | 标签 | 标签前缀）:\n")
            for manual_file, tag, prefix in tag_mismatches:
                f.write(f"{manual_file} | {tag} | {prefix}\n")
        else:
            f.write("全部标签前缀均可在图片名称前缀中找到。\n")

        f.write("-" * 80 + "\n")
        if image_mismatches:
            f.write("【图片名前缀不匹配标签】明细（图片名 | 图片前缀）:\n")
            for image_name, image_prefix in image_mismatches:
                f.write(f"{image_name} | {image_prefix}\n")
        else:
            f.write("全部图片名前缀均可在标签前缀中找到。\n")

    print(f"图片总数: {len(image_files)}")
    print(f"标签总数: {total_tags}")
    print(f"标签前缀不匹配数: {len(tag_mismatches)}")
    print(f"图片名前缀不匹配数: {len(image_mismatches)}")
    print(f"结果已写入: {OUT_FILE}")


if __name__ == "__main__":
    main()
