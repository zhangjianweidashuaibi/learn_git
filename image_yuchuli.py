import os
import json
from PIL import Image

# 统一配置
target_size = (800, 600)
valid_img_dir = r"D:\PyCharm\clean\Data\Images"  # 有效图片文件夹
os.makedirs("./images_standard", exist_ok=True)

# 获取有效图片列表
valid_imgs = [f for f in os.listdir(valid_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

for img_name in valid_imgs:
    img_path = os.path.join(valid_img_dir, img_name)
    # 打开并缩放
    with Image.open(img_path) as img:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        # 转PNG（统一格式）
        new_name = img_name.split(".")[0] + ".png"
        img.save(os.path.join("./images_standard", new_name), "PNG")

print("图片标准化完成：全部转为800x600 PNG")