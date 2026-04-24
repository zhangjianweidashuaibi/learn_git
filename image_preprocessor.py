"""
图像预处理模块 - 专门针对技术文档和手册的OCR优化
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from typing import Union, Optional


class DocumentImagePreprocessor:
    """技术文档图像预处理器"""

    def __init__(self):
        pass

    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        完整的图像预处理流程

        Args:
            image_path: 输入图像路径
            output_path: 可选的输出路径，用于保存处理后的图像

        Returns:
            处理后的图像数组
        """
        try:
            # 使用PIL读取图像（更好的格式兼容性）
            pil_img = Image.open(image_path)

            # 转换为RGB模式（处理RGBA、灰度等格式）
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # 转换为numpy数组
            img_rgb = np.array(pil_img)

            # 1. 自动旋转校正
            img_corrected = self._deskew_image(img_rgb)

            # 2. 调整分辨率
            img_resized = self._resize_image(img_corrected, target_dpi=300)

            # 3. 去噪处理
            img_denoised = self._denoise_image(img_resized)

            # 4. 对比度增强
            img_enhanced = self._enhance_contrast(img_denoised)

            # 5. 二值化处理（针对文本区域）
            img_binary = self._adaptive_threshold(img_enhanced)

            # 6. 边缘锐化
            img_sharpened = self._sharpen_image(img_binary)

            if output_path:
                # 保存处理后的图像
                processed_pil = Image.fromarray(img_sharpened)
                processed_pil.save(output_path)

            return img_sharpened

        except Exception as e:
            print(f"预处理图像 {image_path} 时出错: {e}")
            # 如果预处理失败，返回原始图像
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            return np.array(pil_img)

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """自动旋转校正"""
        try:
            # 转换为灰度图
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # 霍夫变换检测直线
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

            if lines is not None and len(lines) > 0:
                angles = []
                for i in range(min(10, len(lines))):  # 只取前10条线
                    rho, theta = lines[i][0]
                    angle = theta * 180 / np.pi
                    if angle < 45 or angle > 135:
                        angles.append(angle)

                if angles:
                    median_angle = np.median(angles)
                    if median_angle > 90:
                        median_angle -= 180

                    # 旋转图像
                    center = (img.shape[1] // 2, img.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        except Exception as e:
            print(f"旋转校正失败: {e}")

        return img

    def _resize_image(self, img: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """调整图像分辨率到目标DPI"""
        try:
            # 假设原始图像是72 DPI，调整到300 DPI
            scale_factor = target_dpi / 72.0
            if scale_factor != 1.0:
                new_width = int(img.shape[1] * scale_factor)
                new_height = int(img.shape[0] * scale_factor)
                # 添加尺寸限制，避免内存溢出
                if new_width > 4000 or new_height > 4000:
                    scale_factor = min(4000/img.shape[1], 4000/img.shape[0])
                    new_width = int(img.shape[1] * scale_factor)
                    new_height = int(img.shape[0] * scale_factor)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"调整分辨率失败: {e}")
        return img

    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """去噪处理"""
        try:
            # 使用非局部均值去噪
            if len(img.shape) == 3:
                # 彩色图像
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            else:
                # 灰度图像
                denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            return denoised
        except Exception as e:
            print(f"去噪失败: {e}")
            return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """对比度增强"""
        try:
            # 转换为PIL图像进行对比度调整
            pil_img = Image.fromarray(img)

            # 自动对比度调整
            enhancer = ImageEnhance.Contrast(pil_img)
            img_contrast = enhancer.enhance(1.5)  # 增强50%对比度

            # 亮度调整
            enhancer = ImageEnhance.Brightness(img_contrast)
            img_bright = enhancer.enhance(1.1)  # 稍微增加亮度

            return np.array(img_bright)
        except Exception as e:
            print(f"对比度增强失败: {e}")
            return img

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """自适应阈值二值化"""
        try:
            # 转换为灰度图
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # 自适应阈值
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # 转回彩色（如果需要）
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"二值化失败: {e}")
            return img

    def _sharpen_image(self, img: np.ndarray) -> np.ndarray:
        """边缘锐化"""
        try:
            # 定义锐化核
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

            sharpened = cv2.filter2D(img, -1, kernel)
            return sharpened
        except Exception as e:
            print(f"锐化失败: {e}")
            return img


def batch_preprocess_images(input_dir: str, output_dir: str):
    """批量预处理图像"""
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = DocumentImagePreprocessor()

    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"preprocessed_{filename}")

            try:
                preprocessor.preprocess_image(input_path, output_path)
                print(f"处理完成: {filename}")
            except Exception as e:
                print(f"处理失败 {filename}: {e}")


# 简单测试函数
def test_preprocessing():
    """测试图像预处理"""
    preprocessor = DocumentImagePreprocessor()

    # 测试单张图片（你需要替换为实际的图片路径）
    test_image_path = "./images_standard/Manual16_12.png"
    if os.path.exists(test_image_path):
        try:
            processed_img = preprocessor.preprocess_image(test_image_path, "test_processed.png")
            print("预处理测试成功！")
        except Exception as e:
            print(f"预处理测试失败: {e}")
    else:
        print("测试图片不存在，请修改路径")


if __name__ == "__main__":
    test_preprocessing()