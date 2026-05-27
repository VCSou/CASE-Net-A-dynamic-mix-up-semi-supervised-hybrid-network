# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from multiprocessing import Pool, freeze_support

# 定义输入路径和输出路径
input_dir = ''
resized_output_dir = ''

# 创建输出路径
if not os.path.exists(resized_output_dir):
    os.makedirs(resized_output_dir)

# 颜色映射字典
color_map = {
    (0, 0, 0): 0,        # 黑色
    (255, 0, 0): 1,      # 蓝色,LOW
    (0, 255, 0): 2,      # 绿色,HIGH
    (0, 0, 255): 3,      # 红色,MU
}

# 处理单张图片的函数
def process_image(png_file):
    # 构建输入和输出文件的完整路径
    input_path = os.path.join(input_dir, png_file)
    resized_output_path = os.path.join(resized_output_dir, png_file)
    
    # 检查输出文件夹中是否已经存在同名文件
    if os.path.exists(resized_output_path):
        print(f"跳过：{resized_output_path} 已存在。")
        return
    
    # 读取图片
    img = cv2.imread(input_path)
    
    # 检查是否成功读取图片
    if img is None:
        print(f"Failed to read image: {input_path}")
        return
    
    # 缩放图片
    resized_image = cv2.resize(img, (224 224), interpolation=cv2.INTER_NEAREST)
    
    # 初始化空白的灰度图，所有像素赋值为0
    gray_img = np.zeros((resized_image.shape[0], resized_image.shape[1]), dtype=np.uint8)
    
    # 根据颜色映射字典进行赋值
    for i in range(resized_image.shape[0]):
        for j in range(resized_image.shape[1]):
            pixel_color = tuple(resized_image[i, j])
            if pixel_color in color_map:
                gray_img[i, j] = color_map[pixel_color]
    
    # 保存缩放后的灰度图到输出文件夹
    cv2.imwrite(resized_output_path, gray_img)
    print(f"处理并保存：{resized_output_path}")

if __name__ == '__main__':
    freeze_support()
    
    # 获取输入路径下所有png格式的文件
    files = os.listdir(input_dir)
    png_files = [f for f in files if f.endswith('.png')]
    
    # 使用多进程处理图片
    with Pool() as pool:
        pool.map(process_image, png_files)

    print("图片处理完成！")
