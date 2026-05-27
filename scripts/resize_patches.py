# -*- coding: utf-8 -*-

import os
import cv2

# 输入和输出文件夹路径
input_folder = ''
output_folder = ''

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹下的所有图片文件
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 遍历每张图片并进行缩放
for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    
    # 读取图片
    image = cv2.imread(input_path)
    
    # 检查是否成功读取图片
    if image is None:
        print(f"Failed to read image: {input_path}")
        continue
    
    # 缩放图片
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # 保存缩放后的图片到输出文件夹
    cv2.imwrite(output_path, resized_image)
    
print("All images resized and saved successfully.")
