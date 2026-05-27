import os
import shutil
from PIL import Image

# 定义源文件夹路径
source_folder = ''
# 定义目标文件夹路径
target_masks_folder = ''
target_patches_folder =  ''

# 创建目标文件夹
if not os.path.exists(target_masks_folder):
    os.makedirs(target_masks_folder)

if not os.path.exists(target_patches_folder):
    os.makedirs(target_patches_folder)

# 遍历源文件夹中的子文件夹
for subfolder in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder)
    
    # 检查是否是文件夹
    if os.path.isdir(subfolder_path):
        # 源文件夹中的 masks 文件夹路径
        masks_folder_path = os.path.join(subfolder_path, "masks")
        
        # 复制 masks 文件夹中的所有图片到目标文件夹中
        if os.path.exists(masks_folder_path):
            for mask_file in os.listdir(masks_folder_path):
                mask_file_path = os.path.join(masks_folder_path, mask_file)
                # 目标文件夹中对应的 masks 图片路径
                target_mask_file_path = os.path.join(target_masks_folder, mask_file)
                
                # 打开图片
                mask_img = Image.open(mask_file_path)
                
                # 检查图片是否全黑
                is_all_black = all(mask_img.getpixel((x, y)) == (0, 0, 0) for x in range(mask_img.width) for y in range(mask_img.height))
                
                if not is_all_black:
                    shutil.copy(mask_file_path, target_mask_file_path)
        
        # 源文件夹中的 patches 文件夹路径
        patches_folder_path = os.path.join(subfolder_path, "patches")
        
        # 复制 patches 文件夹中的所有图片到目标文件夹中（如果对应的 mask 图片不是全黑）
        if os.path.exists(patches_folder_path):
            for patch_file in os.listdir(patches_folder_path):
                patch_file_path = os.path.join(patches_folder_path, patch_file)
                
                # 获取对应的 mask 文件路径
                mask_filename = os.path.splitext(patch_file)[0] + ".png"
                mask_path = os.path.join(masks_folder_path, mask_filename)
                
                # 检查对应的 mask 图片是否存在且不是全黑
                if os.path.exists(mask_path):
                    mask_img = Image.open(mask_path)
                    is_all_black = all(mask_img.getpixel((x, y)) == (0, 0, 0) for x in range(mask_img.width) for y in range(mask_img.height))
                    if not is_all_black:
                        # 目标文件夹中对应的 patches 图片路径
                        target_patch_file_path = os.path.join(target_patches_folder, patch_file)
                        shutil.copy(patch_file_path, target_patch_file_path)

print("Done")
