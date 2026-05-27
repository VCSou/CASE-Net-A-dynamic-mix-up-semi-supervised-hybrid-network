import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
from functools import partial
import multiprocessing
import matplotlib.colors as mcolors
import sdpc
import json
from PIL import Image
import sys
import matplotlib.pyplot as plt
import math
from shapely.affinity import translate
from shapely.geometry import Polygon,MultiPolygon
# Mask generation by OTSU algorithm
def get_depth(lst):
    """
    Returns the maximum depth of a nested list.
    """
    if isinstance(lst, list):
        return 1 + max(get_depth(item) for item in lst)
    else:
        return 0
def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return (image_open / 255.0).astype(np.uint8)

parser = argparse.ArgumentParser(description='Code to patch WSI using .sdpc files')
parser.add_argument('--json_dir',type=str,default='',help='path of .json files')
parser.add_argument('--save_dir', type=str, default='', help='path to store processed tiles')
parser.add_argument('--tile_size', type=int, default=2048, help='size for processed tiles')
parser.add_argument('--thumbnail_level', type=int, default=4, help='thumbnail level')
parser.add_argument('--patch_level', type=int, default=2, help='level for cutting patches')
parser.add_argument('--overlap_size', type=int, default=0, help='overlap size for processed tiles')
parser.add_argument('--blank_TH', type=float, default=0.8, help='cut patches with blank rate lower than blank_TH')
args = parser.parse_args()

thumbnail_level = args.thumbnail_level
json_dir = args.json_dir
save_dir = args.save_dir
tile_size = args.tile_size
patch_level = args.patch_level
overlap_size = args.overlap_size
blank_TH = args.blank_TH




def generate_patch_mask(o_cut_x,o_cut_y,json_file,origin_demensions,wsi_name,zoom_value):
   downsample_factor =  math.pow(zoom_value,patch_level)
   total_x,total_y = origin_demensions
   total_x = total_x/downsample_factor
   total_y = total_y/downsample_factor
   cut_x = o_cut_x/downsample_factor
   cut_y = o_cut_y/downsample_factor
   num_instances = len(json_file["features"])
   # 维护一个pylygon的列表
   polygon_list = []
   class_list = []
   class_set = {'LOW','HIGH','MU','AD','adnoma'}
   for idx in range(num_instances):
       if json_file["features"][idx]['properties']['classification']['name'] not in class_set:
           continue

           
       polygon_idx_coordinates = json_file["features"][idx]["geometry"]["coordinates"]
       if json_file["features"][idx]['geometry']['type'] != 'MultiPolygon':
          polygon_idx_coordinates = polygon_idx_coordinates[0]
          try:
            polygon_idx_coordinates = [[x/downsample_factor, y/downsample_factor] for x, y in polygon_idx_coordinates]
          except:
              print(f"出现错误！！！{idx}")
              sys.exit()
              
          polygon_idx_coordinates = [tuple(item) for  item in polygon_idx_coordinates]
          polygon_idx = Polygon(polygon_idx_coordinates)
          polygon_list.append(polygon_idx)
          class_list.append(json_file["features"][idx]['properties']['classification']['name'])
       else:
            for polygon_idx_coordinates in polygon_idx_coordinates:
                polygon_idx_coordinates = polygon_idx_coordinates[0]
                polygon_idx_coordinates = [[x/downsample_factor, y/downsample_factor] for x, y in polygon_idx_coordinates]
                polygon_idx_coordinates = [tuple(item) for  item in polygon_idx_coordinates]
                polygon_idx = Polygon(polygon_idx_coordinates)
                polygon_list.append(polygon_idx)
                class_list.append(json_file["features"][idx]['properties']['classification']['name'])
   # 对应的patch的polygon
   square = Polygon([(cut_x, cut_y), (cut_x + tile_size, cut_y), (cut_x + tile_size, cut_y + tile_size), (cut_x, cut_y + tile_size)])
   contain = False
   mask = np.zeros((tile_size, tile_size), dtype=np.int8)
   for idx,polygon in enumerate(polygon_list):
       class_name = class_list[idx]
       label = 0
       if class_name == 'LOW':
           label=1
       elif class_name == 'HIGH':
           label=2
       elif class_name == 'MU':
           label=3
       elif class_name == 'AD':
           label=4
       elif class_name == 'adnoma':
           label=4
       
       if polygon.intersection(square).area > 0:
           print("有交集！！！")
           contain = True
           offset_x = -cut_x
           offset_y = -cut_y
           translate_square = translate(square,offset_x,offset_y)
           translate_polygon = translate(polygon,offset_x,offset_y)
           intersection_polygon = translate_square.intersection(translate_polygon)
           if isinstance(intersection_polygon,Polygon):
               intersection_polygon_coords = list(intersection_polygon.exterior.coords)
               # intersection_polygon_coords = [[y, x] for x, y in intersection_polygon_coords]
               cv2.polylines(mask,np.int32([intersection_polygon_coords]),True,label)
               cv2.fillPoly(mask,np.int32([intersection_polygon_coords]),label)
           if isinstance(intersection_polygon,MultiPolygon):
               print("存在凹标注！！！！")
               for meta_polygon in intersection_polygon.geoms:
                   meta_polygon_coords = list(meta_polygon.exterior.coords)
                   cv2.polylines(mask,np.int32([meta_polygon_coords]),True,label)
                   cv2.fillPoly(mask,np.int32([meta_polygon_coords]),label)
   colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0)]
   cmap = np.array(colors, dtype=np.uint8)
   if contain == True:
       print("要输出了！！！")
       save_mask_path_png = f'{save_dir}/{wsi_name}/masks/{wsi_name}-patch_level-{str(patch_level)}_X-{o_cut_x}_Y-{o_cut_y}.png'
       Image.fromarray(cmap[mask]).save(save_mask_path_png)  
   if contain == False:
       save_mask_path_png = f'{save_dir}/{wsi_name}/masks/{wsi_name}-patch_level-{str(patch_level)}_X-{o_cut_x}_Y-{o_cut_y}.png'
       mask = np.zeros((tile_size, tile_size), dtype=np.int8)
       Image.fromarray(cmap[mask]).save(save_mask_path_png)  
   return contain




def main_process(slide_list):

    idx = 0
    for slide_path in tqdm(slide_list):
        if os.path.exists(os.path.join(save_dir,os.path.basename(slide_path).split('.')[0])):
            continue
        wsi = sdpc.Sdpc(slide_path)
        origin_dimensions = wsi.level_dimensions[0]
        # 创建wsi对应的json文件
        json_path = os.path.join(json_dir,os.path.basename(slide_path).split('.')[0]+".svs.json")
        with open(json_path,"r") as f:
            json_file = json.load(f)
        wsi_name = slide_path.split('/')[-1].split('.')[0]
        zoom_value = wsi.level_downsamples[1] / wsi.level_downsamples[0]
        # level_dimensions其实就是指在当前level下的width和height
        thumbnail = np.array(wsi.read_region((0, 0), thumbnail_level, wsi.level_dimensions[thumbnail_level]))

        # Obtain mask & generate mask image
        black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
        thumbnail[black_pixel] = [255, 255, 255]

        bg_mask = get_bg_mask(thumbnail, kernel_size=5)

        os.makedirs(f'{save_dir}/{wsi_name}/thumbnail', exist_ok=True)
        os.makedirs(f'{save_dir}/{wsi_name}/patches', exist_ok=True)
        os.makedirs(f'{save_dir}/{wsi_name}/masks', exist_ok=True)
        cv2.imwrite(f'{save_dir}/{wsi_name}/thumbnail/thumbnail.png', thumbnail)
        cv2.imwrite(f'{save_dir}/{wsi_name}/thumbnail/mask.png', bg_mask * 255)

        marked_img = thumbnail.copy()

        tile_x = int(tile_size / pow(zoom_value, thumbnail_level - patch_level))
        tile_y = int(tile_size / pow(zoom_value, thumbnail_level - patch_level))

        x_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))
        y_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))

        thumbnail_x, thumbnail_y = wsi.level_dimensions[thumbnail_level]

        total_num = int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1)) * \
                    int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))

        with tqdm(total=total_num, ncols=100) as pbar:
            for i in range(int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1))):
                for j in range(int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))):

                    start_x = int(np.floor(i * (tile_x - x_overlap) / thumbnail_x * bg_mask.shape[1]))
                    start_y = int(np.floor(j * (tile_y - y_overlap) / thumbnail_y * bg_mask.shape[0]))

                    end_x = int(np.ceil((i * (tile_x - x_overlap) + tile_x) / thumbnail_x * bg_mask.shape[1]))
                    end_y = int(np.ceil((j * (tile_y - y_overlap) + tile_y) / thumbnail_y * bg_mask.shape[0]))

                    mask = bg_mask[start_y:end_y, start_x:end_x]

                    if np.sum(mask == 0) / mask.size < blank_TH:
                        cv2.rectangle(marked_img, (end_x, end_y), (start_x, start_y), (255, 0, 0), 2)

                        cut_x = int(start_x * pow(zoom_value, thumbnail_level))  # Coordinate X of layer 0
                        cut_y = int(start_y * pow(zoom_value, thumbnail_level))  # Coordinate Y of layer 0

                        img = wsi.read_region((cut_x, cut_y), patch_level, (tile_size, tile_size))

                        save_img = f'{save_dir}/{wsi_name}/patches/{wsi_name}-patch_level-{str(patch_level)}_X-{cut_x}_Y-{cut_y}.png'


                        has_mask = generate_patch_mask(cut_x,cut_y,json_file,origin_dimensions,wsi_name,zoom_value)
                        cv2.imwrite(save_img, img)


                    pbar.update(1)

        cv2.imwrite(f'{args.save_dir}/{wsi_name}/thumbnail/thumbnail_marked.png', marked_img)
        idx += 1
        print(f'{idx} / {len(slide_list)} {((idx / len(slide_list)) * 100):.2f}%')

if __name__ == '__main__':

    slide_list = ['']
    main_process(slide_list)

