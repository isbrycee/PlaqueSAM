import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from collections import defaultdict


color_map = {
    0: (255, 0, 0), 
    1: (0, 255,0), 
    2: (0,0,255),
    3: (128,128,128)}

# def visualize_and_save_annotations(json_path, image_folder, output_folder):
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 加载 COCO 标注
#     coco = COCO(json_path)
#     # 获取所有类别信息
#     categories = coco.loadCats(coco.getCatIds())
#     category_id_to_name = {cat['id']: cat['name'] for cat in categories}

#     # 初始化类别计数器
#     category_counter = defaultdict(int)

#     # 获取所有图像 ID
#     image_ids = coco.getImgIds()

#     # 遍历每张图像
#     for img_id in image_ids:
#         # 加载图像信息
#         img_info = coco.loadImgs(img_id)[0]
#         image_path = os.path.join(image_folder, img_info['file_name'])
#         print(image_path)
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

#         # 获取该图像的标注
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)

#         # 创建一个空白图像用于绘制掩码
#         mask_image = np.zeros_like(image)

#         # 遍历每个标注
#         for ann in anns:
#             # 获取类别 ID 并更新计数器
#             category_id = ann['category_id']
#             category_name = category_id_to_name[category_id]
#             category_counter[category_name] += 1

#         #     # 获取分割掩码
#             if 'segmentation' in ann:
#                 rle = coco.annToRLE(ann)
#                 m = maskUtils.decode(rle)  # 解码为二值掩码

#                 # 为每个实例生成随机颜色
#                 # color = np.random.randint(0, 256, size=(3,))
#                 # mask_image[m == 1] = color_map[category_id]  # 将掩码区域设置为随机颜色
#                 color = np.random.randint(0, 256, size=(3,))
#                 mask_image[m == 1] = color  # 将掩码区域设置为随机颜色

#         # 将掩码叠加到原图上
#         alpha = 0.5  # 掩码透明度
#         blended_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
        
#         # 保存结果
#         output_path = os.path.join(output_folder, img_info['file_name'])
#         plt.imsave(output_path, blended_image)

#         print(f"Saved: {output_path}")

#     # 打印每个类别的数量
#     print("\nCategory Counts:")
#     for category_name, count in category_counter.items():
#         print(f"{category_name}: {count}")

def visualize_and_save_annotations(json_path, image_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载 COCO 标注
    coco = COCO(json_path)
    # 获取所有类别信息
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # 初始化类别计数器
    category_counter = defaultdict(int)

    # 获取所有图像 ID
    image_ids = coco.getImgIds()

    # 遍历每张图像
    for img_id in image_ids:
        # 加载图像信息
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_folder, img_info['file_name'])
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        # 获取该图像的标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 创建一个空白图像用于绘制掩码
        mask_image = np.zeros_like(image)

        # 遍历每个标注
        for ann in anns:
            # 获取类别 ID 并更新计数器
            category_id = ann['category_id']
            category_name = category_id_to_name[category_id]
            category_counter[category_name] += 1

            # 获取分割掩码
            if 'segmentation' in ann:
                rle = coco.annToRLE(ann)
                m = maskUtils.decode(rle)  # 解码为二值掩码

                # 为每个实例生成随机颜色
                color = np.random.randint(0, 256, size=(3,))
                mask_image[m == 1] = color  # 将掩码区域设置为随机颜色

                # 获取实例的边界框
                bbox = ann['bbox']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # 在图像上绘制类别名称
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(category_name, font, font_scale, thickness)[0]
                text_x = x
                text_y = y - 5 if y - 5 > 0 else y + text_size[1] + 5
                cv2.putText(image, category_name, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            else:
                print("none seg anno")
        # 将掩码叠加到原图上
        alpha = 0.5  # 掩码透明度
        blended_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
        
        # 保存结果
        output_path = os.path.join(output_folder, img_info['file_name'].replace('/', '_'))
        plt.imsave(output_path, blended_image)

        print(f"Saved: {output_path}")

    # 打印每个类别的数量
    print("\nCategory Counts:")
    for category_name, count in category_counter.items():
        print(f"{category_name}: {count}")

# 使用示例
json_path = '/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_split_ToI/test/test_ins_ToI.json'
image_folder = '/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_split_ToI/test/JPEGImages'
output_folder = '/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_split_ToI/test/tmp_vis'

visualize_and_save_annotations(json_path, image_folder, output_folder)
