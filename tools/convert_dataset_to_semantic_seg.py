import os
import shutil
import json

def copy_and_rename_images(src_jpeg, src_annotations, dst_all_image, dst_all_masks):
    # 确保目标文件夹存在
    if not os.path.exists(dst_all_image):
        os.makedirs(dst_all_image)

    # 遍历 JPEGImages 文件夹
    for root, dirs, files in os.walk(src_jpeg):
        for file in files:
            if file.endswith(".jpg"):
                # 获取文件的相对路径
                relative_path = os.path.relpath(root, src_jpeg)
                # 构建新的文件名
                new_filename = os.path.join(relative_path, file).replace(os.sep, "_")
                is_skip = False
                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(root, file)
                json_file = src_file.replace('JPEGImages', 'Json').replace('.jpg', '.json')
                with open(json_file) as f_json:
                    json_data = json.load(f_json)
                for _item in json_data['shapes']:
                    if 'mouth_7' in _item['label']:
                        is_skip=True
                        break
                if is_skip:
                    continue
                dst_file = os.path.join(dst_all_image, new_filename)
                # 复制文件
                shutil.copy(src_file, dst_file)

    # 确保目标文件夹存在
    if not os.path.exists(dst_all_masks):
        os.makedirs(dst_all_masks)  

    # # 遍历 Annotations 文件夹
    for root, dirs, files in os.walk(src_annotations):
        for file in files:
            if file.endswith(".png"):
                # 获取文件的相对路径
                relative_path = os.path.relpath(root, src_annotations)
                # 构建新的文件名
                new_filename = os.path.join(relative_path, file).replace(os.sep, "_")
                is_skip = False
                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(root, file)
                json_file = src_file.replace('Annotations', 'Json').replace('.png', '.json')
                with open(json_file) as f_json:
                    json_data = json.load(f_json)
                for _item in json_data['shapes']:
                    if 'mouth_7' in _item['label']:
                        is_skip=True
                        break
                if is_skip:
                    continue
                dst_file = os.path.join(dst_all_masks, new_filename)
                # 复制文件
                shutil.copy(src_file, dst_file)

# 输入文件夹路径
src_jpeg = "/home/jinghao/projects/dental_plague_detection/dataset/train/JPEGImages"
src_annotations = "/home/jinghao/projects/dental_plague_detection/dataset/train/Annotations/"

dst_all_image = "/home/jinghao/projects/dental_plague_detection/dataset/plague_semantic_segmentation/images/training"
dst_all_masks = "/home/jinghao/projects/dental_plague_detection/dataset/plague_semantic_segmentation/annotations/training"
# 调用函数
copy_and_rename_images(src_jpeg, src_annotations, dst_all_image, dst_all_masks)