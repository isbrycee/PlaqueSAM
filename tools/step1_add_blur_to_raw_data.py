import os
import shutil
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from imagecorruptions import corrupt, get_corruption_names
                        

def contains_letters(s):
    return any(c.isalpha() for c in s)

def gene_blur_images_and_jsons(image_folder_path, anno_folder_path, min_images=8):

    images = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
    image_folder_list = images

    all_jsons = [f for f in os.listdir(anno_folder_path) if os.path.isfile(os.path.join(anno_folder_path, f))]

    # 检查图像数量是否小于指定数量
    if len(image_folder_list) < min_images:
        # 计算需要补充的图像数量
        num_to_add = min_images - len(image_folder_list)
        
        # 检查是否已经有 blur 图片并从待选集合中过滤掉
        is_blur_name_list = []
        for _json in all_jsons:
            src_json = os.path.join(anno_folder_path, _json)
            with open(src_json, 'r') as f_json:
                json_data = json.load(f_json)
            for _item in json_data['shapes']:
                if 'mouth_7' in _item['label'] or 'mouth7' in _item['label']:
                    is_blur_name_list.append(_json.split('.json')[0]+'.JPG')
                    
        select_from_image_folder_list = [item for item in image_folder_list if item not in is_blur_name_list]

        # 随机选择图像进行补充
        selected_images = random.sample(select_from_image_folder_list, k=num_to_add)
        # print(selected_images)
        # import pdb; pdb.set_trace()
        for image in selected_images:
            src_path = os.path.join(image_folder_path, image)
            image_suffix = image[-4:]
            anno_json_file_name = image.replace(image_suffix, '.json')
            anno_path = os.path.join(anno_folder_path, anno_json_file_name)
            img_array = plt.imread(src_path)
            
            blur_method = random.choices(['defocus_blur', 'motion_blur'], k=1)
            corrupted_image = corrupt(img_array, corruption_name=blur_method[0], severity=5)
            
            blurred_image = Image.fromarray(corrupted_image)

            new_save_image_path = os.path.join(image_folder_path, 'blur_' + image)
            blurred_image.save(new_save_image_path)
            # print(new_save_image_path)

            with open(anno_path, 'r') as f_r:
                json_anno = json.load(f_r)
            for item in json_anno['shapes']:
                if 'mouth' in item['label']:
                    item['label'] = 'mouth_7'
                    break
            with open(os.path.join(anno_folder_path, 'blur_' + anno_json_file_name),"w") as f_w:
                json.dump(json_anno, f_w)

            # import pdb; pdb.set_trace()



def organize_data(root_dir):
    # new_folder = os.path.join(root_dir, "../NewFolder")

    # os.makedirs(new_folder, exist_ok=True)

    # subfolders = ["JPEGImages", "Annotations", "Json"]
    # for subfolder in subfolders:
    #     os.makedirs(os.path.join(new_folder, subfolder), exist_ok=True)
    annotation_folder_name = ''
    image_folder_name = ''
    for folder in os.listdir(root_dir):
        if '_post_checked' in folder:
            annotation_folder_name = folder
        elif '_post' in folder:
            continue
        else:
            image_folder_name = folder

    folder_prefixes = []  # Collect folder prefixes for checking
    for folder in os.listdir(root_dir):
        if 'post' in folder:
            continue
        
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            data_folders = os.listdir(folder_path)
            for data_folder in tqdm(data_folders):
                data_path = os.path.join(folder_path, data_folder)

                if os.path.isdir(data_path):
                    if contains_letters(data_folder):
                        folder_prefix = f"{folder}_{data_folder.split('.')[0]}"
                    else:
                        folder_prefix = f"{folder}_{data_folder}"

                    folder_prefixes.append(folder_prefix)
                    
                    images_folder = data_path
                    images = os.listdir(images_folder)

                    index = images_folder.rfind(folder)
                    # if '11_7' in image_folder_name:
                    #     import pdb; pdb.set_trace()
                    if index != -1:
                        anno_folder = images_folder[:index] + images_folder[index:].replace(image_folder_name, annotation_folder_name)
                    else:
                        print('get anno folder error !')
                    if '.' in anno_folder:
                        anno_folder = anno_folder.split('.')[0]
                    anno_json_files_list = os.listdir(anno_folder)
                    print(images_folder)
                    assert len(images) == len(anno_json_files_list)

                    images.sort()
                    anno_json_files_list.sort()

                    gene_blur_images_and_jsons(images_folder, anno_folder)


# 输入多个根目录路径
root_directories = ['/home/jinghao/projects/dental_plague_detection/dataset/15_2_2025_revision/10_8/',
                    '/home/jinghao/projects/dental_plague_detection/dataset/27_1_2025_revision/9_26/' ]
# root_directories = ["/home/jinghao/projects/dental_plague_detection/dataset/15_2_2025_revision_1",  ]
for root_dir in root_directories:
    # single_path = os.path.join(root_directories, root_dir)
    organize_data(root_dir)
