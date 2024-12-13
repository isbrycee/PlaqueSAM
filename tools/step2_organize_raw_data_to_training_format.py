import os
import shutil
import json
from PIL import Image, ImageDraw
from tqdm import tqdm


def correct_coordinates(top_left, bottom_right):
    # 解构坐标点
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # 确保 (x1, y1) 是左上角，(x2, y2) 是右下角
    correct_top_left = (min(x1, x2), min(y1, y2))
    correct_bottom_right = (max(x1, x2), max(y1, y2))
    
    return correct_top_left, correct_bottom_right


def generate_indexed_png(json_path, output_path, palette):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mouth_shape = next((shape for shape in data['shapes'] if shape['label'].startswith('mouth')), None)

    # for point convert cases
    top_left = mouth_shape['points'][0]
    bottom_right = mouth_shape['points'][1]

    top_left, bottom_right = correct_coordinates(top_left, bottom_right)

    mouth_shape['points'][0] = top_left
    mouth_shape['points'][1] = bottom_right

    # 获取图像尺寸
    # width = data['imageWidth']
    # height = data['imageHeight']
    
    width = int(mouth_shape['points'][1][0] - mouth_shape['points'][0][0])
    height = int(mouth_shape['points'][1][1] - mouth_shape['points'][0][1])
    
    # 创建一个新图像，模式为'P'表示索引色图
    img = Image.new('P', (width, height), 0)  # 背景为0
    draw = ImageDraw.Draw(img)
    
    # 定义标签到索引的映射
    label_to_index = {
        'np': 1,
        'p': 2,
        'caries': 3,  # 默认值
        'default': 0
    }
    
    # 将标签映射到图像的调色板
    img.putpalette(palette)
    
    # 遍历所有的shapes
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # 获取多边形的点
            points = [(int(point[0]), int(point[1])) for point in shape['points']]
            # 获取标签
            label = shape['label']
            # 根据标签设置像素值
            pixel_value = label_to_index.get(label, label_to_index['default'])
            # 绘制多边形
            draw.polygon(points, fill=pixel_value)
    
    # 保存PNG图像
    img_name = json_path.split('/')[-1].split('.')[0] + '.png'
    img.save(os.path.join(output_path, img_name))

def generate_json_to_png_mask(folder_path, save_folder_path):
    # 定义调色板，索引从1开始，0为背景
    palette = [0, 0, 0, 3, 255, 255, 255, 0, 0, 255, 215, 0]  # 背景为黑色，1为白色，2为红色，3为绿色
    for i in range(4, 256):
        palette.extend((i, i, i))
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            # 构造输出的PNG文件名
            output_filename = os.path.splitext(filename)[0] + '.png'
            # output_path = os.path.join(folder_path, output_filename)
            # 生成索引色图并保存为PNG
            generate_indexed_png(json_path, save_folder_path, palette)
            # print(f'Processed {filename} and saved as {output_filename}')

def contains_letters(s):
    return any(c.isalpha() for c in s)

def check_counts(new_folder, folder_prefixes):
    for folder_prefix in folder_prefixes:
        image_folder = os.path.join(new_folder, "JPEGImages", folder_prefix)
        json_folder = os.path.join(new_folder, "Json", folder_prefix)

        images = os.listdir(image_folder)
        jsons = os.listdir(json_folder)

        if len(images) != len(jsons):
            print(f"Mismatch in {folder_prefix}: {len(images)} images, {len(jsons)} JSON files.")
        else:
            print(f"Counts match in {folder_prefix}: {len(images)} images, {len(jsons)} JSON files.")

def organize_data(root_dir, new_folder):
    # new_folder = os.path.join(root_dir, "../NewFolder")
    os.makedirs(new_folder, exist_ok=True)
    
    subfolders = ["JPEGImages", "Annotations", "Json"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(new_folder, subfolder), exist_ok=True)
    
    folder_prefixes = []  # Collect folder prefixes for checking
    for folder in os.listdir(root_dir):
        if '_post' in folder:
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
                    images.sort()
                    for i, image in enumerate(images, start=1):
                        image_name, image_ext = os.path.splitext(image)
                        new_image_name = f"{i:03}{'.jpg'}"
                        new_save_folder = os.path.join(new_folder, "JPEGImages", folder_prefix)
                        if not os.path.exists(new_save_folder):
                            os.makedirs(new_save_folder)

                        # Open image
                        img_path = os.path.join(images_folder, image)
                        img = Image.open(img_path)

                        # Load corresponding JSON
                        json_folder = os.path.join(data_path.split(folder)[0], folder, folder+'_post_checked', data_folder.split('.')[0])
                        json_file = os.path.join(json_folder, f"{image_name}.json")
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        # Find mouth_xxx shape for cropping
                        mouth_shape = next((shape for shape in data['shapes'] if shape['label'].startswith('mouth')), None)
                        # for point convert cases
                        if mouth_shape['points'][0][0] > mouth_shape['points'][1][0] and mouth_shape['points'][0][1] > mouth_shape['points'][1][1]:
                            tmp = mouth_shape['points'][0]
                            mouth_shape['points'][0] = mouth_shape['points'][1]
                            mouth_shape['points'][1] = tmp

                        if mouth_shape:
                            points = mouth_shape['points']
                            top_left = points[0]
                            bottom_right = points[1]
                        else:
                            print("bad images and anno:", img_path)
                            
                        
                        top_left, bottom_right = correct_coordinates(top_left, bottom_right)
                        # Crop image
                        cropped_img = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                        cropped_img_path = os.path.join(new_save_folder, new_image_name)
                        cropped_img.save(cropped_img_path)

                        # Update coordinates in JSON
                        x_offset, y_offset = top_left
                        for shape in data['shapes']:
                            # check correctness of point 
                            if shape['shape_type']=='rectangle':
                                if shape['points'][0][0] > shape['points'][1][0] and shape['points'][0][1] > shape['points'][1][1]:
                                    tmp = shape['points'][0]
                                    shape['points'][0] = shape['points'][1]
                                    shape['points'][1] = tmp
                            shape['points'] = [[p[0] - x_offset, p[1] - y_offset] for p in shape['points']]

                        # Save updated JSON
                        new_json_save_folder = os.path.join(new_folder, "Json", folder_prefix)
                        if not os.path.exists(new_json_save_folder):
                            os.makedirs(new_json_save_folder)
                        new_json_path = os.path.join(new_json_save_folder, f"{i:03}.json")
                        with open(new_json_path, 'w') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)

    check_counts(new_folder, folder_prefixes)
    print("Data organization completed.")


# 输入多个根目录路径
# root_directories = ["/home/hust/haojing/dental_plague_dataset/10_8/", "/home/hust/haojing/dental_plague_dataset/10_10/"]
root_directories = ["/home/hust/haojing/dental_plague_dataset/raw_data/9_26/", 
                    "/home/hust/haojing/dental_plague_dataset/raw_data/10_8/",
                    "/home/hust/haojing/dental_plague_dataset/raw_data/10_10/",
                    "/home/hust/haojing/dental_plague_dataset/raw_data/10_24/",
                    "/home/hust/haojing/dental_plague_dataset/raw_data/10_31/"]

save_root_path = '/home/hust/haojing/dental_plague_dataset/plague_for_training_9_10'

for root_dir in root_directories:
    organize_data(root_dir, save_root_path)

# for convert json into png mask
for json_folder in tqdm(os.listdir(os.path.join(save_root_path, 'Json'))):
        save_folder_path = os.path.join(save_root_path, 'Annotations', json_folder)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path, exist_ok=True)
        generate_json_to_png_mask(os.path.join(save_root_path, 'Json', json_folder), save_folder_path)
