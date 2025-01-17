import os
import json

# 遍历文件夹
def check_folders(folder_path):
    for subdir, _, files in os.walk(folder_path):
        json_count = {}
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(subdir, file), 'r') as f:
                    data = json.load(f)
                    for shape in data.get('shapes', []):
                        if 'label' in shape:
                            label = shape['label']
                            if label.startswith("mouth_") and label.split('_')[1].isdigit():
                                num = int(label.split('_')[1])
                                if num in json_count:
                                    json_count[num] += 1
                                else:
                                    json_count[num] = 1

        if len(json_count) != 7 or json_count.get(7, 0) != 2:
            print(f"Folder {subdir} does not meet the requirements.")
        for i in range(1, 7):
            if json_count.get(i, 0) >= 2:
                print(f"Folder {subdir} does not meet the requirements.")

# 传入文件夹路径
folder_path = '/home/jinghao/projects/dental_plague_detection/dataset/train/Json'
check_folders(folder_path)
