import os
import shutil
import random

# 定义数据的根目录
base_dir = '/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_all_bak/'
annotations_dir = os.path.join(base_dir, "Annotations")
images_dir = os.path.join(base_dir, "JPEGImages")
json_dir = os.path.join(base_dir, "Json")

# 定义输出路径
output_dir = "/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_split/"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# 创建训练集和测试集文件夹
os.makedirs(os.path.join(train_dir, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "Json"), exist_ok=True)

os.makedirs(os.path.join(test_dir, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "Json"), exist_ok=True)

# 获取病人文件夹名字
patient_folders = sorted(os.listdir(annotations_dir))  # 获取 Annotations 文件夹中的病人文件夹名称

# 随机打乱病人文件夹名字
random.seed(42)  # 设置随机种子以确保结果可重复
random.shuffle(patient_folders)

# 分割为训练集和测试集
test_size = 100  # 测试集大小
test_patients = patient_folders[:test_size]
train_patients = patient_folders[test_size:]

# 定义一个函数，用于复制病人文件夹
def copy_patient_folders(patient_list, src_dir, dest_dir):
    for patient in patient_list:
        src_path = os.path.join(src_dir, patient)
        dest_path = os.path.join(dest_dir, patient)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dest_path)  # 递归复制整个文件夹
            
# 复制测试集病人文件夹
copy_patient_folders(test_patients, annotations_dir, os.path.join(test_dir, "Annotations"))
copy_patient_folders(test_patients, images_dir, os.path.join(test_dir, "JPEGImages"))
copy_patient_folders(test_patients, json_dir, os.path.join(test_dir, "Json"))

# 复制训练集病人文件夹
copy_patient_folders(train_patients, annotations_dir, os.path.join(train_dir, "Annotations"))
copy_patient_folders(train_patients, images_dir, os.path.join(train_dir, "JPEGImages"))
copy_patient_folders(train_patients, json_dir, os.path.join(train_dir, "Json"))

print("数据集划分完成！")