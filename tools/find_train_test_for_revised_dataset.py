import os
import shutil

def find_and_copy_existing_folders(source_path, target_path, output_path):
    # 定义三个文件夹的名称
    folders = ["Annotations", "JPEGImages", "Json"]
    
    # 创建输出路径（如果不存在）
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for folder in folders:
        # 构建源、目标和输出的子文件夹路径
        source_folder = os.path.join(source_path, folder)
        target_folder = os.path.join(target_path, folder)
        output_folder = os.path.join(output_path, folder)
        
        # 如果源文件夹不存在，跳过
        if not os.path.exists(source_folder):
            print(f"源文件夹 {source_folder} 不存在，跳过...")
            continue
        
        # 如果目标文件夹不存在，跳过
        if not os.path.exists(target_folder):
            print(f"目标文件夹 {target_folder} 不存在，跳过...")
            continue
        
        # 创建输出文件夹（如果不存在）
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 遍历源文件夹中的病人 ID 文件夹
        for patient_id in os.listdir(source_folder):
            source_patient_path = os.path.join(source_folder, patient_id)
            target_patient_path = os.path.join(target_folder, patient_id)
            output_patient_path = os.path.join(output_folder, patient_id)
            
            # 检查是否为目录，且目标路径中是否存在相应的病人文件夹
            if os.path.isdir(source_patient_path) and os.path.exists(target_patient_path):
                print(f"找到存在的病人文件夹：{source_patient_path}")
                
                # 将病人文件夹内容复制到输出路径
                if not os.path.exists(output_patient_path):
                    os.makedirs(output_patient_path)
                
                # 复制文件和文件夹
                for item in os.listdir(source_patient_path):
                    source_item_path = os.path.join(source_patient_path, item)
                    output_item_path = os.path.join(output_patient_path, item)
                    
                    if os.path.isfile(source_item_path):
                        shutil.copy2(source_item_path, output_item_path)  # 复制文件
                    elif os.path.isdir(source_item_path):
                        shutil.copytree(source_item_path, output_item_path)  # 复制文件夹
                
                print(f"已复制：{source_patient_path} -> {output_patient_path}")
            else:
                print(f"目标路径中不存在 {target_patient_path}，跳过...")

# 示例使用
source_path = '/home/jinghao/projects/dental_plague_detection/dataset/plague_for_training_revised'
target_path =  '/home/jinghao/projects/dental_plague_detection/dataset/test'
output_path = '/home/jinghao/projects/dental_plague_detection/dataset/train_test_revised_split/test'

find_and_copy_existing_folders(source_path, target_path, output_path)
