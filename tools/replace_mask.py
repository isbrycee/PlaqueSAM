import os
import shutil

def replace_files_from_A_to_B(folder_A, folder_B):
    # 遍历文件夹 A 中的所有文件
    for filename_A in os.listdir(folder_A):
        if filename_A.endswith(".png"):  # 确保是 .png 文件
            # 解析文件名：将 "9_26_30_002.png" 拆分为 "9_26_30" 和 "002.png"
            parts = filename_A.split("_")
            if len(parts) >= 4:  # 确保文件名符合规则
                subfolder_name = "_".join(parts[:-1])  # 提取子文件夹名，如 "9_26_30"
                file_number = parts[-1]  # 提取文件编号，如 "002.png"

                # 构建文件夹 B 中的目标路径，如 "B/9_26_30/002.png"
                target_path = os.path.join(folder_B, subfolder_name, file_number)

                # 如果目标路径存在，则替换文件
                if os.path.exists(target_path):
                    src_path = os.path.join(folder_A, filename_A)
                    shutil.copy(src_path, target_path)
                    print(f"替换: {src_path} -> {target_path}")
                else:
                    print(f"警告: 目标文件不存在，跳过: {target_path}")
            else:
                print(f"警告: 文件名不符合规则，跳过: {filename_A}")

# 输入文件夹路径
folder_B = "/home/jinghao/projects/dental_plague_detection/dataset/train/Annotations"  # 包含 "9_26_30_002.png" 等文件
folder_A = "/home/jinghao/projects/dental_plague_detection/GEM/datasets/plague/annotations/training"  # 包含 "9_26_30/002.png" 等文件

# 调用函数
replace_files_from_A_to_B(folder_A, folder_B)
