import os
import shutil

def backup_and_rename_files(folder1, folder2, backup_dir, extension1='.JPG', extension2='.png'):
    # 创建备份目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 获取两个文件夹中所有相同名字的子文件夹
    for subdir in os.listdir(folder1):
        subdir1 = os.path.join(folder1, subdir)
        subdir2 = os.path.join(folder2, subdir)
        if os.path.isdir(subdir1) and os.path.isdir(subdir2):
            # 获取两个子文件夹中所有文件的名字（不含后缀）
            files1 = {os.path.splitext(file)[0] for file in os.listdir(subdir1) if file.endswith(extension1)}
            files2 = {os.path.splitext(file)[0] for file in os.listdir(subdir2) if file.endswith(extension2)}
            # 找到两个子文件夹共有的文件名
            common_files = files1.intersection(files2)
            # 对共有的文件名进行排序
            sorted_files = sorted(common_files)
            
            # 备份并重命名文件
            for i, filename in enumerate(sorted_files, start=1):
                src_file1 = os.path.join(subdir1, f"{filename}{extension1}")
                src_file2 = os.path.join(subdir2, f"{filename}{extension2}")
                new_name1 = f"{i:03d}{extension1}"
                new_name2 = f"{i:03d}{extension2}"
                dst_file1 = os.path.join(subdir1, new_name1)
                dst_file2 = os.path.join(subdir2, new_name2)
                
                # 备份原始文件
                backup_file1 = os.path.join(backup_dir, f"{subdir}_{filename}{extension1}")
                backup_file2 = os.path.join(backup_dir, f"{subdir}_{filename}{extension2}")
                shutil.copy2(src_file1, backup_file1)
                shutil.copy2(src_file2, backup_file2)
                
                # 重命名文件
                os.rename(src_file1, dst_file1)
                os.rename(src_file2, dst_file2)
                print(f"Backed up and renamed {src_file1} to {dst_file1} and {src_file2} to {dst_file2}")

# 输入的两个文件夹路径和备份文件夹路径
folder_path1 = '/home/hust/haojing/sam2/datasets/DPS/JPEGImages/'
folder_path2 = '/home/hust/haojing/sam2/datasets/DPS/Annotations/'
backup_folder_path = '/home/hust/haojing/sam2/datasets/DPS_bak'

# 调用函数
backup_and_rename_files(folder_path1, folder_path2, backup_folder_path)
