import os
import json

def check_subfolder(subdir_path):
    """检查子文件夹中是否存在至少一个包含label='p'的JSON文件"""
    json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
    
    for filename in json_files:
        filepath = os.path.join(subdir_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 发现符合要求的标注立即返回True
            if any(shape.get('label') == 'p' for shape in data.get('shapes', [])):
                return True
        except Exception as e:
            print(f"跳过错误文件 {filename}: {str(e)}")
            continue  # 跳过问题文件继续检查其他文件
    
    return False  # 所有文件检查完毕未发现目标标签

def main():
    target_dir = "/home/jinghao/projects/dental_plague_detection/dataset/2025_revised_for_training_split_ToI/test/Json"
    qualified_folders = []
    
    for folder in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, folder)
        if os.path.isdir(folder_path):
            if check_subfolder(folder_path):
                qualified_folders.append(folder)

    with open('list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(qualified_folders))

    print(f"找到 {len(qualified_folders)} 个符合条件的文件夹，结果已保存到list.txt")
    for ii in os.listdir(target_dir):
        if ii not in qualified_folders:
            print(ii)
if __name__ == '__main__':
    main()
