import os
from PIL import Image
import numpy as np

# 定义像素值的映射
mapping = {
    0: 255,
    1: 0,
    2: 1,
    3: 2
}

def remap_mask(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有 PNG 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 打开图像
            with Image.open(input_path) as img:
                # 将图像转换为 numpy 数组
                img_array = np.array(img)
                
                # 创建一个新的数组，用于存放映射后的结果
                remapped_array = np.copy(img_array)
                
                # 根据映射规则替换像素值
                for src_value, dst_value in mapping.items():
                    remapped_array[img_array == src_value] = dst_value
                
                # 将 numpy 数组转换回图像
                remapped_img = Image.fromarray(remapped_array.astype(np.uint8))
                
                # 保存结果到输出路径
                remapped_img.save(output_path)
                print(f"已处理文件: {filename}")


# 输入文件夹路径
input_folder = '/home/jinghao/projects/dental_plague_detection/dataset/plague_semantic_segmentation/annotations/validation_bak/'
output_folder = '/home/jinghao/projects/dental_plague_detection/dataset/plague_semantic_segmentation/annotations/validation/'

# 调用函数处理图像
remap_mask(input_folder, output_folder)