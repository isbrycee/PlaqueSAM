import os
from PIL import Image
import shutil

def check_and_resize_masks(jpg_folder, png_folder, output_folder):
    # 获取两个文件夹中的文件列表
    jpg_files = {f.split('.')[0]: os.path.join(jpg_folder, f) for f in os.listdir(jpg_folder) if f.endswith('.jpg')}
    png_files = {f.split('.')[0]: os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith('.png')}

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检查同名文件并调整分辨率
    for name in jpg_files:
        if name in png_files:
            # 打开 .jpg 和 .png 文件
            jpg_image = Image.open(jpg_files[name])
            png_image = Image.open(png_files[name])

            # 获取分辨率
            jpg_res = jpg_image.size
            png_res = png_image.size

            # 如果分辨率不同，调整 mask 分辨率
            if jpg_res != png_res:
                print(f"调整 {name}.png 的分辨率: {png_res} -> {jpg_res}")
                # resized_mask = png_image.resize(jpg_res, Image.Resampling.NEAREST)  # 使用最近邻插值
                # resized_mask.save(os.path.join(output_folder, f"{name}.png"))
            else:
                # 如果分辨率相同，直接复制到输出文件夹
                print('ok')
                # shutil.copy(png_files[name], os.path.join(output_folder, f"{name}.png"))
        else:
            print(f"警告: {name}.jpg 没有对应的 .png 文件")

    print("处理完成。调整后的 mask 已保存到:", output_folder)

# 输入文件夹路径
jpg_folder = "/home/jinghao/projects/dental_plague_detection/GEM/datasets/plague/images/validation"
png_folder = "/home/jinghao/projects/dental_plague_detection/GEM/datasets/plague/annotations/validation"

output_folder = "/home/jinghao/projects/dental_plague_detection/GEM/datasets/plague/annotations/validation_checked/"

# 调用函数
check_and_resize_masks(jpg_folder, png_folder, output_folder)
