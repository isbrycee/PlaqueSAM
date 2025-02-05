import os
from PIL import Image

def resize_mask_to_match_jpg(jpg_path, png_path):
    # 打开JPEG图像
    with Image.open(jpg_path) as jpg_img:
        jpg_size = jpg_img.size

    # 打开PNG图像（mask）
    with Image.open(png_path) as png_img:
        if png_img.size != jpg_size:
            # 使用最近邻插值调整PNG图像尺寸
            resized_png = png_img.resize(jpg_size, Image.NEAREST)
            resized_png.save(png_path)
            print(f"Resized {png_path} to match {jpg_path}")
        else:
            print(f"{png_path} already matches {jpg_path}")

def check_and_resize_masks(root_folder):
    jpeg_images_folder = os.path.join(root_folder, 'JPEGImages')
    annotations_folder = os.path.join(root_folder, 'Annotations')

    # 遍历JPEGImages文件夹
    for root, dirs, files in os.walk(jpeg_images_folder):
        for file in files:
            if file.endswith('.jpg'):
                jpg_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, jpeg_images_folder)
                png_path = os.path.join(annotations_folder, relative_path, file.replace('.jpg', '.png'))

                if os.path.exists(png_path):
                    resize_mask_to_match_jpg(jpg_path, png_path)
                else:
                    print(f"Mask file not found for {jpg_path}")

# 输入文件夹路径
root_folder = '/home/jinghao/projects/dental_plague_detection/dataset/train'
check_and_resize_masks(root_folder)
