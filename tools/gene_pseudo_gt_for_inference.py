import os
import json
import numpy as np
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 定义类别体系
categories = [
        {
            "id": 0,
            "name": "0_p"
        },
        {
            "id": 1,
            "name": "0_np"
        },
        {
            "id": 2,
            "name": "0_caries"
        },
        {
            "id": 3,
            "name": "1_p"
        },
        {
            "id": 4,
            "name": "1_np"
        },
        {
            "id": 5,
            "name": "1_caries"
        },
        {
            "id": 6,
            "name": "2_p"
        },
        {
            "id": 7,
            "name": "2_np"
        },
        {
            "id": 8,
            "name": "2_caries"
        },
        {
            "id": 9,
            "name": "3_p"
        },
        {
            "id": 10,
            "name": "3_np"
        },
        {
            "id": 11,
            "name": "3_caries"
        },
        {
            "id": 12,
            "name": "4_p"
        },
        {
            "id": 13,
            "name": "4_np"
        },
        {
            "id": 14,
            "name": "4_caries"
        },
        {
            "id": 15,
            "name": "5_p"
        },
        {
            "id": 16,
            "name": "5_np"
        },
        {
            "id": 17,
            "name": "5_caries"
        },
        {
            "id": 18,
            "name": "6_p"
        },
        {
            "id": 19,
            "name": "6_np"
        },
        {
            "id": 20,
            "name": "6_caries"
        },
        {
            "id": 21,
            "name": "7_p"
        },
        {
            "id": 22,
            "name": "7_np"
        },
        {
            "id": 23,
            "name": "7_caries"
        },
        {
            "id": 24,
            "name": "8_p"
        },
        {
            "id": 25,
            "name": "8_np"
        },
        {
            "id": 26,
            "name": "8_caries"
        },
        {
            "id": 27,
            "name": "9_p"
        },
        {
            "id": 28,
            "name": "9_np"
        },
        {
            "id": 29,
            "name": "9_caries"
        },
        {
            "id": 30,
            "name": "10_p"
        },
        {
            "id": 31,
            "name": "10_np"
        },
        {
            "id": 32,
            "name": "10_caries"
        },
        {
            "id": 33,
            "name": "11_p"
        },
        {
            "id": 34,
            "name": "11_np"
        },
        {
            "id": 35,
            "name": "11_caries"
        },
        {
            "id": 36,
            "name": "12_p"
        },
        {
            "id": 37,
            "name": "12_np"
        },
        {
            "id": 38,
            "name": "12_caries"
        },
        {
            "id": 39,
            "name": "13_p"
        },
        {
            "id": 40,
            "name": "13_np"
        },
        {
            "id": 41,
            "name": "13_caries"
        },
        {
            "id": 42,
            "name": "14_p"
        },
        {
            "id": 43,
            "name": "14_np"
        },
        {
            "id": 44,
            "name": "14_caries"
        },
        {
            "id": 45,
            "name": "15_p"
        },
        {
            "id": 46,
            "name": "15_np"
        },
        {
            "id": 47,
            "name": "15_caries"
        },
        {
            "id": 48,
            "name": "16_p"
        },
        {
            "id": 49,
            "name": "16_np"
        },
        {
            "id": 50,
            "name": "16_caries"
        },
        {
            "id": 51,
            "name": "17_p"
        },
        {
            "id": 52,
            "name": "17_np"
        },
        {
            "id": 53,
            "name": "17_caries"
        },
        {
            "id": 54,
            "name": "18_p"
        },
        {
            "id": 55,
            "name": "18_np"
        },
        {
            "id": 56,
            "name": "18_caries"
        },
        {
            "id": 57,
            "name": "19_p"
        },
        {
            "id": 58,
            "name": "19_np"
        },
        {
            "id": 59,
            "name": "19_caries"
        },
        {
            "id": 60,
            "name": "20_p"
        },
        {
            "id": 61,
            "name": "20_np"
        },
        {
            "id": 62,
            "name": "20_caries"
        },
        {
            "id": 63,
            "name": "21_p"
        },
        {
            "id": 64,
            "name": "21_np"
        },
        {
            "id": 65,
            "name": "21_caries"
        },
        {
            "id": 66,
            "name": "22_p"
        },
        {
            "id": 67,
            "name": "22_np"
        },
        {
            "id": 68,
            "name": "22_caries"
        },
        {
            "id": 69,
            "name": "23_p"
        },
        {
            "id": 70,
            "name": "23_np"
        },
        {
            "id": 71,
            "name": "23_caries"
        },
        {
            "id": 72,
            "name": "24_p"
        },
        {
            "id": 73,
            "name": "24_np"
        },
        {
            "id": 74,
            "name": "24_caries"
        },
        {
            "id": 75,
            "name": "25_p"
        },
        {
            "id": 76,
            "name": "25_np"
        },
        {
            "id": 77,
            "name": "25_caries"
        },
        {
            "id": 78,
            "name": "26_p"
        },
        {
            "id": 79,
            "name": "26_np"
        },
        {
            "id": 80,
            "name": "26_caries"
        },
        {
            "id": 81,
            "name": "27_p"
        },
        {
            "id": 82,
            "name": "27_np"
        },
        {
            "id": 83,
            "name": "27_caries"
        },
        {
            "id": 84,
            "name": "28_p"
        },
        {
            "id": 85,
            "name": "28_np"
        },
        {
            "id": 86,
            "name": "28_caries"
        },
        {
            "id": 87,
            "name": "29_p"
        },
        {
            "id": 88,
            "name": "29_np"
        },
        {
            "id": 89,
            "name": "29_caries"
        }
    ]

def generate_infer_ins_toI(jpeg_images_dir, output_json_path):
    images = []
    image_id = 1

    logging.info(f"开始生成 {output_json_path} 文件...")
    # 遍历 JPEGImages 文件夹
    for subdir, _, files in os.walk(jpeg_images_dir):
        for file in sorted(files):
            if file.endswith(".jpg"):
                file_path = os.path.relpath(os.path.join(subdir, file), jpeg_images_dir)
                logging.info(f"处理图像文件: {file_path}")
                with Image.open(os.path.join(jpeg_images_dir, file_path)) as img:
                    width, height = img.size
                images.append({
                    "id": image_id,
                    "file_name": file_path,
                    "width": width,  
                    "height": height  
                })
                image_id += 1

    # 构造 JSON 数据
    data = {
        "images": images,
        "annotations": [],
        "categories": categories
    }

    # 写入 test_ins_ToI.json
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"{output_json_path} 文件生成完成！")

def generate_annotations(jpeg_images_dir, annotations_dir):
    logging.info(f"开始生成 {annotations_dir} 文件夹...")
    # 遍历 JPEGImages 文件夹
    for subdir, _, files in os.walk(jpeg_images_dir):
        for file in sorted(files):
            if file.endswith(".jpg"):
                # 创建对应的子文件夹
                relative_subdir = os.path.relpath(subdir, jpeg_images_dir)
                annotation_subdir = os.path.join(annotations_dir, relative_subdir)
                os.makedirs(annotation_subdir, exist_ok=True)

                # 创建空白 PNG 文件
                png_path = os.path.join(annotation_subdir, file.replace(".jpg", ".png"))
                with Image.open(os.path.join(jpeg_images_dir, relative_subdir, file)) as img:
                    width, height = img.size  # 动态获取图像宽度和高度
                blank_image = np.zeros((height, width), dtype=np.uint8)
                Image.fromarray(blank_image).save(png_path)
                logging.info(f"生成空白 PNG 文件: {png_path}")
    logging.info(f"{annotations_dir} 文件夹生成完成！")

def generate_json_files(jpeg_images_dir, json_dir):
    logging.info(f"开始生成 {json_dir} 文件夹...")
    # 遍历 JPEGImages 文件夹
    for subdir, _, files in os.walk(jpeg_images_dir):
        for file in sorted(files):
            if file.endswith(".jpg"):
                # 创建对应的子文件夹
                relative_subdir = os.path.relpath(subdir, jpeg_images_dir)
                json_subdir = os.path.join(json_dir, relative_subdir)
                os.makedirs(json_subdir, exist_ok=True)
                with Image.open(os.path.join(jpeg_images_dir, relative_subdir, file)) as img:
                    width, height = img.size  # 动态获取图像宽度和高度
                # 创建 JSON 文件
                json_path = os.path.join(json_subdir, file.replace(".jpg", ".json"))
                image_info = {
                    "imagePath": file,
                    "shapes": [ # pseudo info
                        {
                            "label": "mouth_1",
                            "points": [
                                [
                                    0.0,
                                    0.0
                                ],
                                [
                                    100,
                                    100
                                ]
                            ],
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                        },
                    ],
                    "imageHeight": height,  # 假设高度
                    "imageWidth": width   # 假设宽度
                }
                with open(json_path, "w") as f:
                    json.dump(image_info, f, indent=4)
                logging.info(f"生成 JSON 文件: {json_path}")
    logging.info(f"{json_dir} 文件夹生成完成！")

def main(demo_dir=None):
    if demo_dir is None:
        demo_dir = "demo_PlaqueSAM/"  # 默认输入文件夹
    jpeg_images_dir = os.path.join(demo_dir, "JPEGImages")  # 输入文件夹
    output_json_path = os.path.join(demo_dir, "infer_ins_ToI.json")  # 输出 JSON 文件路径
    annotations_dir = os.path.join(demo_dir, "Annotations")  # 输出 Annotations 文件夹
    json_dir = os.path.join(demo_dir, "Json")  # 输出 Json 文件夹

    # 生成 test_ins_ToI.json
    generate_infer_ins_toI(jpeg_images_dir, output_json_path)

    # 生成 Annotations 文件夹及 PNG 文件
    generate_annotations(jpeg_images_dir, annotations_dir)

    # 生成 Json 文件夹及 JSON 文件
    generate_json_files(jpeg_images_dir, json_dir)

if __name__ == "__main__":
    import sys
    demo_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(demo_dir_arg)