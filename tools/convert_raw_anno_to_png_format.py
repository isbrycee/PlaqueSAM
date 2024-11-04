import os
import json
from PIL import Image, ImageDraw

def generate_indexed_png(json_path, output_path, palette):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    width = data['imageWidth']
    height = data['imageHeight']
    
    # 创建一个新图像，模式为'P'表示索引色图
    img = Image.new('P', (width, height), 0)  # 背景为0
    draw = ImageDraw.Draw(img)
    
    # 定义标签到索引的映射
    label_to_index = {
        'np': 1,
        'p': 2,
        'caries': 3,  # 默认值
        'default': 0
    }
    
    # 将标签映射到图像的调色板
    img.putpalette(palette)
    
    # 遍历所有的shapes
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # 获取多边形的点
            points = [(int(point[0]), int(point[1])) for point in shape['points']]
            # 获取标签
            label = shape['label']
            # 根据标签设置像素值
            pixel_value = label_to_index.get(label, label_to_index['default'])
            # 绘制多边形
            draw.polygon(points, fill=pixel_value)
    
    # 保存PNG图像
    img.save(output_path)

def process_folder(folder_path):
    # 定义调色板，索引从1开始，0为背景
    palette = [0, 0, 0, 3, 255, 255, 255, 0, 0, 255, 215, 0]  # 背景为黑色，1为白色，2为红色，3为绿色
    for i in range(4, 256):
        palette.extend((i, i, i))
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            # 构造输出的PNG文件名
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(folder_path, output_filename)
            # 生成索引色图并保存为PNG
            generate_indexed_png(json_path, output_path, palette)
            print(f'Processed {filename} and saved as {output_filename}')

# 调用函数处理指定文件夹
folder_path = '/home/hust/haojing/sam2/datasets/DPS/Annotations'  # 替换为你的文件夹路径
process_folder(folder_path)