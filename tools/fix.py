import os
import shutil
import json
from PIL import Image, ImageDraw
from tqdm import tqdm


palette = [0, 0, 0, 3, 255, 255, 255, 0, 0, 255, 215, 0]  # 背景为黑色，1为白色，2为红色，3为绿色
for i in range(4, 256):
    palette.extend((i, i, i))

json_path = '/home/jinghao/projects/dental_plague_detection/dataset/train/Json/10_24_27/005.json'
    
with open(json_path, 'r') as f:
    data = json.load(f)

mouth_shape = next((shape for shape in data['shapes'] if shape['label'].startswith('mouth')), None)

# for point convert cases
top_left = mouth_shape['points'][0]
bottom_right = mouth_shape['points'][1]

# top_left, bottom_right = correct_coordinates(top_left, bottom_right)

mouth_shape['points'][0] = top_left
mouth_shape['points'][1] = bottom_right

# 获取图像尺寸
# width = data['imageWidth']
# height = data['imageHeight']

width = int(mouth_shape['points'][1][0] - mouth_shape['points'][0][0])
height = int(mouth_shape['points'][1][1] - mouth_shape['points'][0][1])

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
img_name = json_path.split('/')[-1].split('.')[0] + '.png'
img.save('005.png')