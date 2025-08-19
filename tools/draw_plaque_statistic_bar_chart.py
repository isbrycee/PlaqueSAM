import matplotlib.pyplot as plt
import json
from collections import defaultdict
from pycocotools import mask as maskUtils
from matplotlib import rcParams
# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

toothID_to_Number_Map = {
    0: '51',
    1: '52',
    2: '53',
    3: '54',
    4: '55',
    5: '61',
    6: '62',
    7: '63',
    8: '64',
    9: '65',
    10: '71',
    11: '72',
    12: '73',
    13: '74',
    14: '75',
    15: '81',
    16: '82',
    17: '83',
    18: '84',
    19: '85',
    20: '11',
    21: '16',
    22: '21',
    23: '26',
    24: '31',
    25: '36',
    26: '41',
    27: '46',
    28: 'doubleteeth',
    29: 'crown'
}

def get_segmentation_area(segmentation, image_height, image_width):
    if isinstance(segmentation, list):
        # polygon 格式
        rles = maskUtils.frPyObjects(segmentation, image_height, image_width)
        rle = maskUtils.merge(rles)
        area = maskUtils.area(rle)
    elif isinstance(segmentation, dict):
        # RLE 格式
        area = maskUtils.area(segmentation)
    else:
        area = 0
    return float(area)

def statistic_plot_and_save_stacked_counts(data_dict, save_path="stacked_bar_chart.png"):
    """
    根据字典数据绘制堆叠柱状图，并保存到文件。
    
    参数：
        data_dict: dict
            类似 {(1, 0): 0, (1, 5): 2, (1, 6): 1, ... }
            key[1] 表示名字ID，value 取值 0,1,2,3（其中1,2,3会合并为1）
        save_path: str
            保存图片的路径，例如 'chart.png'
    """
    # 统计每个名字的类别数量（1,2,3 合并成 1）
    stats = {}
    for (_, name), pred in data_dict.items():
        category = 0 if pred == 0 else 1
        name = toothID_to_Number_Map[name]
        if 'doubleteeth' in name:
            name = 'DT'
        if name not in stats:
            stats[name] = {0: 0, 1: 0}
        stats[name][category] += 1

    # 准备画图数据
    names = list(stats.keys())
    order_map = {'5': 0, '6': 1, '7': 2, '8': 3, '1': 4, '2': 5,
             '3': 6, '4': 7, 'D': 8}
    # 排序
    names = sorted(names, key=lambda x: (order_map[x[0]], int(x[1:]) if x[1:].isdigit() else x[1:]))

    zeros = [stats[n][0] for n in names]
    ones = [stats[n][1] for n in names]

    # 设置风格：Nature风格（简洁、专业）
    plt.style.use('seaborn-v0_8-whitegrid')  # 选择干净的背景
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (16, 10),
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'black',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })

    # 绘制堆叠柱状图
    colors = ['#c5e7ff', '#90EE90', '#FFB6C1', '#e1e1ff', '#FFD700']
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, zeros, label='w/o plaque', color=colors[0], edgecolor='black', alpha=1, linewidth=1.2)
    ax.bar(x, ones, bottom=zeros, label='w plaque', color=colors[2], edgecolor='black', alpha=1, linewidth=1.2)

    ax.set_xlabel('Tooth ID')
    ax.set_ylabel('Number')
    # ax.set_title('每个名字的类别统计（堆叠图）\n(1,2,3 合并为 1)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"堆叠柱状图已保存到: {save_path}")

def merge_dicts(dict1, dict2):
    merged = dict1.copy()  # 先复制第一个字典
    
    for key, value in dict2.items():
        # 如果键在dict1中已经存在，则修改键的第一个元素
        if key in merged:
            new_key = (key[0] + 10000, *key[1:])
            merged[new_key] = value
        else:
            merged[key] = value
    
    return merged

def load_image_id2size(images):
    id2size = {}
    for img in images:
        id2size[img['id']] = (img['height'], img['width'])
    return id2size

def compute_tooth_grades(annotations, id2size):
    # {(img_id, tooth_idx): {'plaque': float, 'tooth': float, 'caries': float}}
    stats = defaultdict(lambda: {'plaque':0.0, 'tooth':0.0, 'caries':0.0})
    for anno in annotations:
        img_id = anno['image_id']
        cat_id = anno['category_id']
        tooth_idx = cat_id // 3
        type_idx = cat_id % 3  # 0:plaque, 1:tooth, 2:caries
        if img_id not in id2size:
            continue  # 跳过没有图片信息的预测
        h, w = id2size[img_id]
        area = get_segmentation_area(anno['segmentation'], h, w)
        if type_idx == 0:
            stats[(img_id, tooth_idx)]['plaque'] += area
        elif type_idx == 1:
            stats[(img_id, tooth_idx)]['tooth'] += area
        elif type_idx == 2:
            stats[(img_id, tooth_idx)]['caries'] += area
    # 计算分级
    result = {}
    for key, v in stats.items():
        plaque = v['plaque']
        tooth = v['tooth']
        caries = v['caries']
        total = plaque + tooth + caries
        if total == 0 or plaque == 0:
            grade = 0
        else:
            ratio = plaque / total
            if ratio > 0 and ratio < 1/3:
                grade = 1
            elif ratio >= 1/3 and ratio < 2/3:
                grade = 2
            elif ratio >= 2/3:
                grade = 3
            else:
                grade = 0
        result[key] = grade

    return result

if __name__ == "__main__":
    test_gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/test_ins_ToI.json"
    train_gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/train/train_ins_ToI.json"

    with open(train_gt_json_path, "r") as f1:
        train_gt_json = json.load(f1)

    # pred文件可能没有images字段
    images_train = train_gt_json['images']
    id2size_train = load_image_id2size(images_train)
    gt_grades_train = compute_tooth_grades(train_gt_json['annotations'], id2size_train)

    with open(train_gt_json_path, "r") as f2:
        test_gt_json = json.load(f2)

    # pred文件可能没有images字段
    images_test = test_gt_json['images']
    id2size_test = load_image_id2size(images_test)
    gt_grades_test = compute_tooth_grades(test_gt_json['annotations'], id2size_test)


    gt_json = merge_dicts(gt_grades_train, gt_grades_test)

    statistic_plot_and_save_stacked_counts(gt_json, save_path="plaque_statistic_stacked_bar_chart.png")