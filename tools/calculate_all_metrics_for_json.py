import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from collections import defaultdict
from pycocotools.mask import decode
import torch
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import math
from pycocotools import mask as mask_utils
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

def calculate_acc(gt_grades, pred_grades):

    # 1. tooth-level
    correct_tooth, total_tooth = 0, 0
    # 2. image-level
    image_to_teeth = {}
    for (image_id, tooth_idx) in gt_grades:
        image_to_teeth.setdefault(image_id, []).append(tooth_idx)

    correct_image, total_image = 0, 0
    # 3. patient-level
    patient_to_images = {}
    for image_id in image_to_teeth:
        patient_id = image_id // 6
        patient_to_images.setdefault(patient_id, []).append(image_id)

    correct_patient, total_patient = 0, 0

    # 1. Tooth-level
    for key in gt_grades:
        if key in pred_grades:
            # changed to 0 / 1; does not split into three plaque levels
            if gt_grades[key] >= 1:
                gt_grades[key] = 1
            if pred_grades[key] >= 1:
                pred_grades[key] = 1

            if gt_grades[key] == pred_grades[key]:
                correct_tooth += 1
            total_tooth += 1
    tooth_acc = correct_tooth / total_tooth if total_tooth > 0 else 0

    # 2. Image-level
    for image_id, tooth_list in image_to_teeth.items():
        all_correct = True
        for tooth_idx in tooth_list:
            key = (image_id, tooth_idx)
            if key not in pred_grades or gt_grades[key] != pred_grades[key]:
                all_correct = False
                break
        if all_correct:
            correct_image += 1
        total_image += 1
    image_acc = correct_image / total_image if total_image > 0 else 0

    # 3. Patient-level
    for patient_id, image_list in patient_to_images.items():
        all_correct = True
        for image_id in image_list:
            for tooth_idx in image_to_teeth[image_id]:
                key = (image_id, tooth_idx)
                if key not in pred_grades or gt_grades[key] != pred_grades[key]:
                    all_correct = False
                    break
            if not all_correct:
                break
        if all_correct:
            correct_patient += 1
        total_patient += 1
    patient_acc = correct_patient / total_patient if total_patient > 0 else 0

    return {
        'tooth_acc': tooth_acc,
        'image_acc': image_acc,
        'patient_acc': patient_acc
    }


def calculate_tooth_level_Sensitivity_Specificity(gt_grades, pred_grades):
    # 初始化计数器
    tp = tn = fp = fn = 0
    # 遍历 GT 字典
    for key, gt_value in gt_grades.items():
        pred_value = pred_grades.get(key, None)  # 获取对应的 Pred 值，如果不存在默认为 None
        if gt_value >= 1:
            gt_value = 1
        if pred_value is not None:
            if pred_value >= 1:
                pred_value = 1

            if gt_value == 1 and pred_value == 1:  # True Positive
                tp += 1
            elif gt_value == 0 and pred_value == 0:  # True Negative
                tn += 1
            elif gt_value == 0 and pred_value == 1:  # False Positive
                fp += 1
            elif gt_value == 1 and pred_value == 0:  # False Negative
                fn += 1
        else:
            # 没有预测出来时，按照实际类别处理
            if gt_value == 1:
                fn += 1  # 实际有病但没预测，假阴性
            elif gt_value == 0:
                tn += 1  # 实际无病但没预测，真阴性（有争议，但通常按这种方式处理）

    # 计算 Sensitivity 和 Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
     # 输出结果
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    # print(f"Sensitivity: {sensitivity:.2f}")
    # print(f"Specificity: {specificity:.2f}")

    return sensitivity, specificity
   

# def calculate_Accuracy_three_classes_of_dental_plaque(gt_grades, pred_grades, classes=[0,1,2,3]):
#     """
#     计算每个类别的 accuracy
#     :return: dict, {class: accuracy}
#     """
#     result = {}
#     for cls in classes:
#         correct = 0
#         total = 0
#         for key, gt_value in gt_grades.items():
#             pred_value = pred_grades.get(key, None)
#             if pred_value is not None and gt_value == cls:
#                 total += 1
#                 if pred_value == gt_value:
#                     correct += 1
#         result[cls] = correct / total if total > 0 else 0.0
#     return result

def calculate_Accuracy_Sensitivity_Specificity_metrics_three_classes_of_dental_plaque(gt_grades, pred_grades, classes=[0,1,2,3]):
    """
    计算每个类别的 accuracy、sensitivity、specificity
    :return: dict, {class: {'accuracy': x, 'sensitivity': y, 'specificity': z}}
    """
    results = {}
    for cls in classes:
        # 初始化计数器
        TP = FP = TN = FN = 0
        correct = 0
        total = 0
        for key, gt_value in gt_grades.items():
            pred_value = pred_grades.get(key, None)
            if pred_value is None:
                continue
            # Accuracy 统计（仅统计真实为cls的样本）
            if gt_value == cls:
                total += 1
                if pred_value == gt_value:
                    correct += 1
            # TP, FN, FP, TN 统计
            if gt_value == cls and pred_value == cls:
                TP += 1
            elif gt_value == cls and pred_value != cls:
                FN += 1
            elif gt_value != cls and pred_value == cls:
                FP += 1
            elif gt_value != cls and pred_value != cls:
                TN += 1
        # 计算三项指标
        accuracy = correct / total if total > 0 else 0.0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        results[cls] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    return results

def calculate_mAP_Sensitivity_Specificity(gt_json_path, pred_json_path):
    # 初始化COCO验证器
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 计算mAP
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')

    # 指定需要计算的类别名称，例如 'person' 和 'car'
    # cat_ids = []
    # for id in coco_gt.getCatIds():
    #     if id % 3 == 0:
    #         cat_ids.append(id)
    # coco_eval.params.catIds = cat_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 获取 float 值
    mask_mAP = float(coco_eval.stats[0])       # mAP@[.5:.95]
    mask_ap50 = float(coco_eval.stats[1])      # AP@0.5

    # 获取每一类 Recall 数据
    recall = coco_eval.eval['recall']

    # iou_threshold_index = 0  # IoU=0.5
    # catIds = coco_gt.getCatIds()
    # for category_id in catIds:
    #     specific_recall = recall[iou_threshold_index, category_id, 0, 0]
    #     print(f"Recall for category {category_id} at IoU=0.5: {specific_recall}")

    ###### split 计算 sensitivity ####

    # 指定 IoU=0.5 时的 Recall
    iou_threshold_index = 0  # 对应 IoU=0.5
    specific_recall = recall[iou_threshold_index, :, :, 0]  # Shape: [4, K]

    # 遍历感兴趣的类别
    catIds = coco_gt.getCatIds()
    categories_to_merge = [catId for catId in catIds if catId % 3==0]
    area_index = 0  # 'all' 面积范围

    # 获取每个类别的 Recall 和总实例数
    total_tp = 0  # 总召回到的实例数
    total_instances = 0  # 总实例数

    for category_id in categories_to_merge:
        # 获取该类别的 Recall
        category_recall = specific_recall[category_id, area_index]
        
        # 获取该类别的总实例数（TP + FN）
        total_instances_category = len(coco_gt.getAnnIds(catIds=[category_id]))
        # 计算召回的实例数 TP
        if category_recall >= 0:  # 确保 Recall 有效
            tp = category_recall * total_instances_category
            total_tp += tp
            total_instances += total_instances_category
    if total_instances > 0:
        sensitivity = total_tp / total_instances

    ###### split 计算 Specificity ####

    # 指定 IoU=0.5 时的 Recall
    iou_threshold_index = 0  # 对应 IoU=0.5
    specific_recall = recall[iou_threshold_index, :, :, 0]  # Shape: [4, K]

    # 遍历感兴趣的类别
    categories_to_merge = [catId for catId in catIds if not catId % 3==0]
    area_index = 0  # 'all' 面积范围
    # 获取每个类别的 Recall 和总实例数
    total_tp = 0  # 总召回到的实例数
    total_instances = 0  # 总实例数

    for category_id in categories_to_merge:
        # 获取该类别的 Recall
        category_recall = specific_recall[category_id, area_index]
        
        # 获取该类别的总实例数（TP + FN）
        total_instances_category = len(coco_gt.getAnnIds(catIds=[category_id]))
        # 计算召回的实例数 TP
        if category_recall >= 0:  # 确保 Recall 有效
            tp = category_recall * total_instances_category
            total_tp += tp
            total_instances += total_instances_category

    if total_instances > 0:
        specificity = total_tp / total_instances    
    
    return mask_mAP, mask_ap50, sensitivity, specificity


def compute_for_dental_plaque_IoU(gt_json_path, pred_json_path):
    # 加载 COCO ground truth 和 prediction
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 找到所有类别中 ID 是 3 的倍数的类别
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_ids = [cat['id'] for cat in categories if cat['id'] % 3 == 0]
    print(f"合并的类别 ID: {category_ids}")
    
    ious = []
    
    def get_merged_mask(coco, ann_ids, category_ids, h, w):
        anns = coco.loadAnns(ann_ids)
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] in category_ids:
                mask = coco.annToMask(ann)
                merged_mask = np.logical_or(merged_mask, mask)
        return merged_mask.astype(np.uint8)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        
        # 获取gt和pred的ann id
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        pred_ann_ids = coco_pred.getAnnIds(imgIds=img_id)
        
        # 得到合并后的mask
        gt_mask = get_merged_mask(coco_gt, gt_ann_ids, category_ids, h, w)
        pred_mask = get_merged_mask(coco_pred, pred_ann_ids, category_ids, h, w)
        
        # 计算IoU
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        # if gt_mask.sum() == 0:
        #     continue
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou)

    miou = np.mean(ious)

    return miou

def compute_mIoU_for_dental_plaque_IoU_per_category(gt_json_path, pred_json_path, toothID_to_Number_Map):
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_ids = [cat['id'] for cat in categories if cat['id'] % 3 == 0]
    # print(f"合并的类别 ID: {category_ids}")
    
    img_ids = coco_gt.getImgIds()
    cat_ious = {cat_id: [] for cat_id in category_ids}
    
    def get_category_mask(coco, ann_ids, category_id, h, w):
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] == category_id:
                mask = np.logical_or(mask, coco.annToMask(ann))
        return mask.astype(np.uint8)
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        pred_ann_ids = coco_pred.getAnnIds(imgIds=img_id)
        
        for cat_id in category_ids:

            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            gt_ann_cat_ids = [ann['category_id'] for ann in gt_anns]
            if cat_id not in gt_ann_cat_ids:
                # 该类别在此图像中没有标注，跳过
                continue

            gt_mask = get_category_mask(coco_gt, gt_ann_ids, cat_id, h, w)
            pred_mask = get_category_mask(coco_pred, pred_ann_ids, cat_id, h, w)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            cat_ious[cat_id].append(iou)

    # calculate the tooth id and its coressponding numbers that includes dental plaque 
    # for aa, bb in cat_ious.items():
    #     print(toothID_to_Number_Map[aa // 3], len(bb))

    # 计算每个类别的mIoU
    cat_mious = {toothID_to_Number_Map[cat_id // 3]: np.mean(ious) for cat_id, ious in cat_ious.items()}
    return cat_mious

def calculate_Sensitivity_Specificity_per_category(gt_json, pred_json, tooth_id_to_name):
    """
    Calculate Sensitivity, Specificity, and Accuracy for each tooth ID.
    
    Parameters:
        gt_json (dict): Ground truth dictionary, e.g., {(1, 0): 0, (3, 2): 3}.
        pred_json (dict): Prediction dictionary, e.g., {(1, 0): 0, (3, 2): 1}.
        tooth_id_to_name (dict): Mapping of tooth ID to tooth name.
        
    Returns:
        dict: A dictionary with metrics for each tooth ID.
    """
    from collections import defaultdict

    # Initialize metrics storage
    metrics = {tooth_id: {"Sensitivity": 0, "Specificity": 0, "Accuracy": 0} 
               for tooth_id in tooth_id_to_name.values()}
    
    # Initialize counters
    stats = defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    gt_tooth_number = set()
    # Iterate through the ground truth and predictions
    for (image_id, tooth_id), gt_label in gt_json.items():
        gt_tooth_number.add(tooth_id_to_name[tooth_id])
        # Merge classes 1, 2, 3 into 1, keep 0 as is
        gt_label = 0 if gt_label == 0 else 1
        pred_label = pred_json.get((image_id, tooth_id), 0)  # Default to 0 if missing
        pred_label = 0 if pred_label == 0 else 1

        # Update stats for the current tooth ID
        if gt_label == 1 and pred_label == 1:
            stats[tooth_id]["TP"] += 1
        elif gt_label == 0 and pred_label == 0:
            stats[tooth_id]["TN"] += 1
        elif gt_label == 0 and pred_label == 1:
            stats[tooth_id]["FP"] += 1
        elif gt_label == 1 and pred_label == 0:
            stats[tooth_id]["FN"] += 1

    # compute 
    # gt_tooth_number = list(gt_tooth_number)
    # gt_tooth_number.sort()
    # print(gt_tooth_number)

    # Calculate Sensitivity, Specificity, and Accuracy for each tooth ID
    for tooth_id, counts in stats.items():
        TP = counts["TP"]
        TN = counts["TN"]
        FP = counts["FP"]
        FN = counts["FN"]

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Specificity = TN / (TN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # Store metrics
        tooth_name = tooth_id_to_name.get(tooth_id, f"Tooth {tooth_id}")
        metrics[tooth_name] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Accuracy": accuracy,
        }

    return metrics

def draw_bar_charts(data):
    if 'doubleteeth' in data:
        data['DT'] = data.pop('doubleteeth')
    # 提取牙齿编号
    teeth_ids = list(data.keys())
    teeth_ids.remove('11')
    teeth_ids.remove('21')
    teeth_ids.remove('crown')
    

    # 提取每个指标对应的数值
    sensitivity_values = [data[tid]['Sensitivity'] for tid in teeth_ids]
    specificity_values = [data[tid]['Specificity'] for tid in teeth_ids]
    accuracy_values = [data[tid]['Accuracy'] for tid in teeth_ids]
    miou_values = [data[tid]['mIoU'] for tid in teeth_ids]
    
    # 设置参数
    x = np.arange(len(teeth_ids))
    width = 0.5
    
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
    
    # 创建4个子图
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    # 定义颜色
    colors = ['#c5e7ff', '#90EE90', '#FFB6C1', '#e1e1ff', '#FFD700'] # ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3']  # 适合专业风格的配色

    # 绘制Sensitivity
    axes[0].bar(x, sensitivity_values, width=width, color=colors[0], edgecolor='black', alpha=0.7, linewidth=1.2)
    axes[0].set_title('Sensitivity', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(teeth_ids, rotation=0)
    # axes[0].set_ylabel('Value')
    axes[0].set_ylim(0, 1)

    # 绘制Specificity
    axes[1].bar(x, specificity_values, width=width, color=colors[1], edgecolor='black', alpha=0.7, linewidth=1.2)
    axes[1].set_title('Specificity', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(teeth_ids, rotation=0)
    # axes[1].set_ylabel('Value')
    axes[1].set_ylim(0, 1)

    # 绘制Accuracy
    axes[2].bar(x, accuracy_values, width=width, color=colors[2], edgecolor='black', alpha=0.7, linewidth=1.2)
    axes[2].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(teeth_ids, rotation=0)
    # axes[2].set_ylabel('Value')
    axes[2].set_ylim(0, 1)

    # 绘制mIoU
    axes[3].bar(x, miou_values, width=width, color=colors[3], edgecolor='black', alpha=0.7, linewidth=1.2)
    axes[3].set_title('mIoU', fontsize=14, fontweight='bold')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(teeth_ids, rotation=0)
    # axes[3].set_ylabel('Value')
    axes[3].set_ylim(0, 1)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('dental_metrics_barcharts.png', dpi=300, bbox_inches='tight')

def save_grade_per_tooth_to_excel(gt_grades, pred_grades, gt_json, excel_path="output.xlsx"):
    # 创建一个 image_id 到 file_name 的映射
    image_id_to_filename = {image['id']: image['file_name'] for image in gt_json['images']}

    # 合并两个字典
    all_keys = set(gt_grades.keys()).union(pred_grades.keys())  # 获取所有 (image_id, tooth_number) 键

    data = []
    # 遍历所有键，获取预测值和真实值
    for key in all_keys:
        image_id, tooth_number = key
        file_name = image_id_to_filename.get(image_id, "Unknown")  # 如果找不到文件名则返回 "Unknown"
        pred_value = pred_grades.get(key, "N/A")  # 如果预测字典中没有值，用 "N/A" 表示
        gt_value = gt_grades.get(key, "N/A")  # 如果真实值字典中没有值，用 "N/A" 表示
        data.append([file_name, toothID_to_Number_Map[tooth_number], pred_value, gt_value])

    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data, columns=["file_name", "tooth_number", "prediction", "ground_truth"])

    # 保存为 Excel 文件
    df.to_excel(excel_path, index=False)

    print(f"数据已保存到 {excel_path}")


def calculate_ci_95_for_metrics(metrics_dict: dict, sample_size: int):
    """
    根据输入的字典计算多个指标的 95% 置信区间（CI）

    参数:
        metrics_dict (dict): 指标字典，键为指标名称，值为指标值（范围 0 到 1）。
        sample_size (int): 样本量 (n)。

    返回:
        dict: 每个指标对应的 95% 置信区间，格式为 {指标名称: (low, high)}。
    """
    ci_results = {}
    
    for metric_name, metric_value in metrics_dict.items():
        # 计算标准误差（SE）
        se = math.sqrt(metric_value * (1 - metric_value) / sample_size)
        
        # 计算 95% 置信区间
        z_score = 1.96  # 95% 的标准正态分布 Z 值
        lower_bound = metric_value - z_score * se
        upper_bound = metric_value + z_score * se

        # 保证置信区间在合理范围 [0, 1] 内
        lower_bound = max(0, lower_bound)
        upper_bound = min(1, upper_bound)

        # 保存结果
        ci_results[metric_name] = (lower_bound, upper_bound)

    return ci_results


def compute_metrics_for_box_detection(gt_json_path, pred_json_path, iou_threshold=0.5, confidence_threshold=0.3): # iou 0.3/0.7
    """
    计算目标检测的单点 Precision, Recall 和 F1 指标（修正版）。
    关键修正：按置信度降序处理预测框，确保高置信度优先匹配。
    """
    all_targets = torch.load(gt_json_path)
    all_predictions = torch.load(pred_json_path)

    # 初始化 MeanAveragePrecision 类
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(all_predictions, all_targets)
    result = metric.compute()
 
    # # 提取 IOU=0.5 时的平均 Precision
    precision_at_iou_50 = result['map_50']

    # 如果需要 PR 曲线，需要额外计算不同置信度阈值下的 precision 和 recall
    def compute_pr_curve_iou50(preds, targets, conf_steps=50):
        thresholds = torch.linspace(0, 1, conf_steps)
        precisions, recalls = [], []

        for conf_t in thresholds:
            TP, FP, FN = 0, 0, 0
            
            for pred, gt in zip(preds, targets):
                # 过滤低置信度预测
                mask = pred["scores"] >= conf_t
                boxes_pred = pred["boxes"][mask]
                labels_pred = pred["labels"][mask]
                
                boxes_gt = gt["boxes"]
                labels_gt = gt["labels"]

                matched_gt = set()
                
                for i, box_p in enumerate(boxes_pred):
                    label_p = labels_pred[i]
                    ious = box_iou(box_p[None, :], boxes_gt)[0]
                    
                    # 找到匹配的gt
                    max_iou, max_idx = ious.max(0)
                    if max_iou >= 0.5 and label_p == labels_gt[max_idx] and max_idx.item() not in matched_gt:
                        TP += 1
                        matched_gt.add(max_idx.item())
                    else:
                        FP += 1
                
                # 没匹配到的 GT 箱是 FN
                FN += len(boxes_gt) - len(matched_gt)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)

        del precisions[-1]
        del recalls[-1]

        return recalls, precisions

    # 获取 PR 数据
    recalls, precisions = compute_pr_curve_iou50(all_predictions, all_targets)

    # 绘制 PR 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve @IoU=0.5")
    plt.grid(True)
    plt.savefig("teeth_detection_pr_curve.png", dpi=300, bbox_inches='tight')
    print(f"Teeth detection PR Curve is saved in teeth_detection_pr_curve.png")

    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_item, pred_item in zip(all_targets, all_predictions):
        gt_boxes = gt_item['boxes']
        pred_boxes = pred_item['boxes']
        pred_scores = pred_item['scores']
        pred_labels = pred_item['labels']
        gt_labels = gt_item['labels']

        # 置信度过滤
        valid_indices = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[valid_indices]
        pred_scores = pred_scores[valid_indices]  # 保留过滤后的分数
        pred_labels = pred_labels[valid_indices]

        # 关键修正：按置信度降序排序
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        # 计算 IOU
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
        else:
            ious = torch.zeros((len(pred_boxes), len(gt_boxes)))

        # 初始化匹配状态
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        # 按置信度从高到低处理预测框
        for pred_idx in range(len(pred_boxes)):
            # 获取当前预测框对应的最大IOU及其索引
            max_iou, gt_idx = ious[pred_idx].max(0)
            gt_idx = gt_idx.item()
            
            # 检查匹配条件
            if max_iou >= iou_threshold and not matched_gt[gt_idx] and pred_labels[pred_idx] == gt_labels[gt_idx]:
                total_tp += 1
                matched_gt[gt_idx] = True
                # 移除已匹配的GT避免重复匹配
                ious[:, gt_idx] = -1  # 设为负值防止再次匹配
            else:
                total_fp += 1

        total_fn += torch.sum(~matched_gt).item()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 返回结果
    return {
        "map": result['map'],  # 总体 mAP
        "precision_at_iou_50": precision_at_iou_50,  # IOU=0.5 时的平均 Precision
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


# 使用示例
if __name__ == "__main__":
    gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/test_ins_ToI.json"
    pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_woboxtemp_noise0.1/saved_jsons/_pred_val_epoch_000_postprocessed.json" # '/home/jinghao/projects/dental_plague_detection/comparasive_methods/MP-Former/output/inference/coco_instances_results_score_over_0.50.json'
    
    box_gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_woboxtemp_noise0.1/saved_jsons/_box_gt_val_for_calculate_metrics.pt"
    box_pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_woboxtemp_noise0.1/saved_jsons/_box_pred_val_epoch_000_for_calculate_metrics.pt"
    
    # box_gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_gt_val_for_calculate_metrics.pt"
    # box_pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_pred_val_epoch_000_for_calculate_metrics.pt"
    print(compute_metrics_for_box_detection(box_gt_json_path, box_pred_json_path))
    
    # Ours '/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_wboxTemp/saved_jsons/_pred_val_epoch_000_postprocessed.json'
    # MaskDINO '/home/jinghao/projects/dental_plague_detection/MaskDINO/output/inference/coco_instances_results_score_over_0.50.json'
    # Mask2Former '/home/jinghao/projects/dental_plague_detection/comparasive_methods/Mask2Former/output/inference/coco_instances_results_score_over_0.50.json'
    # MaskRCNN '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/tools/output_maskrcnn_resizeShortEdge/inference/coco_instances_results_score_over_0.50.json'
    # '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/projects/PointSup/output/inference/coco_instances_results_score_over_0.50.json'
    # '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/tools/output/inference/coco_instances_results_score_over_0.50.json'
    # '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/tools/output/inference/coco_instances_results_score_over_0.50.json'
    # '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/projects/PointSup/output/inference/coco_instances_results_score_over_0.50.json'
    
    mask_mAP, mask_ap50, sensitivity, specificity = calculate_mAP_Sensitivity_Specificity(
        gt_json_path,
        pred_json_path,
    )
    
    with open(gt_json_path, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_path, "r") as f:
        pred_json = json.load(f)

    # pred文件可能没有images字段
    images = gt_json['images']
    id2size = load_image_id2size(images)

    gt_grades = compute_tooth_grades(gt_json['annotations'], id2size)
    # pred_json 只有annotations
    pred_grades = compute_tooth_grades(pred_json, id2size)

    # import pdb; pdb.set_trace()
    # print(gt_json)
    # print(gt_grades)
    # print(pred_grades)
    # save_grade_per_tooth_to_excel(gt_grades, pred_grades, gt_json, excel_path="output.xlsx")
    # statistic_plot_and_save_category_counts(gt_grades)
    print(calculate_Accuracy_Sensitivity_Specificity_metrics_three_classes_of_dental_plaque(gt_grades, pred_grades))

    accs = calculate_acc(gt_grades, pred_grades)

    sensitivity_per_tooth, specificity_per_tooth = calculate_tooth_level_Sensitivity_Specificity(gt_grades, pred_grades)

    acc_tooth, acc_image, acc_patient = accs['tooth_acc'], accs['image_acc'], accs['patient_acc']

    overall_mIoU_plaque = compute_for_dental_plaque_IoU(gt_json_path, pred_json_path)

    plaque_mIoU_per_category_dict = compute_mIoU_for_dental_plaque_IoU_per_category(gt_json_path, pred_json_path, toothID_to_Number_Map)

    plaque_Sensitivity_Specificity_per_category_dict = calculate_Sensitivity_Specificity_per_category(gt_grades, pred_grades, toothID_to_Number_Map)

    plaque_mIoU_Sensitivity_Specificity_per_category_dict = {}
    for k, v in plaque_Sensitivity_Specificity_per_category_dict.items():
        v['mIoU'] = plaque_mIoU_per_category_dict[k]
        plaque_mIoU_Sensitivity_Specificity_per_category_dict[k] = v
    
    draw_bar_charts(plaque_mIoU_Sensitivity_Specificity_per_category_dict)
    # print(plaque_mIoU_Sensitivity_Specificity_per_category_dict)
    # 生成表格
    header = "|{:^12}|{:^12}|{:^13}|{:^13}|{:^13}|{:^13}|{:^13}|{:^12}|{:^12}|{:^14}|".format(
        "mask_mAP", "mask_ap50", "Mask_Sen", "Mask_Sep", "mIoU", "Tooth_Sen", "Tooth_Sep", "acc_tooth", "acc_image", "acc_patient"
    )
    line = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*14 + "+"
    values = "|{:^12.4f}|{:^12.4f}|{:^13.4f}|{:^13.4f}|{:^13.4f}|{:^13.4f}|{:^13.4f}|{:^12.4f}|{:^12.4f}|{:^14.4f}|".format(
        mask_mAP, mask_ap50, sensitivity, specificity, overall_mIoU_plaque, sensitivity_per_tooth, specificity_per_tooth, acc_tooth, acc_image, acc_patient
    )
    
    print(line)
    print(header)
    print(line)
    print(values)
    print(line)
    
    # for 计算 C.I. 95%
    print('pred sample_size: ', len(pred_grades))
    print('gt sample_size: ', len(gt_grades))
    sample_size = len(pred_grades)  # 样本量
    metric_for_ci_95_dict = {
        'mIoU': overall_mIoU_plaque,
        'Sensitivity': sensitivity_per_tooth,
        'Specificity': specificity_per_tooth,
        'Accuracy': acc_tooth
    } 
    ci_95_results = calculate_ci_95_for_metrics(metric_for_ci_95_dict, sample_size)
    for metric, ci in ci_95_results.items():
        print(f"{metric} 95% 置信区间: {ci[0]:.4f} ~ {ci[1]:.4f}")