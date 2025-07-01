import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from collections import defaultdict

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

def calculate_mAP_Sensitivity_Specificity(gt_json_path, pred_json_path):
    # 初始化COCO验证器
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 计算mAP
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
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


# 使用示例
if __name__ == "__main__":
    gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test/test_ins_ToI.json"
    pred_json_path='/home/jinghao/projects/dental_plague_detection/MaskDINO/output_maskdino_dental_plaque_baseline_2025_May/inference/coco_instances_results_score_over_0.5.json'
    
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
    accs = calculate_acc(gt_grades, pred_grades)

    acc_tooth, acc_image, acc_patient = accs['tooth_acc'], accs['image_acc'], accs['patient_acc']

    # 生成表格
    header = "|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^14}|".format(
        "mask_mAP", "mask_ap50", "sensitivity", "specificity", "acc_tooth", "acc_image", "acc_patient"
    )
    line = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*14 + "+"
    values = "|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^14.4f}|".format(
        mask_mAP, mask_ap50, sensitivity, specificity, acc_tooth, acc_image, acc_patient
    )

    print(line)
    print(header)
    print(line)
    print(values)
    print(line)
    