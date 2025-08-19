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


# ===================================================
# 计算 PR 曲线 (IoU=0.5)
# ===================================================

def compute_metrics_for_box_detection(gt_json_path, pred_json_path_wTemp, pred_json_path_woTemp, iou_threshold=0.3, confidence_threshold=0.0): # iou 0.3/0.7
    """
    计算目标检测的单点 Precision, Recall 和 F1 指标（修正版）。
    关键修正：按置信度降序处理预测框，确保高置信度优先匹配。
    """
    all_targets = torch.load(gt_json_path)
    all_predictions_wTemp = torch.load(pred_json_path_wTemp)
    all_predictions_woTemp = torch.load(pred_json_path_woTemp)

    # 初始化 MeanAveragePrecision 类
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(all_predictions_wTemp, all_targets)
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
        thresholds_list = thresholds.tolist()
        del thresholds_list[-1]
        return thresholds_list, recalls, precisions

    # ===================================================
    # 计算两组 PR 曲线
    # ===================================================
    thresholds1, recalls1, precisions1 = compute_pr_curve_iou50(all_predictions_wTemp, all_targets)
    thresholds2, recalls2, precisions2 = compute_pr_curve_iou50(all_predictions_woTemp, all_targets)

    # ===================================================
    # 找出 Model 1 同时优于 Model 2 的阈值
    # ===================================================
    better_thresholds = []
    for i, t in enumerate(thresholds1):
        if recalls1[i] > recalls2[i] and precisions1[i] > precisions2[i]:
            better_thresholds.append({
                "threshold": t,
                "recall1": recalls1[i],
                "precision1": precisions1[i],
                "recall2": recalls2[i],
                "precision2": precisions2[i]
            })

    # 打印结果
    print("Model 1 同时优于 Model 2 的置信度阈值点：")
    for item in better_thresholds:
        print(
            f"阈值: {item['threshold']:.2f} | "
            f"R1: {item['recall1']:.3f} P1: {item['precision1']:.3f} || "
            f"R2: {item['recall2']:.3f} P2: {item['precision2']:.3f}"
        )

    # ===================================================
    # 绘制对比曲线
    # ===================================================
    plt.figure(figsize=(6, 5))
    plt.plot(recalls1, precisions1, marker="o", label="Model 1")
    plt.plot(recalls2, precisions2, marker="s", label="Model 2")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve @IoU=0.5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("teeth_detection_pr_curve.png", dpi=300, bbox_inches='tight')
    print(f"Teeth detection PR Curve is saved in teeth_detection_pr_curve.png")


    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_item, pred_item in zip(all_targets, all_predictions_wTemp):
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
    pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/exps_FINAL/FINAP_PlaqueSAM_logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_pred_val_epoch_000_postprocessed.json" # '/home/jinghao/projects/dental_plague_detection/comparasive_methods/MP-Former/output/inference/coco_instances_results_score_over_0.50.json'
    
    box_gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/exps_FINAL/FINAP_PlaqueSAM_logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_gt_val_for_calculate_metrics.pt"
    box_pred_json_path_wTemp="/home/jinghao/projects/dental_plague_detection/Self-PPD/exps_FINAL/FINAP_PlaqueSAM_logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_wboxTemp/saved_jsons/_box_pred_val_epoch_000_for_calculate_metrics.pt"
    box_pred_json_path_woTemp="/home/jinghao/projects/dental_plague_detection/Self-PPD/exps_FINAL/FINAP_PlaqueSAM_logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_pred_val_epoch_000_for_calculate_metrics.pt"

    # box_gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_gt_val_for_calculate_metrics.pt"
    # box_pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_woboxTemp/saved_jsons/_box_pred_val_epoch_000_for_calculate_metrics.pt"
    print(compute_metrics_for_box_detection(box_gt_json_path, box_pred_json_path_wTemp, box_pred_json_path_woTemp))