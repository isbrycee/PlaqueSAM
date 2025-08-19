import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.measure import find_contours
import cv2

# 定义颜色映射，每个类别有一个固定颜色
np.random.seed(0)  # 确保颜色一致
num_classes = 90
global_colors_list = np.random.randint(0, 255, (num_classes, 3))  # 生成 90 个随机颜色

def calculate_and_visualize_map(gt_json_path, pred_json_path, image_dir, output_dir):
    """
    只可视化有预测错误和漏预测图片的mAP计算和mask对比
    
    参数:
        gt_json_path (str): COCO格式GT标注文件路径
        pred_json_path (str): 预测结果JSON文件路径
        image_dir (str): 图片文件夹路径
        output_dir (str): 可视化结果保存路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化COCO验证器
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # 计算mAP
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 获取有错误预测和漏预测的图片ID
    error_image_ids = set()
    for eval_img in coco_eval.evalImgs:
        if eval_img is not None:  # 跳过没有评估结果的图片
            # FP: 检查预测中未匹配的部分
            dt_matches = eval_img['dtMatches']  # 预测匹配的结果
            num_predictions = len(dt_matches[0]) if 'dtMatches' in eval_img else 0
            num_false_positives = num_predictions - np.sum(dt_matches)  # 未匹配上的预测
            
            # FN: 检查 GT 中未匹配的部分
            gt_matches = eval_img['gtMatches']  # GT 匹配的结果
            num_gt = len(gt_matches[0]) if 'gtMatches' in eval_img else 0
            num_false_negatives = num_gt - np.sum(gt_matches)  # 未匹配上的 GT
            
            # 如果有 FP 或 FN，则将图片 ID 添加到错误列表
            if num_false_positives > 0 or num_false_negatives > 0:
                error_image_ids.add(eval_img['image_id'])
    
    # 可视化函数
    def visualize_masks(image_id):
        # 加载图片
        
        img_info = coco_gt.loadImgs(int(image_id))[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建画布
        plt.figure(figsize=(20, 10))
        
        # 绘制原图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # 准备标注可视化
        overlay = img.copy()
        alpha = 0.4  # 透明度
        
        # 绘制GT mask
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        annotations = coco_gt.loadAnns(ann_ids)
        for ann in annotations:
            mask = coco_gt.annToMask(ann)
            category_id = ann['category_id']
            color = global_colors_list[category_id % num_classes]  # 从颜色表中选择颜色
            overlay[mask == 1] = color
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color / 255)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # 绘制预测mask
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        ann_ids = coco_pred.getAnnIds(imgIds=image_id)
        annotations = coco_pred.loadAnns(ann_ids)
        for ann in annotations:
            mask = coco_pred.annToMask(ann)
            category_id = ann['category_id']
            color = global_colors_list[category_id % num_classes]  # 使用与 GT 相同类别的颜色
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color / 255)
        plt.title('GT (Random Colors) vs Predictions (Red)')
        
        # 保存结果
        output_path = os.path.join(output_dir, f"vis_{img_info['file_name'].replace('/', '_')}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    # 仅对有错误预测和漏预测的图片进行可视化
    for img_id in error_image_ids:
        # import pdb; pdb.set_trace()
        visualize_masks(img_id)

# 使用示例
if __name__ == "__main__":
    calculate_and_visualize_map(
        gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test/test_ins_ToI.json",
        pred_json_path='/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval/saved_jsons/_pred_val_epoch_000_postprocessed.json',
        image_dir="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test/JPEGImages",
        output_dir="/home/jinghao/projects/dental_plague_detection/Self-PPD/vis_ins_seg_output_ours_bad_case/"
    )