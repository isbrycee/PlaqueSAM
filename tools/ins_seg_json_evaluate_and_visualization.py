import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.measure import find_contours
import cv2

def calculate_and_visualize_map(gt_json_path, pred_json_path, image_dir, output_dir):
    """
    计算mAP并可视化GT和预测的mask
    
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
    
    # 可视化函数
    def visualize_masks(image_id):
        # 加载图片
        img_info = coco_gt.loadImgs(image_id)[0]
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
        print('img_path:', img_path)
        print('img_id:', image_id)
        print('img_shape:', img.shape)
        # 绘制GT mask
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        print('anno_ids:', ann_ids)
        annotations = coco_gt.loadAnns(ann_ids)
        for ann in annotations:
            mask = coco_gt.annToMask(ann)
            color = np.random.rand(3) * 255  # 随机颜色
            print('mask_shape:', mask.shape)
            overlay[mask == 1] = color
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color/255)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # 绘制预测mask
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        ann_ids = coco_pred.getAnnIds(imgIds=image_id)
        annotations = coco_pred.loadAnns(ann_ids)
        for ann in annotations:
            mask = coco_pred.annToMask(ann)
            color = np.array([1.0, 0.0, 0.0])  # 红色表示预测
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)
        plt.title('GT (Random Colors) vs Predictions (Red)')
        
        # 保存结果
        output_path = os.path.join(output_dir, f"vis_{img_info['file_name'].replace('/', '_')}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    # 对所有图片进行可视化
    for img_id in coco_gt.getImgIds():
        visualize_masks(img_id)

# 使用示例
if __name__ == "__main__":
    calculate_and_visualize_map(
        gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/temp_gt.json",
        pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/temp_predictions.json",
        image_dir="/home/jinghao/projects/dental_plague_detection/dataset/temp_debug/test/JPEGImages",
        output_dir="/home/jinghao/projects/dental_plague_detection/Self-PPD/ins_seg_output"
    )
