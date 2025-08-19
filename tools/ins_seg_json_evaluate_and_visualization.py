import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.measure import find_contours
import cv2
from PIL import Image

 # 定义颜色映射，每个类别有一个固定颜色
np.random.seed(0)  # 确保颜色一致
num_classes = 90
global_colors_list = np.random.randint(0, 255, (num_classes, 3))  # 生成 90 个随机颜色

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
    
    # if True:
    #     for ann in coco_pred.dataset['annotations']:
    #         ann['category_id'] = 1  # 将所有类别 ID 设置为 1
    #     for ann in coco_gt.dataset['annotations']:
    #         ann['category_id'] = 1  # 将所有类别 ID 设置为 1

    # 计算mAP
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 可视化函数
    def visualize_masks(image_id):
        # import pdb; pdb.set_trace()
        # 加载图片
        img_info = coco_gt.loadImgs(image_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print('img_path:', img_path)
        print('img_id:', image_id)
        print('img_shape:', img.shape)

        # if "10_8_26" not in img_path:
        #     return

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
        # print('anno_ids:', ann_ids)
        annotations = coco_gt.loadAnns(ann_ids)
        for ann in annotations:
            mask = coco_gt.annToMask(ann)
            category_id = ann['category_id']
            color = global_colors_list[category_id % num_classes]  # 从颜色表中选择颜色
            # color = np.random.rand(3) * 255  # 随机颜色
            # print('mask_shape:', mask.shape)
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
            # color = np.array([1.0, 0.0, 0.0])  # 红色表示预测
            category_id = ann['category_id']
            color = global_colors_list[category_id % num_classes]  # 使用与 GT 相同类别的颜色
            contours = find_contours(mask, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color/255)
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
        gt_json_path="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/test_ins_ToI.json",
        pred_json_path='/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_wboxTemp/saved_jsons/_pred_val_epoch_000_postprocessed_overlap.json',
        image_dir="/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/JPEGImages",
        output_dir="/home/jinghao/projects/dental_plague_detection/Self-PPD/vis_ins_seg_output_ours_sen0.74/"
    )
