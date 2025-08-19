from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import json
from tqdm import tqdm  # 进度条工具，可选

# # 加载你的原始预测结果
# with open(maskdino_pred_json_path) as f:
#     your_data = json.load(f)

# # 降低置信度阈值到0.1（保留更多预测）
# filtered_preds = [ann for ann in your_data if ann["score"] > 0.5]
# print(f"调整后预测数量: {len(filtered_preds)}")


def box_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    box格式: [x1, y1, w, h] → 转换为 [x1, y1, x2, y2]
    """
    # 转换格式
    b1 = np.array([box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]])
    b2 = np.array([box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]])
    
    # 计算交集区域
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集区域
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def nms_per_image_category(preds, iou_thresh=0.5, score_thresh=0.001):
    """
    单张图像单个类别的NMS处理
    :param preds: 该图像该类别下的预测列表，每个元素是包含'bbox'和'score'的字典
    :return: 保留的预测索引列表
    """
    # 按置信度从高到低排序
    sorted_indices = sorted(range(len(preds)), key=lambda i: preds[i]['score'], reverse=True)
    keep = []
    
    while sorted_indices:
        current_idx = sorted_indices.pop(0)
        current_pred = preds[current_idx]
        
        # 先过滤低置信度（根据实际需要可调整）
        if current_pred['score'] < score_thresh:
            continue
            
        keep.append(current_pred)
        
        # 计算与剩余框的IoU
        to_remove = []
        for idx in sorted_indices:
            iou = box_iou(current_pred['bbox'], preds[idx]['bbox'])
            if iou > iou_thresh:
                to_remove.append(idx)
                
        # 移除被抑制的框
        sorted_indices = [i for i in sorted_indices if i not in to_remove]
    
    return keep

def apply_nms_to_json(input_path, output_path, iou_thresh=0.5, score_thresh=0.001):
    """
    主函数：对整个JSON文件应用NMS
    """
    with open(input_path) as f:
        data = json.load(f)
    
    # 按image_id和category_id分组
    grouped = {}
    for ann in data:
        key = (ann['image_id'], ann['category_id'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(ann)
    
    # 处理每个分组
    new_annotations = []
    for key in tqdm(grouped, desc="Processing NMS"):
        image_preds = grouped[key]
        keep_preds = nms_per_image_category(image_preds, iou_thresh, score_thresh)
        new_annotations.extend(keep_preds)
    
    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(new_annotations, f)
    
    print(f"NMS完成！原始预测数: {len(data)}, 处理后预测数: {len(new_annotations)}")

# 使用示例（调整参数）
# apply_nms_to_json(
#     input_path="/home/jinghao/projects/dental_plague_detection/MaskDINO/output_maskdino_r50_iter8w_dental_plague_ins_seg/inference/coco_instances_results.json",
#     output_path="/home/jinghao/projects/dental_plague_detection/MaskDINO/output_maskdino_r50_iter8w_dental_plague_ins_seg/inference/coco_instances_results_over_0.3_nms_0.5.json",
#     iou_thresh=0.5,    # 建议尝试0.5-0.7
#     score_thresh=0.3  # 保留所有预测参与NMS（后续可再过滤）
# )

gt_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_1024_50e_lr1e-4_bblr5e-5_wd_4scales_twostage_100queries_test_512_ToI/saved_jsons/_gt_val.json"
our_pred_json_path="/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_1024_50e_lr1e-4_bblr5e-5_wd_4scales_twostage_100queries_test_512_ToI/saved_jsons/_pred_val_epoch_200.json"
maskdino_pred_json_path='/home/jinghao/projects/dental_plague_detection/MaskDINO/output/inference/coco_instances_results_score_over_0.5.json'

# 加载GT
coco_gt = COCO(gt_json_path)

# 加载两个预测结果
coco_dt1 = coco_gt.loadRes(our_pred_json_path)
coco_dt2 = coco_gt.loadRes(maskdino_pred_json_path)

# 评估不同IoU阈值
for iou_thr in [0.5, 0.75, 0.95]:
    print(f"\nEvaluating at IoU={iou_thr}")
    coco_eval1 = COCOeval(coco_gt, coco_dt1, 'segm')
    coco_eval1.params.iouThrs = [iou_thr]
    coco_eval1.evaluate()
    coco_eval1.accumulate()
    coco_eval1.summarize()
    
    coco_eval2 = COCOeval(coco_gt, coco_dt2, 'segm')
    coco_eval2.params.iouThrs = [iou_thr]
    coco_eval2.evaluate()
    coco_eval2.accumulate()
    coco_eval2.summarize()

# 按类别分析

categories = coco_gt.loadCats(coco_gt.getCatIds())

for cat in categories:
    cat_id = cat['id']
    print(f"\nCategory: {cat['name']}")
    
    # 你的方法
    coco_eval1 = COCOeval(coco_gt, coco_dt1, 'segm')
    coco_eval1.params.catIds = [cat_id]
    coco_eval1.evaluate()
    coco_eval1.accumulate()
    coco_eval1.summarize()
    
    # 其他方法
    coco_eval2 = COCOeval(coco_gt, coco_dt2, 'segm')
    coco_eval2.params.catIds = [cat_id]
    coco_eval2.evaluate()
    coco_eval2.accumulate()
    coco_eval2.summarize()

print(f"gt 数量: {len(coco_gt.dataset['annotations'])}")
print(f"Your method预测数量: {len(coco_dt1.dataset['annotations'])}")
print(f"Other method预测数量: {len(coco_dt2.dataset['annotations'])}")

import numpy as np
import matplotlib.pyplot as plt

# 从预测结果中提取置信度
your_scores = [ann['score'] for ann in coco_dt1.dataset['annotations']]
other_scores = [ann['score'] for ann in coco_dt2.dataset['annotations']]

plt.hist(your_scores, bins=50, alpha=0.5, label='Your Method')
plt.hist(other_scores, bins=50, alpha=0.5, label='MaskDNIO')
plt.legend()
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Confidence Score Distribution')
plt.savefig('Confidence_dist.jpg')