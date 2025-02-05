import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassJaccardIndex

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time 

# 全局颜色映射字典
label_to_color = {}

# 预定义 70 种颜色 (使用 matplotlib 的 tab20 颜色表)
colors = plt.cm.tab20.colors  # 20 种颜色
colors += plt.cm.tab20b.colors  # 再加 20 种颜色
colors += plt.cm.tab20c.colors  # 再加 20 种颜色
colors += plt.cm.tab20.colors[:10]  # 再加 10 种颜色，总共 70 种

# 将颜色转换为 0-255 范围的 RGB 值
colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

# 可视化函数
def visualize_and_save_mask(mask, save_path):
    """
    可视化并保存 mask
    :param mask: 输入的 mask (torch.Tensor, 1024x1024)
    :param save_path: 保存路径
    """
    # 将 mask 转换为 numpy 数组
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # 使用 matplotlib 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_np, cmap='gray')  # 使用灰度图显示
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()
    print(f'The visualization of mask is saved in {save_path}.')

def visualize_and_save_masks_for_instance(input_dict, save_path):

    """
    将所有 masks 画在一张图像上并保存
    :param input_dict: 包含 masks, scores, labels 的字典
    :param save_path: 保存图像的路径
    """
    masks = input_dict["masks"].cpu()  # (N, H, W)
    labels = input_dict["labels"].cpu()  # (N,)
    if "scores" not in input_dict.keys():
        scores = torch.ones_like(labels)
    else:
        scores = input_dict["scores"].cpu()  # (N,)
    
    # 创建一个空白画布 (H, W, 3) 用于叠加 mask
    canvas = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    # 遍历每个 mask
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        # 将 mask 转换为 numpy 数组
        mask_np = mask.numpy()

        # 如果当前 label 还没有分配颜色，则分配一个颜色
        if label.item() not in label_to_color:
            label_to_color[label.item()] = colors[len(label_to_color) % len(colors)]

        # 获取当前 label 对应的颜色
        color = label_to_color[label.item()]

        # 将 mask 叠加到画布上
        canvas[mask_np == 1] = color

    # 可视化并保存
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis('off')  # 关闭坐标轴

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Label: {label}, Score: {score.item():.2f}", 
                   markerfacecolor=np.array(label_to_color[label.item()]) / 255, markersize=10)
        for label, score in zip(labels, scores)
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'The visualization of mask is saved in {save_path}.')

def compute_intersection(box1, box2):
    """
    计算两个矩形的交并比 (IoU)。
    box1 和 box2 的格式为 [x_min, y_min, x_max, y_max]。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # union = box1_area + box2_area - intersection

    return intersection / box1_area

class InstanceSegmentationMetric:
    """
    计算实例分割的指标（AP 和 mIoU）。
    支持在循环中累加样本，最终计算整个数据集的指标。
    """

    def __init__(self, num_box_classes, num_mask_classes, device="cpu"):
        """
        初始化指标计算器。

        :param num_box_classes: int，box 的类别总数（包括背景）。
        :param num_mask_classes: int，mask 的类别总数（包括背景）。
        :param device: str，计算设备（"cpu" 或 "cuda"）。
        """
        self.num_box_classes = num_box_classes
        self.num_mask_classes = num_mask_classes
        self.num_instance_classes = num_box_classes * num_mask_classes  # 实例类别总数
        self.device = device
        self.all_instance_preds = []
        self.all_instance_targets = []
        # 创建 torchmetrics 的 AP 和 mIoU 计算器
        self.mean_ap_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True).to(device)

    def update(self, pred_boxes, target_boxes, pred_masks, target_masks):
        """
        更新指标计算器。

        :param pred_boxes: list[dict]，预测的 box 信息，格式为 [{"boxes": tensor, "scores": tensor, "labels": tensor}, ...]
        :param target_boxes: list[dict]，目标的 box 信息，格式为 [{"boxes": tensor, "labels": tensor}, ...]
        :param pred_masks: tensor; 预测的语义分割 mask, 形状为 (N, H, W), N 是 batch size。
        :param target_masks: tensor; 目标的语义分割 mask, 形状为 (N, H, W), N 是 batch size。
        """
        if len(pred_boxes) != len(target_boxes):
            raise ValueError(f"pred_boxes 和 target_boxes 的长度不一致: pred_boxes={len(pred_boxes)}, target_boxes={len(target_boxes)}")
        
        # 将输入移动到指定设备
        pred_masks = pred_masks.to(self.device)
        target_masks = target_masks.to(self.device)
        
        all_instance_preds = []
        all_instance_targets = []

        for pred_box, target_box, pred_mask, target_mask in zip(pred_boxes, target_boxes, pred_masks, target_masks):

            # for check by visualizing the pred_mask and target_mask
            # visualize_and_save_mask(pred_mask.cpu(), 'pred.png')
            # visualize_and_save_mask(target_mask.cpu(), 'gt.png')

            # 获取预测与目标的 box 和类别信息
            pred_boxes_tensor, pred_scores, pred_labels = pred_box['boxes'].to(self.device), pred_box['scores'].to(self.device), pred_box['labels'].to(self.device)
            target_boxes_tensor, target_labels = target_box['boxes'].to(self.device), target_box['labels'].to(self.device)
            
            H, W = pred_mask.shape[-2:]  # 获取 mask 的高度和宽度

            # 初始化实例掩码和实例类别列表
            pred_instance_masks = []
            pred_instance_labels = []
            pred_instance_scores = []
            target_instance_masks = []
            target_instance_labels = []

            # 遍历预测的 box，为每个 box 生成实例 mask
            for i, (box, box_score, box_label) in enumerate(zip(pred_boxes_tensor, pred_scores, pred_labels)):
                if box_label == 29 or box_score < 0.5: # for filtering the 'Crown'
                    continue
                x1, y1, x2, y2 = box.int()  # 获取 box 的整数边界
                cropped_mask = pred_mask[y1:y2, x1:x2]  # 裁剪出 mask 区域

                # 遍历裁剪出的 mask 中的类别
                for mask_label in torch.unique(cropped_mask):
                    pred_instance_mask = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
                    if mask_label == 0:  # 跳过背景
                        continue
                    # 计算实例类别 ID
                    instance_label = box_label * self.num_mask_classes + mask_label
                    # 生成实例掩码
                    crop_instance_mask = (cropped_mask == mask_label).to(torch.uint8)
                    pred_instance_mask[y1:y2, x1:x2] = crop_instance_mask
                    # 将实例掩码和类别存储
                    pred_instance_masks.append(pred_instance_mask)
                    pred_instance_labels.append(instance_label)
                    pred_instance_scores.append(box_score)
                    
                # for mask_label in torch.unique(cropped_mask):  # 遍历区域内的 mask 类别
                #     if mask_label == 0:  # 跳过背景
                #         continue
                #     instance_label = box_label * self.num_mask_classes + mask_label  # 计算实例类别 ID
                #     pred_instance_mask[i, y1:y2, x1:x2] = (cropped_mask == mask_label).int() * instance_label

            # 遍历目标的 box，为每个 box 生成实例 mask
            for i, (box, box_label) in enumerate(zip(target_boxes_tensor, target_labels)):
                if box_label == 29: # for filtering the 'Crown'
                    continue
                x1, y1, x2, y2 = box.int()  # 获取 box 的整数边界
                cropped_mask = target_mask[y1:y2, x1:x2]  # 裁剪出 mask 区域
                for mask_label in torch.unique(cropped_mask):  # 遍历区域内的 mask 类别
                    target_instance_mask = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
                    if mask_label == 0:  # 跳过背景
                        continue
                    # 计算实例类别 ID
                    instance_label = box_label * self.num_mask_classes + mask_label
                    # 生成实例掩码
                    crop_instance_mask = (cropped_mask == mask_label).to(torch.uint8)
                    target_instance_mask[y1:y2, x1:x2] = crop_instance_mask
                    # 将实例掩码和类别存储
                    target_instance_masks.append(target_instance_mask)
                    target_instance_labels.append(instance_label)

            if not pred_instance_masks or not target_instance_masks:
                continue

            # 将结果存储为 torchmetrics 的输入格式
            if pred_instance_masks:
                all_instance_preds.append({
                    "masks": torch.stack(pred_instance_masks),
                    "scores": torch.stack(pred_instance_scores),
                    "labels": torch.stack(pred_instance_labels),
                })
            if target_instance_masks:
                all_instance_targets.append({
                    "masks": torch.stack(target_instance_masks),
                    "labels": torch.stack(target_instance_labels),
                })

            # visualize_and_save_masks_for_instance(all_instance_preds[-1], save_path='all_masks_pred.png')
            # visualize_and_save_masks_for_instance(all_instance_targets[-1], save_path='all_masks_gt.png')
        
        for item in all_instance_preds:
            visualize_and_save_masks_for_instance(item, save_path=f'/home/jinghao/projects/dental_plague_detection/ins_gt_vis/all_masks_gt_{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))}.png')
        self.mean_ap_metric.update(all_instance_preds, all_instance_targets)
        # 释放不再需要的变量
        del pred_masks, target_masks, all_instance_preds, all_instance_targets
        torch.cuda.empty_cache()


    def compute(self):
        """
        计算累积的指标。

        :return: dict，包含 AP 和 mIoU。
        """
        # 计算 AP
        # self.mean_ap_metric.update(self.all_instance_preds, self.all_instance_targets)
        ap_results = self.mean_ap_metric.compute()

        return {"ap": ap_results}

    def reset(self):
        """
        重置指标计算器。
        """
        self.mean_ap_metric.reset()