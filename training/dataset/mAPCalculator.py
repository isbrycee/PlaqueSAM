import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


# 定义一个类来收集数据和计算 mAP
class MAPCalculator:
    def __init__(self, class_agnostic=False):
        self.class_agnostic =class_agnostic
        # 初始化 mAP 计算器
        self.metric = MeanAveragePrecision(class_metrics=True)
        # 存储所有预测和标注信息
        self.all_predictions = []
        self.all_targets = []
        self.target_labels = []
        self.pred_labels = []
        class_name_to_idx_map = {'51':0, '52':1, '53':2, '54':3, '55':4, 
                            '61':5, '62':6, '63':7, '64':8, '65':9, 
                            '71':10, '72':11, '73':12, '74':13, '75':14,
                            '81':15, '82':16, '83':17, '84':18, '85':19,

                            '11': 20, '16': 21,
                            '21': 22, '26': 23,
                            '31': 24, '36': 25,
                            '41': 26, '46': 27,
                            
                            'doubleteeth': 28,
                            'crown': 29,
                            }
        self.class_idx_to_name_map = dict(zip(class_name_to_idx_map.values(), class_name_to_idx_map.keys()))
        
    def update(self, preds, targets, img_batch, video_name, is_visualized=False):
        """
        更新每个 batch 的预测和标注。
        :param preds: 预测列表，包含每个样本的字典 (e.g., [{"boxes": tensor, "scores": tensor, "labels": tensor}, ...])
        :param targets: 标注列表，包含每个样本的字典 (e.g., [{"boxes": tensor, "labels": tensor}, ...])
        """
        # 忽略类别标签，将所有类别视为同一类
        if self.class_agnostic:
            for pred in preds:
                pred["labels"] = torch.zeros_like(pred["labels"])  # 将预测框的类别设置为 0
            for target in targets:
                target["labels"] = torch.zeros_like(target["labels"])  # 将标注框的类别设置为 0
        
        for pred in preds:
            self.pred_labels += pred["labels"].cpu().numpy().tolist()  # 将预测框的类别设置为 0
        for target in targets:
            self.target_labels += target["labels"].cpu().numpy().tolist()  # 将预测框的类别设置为 0

        self.all_predictions.extend(preds)
        self.all_targets.extend(targets)

        if is_visualized:
            output_dir = '/home/jinghao/projects/dental_plague_detection/Self-PPD/bad_visual/'
            self.visualize_and_save(img_batch, output_dir, video_name.replace('/', '_'), preds, targets, self.class_idx_to_name_map, max_images=10000)

    def compute(self):
        """
        计算整个数据集的 mAP。
        """
        self.metric.update(self.all_predictions, self.all_targets)
        # print(set(self.pred_labels))
        # print(set(self.target_labels))
        return self.metric.compute()
    
    def denormalize(self, tensor, mean, std):
        """
        对归一化后的图像进行反归一化，恢复原始像素值范围。

        Args:
            tensor (torch.Tensor): 被归一化的图像张量，形状 (C, H, W)。
            mean (list or tuple): 每个通道的均值。
            std (list or tuple): 每个通道的标准差。

        Returns:
            torch.Tensor: 反归一化后的图像张量，形状 (C, H, W)。
        """
        device = tensor.device
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)  # 调整形状以广播
        std = torch.tensor(std).view(-1, 1, 1).to(device)    # 调整形状以广播
        return tensor * std + mean

    def visualize_and_save(
        self, 
        images,  # 输入图像张量列表 (shape: [C, H, W])
        output_dir: str, 
        video_name: str,
        preds: list, 
        targets: list, 
        class_names: list = None,  # 可选的类别名称列表
        max_images: int = 10,       # 最多保存多少张图像
    ):
        """
        可视化预测框和真实框，并保存到指定目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, (image, pred, target) in enumerate(zip(images, preds, targets)):
            if idx >= max_images:  # 控制最大保存数量
                break

            # 将图像转换为 uint8 格式
            if image.is_floating_point():
                image = self.denormalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).squeeze(0)
                # image = (image * 255).to(torch.uint8).squeeze(0)
            
            # ---- 绘制真实框 ----
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()
            
            # 处理 class-agnostic 标签
            if self.class_agnostic:
                gt_labels = [0] * len(gt_labels)
            elif class_names is not None:  # 如果有类别名称
                gt_labels = [class_names[label.item()] for label in gt_labels]
            else:
                gt_labels = [f"Class {label}" for label in gt_labels]
            
            # 在图像上绘制真实框 (绿色)
            gt_image = draw_bounding_boxes(
                image, 
                boxes=gt_boxes,
                labels=gt_labels,
                colors="green",
                font='/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf',
                font_size=30,
                width=2
            )
            
            # ---- 绘制预测框 ----
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu().tolist()
            pred_labels = pred["labels"].cpu()
            
            # 处理 class-agnostic 标签
            if self.class_agnostic:
                pred_labels = [0] * len(pred_labels)
            elif class_names is not None:
                pred_labels = [class_names[label.item()] for label in pred_labels]
            else:
                pred_labels = [f"Class {label}" for label in pred_labels]
            
            # 组合标签和置信度
            pred_labels_with_score = [
                f"{label} ({score:.2f})" 
                for label, score in zip(pred_labels, pred_scores)
            ]
            
            # 在图像上绘制预测框 (红色)
            pred_image = draw_bounding_boxes(
                image, 
                boxes=pred_boxes,
                labels=pred_labels_with_score,
                font='/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf',
                font_size=24,
                colors="red",
                width=2
            )
            
            # ---- 合并图像并保存 ----
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            
            # 显示真实框
            ax[0].imshow(to_pil_image(gt_image))
            ax[0].set_title("Ground Truth")
            ax[0].axis("off")
            
            # 显示预测框
            ax[1].imshow(to_pil_image(pred_image))
            ax[1].set_title("Predictions")
            ax[1].axis("off")
            
            # 保存图像
            output_path = os.path.join(output_dir, f"vis_{video_name}_{idx}.png")

            if not set(pred_labels) == set(gt_labels):
                plt.savefig(output_path, bbox_inches="tight")

            plt.close()