import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# 定义一个类来收集数据和计算 mAP
class MAPCalculator:
    def __init__(self, class_agnostic=False):
        self.class_agnostic =class_agnostic
        # 初始化 mAP 计算器
        self.metric = MeanAveragePrecision()
        # 存储所有预测和标注信息
        self.all_predictions = []
        self.all_targets = []
        
    def update(self, preds, targets):
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
        # print('label')
        # for pred in preds:
        #     print(pred["labels"])
        #     print(pred["boxes"])
        # print('target')
        # for target in targets:
        #     print(target["labels"])
        #     print(target["boxes"])
        self.all_predictions.extend(preds)
        self.all_targets.extend(targets)

    def compute(self):
        """
        计算整个数据集的 mAP。
        """
        self.metric.update(self.all_predictions, self.all_targets)
        return self.metric.compute()