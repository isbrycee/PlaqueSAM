import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassJaccardIndex

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time 
import json
import torchvision.transforms.functional as F
from torch.nn import functional

from pycocotools import mask as mask_utils
from skimage.measure import find_contours
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import random
from pycocotools.cocoeval import maskUtils
from torchvision.transforms import InterpolationMode
from training.utils import box_ops
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

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

            # 遍历 gt 的 box，为每个 box 生成实例 mask
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
        
        # for item in all_instance_preds:
        #     visualize_and_save_masks_for_instance(item, save_path=f'/home/jinghao/projects/dental_plague_detection/ins_gt_vis/all_masks_gt_{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))}.png')
        
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


class Postprocessor_for_Instance_Segmentation:
    """
    计算实例分割的指标（AP 和 mIoU）。
    支持在循环中累加样本，最终计算整个数据集的指标。
    """

    def __init__(self, gt_json_path, num_mask_classes, saved_jsons_dir, val_ins_seg_class_agnostic, image_size, threshold_for_masks=0.5, device="cpu"):
        """
        初始化指标计算器。

        :param num_box_classes: int，box 的类别总数（包括背景）。
        :param num_mask_classes: int，mask 的类别总数（包括背景）。
        :param device: str，计算设备（"cpu" 或 "cuda"）。
        """
        self.device = device
        self.num_mask_classes = num_mask_classes
        self.gt_json_path = gt_json_path
        self.saved_jsons_dir = saved_jsons_dir
        self.val_ins_seg_class_agnostic = val_ins_seg_class_agnostic

        with open(gt_json_path, "r") as f:
            coco_data = json.load(f)
            self.coco_data = coco_data
        
        self.filename_to_image_id = self._extract_filename_to_image_id_mapping(self.coco_data)
        self.pred_ins = []
        self.image_size = image_size
        self.num_classes = 3 # TODO
        self.threshold_for_masks = threshold_for_masks

    def _extract_filename_to_image_id_mapping(self, coco_data):
        # 初始化映射字典
        filename_to_image_id_dict = {}
        
        # 遍历 images 部分
        for image_info in coco_data["images"]:
            file_name = image_info["file_name"]
            image_id = image_info["id"]
            width = image_info["width"]
            height = image_info["height"]
            filename_to_image_id_dict[file_name] = (image_id, (width, height))
        
        return filename_to_image_id_dict
    

    # def update(self, indices_to_reserve, outputs, valid_box_mask_pairs, img_batch, video_name, is_visualized=False):

    #     for t_frame, pred_boxes in valid_box_mask_pairs.items():
    #         img_file_name = video_name + '/' + str(indices_to_reserve[t_frame]+1).zfill(3) + '.jpg'
    #         # filter those images that are classfied wrong
    #         if img_file_name in self.filename_to_image_id.keys():
    #             img_id, (width, height) = self.filename_to_image_id[img_file_name]
    #         else:
    #             print("bad ", img_file_name)
    #             continue
    #         for ids, (pred_cls, (pred_box, pred_score)) in enumerate(pred_boxes.items()):

    #             all_pred_masks = outputs[t_frame]['multistep_pred_multimasks_high_res']

    #             for pred_masks in all_pred_masks:
    #                 pred_mask = pred_masks[ids]

    #                 pred_mask = torch.argmax(pred_mask, dim=0)
    #                 pred_mask = pred_mask.unsqueeze(0)
                    
    #                 foreground_masks = {}
    #                 for cls in range(1, self.num_mask_classes + 1):
    #                     cls_mask = (pred_mask == cls).long()
    #                     if cls_mask.sum().item() > 0:
    #                         foreground_masks[cls] = cls_mask.squeeze(0)

    #                 for mask_id, mask_per_cls in foreground_masks.items():
    #                     mask_id_map = {1: 2, 2: 1, 3: 3}

    #                     category_id = pred_cls * self.num_mask_classes + (mask_id_map[mask_id] - 1)
    #                     resized_mask = F.resize(mask_per_cls[None, None], (height, width)).squeeze().contiguous().cpu().numpy().astype(np.uint8)
    #                     # resized_mask = self.binary_mask_to_polygon(resized_mask)
    #                     resized_mask_rle = mask_utils.encode(np.asfortranarray(resized_mask))
    #                     resized_mask_rle["counts"] = resized_mask_rle["counts"].decode("utf-8")

    #                     pred_ins = {
    #                                 "image_id": img_id,
    #                                 "category_id": category_id,
    #                                 "score": pred_score,
    #                                 # "bbox": mask_utils.toBbox(resized_mask_rle).tolist(),  # 从 RLE 生成检测框, # error box coors
    #                                 "segmentation": resized_mask_rle,
    #                             }
    #                     self.pred_ins.append(pred_ins)


    def update(self, indices_to_reserve, outputs, valid_box_mask_pairs, img_batch, video_name, is_visualized=False):

        for t_frame, pred_boxes in valid_box_mask_pairs.items():
            img_file_name = video_name + '/' + str(indices_to_reserve[t_frame]+1).zfill(3) + '.jpg'
            # filter those images that are classfied wrong
            if img_file_name in self.filename_to_image_id.keys():
                img_id, (width, height) = self.filename_to_image_id[img_file_name]
            else:
                print("bad ", img_file_name)
                continue
            for ids, (pred_cls, (pred_box, pred_score)) in enumerate(pred_boxes.items()):

                all_pred_masks = outputs[t_frame]['multistep_pred_multimasks_high_res']
                all_is_obejct_appearing = outputs[t_frame]['multistep_object_score_logits']
                # all_pred_masks = [outputs[t_frame]['multistep_aux_pred_multimasks'][-1]]
                for pred_masks, is_obejct_appearing in zip(all_pred_masks, all_is_obejct_appearing):
                    
                    # pred_masks = torch.nn.functional.interpolate(
                    #     pred_masks.float(),
                    #     size=(1024, 1024),
                    #     mode="nearest",
                    # )

                    pred_masks = functional.interpolate(pred_masks, size=(height, width), mode="bilinear", align_corners=False)

                    # pred_mask = pred_masks[ids].sigmoid() # get first three masks
                    pred_mask = pred_masks[ids] # get first three masks
                    is_obejct_appearing_for_single_box_rompt = is_obejct_appearing[ids].sigmoid()
                    pred_mask = (pred_mask > 0).float()
                    
                    foreground_masks = {}
                    for mask_ids in range(pred_mask.shape[0]):
                        cls = mask_ids + 1
                        cls_mask = (pred_mask[mask_ids] == 1).long()
                        if cls_mask.sum().item() > 0:
                            foreground_masks[cls] = cls_mask.squeeze(0)

                    for mask_id, mask_per_cls in foreground_masks.items():
                        # filter is not object mask
                        if is_obejct_appearing_for_single_box_rompt[mask_id-1] < self.threshold_for_masks:
                            continue

                        mask_id_map = {1: 2, 2: 1, 3: 3}
                        category_id = pred_cls * self.num_mask_classes + (mask_id_map[mask_id] - 1)
                        resized_mask = F.resize(mask_per_cls[None, None], (height, width), interpolation=InterpolationMode.NEAREST).squeeze().contiguous().cpu().numpy().astype(np.uint8)
                        # resized_mask = self.binary_mask_to_polygon(resized_mask)
                        resized_mask_rle = mask_utils.encode(np.asfortranarray(resized_mask))
                        resized_mask_rle["counts"] = resized_mask_rle["counts"].decode("utf-8")

                        pred_ins = {
                                    "image_id": img_id,
                                    "category_id": category_id,
                                    "score": pred_score,
                                    # "bbox": mask_utils.toBbox(resized_mask_rle).tolist(),  # 从 RLE 生成检测框, # error box coor=s
                                    "segmentation": resized_mask_rle,
                                }
                        self.pred_ins.append(pred_ins)


    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        num_queries = 300
        device = mask_pred.device
        labels = torch.arange(self.num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1) # TODO
        test_topk_per_image = 3
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]
        
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def instance_inference_for_MaskDINO(self, outputs, real_size):
        
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]
        # upsample masks
        mask_pred_results = functional.interpolate(
            mask_pred_results,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        real_width, real_height = real_size
        processed_results = []
        for mask_cls_result, mask_pred_result, mask_box_result in zip(
            mask_cls_results, mask_pred_results, mask_box_results
        ):  # image_size is augmented size, not divisible to 32
            processed_results.append({})
            new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)
            
            mask_box_result = mask_box_result.to(mask_pred_result)
            height = new_size[0]/self.image_size[0]*real_height
            width = new_size[1]/self.image_size[1]*real_width
            mask_box_result = self.box_postprocess(mask_box_result, height, width)

            instance_r = self.instance_inference(mask_cls_result, mask_pred_result, mask_box_result)
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def update_for_MaskDINO(self, indices_to_reserve, outputs, valid_box_mask_pairs, img_batch, video_name, is_visualized=False):

        for t_frame, pred_boxes in valid_box_mask_pairs.items():
            img_file_name = video_name + '/' + str(indices_to_reserve[t_frame]+1).zfill(3) + '.jpg'
            # filter those images that are classfied wrong
            if img_file_name in self.filename_to_image_id.keys():
                img_id, (real_width, real_height) = self.filename_to_image_id[img_file_name]
            else:
                print("bad ", img_file_name)
                continue
            for ids, (pred_cls, (pred_box, pred_score_box)) in enumerate(pred_boxes.items()):

                all_pred_masks = outputs[t_frame]['pred_masks']
                processed_results = self.instance_inference_for_MaskDINO(all_pred_masks, (real_width, real_height))

                # for pred_masks in processed_results:
                pred_masks = processed_results[ids]
                pred_masks_list = torch.unbind(pred_masks['instances'].pred_masks, dim=0)
                pred_scores_list = torch.unbind(pred_masks['instances'].scores, dim=0)
                pred_labels_list = torch.unbind(pred_masks['instances'].pred_classes, dim=0)

                for pred_mask, pred_label, pred_score_mask in zip(pred_masks_list, pred_labels_list, pred_scores_list):
                    # print(pred_label)
                    if pred_label == 3:
                        continue
                    if pred_score_mask.item() < self.threshold_for_masks:
                        continue
                    mask_id_map = {1: 2, 2: 1, 3: 3}

                    category_id = pred_cls * self.num_mask_classes + (mask_id_map[pred_label.item()+1] - 1)
                    resized_mask = F.resize(pred_mask[None, None], (real_height, real_width), interpolation=InterpolationMode.NEAREST).squeeze().contiguous().cpu().numpy().astype(np.uint8)
                    # resized_mask = self.binary_mask_to_polygon(resized_mask)
                    resized_mask_rle = mask_utils.encode(np.asfortranarray(resized_mask))
                    resized_mask_rle["counts"] = resized_mask_rle["counts"].decode("utf-8")

                    pred_ins = {
                                "image_id": img_id,
                                "category_id": category_id,
                                "score": pred_score_mask.item(),
                                # "bbox": mask_utils.toBbox(resized_mask_rle).tolist(),  # 从 RLE 生成检测框, # error box coor=s
                                "segmentation": resized_mask_rle,
                            }
                    self.pred_ins.append(pred_ins)


    def visualize_coco_results(self, image_id, coco_gt, coco_dt, save_path):
        """
        Visualize COCO GT and prediction results side by side.

        Parameters:
            image_id (int): The COCO image ID to visualize.
            coco_gt (COCO): COCO object for ground truth.
            coco_dt (COCO): COCO object for detection results.
            save_path (str): Path to save the visualization.
        """
        # Load image metadata
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = image_info['file_name']
        
        # Load ground truth annotations for the image
        gt_ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        
        # Load prediction annotations for the image
        dt_ann_ids = coco_dt.getAnnIds(imgIds=image_id)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        # Load image
        img = plt.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        # Create a plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(img)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # Visualize GT annotations
        for ann in gt_anns:
            mask = coco_gt.annToMask(ann)
            axes[0].imshow(mask, alpha=0.5, cmap='jet')

        # Visualize predicted annotations
        for ann in dt_anns:
            mask = coco_dt.annToMask(ann)
            axes[1].imshow(mask, alpha=0.5, cmap='jet')

        # Save the visualization
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    
    def compute_coco_metrics_for_ins_seg(self, epoch):
        """
        使用 COCO API 计算实例分割的 mAP 和相关指标。
        
        Parameters:
            predictions (list): 预测结果列表，COCO 格式，包含以下字段：
                - 'image_id': 图像 ID
                - 'category_id': 类别 ID
                - 'score': 预测得分
                - 'segmentation': RLE 格式的分割结果
            coco_annotation_file (str): COCO 格式的标签文件路径。
        
        Returns:
            dict: 包含 mAP 和其他指标的结果。
        """
        # 加载 COCO 格式的标签文件
        if not os.path.exists(self.saved_jsons_dir):
            os.makedirs(self.saved_jsons_dir)

        _gt_json_path = os.path.join(self.saved_jsons_dir, "_gt_val.json")

        # if not (os.path.exists(_gt_json_path) and os.path.isfile(_gt_json_path)):
        coco_annotations = self.convert_polygon_to_rle(self.coco_data)
        # 将预测结果保存为 COCO 格式的 JSON 文件
        with open(_gt_json_path, "w") as f:
            json.dump(coco_annotations, f)
        
        # 创建 COCO 对象
        coco_gt = COCO(_gt_json_path)
        epoch = str(epoch).zfill(3)
        _pred_json_path = os.path.join(self.saved_jsons_dir, f"_pred_val_epoch_{epoch}.json")
        # 将预测结果保存为 COCO 格式的 JSON 文件
        with open(_pred_json_path, "w") as f: 
            json.dump(self.pred_ins, f)
        
        # 加载预测结果
        coco_dt = coco_gt.loadRes(_pred_json_path)
        
        if self.val_ins_seg_class_agnostic:
            for ann in coco_dt.dataset['annotations']:
                ann['category_id'] = 1  # 将所有类别 ID 设置为 1
            for ann in coco_gt.dataset['annotations']:
                ann['category_id'] = 1  # 将所有类别 ID 设置为 1

        # 初始化 COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        
        # 运行评估
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 提取指标
        metrics = {
            'AP': coco_eval.stats[0],  # AP @ [IoU=0.50:0.95]
            'AP50': coco_eval.stats[1],  # AP @ [IoU=0.50]
            'AP75': coco_eval.stats[2],  # AP @ [IoU=0.75]
            'AP_small': coco_eval.stats[3],  # AP for small objects
            'AP_medium': coco_eval.stats[4],  # AP for medium objects
            'AP_large': coco_eval.stats[5],  # AP for large objects
            'AR1': coco_eval.stats[6],  # AR @ max_dets=1
            'AR10': coco_eval.stats[7],  # AR @ max_dets=10
            'AR100': coco_eval.stats[8],  # AR @ max_dets=100
            'AR_small': coco_eval.stats[9],  # AR for small objects
            'AR_medium': coco_eval.stats[10],  # AR for medium objects
            'AR_large': coco_eval.stats[11],  # AR for large objects
        }
        return metrics


    def convert_polygon_to_rle(self, coco_annotations):
        """
        将 COCO 格式的标签文件中的多边形分割转换为 RLE 格式（修复图像尺寸问题版）
        
        Parameters:
            coco_annotations (dict): COCO 格式的标签文件内容
        
        Returns:
            dict: 新的 COCO 标签文件内容，其中分割部分转换为 RLE 格式
        """
        # 创建image_id到图像信息的映射字典
        image_map = {img['id']: img for img in coco_annotations['images']}
        
        for ann in coco_annotations['annotations']:
            # 获取对应的图像信息
            img_info = image_map.get(ann['image_id'])
            if not img_info:
                raise ValueError(f"Image ID {ann['image_id']} not found in images list")
            
            # 提取当前图像的实际尺寸
            height, width = img_info['height'], img_info['width']
            
            # 转换多边形到RLE
            polygons = ann['segmentation']
            rle = mask_utils.frPyObjects(polygons, height, width)
            merged_rle = mask_utils.merge(rle)
            
            # 转换counts为字符串格式
            merged_rle['counts'] = merged_rle['counts'].decode('utf-8')
            ann['segmentation'] = merged_rle
            
        return coco_annotations

    def binary_mask_to_polygon(self, binary_mask):
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # 至少需要3个点（6个坐标）
                segmentation.append(contour)
        return segmentation
    
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
