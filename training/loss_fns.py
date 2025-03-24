# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
from pycocotools import mask as mask_utils
import numpy as np
import matplotlib.pyplot as plt

from training.utils.mask_RLE_utils import encode_mask_rle, decode_mask_rle

####################################### Derived from MaskDIN ##################################################

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

def sigmoid_focal_loss_MaskDINO(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def dice_loss_MaskDINO(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss_MaskDINO
)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss_MaskDINO(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss_MaskDINO
)  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

############################################ End ####################################################

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()# changed by bryce
    # inputs = F.softmax(inputs, dim=0)
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def dice_loss_semantic_seg(pred, target, smooth=1e-6):
    # pred: [batch_size, C, H, W] (after softmax)
    # target: [batch_size, H, W]
    num_classes = pred.size(1)
    device = pred.device
    pred = torch.softmax(pred, dim=1)  # Apply softmax to logits
    target_one_hot = torch.eye(num_classes, device=device)[target].permute(0, 3, 1, 2).to(device)  # One-hot encoding
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_non_background = dice[:, 1:]  # remove background
    return 1 - dice_non_background.mean()


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    for_object_score_compute=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class). # size(3,1,256,256)
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    # prob = inputs.sigmoid() # size(3,1,256,256) raw ; changed by bryce
    
    # remove background
    # inputs = inputs[1:, :]
    # targets = targets[1:, :]
    # targets = targets[0:1, :] # changed by bryce ; !!! note !!!
    
    if not for_object_score_compute:
        input_softmax = F.softmax(inputs, dim=0)
        prob = input_softmax # size(3,1,256,256)
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # changed by bryce
        ce_loss = F.binary_cross_entropy(input_softmax, targets, reduction="none") # changed by bryce
    else:
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    
    return loss.mean(1).sum() / num_objects




def focal_loss_for_semantic_seg(logits, targets, alpha=1.0, gamma=2.0, class_weights=None, reduction='mean'):
    """
    Args:
        logits (torch.Tensor):  预测张量，形状为 (B, C, H, W)
        targets (torch.Tensor): 目标张量，形状为 (B, H, W)
        alpha (float):          Focal Loss 的平衡参数
        gamma (float):          Focal Loss 的难样本聚焦参数
        class_weights (list):   可选的手动指定类别权重（长度需与类别数一致）
        reduction (str):        损失聚合方式（'mean' 或 'sum'）
    """
    device = logits.device
    B, C, H, W = logits.shape  # C 是类别数（包括背景）

    # 检查目标中的类别索引是否合法
    if torch.any(targets >= C):
        raise ValueError("Ground truth contains invalid class indices.")

    # ------------------------------------------
    # 自动计算类别权重（基于整个 batch 的统计）
    # ------------------------------------------
    if class_weights is None:
        # 统计整个 batch 中每个类别的像素数量
        class_counts = torch.bincount(targets.view(-1), minlength=C)  # (C,)

        # 计算权重：1 / 出现次数（忽略 count=0 的类别）
        epsilon = 1e-6  # 防止除以零
        foreground_weights = torch.zeros_like(class_counts, dtype=torch.float)
        valid_mask = class_counts > 0
        foreground_weights[valid_mask] = 1.0 / (class_counts[valid_mask] + epsilon)

        # 归一化权重（使权重和为1）
        if valid_mask.sum() > 0:
            final_class_weights = foreground_weights / foreground_weights.sum()
    else:
        # 使用手动指定的类别权重
        final_class_weights = class_weights.to(device)

    # ------------------------------------------
    # 计算 Focal Loss
    # ------------------------------------------
    # 1. 计算 softmax 概率
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # 2. 获取目标类别对应的预测概率
    targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2)  # 转换为 one-hot 格式 (B, C, H, W)
    targets_one_hot = targets_one_hot.to(device, dtype=probs.dtype)

    # 3. 计算 focal loss 的 (1 - p_t)^\gamma 和 log(p_t)
    pt = (probs * targets_one_hot).sum(dim=1)  # 只保留目标类别的概率 (B, H, W)
    log_pt = torch.log(pt + 1e-6)  # 防止 log(0)
    focal_factor = (1 - pt) ** gamma  # (1 - p_t)^\gamma

    # 4. 加入 alpha 平衡参数
    if class_weights is not None:
        # 获取每个像素点的对应类别权重
        alpha_t = final_class_weights.gather(0, targets.view(-1)).view_as(targets)  # (B, H, W)
    else:
        alpha_t = alpha  # 使用固定 alpha 值

    # 5. 计算最终的 focal loss
    focal_loss = -alpha_t * focal_factor * log_pt  # 按公式计算 Focal Loss

    # ------------------------------------------
    # 损失聚合
    # ------------------------------------------
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4

    pred_mask = inputs.flatten(2) > 0
    
    # add by bryce; note here, 0.5 threshold
    # pred_mask = F.softmax(inputs, dim=0).flatten(2) > 0.5
    # end 
    
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def iou_loss_for_semantic_seg(logits, targets, smooth=1e-6):
    """
    IoU Loss for semantic segmentation.

    Args:
        logits: [batch_size, num_classes, H, W] - raw output from the model.
        targets: [batch_size, H, W] - ground truth labels (integer class indices).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Computed IoU Loss.
    """
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)  # [batch_size, num_classes, H, W]
    
    # One-hot encode targets
    num_classes = logits.size(1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [batch_size, num_classes, H, W]

    # Compute Intersection and Union
    intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # [batch_size, num_classes]
    union = (probs + targets_one_hot - probs * targets_one_hot).sum(dim=(2, 3))  # [batch_size, num_classes]

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)  # [batch_size, num_classes]

    # Compute IoU Loss
    loss = 1 - iou
    loss_non_background = loss[:, 1:]  # remove background
    return loss_non_background.mean()  # Average over batch and classes

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        focal_alpha_for_box=0.25,
        focal_gamma_for_box=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict: # not here
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor, use_one_box_per_prompt: torch.bool, mode: str):
        assert len(outs_batch) == len(targets_batch) and len(outs_batch) > 0
        if mode == 'val' and use_one_box_per_prompt:
            return {'core_loss': torch.tensor(-1, dtype=torch.float, device=outs_batch[0]['multistep_pred_multimasks_high_res'][0].device), }
        
        if use_one_box_per_prompt:
            losses = defaultdict(int)
            device = outs_batch[0]['multistep_pred_multimasks_high_res'][0].device
            for outs, targets in zip(outs_batch, targets_batch):
                num_objects = torch.tensor(
                (len(targets)), device=device, dtype=torch.float
                )  # Number of objects is fixed within a batch
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_objects)
                num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()
                targets_mask = torch.stack([torch.from_numpy(decode_mask_rle(encoded_mask)) for _, encoded_mask in targets.values()]).to(device)
                # self.visualize_semantic_segmentation(targets_mask) # gt is correct
                cur_losses = self._forward(outs, targets_mask, num_objects, mode) # for each frame
                for k, v in cur_losses.items():
                    losses[k] += v
        else:
            num_objects = torch.tensor(
                (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
            )  # Number of objects is fixed within a batch
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_objects)
            num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

            losses = defaultdict(int)
            for outs, targets in zip(outs_batch, targets_batch):
                cur_losses = self._forward(outs, targets, num_objects) # for each frame
                for k, v in cur_losses.items():
                    losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects, mode):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_low_res_list = outputs["multistep_pred_multimasks"]
        ious_list = outputs["multistep_pred_ious"] # len()=6, each (6, 4)
        object_score_logits_list = outputs["multistep_object_score_logits"] # len()=1, each (6, 1)
        aux_src_masks_list = outputs['multistep_aux_pred_multimasks']

        assert len(src_masks_low_res_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks_low_res, ious, object_score_logits in zip(
            src_masks_low_res_list, ious_list, object_score_logits_list
        ):
            # self._update_losses(
            #     losses, src_masks, target_masks, ious, num_objects, object_score_logits
            # )

            # draw_Data = torch.argmax(src_masks, dim=1).int()
            # self.visualize_semantic_segmentation(draw_Data)

            # self._update_losses_for_semantic_segmentation(
            #     losses, src_masks, target_masks, ious, num_objects, object_score_logits, mode
            # )

            self._update_losses_for_MaskDINO_ins_seg_loss(
                losses, src_masks_low_res, target_masks, num_objects, ious, object_score_logits, mode
            )
        
        for aux_src_masks in aux_src_masks_list:
            self._update_losses_for_MaskDINO_ins_seg_loss(
                losses, aux_src_masks, target_masks, num_objects=num_objects,
            )

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses_for_MaskDINO_ins_seg_loss(
        self, losses, src_masks, target_masks, num_objects, ious=None, object_score_logits=None, mode=None
    ):
        B, C, H, W = src_masks.shape
        _, _, gt_H, gt_W = target_masks.shape

        target_masks = target_masks.squeeze(1).to(torch.int64) #(6,512,512)
        target_masks = F.one_hot(target_masks, num_classes=C+1).permute(0, 3, 1, 2)
        src_masks = src_masks.reshape(-1, H, W)
        target_masks = target_masks[:, 1:].reshape(-1, gt_H, gt_W)
    
        # self.visualize_semantic_segmentation(target_masks, save_path='gt.png')
        # self.visualize_semantic_segmentation(src_masks, save_path='our_pred.png')

        # filter no foreground mask on both targets and preds
        exist_foreground_flag_mask = torch.any(target_masks, dim=(1, 2))  # 检查每个矩阵是否有至少一个1
        target_masks = target_masks[exist_foreground_flag_mask]
        src_masks = src_masks[exist_foreground_flag_mask]

        src_masks = src_masks[:, None].float() # (14,1,256,256)
        target_masks = target_masks[:, None].float() # (14,1,1024,1024)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)


        losses["loss_mask"] += sigmoid_ce_loss_jit(point_logits, point_labels, num_objects)
        losses["loss_dice"] += dice_loss_jit(point_logits, point_labels, num_objects)

        del src_masks
        del target_masks
        torch.cuda.empty_cache()
        # if mode == 'val':
        #     src_masks, _ = torch.max(src_masks, dim=0, keepdim=True) # (1, 512, 512)
        #     target_masks, _ = torch.max(target_masks, dim=0, keepdim=True) # (1, 512, 512)

        loss_multiiou = torch.tensor(
                0.0, dtype=losses['loss_mask'].dtype, device=losses['loss_mask'].device
            )
        loss_class = torch.tensor(
            0.0, dtype=losses['loss_mask'].dtype, device=losses['loss_mask'].device
        )

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_iou"] += loss_multiiou
        losses["loss_class"] += loss_class # torch.tensor(0.0, device=target_masks.device) # loss_class


    def _update_losses_for_semantic_segmentation(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, mode
    ):
        target_masks = target_masks.squeeze(1).to(torch.int64) #(6,512,512)
        # src_masks = src_masks.transpose(1,0).contiguous() # (4,6,512,512)
        # target_masks = target_masks.expand_as(src_masks) # (3,1,256,256)
        # get focal, dice and iou loss on all output masks in a prediction step
        class_weights = torch.tensor([1.0, 5.0, 20.0, 20.0])

        # if mode == 'val':
        #     src_masks, _ = torch.max(src_masks, dim=0, keepdim=True) # (1, 512, 512)
        #     target_masks, _ = torch.max(target_masks, dim=0, keepdim=True) # (1, 512, 512)

        loss_multimask = focal_loss_for_semantic_seg(src_masks, target_masks, alpha=self.focal_alpha, gamma=self.focal_gamma, class_weights=class_weights, reduction='mean')
        # loss_multimask = focal_loss_for_semantic_seg(src_masks.transpose(0,1).contiguous(), target_masks)
        loss_multidice = dice_loss_semantic_seg(src_masks, target_masks)
        loss_multiiou = iou_loss_for_semantic_seg(src_masks, target_masks)

        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                target_masks.shape[0], # loss_multimask.shape[0], ; changed by bryce 
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
            #     ..., None
            # ].float() # size(3, 1) value [[1,1,1,]]
            # add by bryce
            # num_classes, num_prompts, H, W = src_masks.shape
            num_prompts = object_score_logits.shape[0]
            # target_obj = torch.any(target_masks.unsqueeze(0) == torch.arange(num_classes, device=target_masks.device).view(num_classes, 1, 1, 1), dim=(1, 2, 3)).float().view(num_classes, 1)
            target_obj = torch.ones(num_prompts, device=src_masks.device).unsqueeze(1)
            loss_class = sigmoid_focal_loss(
                object_score_logits, # changed by bryce
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                for_object_score_compute=True
            )

        # assert loss_multimask.dim() == 2
        # assert loss_multidice.dim() == 2
        # assert loss_multiiou.dim() == 2
        # if loss_multimask.size(1) > 1:
        #     # take the mask indices with the smallest focal + dice loss for back propagation
        #     loss_combo = (
        #         loss_multimask * self.weight_dict["loss_mask"]
        #         + loss_multidice * self.weight_dict["loss_dice"]
        #     )
        #     best_loss_inds = torch.argmin(loss_combo, dim=-1)
        #     batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
        #     loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
        #     loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
        #     # calculate the iou prediction and slot losses only in the index
        #     # with the minimum loss for each mask (to be consistent w/ SAM)
        #     if self.supervise_all_iou:
        #         loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
        #     else:
        #         loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        # else:
        #     loss_mask = loss_multimask
        #     loss_dice = loss_multidice
        #     loss_iou = loss_multiiou
        
        loss_mask = loss_multimask
        loss_dice = loss_multidice # torch.tensor(0.0, device=target_masks.device) # loss_multidice
        loss_iou = loss_multiiou # torch.tensor(0.0, device=target_masks.device) # loss_multiiou
        
        # backprop focal, dice and iou loss only if obj present
        # changed by bryce
        # target_obj[0, 0] = 0 # remove background
        # loss_mask = loss_mask * target_obj
        # loss_dice = (loss_dice * target_obj).sum()
        # loss_iou = (loss_iou * target_obj).sum()
        # loss_class = (loss_class * target_obj).sum()

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask
        losses["loss_dice"] += loss_dice
        losses["loss_iou"] += loss_iou
        losses["loss_class"] += loss_class # torch.tensor(0.0, device=target_masks.device) # loss_class

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks) # (3,1,256,256)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss( # （3, 1）
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
            for_object_score_compute=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float() # size(3, 1) value [[1,1,1,]]
            loss_class = sigmoid_focal_loss(
                object_score_logits, # changed by bryce
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                for_object_score_compute=True
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if 'boxes' in loss_key or 'image_classify' in loss_key: # add by bryce
                continue
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
    
    def visualize_semantic_segmentation(self, tensor, save_path="output.png"):
        """
        可视化语义分割结果，将 (N, 1024, 1024) 的张量可视化为 N 张子图，并保存到一张大图上。

        参数:
            tensor (torch.Tensor): 输入的张量，形状为 (N, 1024, 1024)，像素值为 0/1/2/3。
            save_path (str): 保存图像的路径，默认为 "output.png"。
        """
        # 确保输入张量的形状正确
        assert tensor.ndim == 3 , "输入张量的形状必须为 (N, 1024, 1024)"

        num_images, h, w = tensor.shape  # 获取图像数量
        rows = int(np.ceil(np.sqrt(num_images)))  # 计算子图的行数
        cols = int(np.ceil(num_images / rows))    # 计算子图的列数

        # 定义类别对应的颜色映射
        cmap = plt.cm.get_cmap('viridis', 4)  # 4 个类别 (0, 1, 2, 3)
        class_colors = {
            0: cmap(0),  # 类别 0 的颜色
            1: cmap(1),  # 类别 1 的颜色
            2: cmap(2),  # 类别 2 的颜色
            3: cmap(3),  # 类别 3 的颜色
        }

        # 创建一个大图，动态调整子图布局
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.ravel()  # 将二维的 axes 展平为一维

        # 遍历每个子图并绘制
        for i in range(num_images):
            ax = axes[i]
            img = tensor[i].cpu().detach().numpy()  # 将张量转换为 NumPy 数组
            colored_img = np.zeros((h, w, 3))  # 创建一个 RGB 图像

            # 根据类别值填充颜色
            for class_id, color in class_colors.items():
                colored_img[img == class_id] = color[:3]  # 只取 RGB 值，忽略 alpha

            ax.imshow(colored_img)
            ax.set_title(f"Image {i+1}")
            ax.axis('off')  # 关闭坐标轴

        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            axes[i].axis('off')

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path)
