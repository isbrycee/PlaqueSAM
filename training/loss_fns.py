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
    targets = targets[0:1, :] # changed by bryce ; !!! note !!!
    
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

    # 检查 gt 中的值是否在合法范围内
    if torch.any(targets >= logits.shape[1]):
        raise ValueError("Ground truth contains invalid class indices.")
    device = logits.device
    num_classes = logits.shape[1]
    class_counts = torch.bincount(targets.view(-1), minlength=num_classes)

    # 只考虑前景类别（1, 2, 3）
    foreground_counts = class_counts[1:4]
    # 计算权重，忽略数量为 0 的类别
    class_weights = torch.zeros(3)  # 初始化前景类别的权重
    for i, count in enumerate(foreground_counts):
        if count > 0:  # 忽略数量为 0 的类别
            class_weights[i] = 1.0 / count

    # 归一化权重，使得权重之和为1
    if class_weights.sum() > 0:  # 确保有权重被计算
        class_weights = class_weights / class_weights.sum()

    # 构建长度为 4 的权重张量，背景类别的权重设为 0
    final_class_weights = torch.zeros(4)  # 背景类别权重为 0
    final_class_weights[1:4] = class_weights  # 填充前景类别的权重

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=final_class_weights.to(device))
    
    # 计算损失
    loss = criterion(logits, targets.long())  # gt.squeeze(1) 将 (1, 1, 512, 512) 变为 (1, 512, 512)


    return loss


# def focal_loss_for_semantic_seg(logits, targets, alpha=1.0, gamma=2.0, class_weights=None, reduction='mean'):
#     """
#     Focal Loss for multi-class classification.

#     Args:
#         logits: [batch_size, num_classes, H, W] - raw output from the model.
#         targets: [batch_size, H, W] - ground truth labels (integer class indices).
#         alpha (float): Weighting factor for classes.
#         gamma (float): Focusing parameter.
#         reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'.

#     Returns:
#         torch.Tensor: Computed Focal Loss.
#     """
#     # changed by bryce; 20250212
#     # probs = logits.sigmoid()
#     # Apply softmax to get probabilities
#     probs = F.softmax(logits, dim=1)  # [batch_size, num_classes, H, W]
    
#     # One-hot encode targets
#     num_classes = logits.size(1)  # Number of classes
#     targets_one_hot = torch.zeros_like(probs)  # Create a tensor of the same shape as probs
#     targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # Convert targets to one-hot encoding

#     # remove background
#     # probs = probs[:, 1:, :, :]
#     # targets_one_hot = targets_one_hot[:, 1:, :, :]

#     # Gather probabilities of the target class
#     probs_target = (probs * targets_one_hot).sum(dim=1)  # [batch_size, H, W]

#     # Compute Focal Loss
#     focal_weight = (1 - probs_target) ** gamma  # [batch_size, H, W]
#     log_probs = torch.log(probs_target + 1e-8)  # Avoid log(0)
#     loss = -alpha * focal_weight * log_probs  # [batch_size, H, W]

#     # Apply class weights if provided
#     if class_weights is not None:
#         # Gather class weights for each pixel
#         class_weights = class_weights.to(logits.device)  # Ensure weights are on the same device
#         weights = class_weights[targets]  # [batch_size, H, W]
#         loss = loss * weights  # Weight the loss for each pixel

#     # Apply reduction
#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     else:  # 'none'
#         return loss

# def focal_loss_for_semantic_seg(logits, gt, ignore_index=0):
#     """
#     计算语义分割的监督损失，并忽略背景类的像素。
#     修复了可能导致 NaN 的问题。
    
#     Args:
#         logits (torch.Tensor): 模型输出的 logits，shape 为 (N, C, H, W)，其中 C 是类别数。
#         gt (torch.Tensor): Ground Truth 标签，shape 为 (N, H, W)。
#         ignore_index (int): 要忽略的类别索引，默认为 0（背景类）。
    
#     Returns:
#         loss (torch.Tensor): 计算的监督损失。
#     """
#     # 确保 logits 的 shape 是 (N, C, H, W)，gt 的 shape 是 (N, H, W)
#     assert logits.shape[0] == gt.shape[0], "Batch size must match between logits and gt"
#     assert logits.shape[2:] == gt.shape[1:], "Spatial dimensions must match between logits and gt"
#     assert logits.shape[1] > ignore_index, "Logits must have at least ignore_index+1 channels"

#     # 检查 gt 中的值是否在合法范围内
#     if torch.any(gt >= logits.shape[1]):
#         raise ValueError("Ground truth contains invalid class indices.")

#     num_classes = logits.shape[1]
#     class_counts = torch.bincount(gt.view(-1), minlength=num_classes)
#     class_weights = 1.0 / (class_counts.float() + 1e-6)  # 避免除以 0
#     class_weights /= class_weights.sum()  # 归一化

#     # # 避免 logits 值过大或过小导致数值不稳定
#     # logits = torch.clamp(logits, min=-100, max=100)

#     # # 使用 CrossEntropyLoss，设置 ignore_index 来忽略背景类
#     loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
#     # # 计算损失
#     loss = loss_fn(logits, gt)
    
#     # 创建一个掩码，忽略背景类（第 0 类）
#     # mask = (gt != 0).float()
    
#     # # 计算交叉熵损失
#     # criterion = nn.CrossEntropyLoss(reduction='none')  # 使用 reduction='none' 来获取每个像素的损失
#     # loss = criterion(logits, gt)

#     # # 应用掩码，忽略背景类的损失
#     # loss = (loss * mask).sum() / mask.sum()  # 只计算非背景类的平均损失

#     # # 检查损失是否为 NaN
#     # if torch.isnan(loss):
#     #     return torch.tensor(0.0, device=logits.device)
#     #     raise ValueError("Loss became NaN. Check your logits and ground truth data.")
#     print("mask loss: ", loss)
#     return loss

# def focal_loss_for_semantic_seg(inputs, target, num_classes=0, alpha=0.5, gamma=2):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()
#     # if h != ht and w != wt:
#     #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)

#     logpt  = -nn.CrossEntropyLoss(ignore_index=num_classes, reduction='none')(temp_inputs, temp_target) # weight=cls_weights,
#     pt = torch.exp(logpt)
#     if alpha is not None:
#         logpt *= alpha
#     loss = -((1 - pt) ** gamma) * logpt
#     loss = loss.mean()
#     return loss


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

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
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

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
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
        src_masks_list = outputs["multistep_pred_multimasks_high_res"] # len()=6, each (3, 1, 256, 256)
        ious_list = outputs["multistep_pred_ious"] # len()=6, each (3, 1)
        object_score_logits_list = outputs["multistep_object_score_logits"] # len()=6, each (3, 1)

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # self._update_losses(
            #     losses, src_masks, target_masks, ious, num_objects, object_score_logits
            # )
            self._update_losses_for_semantic_segmentation(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses_for_semantic_segmentation(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.squeeze(1).to(torch.int64)
        src_masks = src_masks.transpose(1,0).contiguous()
        # target_masks = target_masks.expand_as(src_masks) # (3,1,256,256)
        # get focal, dice and iou loss on all output masks in a prediction step
        class_weights = torch.tensor([1.0, 5.0, 10.0, 10.0])
        loss_multimask = focal_loss_for_semantic_seg(src_masks.transpose(0,1).contiguous(), target_masks, alpha=self.focal_alpha, gamma=self.focal_gamma, class_weights=class_weights, reduction='mean')
        # loss_multimask = focal_loss_for_semantic_seg(src_masks.transpose(0,1).contiguous(), target_masks)
        loss_multidice = dice_loss_semantic_seg(src_masks.transpose(0,1).contiguous(), target_masks)
        loss_multiiou = iou_loss_for_semantic_seg(src_masks.transpose(0,1).contiguous(), target_masks)

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
            # target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
            #     ..., None
            # ].float() # size(3, 1) value [[1,1,1,]]
            # add by bryce
            num_classes = src_masks.shape[0]
            target_obj = torch.any(target_masks.unsqueeze(0) == torch.arange(num_classes, device=target_masks.device).view(num_classes, 1, 1, 1), dim=(1, 2, 3)).float().view(num_classes, 1)
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
