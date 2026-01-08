# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)

from training.setCriterion import SetCriterion, HungarianMatcher, ImageClassificationLoss
from sam2.modeling.box_decoder import PostProcess
from training.dataset.mAPCalculator import MAPCalculator
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
from training.utils.MaskAP_instance_metrics import InstanceSegmentationMetric, Postprocessor_for_Instance_Segmentation
from training.utils.box_ops import box_norm_cxcywh_to_unnorm_xyxy, box_normalize_xyxy_to_cxcywh, box_xyxy_to_cxcywh
import matplotlib.pyplot as plt
from sam2.modeling.criterion import SetCriterion_MaskDINO, HungarianMatcher_MaskDINO
from training.utils.mask_RLE_utils import encode_mask_rle, decode_mask_rle
from collections import defaultdict
from pycocotools import mask as mask_utils



CORE_LOSS_KEY = "core_loss"


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None
    param_allowlist: Optional[Set[str]] = None # add by bryce

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    saved_jsons_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        val_boxes_class_agnostic: bool = False, 
        val_ins_seg_class_agnostic: bool = False, 
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):

        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.best_val_ins_seg_50 = -1
        self.best_val_box_map_50 = -1
        self.best_val_cls_acc = -1
        self.val_boxes_class_agnostic = val_boxes_class_agnostic
        self.val_ins_seg_class_agnostic = val_ins_seg_class_agnostic
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0
        # add by bryce
        
        # losses = ['labels', 'boxes'] # ['labels', 'boxes', 'cardinality']
        self.loss_for_box = SetCriterion(num_classes=self.model_conf.box_decoder.num_classes, matcher=HungarianMatcher(), \
                                         weight_dict=self.loss_conf.all.weight_dict, focal_alpha=self.loss_conf.all.focal_alpha_for_box, \
                                            gamma=self.loss_conf.all.focal_gamma_for_box, losses=['labels', 'boxes'], image_size=model['image_size'])
        self.loss_for_image_classify = ImageClassificationLoss(class_weights=self.loss_conf.all.weight_dict['loss_image_classify'], \
                                                               class_weights_for_each_class=None)
        matcher = HungarianMatcher_MaskDINO(
            cost_class=4.0,
            cost_mask=5.0,
            cost_dice=5.0,
            cost_box=5.0,
            cost_giou=2.0,
            num_points=112 * 112,
        )
        weight_dict = {"loss_ce": 4.0}
        weight_dict.update({"loss_mask": 20.0, "loss_dice": 10.0})
        weight_dict.update({"loss_bbox":5.0,"loss_giou":2.0})
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)
        aux_weight_dict = {}
        for i in range(9):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        no_object_weight = 0.1
        
        self.loss_for_MaskDINO = SetCriterion_MaskDINO(
            3,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=['labels', 'boxes', 'masks'],
            num_points=112 * 112,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            dn='no',
            dn_losses=[],
            panoptic_on=False,
            semantic_ce_loss=False,
        )
        self._setup_image_classify_to_ToI_mapping()

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        
        barrier()

    # add by bryce
    def _setup_image_classify_to_ToI_mapping(self):
        """
        Initializes the relationship between image angles and ToI
        """
        # self.image_angle_to_ToI_boxes_mapping = {
        #     1: [0, 1, 5, 6, 20, 22, 28, 29],
        #     2: [10, 11, 15, 16, 24, 26, 28, 29],
        #     3: [2, 3, 4, 21, 28, 29],
        #     4: [17, 18, 19, 27, 28, 29],
        #     5: [7, 8, 9, 23, 28, 29],
        #     6: [12, 13, 14, 25, 28, 29],
        # }
        self.image_angle_to_ToI_boxes_mapping = {
            1: [0, 1, 5, 6, 20, 22, 28],
            2: [10, 11, 15, 16, 24, 26, 28],
            3: [2, 3, 4, 21, 28],
            4: [17, 18, 19, 27, 28],
            5: [7, 8, 9, 23, 28],
            6: [12, 13, 14, 25, 28],
        }

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):

        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )
    
        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def visualize_single_dim_tensor(self, tensor, save_path="output.png"):
        """
        可视化语义分割结果，将 (512, 512) 的掩码张量可视化为彩色图像并保存。

        参数:
            tensor (torch.Tensor): 输入的张量，形状为 (512, 512)，像素值为 0/1/2/3.
            save_path (str): 保存图像的路径，默认为 "output.png".
        """
        # 确保输入张量的形状正确
        assert tensor.ndim == 2, "输入张量的形状必须为 (512, 512)"
        w, h = tensor.shape
        # 定义类别对应的颜色映射
        cmap = plt.cm.get_cmap('viridis', 4)  # 4 个类别 (0, 1, 2, 3)
        class_colors = {
            0: cmap(0),  # 类别 0 的颜色 (RGB)
            1: cmap(1),  # 类别 1 的颜色
            2: cmap(2),  # 类别 2 的颜色
            3: cmap(3),  # 类别 3 的颜色
        }

        # 创建画布和子图
        fig, ax = plt.subplots(figsize=(5, 5))

        # 将张量转换为 NumPy 数组
        img = tensor.cpu().numpy()
        colored_img = np.zeros((h, w, 3))  # 创建 RGB 图像

        # 根据类别值填充颜色
        for class_id, color in class_colors.items():
            colored_img[img == class_id] = color[:3]  # 只取 RGB 值，忽略 alpha

        # 可视化并关闭坐标轴
        ax.imshow(colored_img)
        ax.set_title("Tensor Segmentation Mask")
        ax.axis('off')

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像释放内存


    def visualize_semantic_segmentation(self, tensor, save_path="output.png"):
        """
        可视化语义分割结果，将 (N, 1024, 1024) 的张量可视化为 N 张子图，并保存到一张大图上。

        参数:
            tensor (torch.Tensor): 输入的张量，形状为 (N, 1024, 1024)，像素值为 0/1/2/3。
            save_path (str): 保存图像的路径，默认为 "output.png"。
        """
        # 确保输入张量的形状正确
        assert tensor.ndim == 3, "输入张量的形状必须为 (N, 1024, 1024)"

        num_images, h, w = tensor.shape # 获取图像数量
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
            img = tensor[i].cpu().numpy()  # 将张量转换为 NumPy 数组
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
        plt.close()  # 关闭图像释放内存

    def _get_pred_classification_info_for_evaluate(self, outputs_for_image_classify, targets_image_classify):
        targets_classes = targets_image_classify.to(self.device)
        if outputs_for_image_classify is not None:
            pred_image_classify_processed = outputs_for_image_classify.to(self.device)
        else:
            pred_image_classify_processed = targets_classes
        return pred_image_classify_processed, targets_classes
        

    def _get_pred_masks_info_for_evaluate(self, outputs_for_masks, gt_for_masks, score_threshod=0.5):
        masks_pred_list = []
        masks_pred_sig_list = []
        for pred_mask_logits in outputs_for_masks:
            pred_mask_sigmoid = torch.sigmoid(pred_mask_logits['pred_masks_high_res'].squeeze(1))
            pred_mask_sigmoid_bool = pred_mask_sigmoid > score_threshod
            masks_pred_sig_list.append(pred_mask_sigmoid)
            masks_pred_list.append(pred_mask_sigmoid_bool)
            
        # for pred
        pred_masks_for_eval = torch.stack(masks_pred_list, dim=0).int()
        pred_masks_sig_for_eval = torch.stack(masks_pred_sig_list, dim=0)
        
        all_zeros_pred = pred_masks_for_eval == 0
        all_zeros_pred = all_zeros_pred.all(dim=1)
        pred_background_for_eval = all_zeros_pred.int().unsqueeze(1) # (6, 1, 256, 256)

        pred_masks_for_eval = torch.cat((pred_background_for_eval, pred_masks_sig_for_eval), dim=1)
        pred_masks_for_eval = torch.argmax(pred_masks_for_eval, dim=1) # (6, 256, 256) [0,1,2,3]
        
        # for gt
        gt_masks_for_eval = gt_for_masks.int()

        all_zeros = gt_masks_for_eval == 0
        all_zeros = all_zeros.all(dim=1)
        gt_background_for_eval = all_zeros.int().unsqueeze(1)
        gt_masks_for_eval = torch.cat((gt_background_for_eval, gt_masks_for_eval), dim=1)
        gt_masks_for_eval = torch.argmax(gt_masks_for_eval, dim=1) # (6, 256, 256) [0,1,2,3]

        return pred_masks_for_eval, gt_masks_for_eval


    def _get_pred_masks_info_for_evaluate_semantic_Seg(self, outputs_for_masks, gt_for_masks, use_one_box_per_prompt):
        masks_pred_list = []
        for pred_mask_logits in outputs_for_masks:
            pred_mask_softmax = torch.softmax(pred_mask_logits['multistep_pred_multimasks_high_res'][0], dim=1)
            
            # 沿着类别通道维度（dim=1）找到最大索引
            pred_mask = torch.argmax(pred_mask_softmax, dim=1).int()  # 直接保持维度
            # self.visualize_semantic_segmentation(pred_mask)
            if use_one_box_per_prompt:
                pred_mask, _ = torch.max(pred_mask, dim=0, keepdim=True)

            masks_pred_list.append(pred_mask)
            
        # for pred
        pred_masks_for_eval = torch.stack(masks_pred_list, dim=0).int()

        return pred_masks_for_eval.squeeze(1), gt_for_masks.squeeze(1) # (); (6, 1024, 1024)

    def _get_pred_boxes_info_for_evaluate(self, outputs_for_boxes, gt_for_boxes, score_threshod=0.5, pred_image_classify=None):
        
        # box_decoder_pred = {'pred_logits': outputs_for_boxes['box_decoder_pred_cls'],
        #                         'pred_boxes': outputs_for_boxes['box_decoder_pred_boxes']}
        device = outputs_for_boxes['pred_logits'].device
        target_sizes = torch.tensor((self.model_conf.image_size, self.model_conf.image_size)).repeat((len(outputs_for_boxes['pred_logits']), 1)).to(device)
        Boxes_Decoder_PostProcess = PostProcess(self.model_conf.box_decoder_max_num_select, self.model_conf.box_decoder_postprocess_nms_iou_threshold) # 0.75
        results = Boxes_Decoder_PostProcess(outputs_for_boxes, target_sizes=target_sizes, not_to_xyxy=False, test=False)
        
        pred_boxes_per_frame = []
        for idx, pred_boxed_info in enumerate(results):
            scores = pred_boxed_info['scores']
            labels = pred_boxed_info['labels']
            boxes = pred_boxed_info['boxes']
            ToI_per_image_angle_list = self.image_angle_to_ToI_boxes_mapping[(pred_image_classify[idx]+1).item()]
            select_mask = torch.isin(labels, torch.tensor(ToI_per_image_angle_list).to(device)) & (scores > score_threshod)
            # select_mask = scores > score_threshod
            pred_dict = {
                'boxes': boxes[select_mask],
                'size': target_sizes,
                'box_label': labels[select_mask]
            }
            pred_boxes_per_frame.append({'labels':pred_dict['box_label'], 'scores':scores[select_mask], 'boxes': pred_dict['boxes']})
        
        gt_boxes_per_frame = []
        for target in gt_for_boxes:
            # target_box = box_norm_cxcywh_to_unnorm_xyxy(target['boxes'], (self.model_conf.image_size, self.model_conf.image_size))
            gt_boxes_per_frame.append({'labels':target['labels'].to(device), 'boxes': target['boxes'].to(device)})

        return pred_boxes_per_frame, gt_boxes_per_frame

    def _set_aux_loss_for_box_decoder(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
    def _set_interm_loss_for_box_decoder(self, output_for_two_stage):
        interm_class, ref_enc, init_box_proposal = output_for_two_stage
        interm_coord = ref_enc[-1]

        return  {'pred_logits': interm_class, 'pred_boxes': interm_coord}
    
    def generate_instance(self, mask_tensor, image_size_xyxy, device):
        """
        根据输入的语义分割 mask tensor 生成目标实例，包括 labels、masks 和 boxes。

        参数:
            mask_tensor: (H, W) 的 tensor，值为 0 (背景类)、1、2、3。
            box_xyxy_to_cxcywh: 函数，将 xyxy 坐标转换为 cxcywh 格式。
            image_size_xyxy: (W, H)，表示图像的宽高，用于归一化 boxes。

        返回:
            instance: 包含 "labels", "masks", 和 "boxes" 的字典。
        """
        # 忽略背景类 (0)
        unique_labels = torch.unique(mask_tensor)
        unique_labels = unique_labels[unique_labels != 0]  # 去除背景

        labels = []
        masks = []
        boxes = []
        for label in unique_labels:
            # 获取当前类别的 mask (前景为 1，背景为 0)
            binary_mask = (mask_tensor == label).float()
            
            # 找到该类别的 bounding box
            pos = torch.nonzero(binary_mask, as_tuple=False)  # 获取前景的坐标
            if pos.size(0) == 0:
                continue  # 如果前景不存在，跳过
            x_min, y_min = pos[:, 1].min().item(), pos[:, 0].min().item()
            x_max, y_max = pos[:, 1].max().item(), pos[:, 0].max().item()

            # 保存 labels, masks, 和 boxes
            labels.append(label-1)
            masks.append(binary_mask)
            boxes.append(torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32))

        # if len(labels) == 0:
        #     # 如果没有前景类，返回空的结构
        #     return {
        #         "labels": torch.tensor([], dtype=torch.int64),
        #         "masks": torch.empty(0, *mask_tensor.shape, dtype=torch.float32),
        #         "boxes": torch.empty(0, 4, dtype=torch.float32)
        #     }

        # 转换为 tensor
        labels = torch.tensor(labels, dtype=torch.int64, device=device)
        masks = torch.stack(masks, dim=0).to(device) # 将多个 (H, W) 的 mask 堆叠成 (N, H, W)
        boxes = torch.stack(boxes, dim=0).to(device)  # 将多个 (4,) 的 box 堆叠成 (N, 4)

        # 转换 boxes 到 cxcywh 格式并归一化
        boxes = box_xyxy_to_cxcywh(boxes)/image_size_xyxy

        return {
            "labels": labels,
            "masks": masks,
            "boxes": boxes
        }

    def _perpare_target_for_MaskDINO(self, targets_for_semantic_seg_per_box_prompt, device):

        new_targets_batched = []
        
        for targets in targets_for_semantic_seg_per_box_prompt:
            new_targets_per_image = []
            for _, encoded_mask in targets.values():
                target_mask = torch.from_numpy(decode_mask_rle(encoded_mask))
                w, h = target_mask.shape
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                new_targets_per_image.append(self.generate_instance(target_mask, image_size_xyxy, device))
            new_targets_batched.append(new_targets_per_image)

        return new_targets_batched

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        (outputs, outputs_for_boxes, output_for_two_stage, 
            pred_image_classifiy_logits, 
            pred_image_classify_processed, 
            indices_to_reserve, 
            valid_box_mask_pairs) = model(batch, phase)
        
        targets = batch.masks[indices_to_reserve] # (6, 3, 256, 256)
        targets_for_semantic_seg = batch.masks_for_semantic_seg[indices_to_reserve].to(torch.int64)
        targets_for_semantic_seg_per_box_prompt = [batch.box_mask_pairs[ids] for ids in indices_to_reserve]
        
        # for visualize gt mask
        # import matplotlib.pyplot as plt
        # ================================ for visual gt ================================
        # draw_Data = torch.where((targets.int() > 0.5).to(float)==1, 255, 0)
        # tensor_np = draw_Data.reshape(-1, self.model_conf.image_size, self.model_conf.image_size).cpu().detach().numpy()  # 转换为(18, 256, 256)

        # ================================ for visual pred ================================
        # draw_Data = torch.argmax(outputs[0]['multistep_pred_multimasks_high_res'][0], dim=1).int()
        # torch.max(pred_mask, dim=0, keepdim=True)

        # draw_Data = torch.where((draw_Data > 0.5).to(float)==1, 255, 0)

        # self.visualize_semantic_segmentation(draw_Data) # (8, 512, 512)
        # tensor_np = draw_Data.reshape(-1, self.model_conf.image_size, self.model_conf.image_size).cpu().detach().numpy()  # 转换为(18, 256, 256)
        # # 设置图像显示的行数和列数
        # nrows = 4
        # ncols = 6
        
        # # 创建一个图形窗口
        # plt.figure(figsize=(ncols * 2, nrows * 2))

        # # 遍历每个图像并显示
        # for i in range(tensor_np.shape[0]):
        #     plt.subplot(nrows, ncols, i + 1)
        #     plt.imshow(tensor_np[i], cmap='gray')  # 显示灰度图像
        #     plt.axis('off')  # 不显示坐标轴

        # plt.tight_layout()
        # plt.savefig('saved_images.png')  # 保存为PNG格式，DPI为300 , dpi=300
        # print("save img")
        # import pdb; pdb.set_trace()
        # end

        batch_size = targets.shape[0]
        device = targets.device
        key = batch.dict_key  # key for dataset
        # loss = self.loss[key](outputs, targets)
        
        use_MaskDINO_decoder = False
        
        #TODO, add loss for val stage
        if phase == 'val':
            loss = {'val': torch.tensor(-1, device=device), 
                    'core_loss': torch.tensor(-1, device=device)}
        else:
            if use_MaskDINO_decoder:
                targets_for_MaskDINO = self._perpare_target_for_MaskDINO(targets_for_semantic_seg_per_box_prompt, device)
                loss = defaultdict(int)
                for output, target in zip(outputs, targets_for_MaskDINO):
                    outputs_for_MaskDINO = output['pred_masks']
                    tmp_loss = self.loss_for_MaskDINO(outputs_for_MaskDINO, target)
                    for k in list(tmp_loss.keys()):
                        if k in self.loss_for_MaskDINO.weight_dict:
                            tmp_loss[k] *= self.loss_for_MaskDINO.weight_dict[k]
                        else:
                            # remove this loss if not specified in `weight_dict`
                            tmp_loss.pop(k)
                    for k, value in tmp_loss.items():
                        loss[k] = value
        
            elif self.model_conf.use_one_box_per_prompt:
                loss = self.loss[key](outputs, targets_for_semantic_seg_per_box_prompt, use_one_box_per_prompt=True, mode=phase) # add by bryce
            else:
                loss = self.loss[key](outputs, targets_for_semantic_seg, use_one_box_per_prompt=False, mode=phase) # add by bryce
        
        # add by bryce; loss for boxes & evaluation for box
        targets_boxes = [batch.boxes[indx] for indx in indices_to_reserve] # dict({0:size(9,2,2)})
        targets_boxes_xyxy_unnorm = copy.deepcopy(targets_boxes)
        # add by bryce
        # convert tgt box from unnormalized xyxy to normalized cxcywh
        for target in targets_boxes:
            boxes_anno_for_each_frame = target['boxes']
            target['boxes'] = box_normalize_xyxy_to_cxcywh(boxes_anno_for_each_frame, self.model_conf.image_size).to(device)
            target['labels'] = target['labels'].to(device)
        targets_boxes_cxcywh = copy.deepcopy(targets_boxes)
        # end
        
        # for loss of boxes final prediction supervision
        outputs_boxes = {'pred_logits': outputs_for_boxes['box_decoder_pred_cls'][-1], 
                        'pred_boxes': outputs_for_boxes['box_decoder_pred_boxes'][-1]}
        loss_boxes = self.loss_for_box(outputs_boxes, targets_boxes_cxcywh)
        for k, v in loss_boxes.items():
            new_key = k.split('_')[0] + '_boxes_' + k.split('_')[1]
            loss[new_key] = v
        # for aux box prediction supervision
        aux_outputs_boxes = self._set_aux_loss_for_box_decoder(outputs_for_boxes['box_decoder_pred_cls'], 
                                               outputs_for_boxes['box_decoder_pred_boxes'])
        aux_loss_boxes_list = []
        for i, aux_outputs_box in enumerate(aux_outputs_boxes):
            loss_boxes_aux = self.loss_for_box(aux_outputs_box, targets_boxes_cxcywh)
            aux_loss_boxes_list.append(loss_boxes_aux)
            for k, v in loss_boxes_aux.items():
                new_key = k.split('_')[0] + f'_boxes_dec_layer{i}_' + k.split('_')[1]
                loss[new_key] = v
        
        # for box two-stage
        interm_outputs_boxes = self._set_interm_loss_for_box_decoder(output_for_two_stage)
        loss_interm_boxes = self.loss_for_box(interm_outputs_boxes, targets_boxes_cxcywh)
        for k, v in loss_boxes.items():
            new_key = k.split('_')[0] + '_interm_boxes_' + k.split('_')[1]
            loss[new_key] = v
        # end loss of boxes 
        
        # add by bryce; loss for image_classify
        targets_image_classify = torch.tensor(batch.image_classify).to(device)
        loss_image_classify = self.loss_for_image_classify(pred_image_classifiy_logits, targets_image_classify)
        for k, v in loss_image_classify.items():
            loss[k] = v
        # end
        
        # add by bryce; for eval
        if phase == 'val':
            outputs_boxes = {'pred_logits': outputs_for_boxes['box_decoder_pred_cls'][-1], 
                        'pred_boxes': outputs_for_boxes['box_decoder_pred_boxes'][-1]}
            preds_boxes, targets_boxes = self._get_pred_boxes_info_for_evaluate(outputs_boxes, targets_boxes_xyxy_unnorm, self.model_conf.threshold_for_boxes, pred_image_classify_processed)
            ############## for box vis, check box ###################
            # 绘制图像
            # boxes = preds_boxes[0]['boxes'].cpu()
            # labels = preds_boxes[0]['labels'].cpu()
            # import matplotlib.pyplot as plt
            # import matplotlib.patches as patches
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.set_xlim(0, 1024)
            # ax.set_ylim(0, 1024)
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')
            # ax.set_title('Bounding Boxes with Labels')

            # # 遍历每个 box 和类别，并绘制
            # for box, label in zip(boxes, labels):
            #     x_min, y_min, x_max, y_max = box
            #     width = x_max - x_min
            #     height = y_max - y_min
            #     # 添加矩形框
            #     rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
            #     ax.add_patch(rect)
            #     # 添加类别标签
            #     ax.text(x_min, y_min - 10, f'Class {label}', color='blue', fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

            # # 翻转 Y 轴（因为图像坐标系通常以左上角为原点，而 matplotlib 以左下角为原点）
            # plt.gca().invert_yaxis()
            # plt.savefig("output.png", dpi=300, bbox_inches='tight')
            # plt.close()  # 关闭绘图窗口
            ############## end for box vis, check box ###################
            
            # preds_masks, targets_masks = self._get_pred_masks_info_for_evaluate(outputs, targets, self.model_conf.threshold_for_masks)

            preds_masks, targets_masks = None, None
            # preds_masks, targets_masks = self._get_pred_masks_info_for_evaluate_semantic_Seg(outputs, targets_for_semantic_seg, self.model_conf.use_one_box_per_prompt)

            preds_classes, targets_classes = self._get_pred_classification_info_for_evaluate(pred_image_classify_processed, targets_image_classify)
        # 
        
        loss_str = f"Losses/{phase}_{key}_loss"
        
        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )

            # if not self.epoch >= self.max_epochs:
            #     # seg loss
            #     loss = self._log_loss_detailed_and_return_core_loss(
            #         loss, loss_log_str, self.steps[phase]
            #     )
            # elif not self.epoch < self.max_epochs:
            #     # add by bryce; box loss
            #     loss = 0.0
            #     loss = self._add_boxes_loss_into_core_loss(
            #         loss, loss_boxes, aux_loss_boxes_list, loss_interm_boxes, loss_log_str, self.steps[phase]
            #     )
            #     # add by bryce; image loss
            #     loss = self._add_image_classify_loss_into_core_loss(
            #         loss, loss_image_classify, loss_log_str, self.steps[phase]
            #     )

            # for computing all loss; 
            # seg loss
            if use_MaskDINO_decoder:
                seg_loss = self._log_loss_detailed_and_return_core_loss_for_MaskDINO(
                    loss, loss_log_str, self.steps[phase]
                )
            else:
                seg_loss = self._log_loss_detailed_and_return_core_loss(
                    loss, loss_log_str, self.steps[phase]
                )
            # add by bryce; box loss
            box_loss = self._add_boxes_loss_into_core_loss(
                device, loss_boxes, aux_loss_boxes_list, loss_interm_boxes, loss_log_str, self.steps[phase]
            )
            # add by bryce; image loss
            image_cls_loss = self._add_image_classify_loss_into_core_loss(
                device, loss_image_classify, loss_log_str, self.steps[phase]
            )

            if self.model_conf.train_stage == '1st':
                loss_all = image_cls_loss
            elif self.model_conf.train_stage == '2nd':
                loss_all = box_loss
            elif self.model_conf.train_stage == '3rd':
                loss_all = seg_loss
            else: 
                loss_all = seg_loss + box_loss + image_cls_loss

            # loss_all = seg_loss + box_loss + image_cls_loss
            # end
        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss_all,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss_all}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        del targets
        del targets_for_semantic_seg
        del targets_for_semantic_seg_per_box_prompt
        torch.cuda.empty_cache()
       
        if phase == 'train':
            return ret_tuple
        elif phase == 'val':
            return ret_tuple, preds_boxes, targets_boxes, preds_masks, targets_masks, preds_classes, targets_classes, indices_to_reserve, outputs, valid_box_mask_pairs, outputs_for_boxes['init_cond_frames']

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
        
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        start_time = time.time()  
        self.train_dataset = None
        self.val_dataset = None
        if self.mode in ["val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
            # add by bryce
            self.train_dataset = None
            elapsed_time = time.time() - start_time 
            print(f"The time for loading datasets: {elapsed_time:.2f} s")
            return
        
        if self.mode in ["train"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
            # add by bryce
            self.train_dataset = instantiate(self.data_conf.train)
            elapsed_time = time.time() - start_time 
            print(f"The time for loading datasets: {elapsed_time:.2f} s")
            return
        
        if self.mode in ["train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)
            elapsed_time = time.time() - start_time
            print(f"The time for loading datasets: {elapsed_time:.2f} s")

    def run_train(self):

        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        # add by bryce 
        # dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))

        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()
        
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )
        
        end = time.time()
        # add by bryce 
        BoxmAP_calculator = MAPCalculator(saved_jsons_dir=self.logging_conf.saved_jsons_dir, class_agnostic=self.val_boxes_class_agnostic)
        # +1 for background
        mIoU_calculator = MulticlassJaccardIndex(num_classes=self.model_conf.num_classes_for_mask+1, average=None, ignore_index=0).to(self.device)
        accuracy_calculator = Accuracy(task="multiclass", num_classes=self.model_conf.image_classify_decoder.num_classes).to(self.device)
        MaskmAP_calculator = InstanceSegmentationMetric(num_box_classes=30, num_mask_classes=self.model_conf.num_classes_for_mask, device=self.device)

        gt_json_path = self.data_conf.val.datasets[0].dataset.datasets[0].video_dataset.gt_ins_seg_json
        Ins_Seg_postprocessor = Postprocessor_for_Instance_Segmentation(gt_json_path, self.model_conf.num_classes_for_mask, self.logging_conf.saved_jsons_dir, \
                                                                        self.val_ins_seg_class_agnostic, \
                                                                        image_size=(self.model_conf.image_size, self.model_conf.image_size),\
                                                                        threshold_for_masks=self.model_conf.threshold_for_masks, \
                                                                        device=self.device)

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            batch = batch.to(self.device, non_blocking=True)
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        ret_tuple, preds_boxes, targets_boxes, preds_masks, targets_masks, preds_classes, \
                        targets_classes, indices_to_reserve, outputs, valid_box_mask_pairs, init_cond_frames = self._step(
                            batch,
                            model,
                            phase,
                        )
                        indices_to_reserve_for_visual = [i for i, value in enumerate(batch.image_classify) if value != 6]
                        BoxmAP_calculator.update(preds_boxes, targets_boxes, batch.img_batch[indices_to_reserve_for_visual], batch.metadata.video_name, is_visualized=self.model_conf.is_visualize_bad_cases_from_boxes) # add by bryce for compute box mAP
                        # mIoU_calculator.update(preds_masks, targets_masks) # add by bryce for compute mIoU
                        if not torch.equal(preds_classes, targets_classes):
                            print(batch.metadata.video_name)
                            print('pred:', preds_classes)
                            print('target:', targets_classes)
                        accuracy_calculator.update(preds_classes, targets_classes) # add by bryce for compute image classification accuracy
                        # MaskmAP_calculator.update(preds_boxes, targets_boxes, preds_masks, targets_masks)
                        # Ins_Seg_postprocessor.update_for_MaskDINO(indices_to_reserve, outputs, valid_box_mask_pairs, batch.img_batch[indices_to_reserve_for_visual], batch.metadata.video_name, is_visualized=False)
                        Ins_Seg_postprocessor.update(indices_to_reserve, outputs, valid_box_mask_pairs, batch.img_batch[indices_to_reserve_for_visual], batch.metadata.video_name, is_visualized=False)

                        loss_dict, batch_size, extra_losses = ret_tuple
                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        
        # add by bryce; for Metrics logging; for compute accuracy
        classification_acc_results = accuracy_calculator.compute()
        logging.info(f"Image Classification ACC Results: {classification_acc_results}")
        # add by bryce; for Metrics logging; for compute mAP
        box_mAP_results = BoxmAP_calculator.compute_and_save_json(self.epoch, is_save_json_for_boxes_prediction=self.model_conf.is_save_json_for_boxes_prediction)
        logging.info(f"Object Detection mAP Results: {box_mAP_results}")

        # add by bryce; for Metrics logging; 计算平均 IoU（mIoU）
        # mask_mIoU_results_per_class = mIoU_calculator.compute()
        # mask_mIoU_results = mask_mIoU_results_per_class[1:].mean()
        mask_mIoU_results = -1.0
        # logging.info(f"Mask IoU Results Per Class: {mask_mIoU_results_per_class}")
        # logging.info(f"Mask Mean IoU (mIoU) Results: {mask_mIoU_results}")

        # add by bryce; for Metrics logging; for compute MaskmAP
        # box_MaskmAP_results = MaskmAP_calculator.compute()
        # logging.info(f"Instance Segmentation mAP Results: {box_MaskmAP_results}")

        # add by bryce; for Metrics logging; for compute ins seg ap
        ins_seg_metrics = Ins_Seg_postprocessor.compute_coco_metrics_for_ins_seg(self.epoch)

        header1 = "|   Acc   |  Box mAP  |  Box AP50  | Semantic mIoU |"
        separator1 = "+---------+-----------+------------+---------------+"
        data_row1 = f"| {classification_acc_results:^7.3f} | {box_mAP_results['map']:^9.3f} | {box_mAP_results['map_50']:^10.3f} | {mask_mIoU_results:^13.3f} |"

        header2 = "|   mAP   |   AP50   |   AP75   |  AP_small | AR_medium | AR_large |"
        separator2 = "+---------+----------+----------+-----------+-----------+----------+"
        data_row2 = f"| {ins_seg_metrics['AP']:^7.3f} | {ins_seg_metrics['AP50']:^8.3f} | {ins_seg_metrics['AP75']:^8.3f} | {ins_seg_metrics['AP_small']:^9.3f} | {ins_seg_metrics['AR_medium']:^9.3f} | {ins_seg_metrics['AR_large']:^8.3f} |"
        # 输出表格
        logging.info(separator1)
        logging.info(header1)
        logging.info(separator1)
        logging.info(data_row1)
        logging.info(separator1)

        logging.info(separator2)
        logging.info(header2)
        logging.info(separator2)
        logging.info(data_row2)
        logging.info(separator2)
        

        if classification_acc_results > self.best_val_cls_acc:
            self.best_val_cls_acc = classification_acc_results
            self.save_checkpoint(self.epoch + 1, ['best_img_cls_acc'])
            logging.info(f"Best checkpoint has been saved in epoch {self.epoch}. The current best image classification accuracy is : {self.best_val_cls_acc}")


        if box_mAP_results['map_50'] > self.best_val_box_map_50:
            self.best_val_box_map_50 = box_mAP_results['map_50']
            self.save_checkpoint(self.epoch + 1, ['best_box_map50'])
            logging.info(f"Best checkpoint has been saved in epoch {self.epoch}. The current best box detection map_50 is : {self.best_val_box_map_50}")


        # save ckp; add by bryce
        if ins_seg_metrics['AP50'] > self.best_val_ins_seg_50:
            self.best_val_ins_seg_50 = ins_seg_metrics['AP50']
            self.save_checkpoint(self.epoch + 1, ['best_ins_seg_map50'])
            logging.info(f"Best checkpoint has been saved in epoch {self.epoch}. The current best instance segmentatiom map_50 is : {self.best_val_ins_seg_50}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):

        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        end = time.time()

        # for name, param in self.model.named_parameters():
            # print(name)
        #     print(param.requires_grad)

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(
                self.device, non_blocking=True
            )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)

        # print for debug video name
        # print(batch.metadata.video_name)

        with torch.cuda.amp.autocast(
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()
        
        # print(loss)

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        self.scaler.scale(loss).backward(retain_graph=True) # retain_graph=True
    
        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

        torch.cuda.empty_cache()

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                assert all([k in loss_keys for k in val_keys]), (
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    def _setup_components(self):

        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        
        # add by bryce; for seperate training; twostage
        logging.info(f"Current training stage is {self.model_conf.train_stage}.")
        _set_trainable_params(self.model, self.model_conf.train_stage)
        # End

        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.cuda.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            self.optim_conf.param_allowlist, # add by bryce
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    # add by bryce;
    def _log_loss_detailed_and_return_core_loss_for_MaskDINO(self, loss, loss_str, step):
        core_loss = 0.0
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)

        for loss_key, weight in loss.items():
            core_loss += weight

        return core_loss

    # add by bryce
    def _add_boxes_loss_into_core_loss(self, device, loss_boxes, aux_loss_boxes_list, loss_interm_boxes, loss_str, step):
        core_loss = torch.tensor(0.0, device=device)
        
        for k, v in loss_boxes.items():
            new_key = k.split('_')[0] + '_boxes_' + k.split('_')[1]
            if 'error' not in new_key:
                core_loss += v * self.loss_for_box.weight_dict[new_key]

        for aux_loss_boxes in aux_loss_boxes_list:
            for k, v in aux_loss_boxes.items():
                new_key = k.split('_')[0] + '_boxes_' + k.split('_')[1]
                if 'error' not in new_key:
                    core_loss += v * self.loss_for_box.weight_dict[new_key]

        for k, v in loss_interm_boxes.items():
            new_key = k.split('_')[0] + '_boxes_' + k.split('_')[1]
            if 'error' not in new_key:
                core_loss += v * self.loss_for_box.weight_dict[new_key]

        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss_boxes:
                if not k.startswith('loss_'):
                    continue
                log_str = os.path.join(loss_str, k)
                new_key = k.split('_')[0] + '_boxes_' + k.split('_')[1]
                weight = self.loss_for_box.weight_dict[new_key]
                self.logger.log(log_str, loss_boxes[k] * weight, step)

        return core_loss
    # end 

    # add by bryce
    def _add_image_classify_loss_into_core_loss(self, device, loss_image_classify, loss_str, step):
        # core_loss = loss
        core_loss = torch.tensor(0.0, device=device)
        for k, v in loss_image_classify.items():
            if 'error' in k:
                continue
            core_loss += v * self.loss_for_image_classify.weight_dict[k]

        if step % self.logging_conf.log_scalar_frequency == 0:
            for key in loss_image_classify:
                log_str = os.path.join(loss_str, key)
                weight = self.loss_for_image_classify.weight_dict['loss_image_classify']
                self.logger.log(log_str, loss_image_classify[key] * weight, step)

        return core_loss
    # end 

def _set_trainable_params(model: torch.nn.Module, train_stage: str = ""):
    
    # set all params to requires_grad=False
    for name, param in model.named_parameters():
        # print(name)
        param.requires_grad = False

    first_stage_trained_params = ['image_classify_decoder'] 
    second_stage_trained_params = ['box_decoder', 'conv_s0', 'conv_s1', 'obj_ptr_proj', 'obj_ptr_tpos_proj']
    third_stage_trained_params = ['sam_prompt_encoder', 'sam_mask_decoder', 'no_mem_embed', 'no_mem_pos_enc', 'mask_downsample', 'memory_attention'] # 
    
    if train_stage == '1st':
        # 冻结mask decoder，解冻其他参数
        for name, param in model.named_parameters():
            param.requires_grad = any(s in name for s in first_stage_trained_params)

    elif train_stage == '2nd':
        # 解冻mask decoder，冻结其他参数
        for name, param in model.named_parameters():
            param.requires_grad = any(s in name for s in second_stage_trained_params)
            
    elif train_stage == '3rd':
        # 解冻mask decoder，冻结其他参数
        for name, param in model.named_parameters():
            param.requires_grad = (
                any(s in name for s in third_stage_trained_params) and
                all(s not in name for s in second_stage_trained_params)
            )
            
            print(name, param.requires_grad)
            # param.requires_grad = any(s in name for s in third_stage_trained_params)
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
