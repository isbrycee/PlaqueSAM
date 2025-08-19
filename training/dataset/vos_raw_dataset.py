# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd
import json
import base64
import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
    JsonBBoxLoader,
)

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np
import cv2

from training.utils.mask_RLE_utils import encode_mask_rle, decode_mask_rle, decode_str



@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        gt_box_folder, # add by bryce
        gt_ins_seg_json, # add by bryce
        gt_ins_box2innerMask_json_path, # add by bryce
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.gt_box_folder = gt_box_folder
        self.gt_ins_seg_json_path = gt_ins_seg_json
        self.gt_ins_box2innerMask_json_path = gt_ins_box2innerMask_json_path
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult
        
        # add by bryce
        # {10_10_11/005:{1:((x1,y1,w,h), numpy(mask))}, ...}
        self.box_mask_pairs_dict = self._load_box2innerMask_per_img(self.gt_ins_seg_json_path, self.gt_ins_box2innerMask_json_path)

    def visualize_masks(self, result_dict, save_dir, color_map=None):
        """
        可视化合并后的mask和bbox
        :param result_dict: 前一个函数返回的字典
        :param save_dir: 可视化结果保存路径
        :param color_map: 自定义颜色映射字典，格式为 {子类值: (B,G,R)}
        """
        # 默认颜色映射（BGR格式）
        default_colors = {
            1: (0, 0, 255),    # 红色：0_np
            2: (0, 255, 0),    # 绿色：0_p
            3: (255, 0, 0)     # 蓝色：0_caries
        }
        color_map = color_map or default_colors
        
        os.makedirs(save_dir, exist_ok=True)

        for img_id, img_data in result_dict.items():
            # 获取图片尺寸（假设所有类别的mask尺寸相同）
            sample_mask = next(iter(img_data.values()))[1]
            h, w = sample_mask.shape
            # 创建空白画布
            canvas = np.zeros((h, w, 3), dtype=np.uint8)


            for merged_id, (bbox, class_mask) in img_data.items():
                # 绘制子类区域
                for value, color in color_map.items():
                    canvas[class_mask == value] = color
                
                # 绘制边界框（跳过无效bbox）
                if bbox != [0, 0, 0, 0]:
                    x, y, bw, bh = [int(v) for v in bbox]
                    cv2.rectangle(canvas, 
                                (x, y),
                                (x + bw, y + bh),
                                color=(255, 255, 255),  # 白色边框
                                thickness=2)
                
            # 保存结果
            img_id = img_id.replace('/', '_')
            filename = f"img_{img_id}.png"
            cv2.imwrite(os.path.join(save_dir, filename), canvas)
            print(f'save to {filename}')


    def _load_box2innerMask_json(self, json_path):
        """
        从JSON文件加载并恢复原始数据结构
        
        Args:
            json_path: 保存的JSON文件路径
        Returns:
            与原函数返回结构相同的字典
        """
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        result_dict = {}
        for img_key, img_data in loaded_data.items():
            restored_img = {}
            for merged_id_str, entry in img_data.items():
                merged_id = int(merged_id_str)
                bbox = entry['bbox']
                mask_info = entry['mask']
                
                if mask_info['type'] == 'ndarray':
                    # 恢复numpy数组
                    mask = np.array(
                        mask_info['data'], 
                        dtype=np.dtype(mask_info['dtype'])
                    )
                elif mask_info['type'] == 'rle':
                    # 恢复RLE格式
                    mask = {
                        'rle': mask_info['rle'],
                        'shape': mask_info['shape']
                    }
                else:
                    raise ValueError("Unknown mask type in JSON")
                
                restored_img[merged_id] = (bbox, mask)
            result_dict[img_key] = restored_img
        
        return result_dict

    # add by bryce
    def _load_box2innerMask_per_img(self, gt_ins_seg_json_path, gt_ins_box2innerMask_json_path=None):
        """
        load box-semantic mask from the coco_ins_seg_json
        Args:
            gt_ins_seg_json_path: json path following COCO Instance Seg
        Return:
            box-mask pairs: dict('10_20_1/001': numpy(1, img_w, img_h))
        """

        if gt_ins_box2innerMask_json_path and os.path.isfile(gt_ins_box2innerMask_json_path):
            return self._load_box2innerMask_json(gt_ins_box2innerMask_json_path)

        # 加载COCO数据集
        coco = COCO(gt_ins_seg_json_path)
        
        # 构建类别映射字典
        category_mapping = {}
        suffix_values = {'p': 2, 'np': 1, 'caries': 3}
        for cat in coco.dataset['categories']:
            prefix, suffix = cat['name'].split('_')
            merged_id = int(prefix)
            sub_value = suffix_values[suffix]
            category_mapping[cat['id']] = (merged_id, sub_value)
        
        # 初始化结果字典
        result_dict = {}
        
        # 处理每张图片
        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            
            # 按合并后的类别分组
            merged_groups = defaultdict(list)
            for ann in annotations:
                original_cat_id = ann['category_id']
                merged_id, sub_value = category_mapping[original_cat_id]
                merged_groups[merged_id].append((ann, sub_value))
            
            # 处理每个合并后的类别组
            img_dict = {}
            for merged_id, group in merged_groups.items():
                # 创建全零矩阵
                height, width = img_info['height'], img_info['width']
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                class_mask = np.zeros((height, width), dtype=np.uint8)
                
                # 合并mask并生成类别矩阵
                for ann, sub_value in group:
                    ann_mask = coco.annToMask(ann)
                    combined_mask = np.logical_or(combined_mask, ann_mask)
                    class_mask[ann_mask == 1] = sub_value  # 最后出现的会覆盖之前的
                    
                # 计算外接矩形
                if np.any(combined_mask):
                    rle = mask_utils.encode(np.asfortranarray(combined_mask))
                    bbox = mask_utils.toBbox(rle).tolist()
                    # print(np.unique(class_mask))
                    rle_box_mask_pair = encode_mask_rle(np.asfortranarray(class_mask))
                    img_dict[merged_id] = (bbox, rle_box_mask_pair)
                else:
                    img_dict[merged_id] = ([0, 0, 0, 0], class_mask)
            
            result_dict[img_info['file_name'][:-4]] = img_dict

        # self.visualize_masks(result_dict, "/home/jinghao/projects/dental_plague_detection/dataset/visual_tmp")
        return result_dict


    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        
        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )
        # add by bryce
        video_box_root = os.path.join(self.gt_box_folder, video_name)

        # get annos for specific person
        box_mask_pairs_dict_per_video = {}
        for k, v in self.box_mask_pairs_dict.items():
            if video_name in k:
                box_mask_pairs_dict_per_video[k] = v

        bbox_loader = JsonBBoxLoader(video_box_root, box_mask_pairs_dict_per_video)

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader, bbox_loader, video_name

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
