# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    boxes: dict # id: box[[x1, y1], [x2, y2]] # add by bryce
    image_classify: list
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask
    


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    boxes: dict
    image_classify: int


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    # step_t_boxes = {i:None for i in range(T)} # add by bryce
    step_t_boxes = [] # add by bryce
    step_t_image_classify = [] # add by bryce
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step
    
    # add by bryce
    class_name_to_idx_map = {'51':0, '52':1, '53':2, '54':3, '55':4, 
                             '61':5, '62':6, '63':7, '64':8, '65':9, 
                             '71':10, '72':11, '73':12, '74':13, '75':14,
                             '81':15, '82':16, '83':17, '84':18, '85':19,

                             '51_stain':0,'52_stain':1, '53_stain':2, '54_stain':3, '55_stain':4, 
                             '61_stain':5, '62_stain':6, '63_stain':7, '64_stain':8, '65_stain':9, '63_stan':7,
                             '71_stain':10, '72_stain':11, '73_stain':12, '74_stain':13, '75_stain':14, 
                             '81_stain':15, '82_stain':16, '83_stain':17, '84_stain':18, '85_stain':19, 

                             '52_retainedteeth':1,
                             '53_retainedteeth':2,
                             '75_discoloration':14,
                             '51_discoloration':0,
                             '51_retainedteeth':0,
                             '61_retainedteeth':5,
                             '62_retainedteeth':6,
                             '64_retainedteeth':8,
                             '63_retainedteeth':7,
                             '54_retainedteeth':3,
                             '74_retainedteeth':13,
                             '61_discoloration':5,

                             '55_crown':4,
                             '84_crown':18,
                             '74_crown':13,
                             
                             "55'":4,
                             '622':6,
                             '585':19,
                             '875':14,

                             '72\\3':11,
                             '72/3':11,
                             '82/83':16,

                             '42':16,
                             '32':11,
                             '11': 0,
                             '31': 10, '36':14, '41': 15, '46':19, 
                             
                             }

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            # handle for boxes gt
            meta_box_info = frame.boxes # add by bryce
            # for boxes
            merged_meta_box_info = [item for sublist in meta_box_info.values() for item in sublist]
            merged_meta_box_info = torch.tensor(merged_meta_box_info, dtype=torch.float64)
            # for classes
            # for filter bad cagegory names
            for sublist in meta_box_info.keys():
                if '_' in sublist:
                    sublist = sublist.split('_')[0]


            merged_meta_class_info = [class_name_to_idx_map[sublist] for sublist in meta_box_info.keys()]
            merged_meta_class_info = torch.tensor(merged_meta_class_info, dtype=torch.int64)
            # step_t_boxes[t] = merged_meta_box_info.reshape(-1, 2, 2)
            step_t_boxes.append({'labels':merged_meta_class_info, 'boxes': merged_meta_box_info.reshape(-1, 4)}) 
            step_t_image_classify.append(frame.image_classify - 1) # index start from 0; 
            # end; add by bryce
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    boxes = step_t_boxes # add by bryce
    image_classify = step_t_image_classify # add by bryce

    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        boxes=boxes, # add by bryce
        image_classify=image_classify, # add by bryce
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
