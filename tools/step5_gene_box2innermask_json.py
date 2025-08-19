import json
import base64
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from collections import defaultdict
from training.utils.mask_RLE_utils import encode_mask_rle, decode_mask_rle, encode_list, decode_str
import cv2
import os


def save_box2innerMask_json(coco_json_path, output_json_path):
    """
    将_load_box2innerMask_per_img函数的输出结果保存为JSON文件
    
    Args:
        coco_json_path: COCO格式的实例分割标注文件路径
        output_json_path: 输出JSON文件路径
    """
    # 实例化包含原函数的类（这里需要根据实际类名修改）
    class DataProcessor:
        
        def _load_box2innerMask_per_img(self, gt_ins_seg_json_path):
            """
            load box-semantic mask from the coco_ins_seg_json
            Args:
                gt_ins_seg_json_path: json path following COCO Instance Seg
            Return:
                box-mask pairs: dict('10_20_1/001': numpy(1, img_w, img_h))
            """
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
                        # rle_box_mask_pair = mask_utils.encode(np.asfortranarray(class_mask))
                        rle_box_mask_pair = encode_mask_rle(np.asfortranarray(class_mask))
                        img_dict[merged_id] = (bbox, rle_box_mask_pair)
                    else:
                        img_dict[merged_id] = ([0, 0, 0, 0], class_mask)
                
                result_dict[img_info['file_name'][:-4]] = img_dict

            # self.visualize_masks(result_dict, "/home/jinghao/projects/dental_plague_detection/dataset/visual_tmp")
            return result_dict
        
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

    processor = DataProcessor()
    result_dict = processor._load_box2innerMask_per_img(coco_json_path)

    # 转换字典为可序列化格式
    serializable_dict = {}
    for img_key, img_data in result_dict.items():
        img_serialized = {}
        for merged_id, (bbox, mask) in img_data.items():
            mask_info = {}
            if isinstance(mask, np.ndarray):
                # 处理numpy数组
                mask_info = {
                    'type': 'ndarray',
                    'data': mask.tolist(),
                    'dtype': str(mask.dtype)
                }
            elif isinstance(mask, dict) and 'rle' in mask:
                # 处理RLE格式
                mask_info = {
                    'type': 'rle',
                    'rle': mask['rle'],
                    'shape': mask['shape']
                }
            else:
                raise ValueError("Unsupported mask type")

            img_serialized[str(merged_id)] = {
                'bbox': bbox,
                'mask': mask_info
            }
        serializable_dict[img_key] = img_serialized

    # 保存到JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)

def load_box2innerMask_json(json_path):
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

mode = "test"
input_json = f"/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/{mode}/{mode}_ins_ToI.json"
output_json = f"/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/{mode}/{mode}_ins_box2innerMask.json"

input_json = f"/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/test_ins_ToI.json"
output_json = f"/home/jinghao/projects/dental_plague_detection/dataset/2025_May_revised_training_split/test_2025_July_revised/test_ins_box2innerMask.json"

save_box2innerMask_json(input_json, output_json)

restored_data = load_box2innerMask_json(output_json)