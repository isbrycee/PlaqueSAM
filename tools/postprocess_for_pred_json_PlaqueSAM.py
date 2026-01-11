import json
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
import numpy as np
from scipy.ndimage import label, sum as ndi_sum
from PIL import Image
import os
from scipy.ndimage import binary_fill_holes

def keep_largest_component(mask):
    """
    保留掩码中面积最大的连通域，其余设为0
    
    参数:
        mask (numpy.ndarray): 输入的二值/多值掩码
        
    返回:
        numpy.ndarray: 只包含最大连通域的掩码
    """
    # 二值化处理：非零值视为前景
    binary_mask = mask != 0
    
    # 标记连通域（使用最大连通性）
    labeled, num_components = label(binary_mask)
    
    # 如果没有连通域，返回全零数组
    if num_components == 0:
        return np.zeros_like(mask)
    
    # 如果只有一个连通域，直接返回原掩码
    if num_components == 1:
        return mask.copy()
    
    # 计算每个连通域的面积（像素数量）
    areas = ndi_sum(binary_mask.astype(int), 
                   labels=labeled, 
                   index=np.arange(1, num_components+1))
    
    # 找到最大面积的连通域索引（+1 对应标签值）
    max_label = np.argmax(areas) + 1
    
    # 创建结果数组（初始全零）
    result = np.zeros_like(mask)
    
    # 保留最大连通域的原值
    max_component_mask = (labeled == max_label)
    result[max_component_mask] = mask[max_component_mask]
    
    return result

def filter_masks_by_area(gt_json_path, pred_json_path, output_json_path, area_threshold, skip_coco_eval=False):
    """
    Filters out masks with area smaller than a given threshold and saves the processed results.

    Parameters:
        gt_json_path (str): Path to the ground truth JSON file.
        pred_json_path (str): Path to the predictions JSON file.
        output_json_path (str): Path to save the processed predictions JSON file.
        area_threshold (int): Minimum area required for a mask to be retained.
        skip_coco_eval (bool): If True, skip COCO evaluation (for inference mode).
    """


    # Load the ground truth and predictions
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # Load predictions as a list of annotations
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    # Process each annotation
    filtered_predictions = []
    for i, annotation in enumerate(predictions):
        # Decode the RLE mask
        rle = annotation['segmentation']
        binary_mask = mask_utils.decode(rle)

        # fill_holes
        binary_mask = keep_largest_component(binary_mask)
        binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)
              
        # # 创建PIL图像对象
        # image_array = (result * 255).astype(np.uint8)
        # img = Image.fromarray(image_array)
        # filename = f'{i}.jpg'
        # # 保存图像
        # img.save(os.path.join('/home/jinghao/projects/dental_plague_detection/Self-PPD/tmp', filename))
        # print(f"图像已保存至: {filename}")

        binary_mask_rle = mask_utils.encode(np.asfortranarray(binary_mask))

        # Calculate the area of the mask
        area = mask_utils.area(binary_mask_rle)
        
        # If the area is above the threshold, keep the annotation
        if area >= area_threshold:
            annotation['segmentation'] = binary_mask_rle
            filtered_predictions.append(annotation)
    
    # Group predictions by image_id
    predictions_by_image = {}
    for annotation in filtered_predictions:
        image_id = annotation['image_id']
        # if image_id != 82:
        #     continue
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(annotation)

    # Resolve overlaps within each image
    adjusted_predictions = []
    for image_id, annotations in predictions_by_image.items():
        # Decode all masks for this image
        decoded_masks = []
        for annotation in annotations:
            rle = annotation['segmentation']
            decoded_masks.append(mask_utils.decode(rle))

        # Create an empty array to track overlaps
        height, width = decoded_masks[0].shape
        overlap_mask = np.zeros((height, width), dtype=np.int32)

        # Assign each mask a unique ID
        mask_ids = np.zeros((len(decoded_masks), height, width), dtype=np.uint8)
        for i, mask in enumerate(decoded_masks):

            mask_ids[i] = (mask > 0).astype(np.uint8) * (i + 1)
            overlap_mask += mask_ids[i]

        # Find overlap regions
        overlap_regions = overlap_mask > 1

        ########## Resolve overlaps based on category rules; for better visualization
        for i in range(len(decoded_masks)):
            for j in range(i + 1, len(decoded_masks)):
                # Find overlapping regions between mask i and mask j
                overlap = (mask_ids[i] > 0) & (mask_ids[j] > 0)
                if not np.any(overlap):
                    continue
                
                # Get categories of the two overlapping masks
                category_i = annotations[i]['category_id'] % 3
                category_j = annotations[j]['category_id'] % 3

                # Determine which category should dominate the overlap
                if category_i == 0 and category_j == 1:
                    dominant_category = 0
                elif category_i == 0 and category_j == 2:
                    dominant_category = 0
                elif category_i == 1 and category_j == 2:
                    dominant_category = 2 # 2
                elif category_j == 0 and category_i == 1:
                    dominant_category = 0
                elif category_j == 0 and category_i == 2:
                    dominant_category = 0
                elif category_j == 1 and category_i == 2:
                    dominant_category = 2 # 2
                else:
                    continue  # No overlap adjustment needed

                # Apply the dominant category to the overlap region
                if dominant_category == category_i:
                    mask_ids[j][overlap] = 0  # Remove overlap from mask j
                else:
                    mask_ids[i][overlap] = 0  # Remove overlap from mask i


        # Re-encode each adjusted mask and update annotations
        for i, mask in enumerate(mask_ids):
            binary_mask = (mask > 0).astype(np.uint8)
            adjusted_rle = mask_utils.encode(np.asfortranarray(binary_mask))
            adjusted_rle['counts'] = adjusted_rle['counts'].decode('utf-8')  # Ensure JSON serializable
            annotations[i]['segmentation'] = adjusted_rle

        # Append adjusted annotations to the final list
        adjusted_predictions.extend(annotations)


    # Save the filtered predictions to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(adjusted_predictions, f, indent=4)

    print(f"Processed predictions saved to {output_json_path}")

    # Run COCO evaluation if not skipped (skip for inference mode)
    if not skip_coco_eval:
        coco_pred = coco_gt.loadRes(output_json_path)
        coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postprocess PlaqueSAM predictions')
    parser.add_argument('--gt-json', required=True, help='Path to ground truth JSON')
    parser.add_argument('--pred-json', required=True, help='Path to predictions JSON')
    parser.add_argument('--output-json', required=True, help='Path to save processed predictions')
    parser.add_argument('--area-threshold', type=int, default=1000, help='Minimum area threshold (default: 1000)')
    parser.add_argument('--skip-coco-eval', action='store_true', help='Skip COCO evaluation (for inference mode)')
    
    args = parser.parse_args()
    
    filter_masks_by_area(args.gt_json, args.pred_json, args.output_json, args.area_threshold, args.skip_coco_eval)
