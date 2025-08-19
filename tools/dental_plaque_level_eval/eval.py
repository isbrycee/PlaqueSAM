import json

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from tqdm import tqdm

from mapping import ImageToothMapping


def calculate_plaque_levels(coco_predictions, conf_thresh=None):
    img_tooth_mapping = ImageToothMapping()

    # Group predictions by image id and filter by score threshold
    predictions = {
        img_id: [
            pred for pred in coco_predictions
            if pred['image_id'] == img_id and (conf_thresh is None or pred['score'] >= conf_thresh)
        ]
        for img_id in set([pred['image_id'] for pred in coco_predictions])
    }

    plaque_levels = []
    for img_id, preds in tqdm(predictions.items()):
        toi_categories = img_tooth_mapping[img_id]
        for tooth_id, tooth_category_ids in toi_categories.items():
            plaque_area = 0
            total_area = 0
            num_instances = {'plaque': 0, 'non_plaque': 0, 'caries': 0}
            for pred in preds:
                category = tooth_category_ids.get(pred['category_id'], None)
                if category is not None:
                    size = pred['segmentation']['size']
                    counts = pred['segmentation']['counts']
                    mask = torch.as_tensor(mask_util.decode({'size': size, 'counts': counts}), dtype=torch.bool)
                    area = mask.sum().item()
                    total_area += area
                    num_instances[category] += 1
                    if category == 'plaque':
                        plaque_area += area

            if total_area > 0:
                plaque_levels.append({
                    'image_id': img_id,
                    'tooth_id': tooth_id,
                    'plaque_level': plaque_area / total_area,
                    'num_instances': num_instances
                })

    return plaque_levels


def plaque_level_metrics(gt_plaque_levels, plaque_levels):

    def _get_level_cat(p_level):
        if p_level is None or p_level == 0:
            return 0
        elif 0 < p_level <= 1/3:
            return 1
        elif 1/3 < p_level <= 2/3:
            return 2
        else:
            return 3

    gt_plaque_levels = {(x['image_id'], x['tooth_id']): x['plaque_level'] for x in gt_plaque_levels}
    plaque_levels = {(x['image_id'], x['tooth_id']): x['plaque_level'] for x in plaque_levels}

    # Add missing keys to ground truth
    missing_keys = set(plaque_levels.keys()) - set(gt_plaque_levels.keys())
    gt_plaque_levels.update({key: None for key in missing_keys})
    print("Missing keys in ground truth:", missing_keys)

    tooth_metrics = {}
    for (img_id, tooth_id), gt_level in gt_plaque_levels.items():
        pred_level = plaque_levels.get((img_id, tooth_id), None)
        gt_level_cat = _get_level_cat(gt_level)
        pred_level_cat = _get_level_cat(pred_level)
        if tooth_id not in tooth_metrics:
            tooth_metrics[tooth_id] = {'correct': 0, 'count': 0, 'error': 0}
        tooth_metrics[tooth_id]['correct'] += \
            int(gt_level_cat == pred_level_cat)
        tooth_metrics[tooth_id]['error'] += \
            abs((gt_level or 0) - (pred_level or 0))
        tooth_metrics[tooth_id]['count'] += 1

    for tooth_id, metrics in tooth_metrics.items():
        metrics['accuracy'] = metrics['correct'] / metrics['count']
        metrics['mae'] = metrics['error'] / metrics['count']

    total_count = sum([metrics['count'] for metrics in tooth_metrics.values()])
    total_correct = sum([metrics.pop('correct') for metrics in tooth_metrics.values()])
    total_error = sum([metrics.pop('error') for metrics in tooth_metrics.values()])

    total_metrics = {
        'count': total_count,
        'accuracy': total_correct / total_count,
        'mae': total_error / total_count
    }

    return tooth_metrics, total_metrics


def evaluate(gt_path, results_path, conf_thresh):
    # Load ground truth annotations
    coco_gt = COCO(gt_path)

    # Load predictions
    if results_path.endswith('.pth'):
        predictions = torch.load(results_path)
        coco_predictions = [instance for prediction in predictions for instance in prediction['instances']]
    elif results_path.endswith('.json'):
        with open(results_path) as f:
            coco_predictions = json.load(f)
    else:
        raise ValueError("Results file must be a .pth or .json file")

    print("Number of predictions:", len(coco_predictions))

    # Fix bounding boxes
    for prediction in coco_predictions:
        if 'bbox' in prediction and (prediction['bbox'] is None or sum(prediction['bbox']) == 0):
            prediction.pop('bbox')

    # Create COCO results object
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Initialize COCOeval object for bounding box evaluation
    print("Evaluating bounding box metrics...")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    # Initialize COCOeval object for mask evaluation
    print("Evaluating mask metrics...")
    coco_eval_mask = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()

    # Calculate plaque levels
    print("Calculating plaque levels of the ground truth...")
    gt_plaque_levels = calculate_plaque_levels(coco_gt.dataset['annotations'])
    print("Calculating plaque levels of the results...")
    plaque_levels = calculate_plaque_levels(coco_predictions, conf_thresh)

    # Calculate plaque level accuracy
    tooth_metrics, total_metrics = plaque_level_metrics(gt_plaque_levels, plaque_levels)
    print("Total Plaque Level Metrics:", total_metrics)
    print("Per-Tooth Plaque Level Metrics:")
    for tooth_id, metrics in sorted(tooth_metrics.items()):
        print(f"    Tooth {tooth_id}: {metrics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate results in the COCO JSON format.")
    parser.add_argument('-p', "--predictions_path", type=str, help="Path to the predictions JSON or PTH file.")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold for predictions when calculating plaque level.")
    args = parser.parse_args()
    gt_path = "instances_test.json"
    evaluate(gt_path, args.predictions_path, args.conf_thresh)
