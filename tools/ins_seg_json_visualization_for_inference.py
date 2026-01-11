import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pycocotools.coco import COCO

box_class_color_dict = {
    # 5x: 冷色（蓝-紫-青）
    '51': (72, 61, 139),        # 深蓝紫
    '52': (65, 105, 225),       # 亮蓝
    '53': (0, 191, 255),        # 天蓝
    '54': (0, 139, 139),        # 深青
    '55': (0, 255, 255),        # 青色

    # 6x: 绿青色（青绿-黄绿-嫩绿）
    '61': (0, 255, 128),        # 鲜绿
    '62': (60, 179, 113),       # 中绿
    '63': (127, 255, 0),        # 黄绿色
    '64': (173, 255, 47),       # 青黄绿
    '65': (192, 255, 62),       # 嫩黄绿

    # 7x: 暖色（黄-橙-橙红）
    '71': (255, 255, 0),        # 纯黄
    '72': (255, 215, 0),        # 金黄
    '73': (255, 165, 0),        # 橙色
    '74': (255, 140, 0),        # 深橙
    '75': (255, 99, 71),        # 番茄红

    # 8x: 红色系（红橙-大红-深红）
    '81': (255, 69, 0),         # 橙红
    '82': (255, 0, 0),          # 纯红
    '83': (220, 20, 60),        # 猩红
    '84': (178, 34, 34),        # 深红
    '85': (139, 0, 0),          # 暗红

    # 其余色号保持不变或自定义
    '11': (255, 156, 15),
    '21': (255, 145, 0),
    '31': (255, 134, 0),
    '41': (255, 123, 0),
    '16': (255, 112, 0),
    '26': (255, 101, 0),
    '36': (255, 90, 0),
    '46': (255, 79, 0),
    'crown': (255, 68, 0),
    'doubleteeth': (255, 57, 0)
}

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return inter / union

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def adjust_color(color, factor=0.7):
    """
    调整颜色亮度：生成一个变暗或变浅的颜色，用于文字背景。
    
    参数:
        color: (B, G, R) 颜色值
        factor: 颜色调整因子 (0~1, 越小颜色越暗)
    返回:
        调整后的颜色 (B, G, R)
    """
    return tuple(max(0, min(255, int(c * factor))) for c in color)

def draw_box(img, bbox, label, color, thickness=2, font_scale=1.5):
    """
    绘制美观的边界框，并显示标签（文字背景颜色与边框颜色协调）。
    
    参数:
        img (ndarray): 图像。
        bbox (tuple): 边界框坐标 (x1, y1, x2, y2)。
        label (str): 标签文本。
        color (tuple): 边界框颜色 (B, G, R)。
        thickness (int): 边界框线条粗细。
        font_scale (float): 标签文本的字体大小。
    """
    x1, y1, x2, y2 = [int(round(b)) for b in bbox]
    
    # 绘制边框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    
    # 计算标签的尺寸
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=1)[0]
    text_width, text_height = text_size
    text_offset_x, text_offset_y = x1, max(0, y1 - text_height - 10)
    
    # 标签背景的坐标
    pad_x, pad_y = 10, 5  # 背景填充的内边距
    box_coords = (
        (text_offset_x, text_offset_y),  # 左上角
        (text_offset_x + text_width + 2 * pad_x, text_offset_y + text_height + 2 * pad_y)  # 右下角
    )
    
    # 计算文字背景颜色（基于边界框颜色，稍微调暗）
    bg_color = adjust_color(color, factor=0.7)
    
    # 绘制半透明的文字背景
    overlay = img.copy()
    alpha = 0.7  # 半透明度
    cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 动态调整文字颜色：根据背景颜色的亮度选择白色或黑色
    brightness = np.mean(bg_color)
    text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
    
    # 绘制标签文字
    cv2.putText(img, label, (text_offset_x + pad_x, text_offset_y + text_height + pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

def scale_bbox(bbox, scale_x, scale_y):
    # bbox: [x, y, w, h]
    x, y, w, h = bbox
    x = x * scale_x
    y = y * scale_y
    w = w * scale_x
    h = h * scale_y
    return [x, y, w, h]

def draw_mask_contour(img, mask, color, thickness=4):
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)

def draw_mask(img, mask, color, alpha=0.8, thickness=None):
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    img[mask == 1] = (img[mask == 1] * (1 - alpha) + color_arr * alpha).astype(np.uint8)

def draw_dotted_contour(image, contour, color, thickness=2, dash_len=5, space_len=3):
    """
    dash_len: 每个短线的像素长度
    space_len: 线段之间空白的像素长度
    """
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        dist = int(np.linalg.norm(p2 - p1))
        # 计算方向向量
        direction = (p2 - p1) / dist if dist > 0 else np.array([0, 0])
        
        pos = 0
        while pos < dist:
            start = p1 + direction * pos
            end = p1 + direction * min(pos + dash_len, dist)
            cv2.line(
                image,
                tuple(start.astype(int)),
                tuple(end.astype(int)),
                color,
                thickness
            )
            pos += dash_len + space_len  # 前进 dash + space

def vis_one_image(img_path, pred_anns, pred_boxes, gt_anns, coco_gt, cat_id_to_name, save_path, box_base_size=512, iou_thr=0.3):
    img = cv2.imread(img_path)
    if img is None:
        print('Image not found:', img_path)
        return
    
    h, w = img.shape[:2]
    
    # 1. 画box（缩放到当前图片大小）
    scale_x = w / box_base_size
    scale_y = h / box_base_size
    for box_info in pred_boxes:
        box = box_info['bbox']
        label = str(box_info['category'])
        # score = box_info['score']
        scaled_box = scale_bbox(box, scale_x, scale_y)
        draw_box(img, scaled_box, label, box_class_color_dict[label], 4)

    # 2. 可视化 category_id 能被 2 整除的 mask
    mask_anns = [ann for ann in pred_anns if ann['category_id'] % 3 == 0 or ann['category_id'] % 3 == 1]
    mask_colors = [
        (0, 0, 255),  # for tooth
        (255, 254, 2),  # for plaque
    ]
    overlay = img.copy()

    # --- category_id % 3 == 1 的 instance，保留原有mask染色可视化 ---
    mask_colors = (255, 254, 2)  # 你可以根据实际需求调整
    for ann in gt_anns: # pred_anns
        if ann['category_id'] % 3 == 1:
            mask = coco_gt.annToMask(ann)
            # for drawing tooth mask
            for c in range(3):
                overlay[:,:,c][mask==1] = mask_colors[c]

            # for drawing tooth edge
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (238, 238, 175), thickness=4)

    # for drawing tooth mask
    # img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    # for drawing tooth edge
    # img = cv2.addWeighted(overlay, 1.0, img, 1.0, 0)

    ######################### for TP/FP/FN vis #########################
    # --- 针对 category_id % 3 == 0 的 instance，计算 TP/FP/FN 并可视化 ---
    # 1. 筛选
    pred_target = [ann for ann in pred_anns if ann['category_id'] % 3 == 0]
    gt_target = [ann for ann in gt_anns if ann['category_id'] % 3 == 0]
    
    # 2. 生成mask
    pred_masks = [coco_gt.annToMask(ann) for ann in pred_target]
    # pred_masks = [] # for drawing gt
    gt_masks = [coco_gt.annToMask(ann) for ann in gt_target]    

    # 3. 匹配IoU
    gt_matched = set()
    pred_matched = set()
    tp_pairs = []
    for i, p_mask in enumerate(pred_masks):
        best_iou = 0
        best_j = -1
        for j, g_mask in enumerate(gt_masks):
            if j in gt_matched:
                continue
            iou = compute_iou(p_mask, g_mask)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j != -1:
            tp_pairs.append((i, best_j))
            gt_matched.add(best_j)
            pred_matched.add(i)
    tp_pred_idx = set(i for i, _ in tp_pairs)
    tp_gt_idx = set(j for _, j in tp_pairs)
    fp_pred_idx = set(range(len(pred_masks))) - tp_pred_idx
    fn_gt_idx = set(range(len(gt_masks))) - tp_gt_idx

    # 4. 可视化
    for i in tp_pred_idx:
        draw_mask_contour(img, pred_masks[i], (0,0,255), thickness=6)     # TP: 绿色
    for i in fp_pred_idx:
        draw_mask_contour(img, pred_masks[i], (0,0,255), thickness=6)     # FP: 红色 (0,0,255)
    for j in fn_gt_idx:
        draw_mask_contour(img, gt_masks[j],  (0,0,255), thickness=6)     # FN: 黄色  # 画 gt 时使用紫色 (155, 121, 255)

    # for idx, ann in enumerate(mask_anns):
    #     mask = coco_gt.annToMask(ann)
    #     assert ann['category_id'] % 3 in [0, 1]
    #     color = mask_colors[ann['category_id'] % 3]
    #     for c in range(3):
    #         overlay[:, :, c][mask == 1] = color[c]

    ######################### END vis #########################

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def main(gt_json_path, pred_mask_json_path, pred_box_json_path, image_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gt_json = load_json(gt_json_path)
    pred_mask_json = load_json(pred_mask_json_path)
    pred_box_json = load_json(pred_box_json_path)

    # 用COCO对象读取gt，方便annToMask
    coco_gt = COCO(gt_json_path)

    # file_name <-> image_id 映射
    file_name2id = {img['file_name']: img['id'] for img in gt_json['images']}

    # 预测分割的anns按image_id分组
    pred_mask_anns_per_img = defaultdict(list)
    for ann in pred_mask_json:
        pred_mask_anns_per_img[ann['image_id']].append(ann)

    # box按file_name分组
    pred_boxes_per_file = defaultdict(list)
    for box in pred_box_json:
        pred_boxes_per_file[box['file_name']].append(box)

    # gt anns按image_id分组
    gt_mask_anns_per_img = defaultdict(list)
    for ann in gt_json['annotations']:
        gt_mask_anns_per_img[ann['image_id']].append(ann)

    # 类别id到名字
    cat_id_to_name = {cat['id']: cat['name'] for cat in gt_json['categories']}

    for file_name in tqdm(file_name2id):
        image_id = file_name2id[file_name]
        img_path = os.path.join(image_root, file_name)
        pred_anns = pred_mask_anns_per_img[image_id]
        pred_boxes = pred_boxes_per_file[file_name]
        gt_anns = gt_mask_anns_per_img[image_id]
        # import pdb; pdb.set_trace()
        # save_path = os.path.join(out_dir, file_name.replace('/', '_'))
        save_path = os.path.join(out_dir, file_name)
        vis_one_image(img_path, pred_anns, pred_boxes, gt_anns, coco_gt, cat_id_to_name, save_path)


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Base path is two levels up from tools/
    base_path = os.path.abspath(os.path.join(script_dir, '..'))
    
    gt_json_path = os.path.join(base_path, 'demo_PlaqueSAM/infer_ins_ToI.json')
    pred_mask_json_path = os.path.join(base_path, 'logs_infer_demo/saved_jsons/_pred_val_epoch_000.json')
    pred_box_json_path = os.path.join(base_path, 'logs_infer_demo/saved_jsons/_box_pred_val_epoch_000_for_visualization.json')
    image_root = os.path.join(base_path, 'demo_PlaqueSAM/JPEGImages')
    out_dir = os.path.join(base_path, 'demo_PlaqueSAM/visualizations/')
    
    main(gt_json_path, pred_mask_json_path, pred_box_json_path, image_root, out_dir)
