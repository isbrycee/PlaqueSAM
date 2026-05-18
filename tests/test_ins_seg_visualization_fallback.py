import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import ins_seg_json_visualization_for_inference as vis


class FakeCoco:
    def annToMask(self, ann):
        return ann['mask']


def mask_with_rect(x1, y1, x2, y2, shape=(20, 20)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = 1
    return mask


def run_vis(monkeypatch, pred_anns, pred_boxes):
    calls = []

    def capture_draw_box(_img, bbox, label, color, thickness=2, font_scale=1.5):
        calls.append({
            'bbox': tuple(bbox),
            'label': label,
            'color': color,
            'thickness': thickness,
        })

    monkeypatch.setattr(vis.cv2, 'imread', lambda _path: np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(vis.cv2, 'imwrite', lambda _path, _img: True)
    monkeypatch.setattr(vis.os, 'makedirs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vis, 'draw_box', capture_draw_box)

    vis.vis_one_image(
        'image.png',
        pred_anns,
        pred_boxes,
        [],
        FakeCoco(),
        {},
        'out/image.png',
    )
    return calls


def test_fallback_labels_non_caries_masks_and_skips_caries(monkeypatch):
    calls = run_vis(
        monkeypatch,
        [
            {'category_id': 0, 'mask': mask_with_rect(2, 3, 5, 7)},
            {'category_id': 1, 'mask': mask_with_rect(8, 2, 12, 4)},
            {'category_id': 2, 'mask': mask_with_rect(1, 1, 3, 3)},
        ],
        [],
    )

    assert [call['label'] for call in calls] == ['51', '51']
    assert calls[0]['bbox'] == (2, 3, 5, 7)
    assert calls[0]['color'] == vis.box_class_color_dict['51']


def test_existing_box_label_prevents_duplicate_fallback_box(monkeypatch):
    calls = run_vis(
        monkeypatch,
        [
            {'category_id': 0, 'mask': mask_with_rect(2, 3, 5, 7)},
            {'category_id': 3, 'mask': mask_with_rect(8, 2, 12, 4)},
        ],
        [{'file_name': 'image.png', 'category': '51', 'bbox': [1, 1, 6, 6]}],
    )

    assert [call['label'] for call in calls] == ['51', '52']


def test_missing_or_incomplete_box_json_degrades_to_fallback(monkeypatch):
    assert vis.load_box_json('missing-boxes.json') == []

    calls = run_vis(
        monkeypatch,
        [{'category_id': 60, 'mask': mask_with_rect(4, 5, 9, 10)}],
        [{'file_name': 'image.png'}, {'category': '11', 'bbox': [1, 2, 3, 4]}],
    )

    assert [call['label'] for call in calls] == ['11']
    assert calls[0]['color'] == vis.box_class_color_dict['11']


def test_unknown_fdi_label_uses_safe_fallback_color_and_box_clamps_to_image():
    assert vis.label_for_category_id(999) == '333'
    assert vis.color_for_label('333') == vis.DEFAULT_BOX_COLOR
    assert vis.clamp_bbox_to_image((-5, -2, 30, 25), (20, 20, 3)) == (0, 0, 19, 19)
