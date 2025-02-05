import os
import json
import csv
from shapely.geometry import Polygon, box
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

# train or test root, containing the dirs:/JPEGImages, /Annotaions, /Json
root_dir = "/train_root/"

images_dir = os.path.join(root_dir, "JPEGImages")
annotations_dir = os.path.join(root_dir, "Json")

output_name = "train_revised"

coco_data = {
    "info": {
        "description": "Custom Dataset",
        "url": "",
        "version": "1.0",
        "year": 2025,
        "contributor": "",
        "date_created": "2025-01-22"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

category_mapping = {f"{tooth}_{label}": i for tooth in range(30) for i, label in
                    enumerate(["p", "np", "caries"], start=tooth * 3)}

coco_data["categories"] = [
    {"id": idx, "name": name} for name, idx in category_mapping.items()
]

class_name_to_idx_map = {'51': 0, '52': 1, '53': 2, '54': 3, '55': 4,
                         '61': 5, '62': 6, '63': 7, '64': 8, '65': 9,
                         '71': 10, '72': 11, '73': 12, '74': 13, '75': 14,
                         '81': 15, '82': 16, '83': 17, '84': 18, '85': 19,

                         '11': 20, '16': 21,
                         '21': 22, '26': 23,
                         '31': 24, '36': 25,
                         '41': 26, '46': 27,

                         'doubleteeth': 28,
                         'crown': 29,

                         '51_stain': 0, '52_stain': 1, '53_stain': 2, '54_stain': 3, '55_stain': 4,
                         '61_stain': 5, '62_stain': 6, '63_stain': 7, '64_stain': 8, '65_stain': 9, '63_stan': 7,
                         '71_stain': 10, '72_stain': 11, '73_stain': 12, '74_stain': 13, '75_stain': 14,
                         '81_stain': 15, '82_stain': 16, '83_stain': 17, '84_stain': 18, '85_stain': 19,
                         '71_stian': 10,

                         '52_retainedteeth': 1,
                         '53_retainedteeth': 2,
                         '75_discoloration': 14,
                         '51_discoloration': 0,
                         '51_retainedteeth': 0,
                         '61_retainedteeth': 5,
                         '62_retainedteeth': 6,
                         '64_retainedteeth': 8,
                         '63_retainedteeth': 7,
                         '54_retainedteeth': 3,
                         '74_retainedteeth': 13,
                         '61_discoloration': 5,

                         '55_crown': 29,
                         '84_crown': 29,
                         '74_crown': 29,

                         "55'": 4,
                         '622': 6,

                         # '585':19,
                         # '875':14,

                         '72\\3': 28,
                         '72/3': 28,
                         '82/83': 28,
                         '81/82': 28,

                         '110': 15,

                         # '42':16,
                         # '32':11,
                         # '22': 0,
                         # '23': 0,
                         # '24': 0,
                         # '25': 0,

                         }

polyg_tooth_match_issues = []
poly_shape_not_valid = []


def get_annotations(json_path, image_id, annotation_id):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    shapes = json_data['shapes']

    # Skip the image if 'mouth_7'
    if any(shape['label'] == 'mouth_7' for shape in shapes):
        return None, annotation_id

    polygons = [
        {
            "label": shape["label"],
            "points": shape["points"]
        }
        for shape in shapes if shape['shape_type'] == 'polygon' and shape['label'] in {"p", "np", "caries"}
    ]
    teeth = [
        {
            "label": shape["label"],
            "points": shape["points"]
        }
        for shape in shapes if shape['shape_type'] == 'rectangle' and not shape['label'].startswith('mouth')
    ]

    annotations = []
    for polygon in polygons:
        poly_iss = False
        poly_shape = Polygon(polygon["points"])
        if not poly_shape.is_valid:
            poly_iss = True
            area_before_fix = poly_shape.area
            print(f"\nInvalid polygon detected in file '{os.path.relpath(json_path, root_dir)}'. Attempting to fix...")
            print(f"  Polygon area before fix: {poly_shape.area}, Label: '{polygon.get('label', 'unknown')}'")
            poly_shape = poly_shape.buffer(0)
            area_after_fix = poly_shape.area

            if not poly_shape.is_valid:
                status = "Not Fixed"
                print(
                    f"Failed to fix invalid polygon geometry in file '{os.path.relpath(json_path, root_dir)}': {polygon['points']}")
                continue
            else:
                status = "Fixed"
                print("Fixed")
                print(f"  Polygon area  after fix: {poly_shape.area}, Label: '{polygon.get('label', 'unknown')}'")

        polygon_area = poly_shape.area
        matches = []
        for tooth in teeth:
            x1, y1 = tooth['points'][0]
            x2, y2 = tooth['points'][1]
            tooth_box = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            intersection_area = poly_shape.intersection(tooth_box).area
            if intersection_area > 0:
                matches.append({
                    "tooth_label": tooth["label"],
                    "intersection_area": intersection_area
                })

        if matches:
            matches.sort(key=lambda x: x["intersection_area"], reverse=True)
            best_match = matches[0]
            second_best = matches[1] if len(matches) > 1 else None

            if best_match["intersection_area"] / polygon_area < 0.97:
                relative_path = os.path.relpath(json_path, root_dir)
                polyg_tooth_match_issues.append({
                    "file": relative_path,
                    "polygon_label": polygon['label'],
                    "tooth_label": best_match['tooth_label'],
                    "intersection_percentage": best_match["intersection_area"] / polygon_area
                })
                print(f"\nWarning: Potential issue with polygon '{polygon['label']}' in file: '{relative_path}'.")
                print(
                    f"  Best match: tooth '{best_match['tooth_label']}', intersection: ({(best_match['intersection_area'] / polygon_area) * 100:.2f}%)")
                if second_best:
                    print(
                        f"  Second best: '{second_best['tooth_label']}' ({(second_best['intersection_area'] / polygon_area) * 100:.2f}%)")
            tooth_id = class_name_to_idx_map[best_match["tooth_label"]]
            category_id = category_mapping[f"{tooth_id}_{polygon['label']}"]

            x_coords = [point[0] for point in polygon["points"]]
            y_coords = [point[1] for point in polygon["points"]]
            x_min, y_min = min(x_coords), min(y_coords)
            bbox_width, bbox_height = max(x_coords) - x_min, max(y_coords) - y_min

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [[coord for point in polygon["points"] for coord in point]],
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": polygon_area,
                "iscrowd": 0
            })
            annotation_id += 1
            if poly_iss:
                poly_shape_not_valid.append({
                    "file": os.path.relpath(json_path, root_dir),
                    "label": f"{tooth_id}_{polygon['label']}",
                    "area_before_fix": area_before_fix,
                    "area_after_fix": area_after_fix,
                    "status": status
                })
    return annotations, annotation_id


image_id = 1
annotation_id = 1

for dir_name in sorted(os.listdir(images_dir)):
    dir_path = os.path.join(images_dir, dir_name)

    for image_name in sorted(os.listdir(dir_path)):

        name_in_coco = f"{dir_name}/{image_name}"
        image_path = os.path.join(dir_path, image_name)
        with Image.open(image_path) as img:
            width, height = img.size

        json_file_path = os.path.join(annotations_dir, dir_name, f"{os.path.splitext(image_name)[0]}.json")
        if os.path.exists(json_file_path):
            new_annotations, annotation_id = get_annotations(json_file_path, image_id, annotation_id)
            if new_annotations is None:  # Skip the image (when mouth_7)
                continue
            coco_data["images"].append({
                "id": image_id,
                "file_name": name_in_coco,
                "width": width,
                "height": height
            })
            coco_data["annotations"].extend(new_annotations)

            image_id += 1


def save_coco_file(output_path):
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


def save_csv(data, output_path, fieldnames):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


output_coco_path = os.path.join(root_dir, f"{output_name}.json")
save_coco_file(output_coco_path)

save_csv(polyg_tooth_match_issues, os.path.join(root_dir, f"{output_name}_polyg_tooth_match_Issues.csv"),
         ["file", "polygon_label", "tooth_label", "intersection_percentage"])
save_csv(poly_shape_not_valid, os.path.join(root_dir, f"{output_name}_poly_shape_notValid.csv"),
         ["file", "label", "area_before_fix", "area_after_fix", "status"])





# load the CoCo file & and print the category names

with open(output_coco_path, 'r') as f:
    coco_data = json.load(f)

print("\nCategory names in the COCO file:")
for category in coco_data.get("categories", []):
    print(f"ID: {category['id']}, Name: {category['name']}")
