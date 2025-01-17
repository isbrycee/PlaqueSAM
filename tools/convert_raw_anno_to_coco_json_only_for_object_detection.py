import os
import json
from datetime import datetime

class_name_to_idx_map = {'51':0, '52':1, '53':2, '54':3, '55':4, 
                            '61':5, '62':6, '63':7, '64':8, '65':9, 
                            '71':10, '72':11, '73':12, '74':13, '75':14,
                            '81':15, '82':16, '83':17, '84':18, '85':19,

                            '11': 20, '16': 21,
                            '21': 22, '26': 23,
                            '31': 24, '36': 25,
                            '41': 26, '46': 27,

                            'doubleteeth': 28,
                            'crown': 29,

                            '51_stain':0,'52_stain':1, '53_stain':2, '54_stain':3, '55_stain':4, 
                            '61_stain':5, '62_stain':6, '63_stain':7, '64_stain':8, '65_stain':9, '63_stan':7,
                            '71_stain':10, '72_stain':11, '73_stain':12, '74_stain':13, '75_stain':14, 
                            '81_stain':15, '82_stain':16, '83_stain':17, '84_stain':18, '85_stain':19, 
                            '71_stian':10, '36_stain': 25, '46_stain': 27, 

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

                            '55_crown':29,
                            '84_crown':29,
                            '74_crown':29,
                            
                            "55'":4,
                            '622':6,

                            '72\\3':28,
                            '72/3':28,
                            '82/83':28,
                            '81/82': 28,

                            'mouth_1': 30,
                            'mouth_2': 31,
                            'mouth_3': 32,
                            'mouth_4': 33,
                            'mouth_5': 34,
                            'mouth_6': 35,
                            'mouth_7': 36,
                            }

def convert_to_coco_format(input_folder, output_file):
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "COCO format dataset",
            "contributor": "",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_mapping = {}
    annotation_id = 1
    image_id = 1

    # Loop through subfolders and JSON files
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)

                # Load JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Get image information (assuming image file has same name as JSON but with .jpg/.png extension)
                image_name = os.path.splitext(file)[0] + ".JPG"
                image_width = 4032  # Replace with actual width if known
                image_height = 3024  # Replace with actual height if known

                # Add image info to COCO format
                coco_format["images"].append({
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": image_name,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": datetime.now().strftime("%Y-%m-%d")
                })

                # Process annotations
                for shape in data.get("shapes", []):
                    label = shape["label"]
                    if label in ['np', 'p', 'caries']:
                        continue
                    if label not in class_name_to_idx_map.keys():
                        print(label)
                    label = class_name_to_idx_map[label]
                    # print(label)
                    points = shape["points"]  # Two points define the rectangle

                    # COCO bbox format: [x_min, y_min, width, height]
                    x_min = min(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    x_max = max(points[0][0], points[1][0])
                    y_max = max(points[0][1], points[1][1])
                    width = x_max - x_min
                    height = y_max - y_min
                    if width <=0 or height <=0:
                        print("11111")
                    # Add category if not already present
                    if label not in category_mapping:
                        category_mapping[label] = len(category_mapping) + 1
                        coco_format["categories"].append({
                            "id": category_mapping[label],
                            "name": label,
                            "supercategory": "none"
                        })

                    # Add annotation
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_mapping[label],
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

                # Increment image ID
                image_id += 1
    print(len(category_mapping))
    # Save COCO format JSON to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=4)

# Input folder containing subfolders with JSON files
input_folder = "/home/jinghao/projects/dental_plague_detection/dataset/11_12_for_pseudo_labels_bbox/"
# Output COCO JSON file
output_file = "output_coco.json"

convert_to_coco_format(input_folder, output_file)