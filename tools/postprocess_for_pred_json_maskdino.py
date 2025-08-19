import json

# 读取原始 JSON 文件
input_file = '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/projects/PointSup/output/inference/coco_instances_results.json'  # 输入文件名
output_file = '/home/jinghao/projects/dental_plague_detection/MaskDINO/detectron2/projects/PointSup/output/inference/coco_instances_results_score_over_0.50.json'  # 输出文件名

# 设定过滤阈值
threshold = 0.50

# 读取 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤 score 大于 0.7 的元素
filtered_data = [item for item in data if item.get('score', 0) > threshold]

print(f'{len(filtered_data)} instances reserved !')

# 将过滤后的数据写入新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"过滤完成，共保留 {len(filtered_data)} 个元素，结果已保存到 {output_file}")