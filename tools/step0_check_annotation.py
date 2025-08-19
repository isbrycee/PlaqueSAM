import os
import shutil
import json
from tqdm import tqdm

angle_tooth_id_check_dict = {
    '3': [0,1,2,3,4,20,21, 28, 29],
    '4': [15,16,17,18,19,26,27, 28, 29],
    '5': [5,6,7,8,9,22,23, 28, 29],
    '6': [10,11,12,13,14, 24,25, 28, 29],
}


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
                            '71_stian':10,

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
                            '110': 15, # 81
                            # '585':19,
                            # '875':14,

                            '72\\3':28,
                            '72/3':28,
                            '82/83':28,
                            '81/82': 28,
                
                            # '42':16,
                            # '32':11,
                            # '22': 0,
                            # '23': 0, 
                            # '24': 0,
                            # '25': 0,

                            }

def contains_letters(s):
    return any(c.isalpha() for c in s)

def correct_coordinates(top_left, bottom_right):
    # 解构坐标点
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # 确保 (x1, y1) 是左上角，(x2, y2) 是右下角
    correct_top_left = (min(x1, x2), min(y1, y2))
    correct_bottom_right = (max(x1, x2), max(y1, y2))
    
    return correct_top_left, correct_bottom_right

def check_and_resave_jsons(json_folder, save_folder):

    images = [f for f in os.listdir(json_folder) if os.path.isfile(os.path.join(json_folder, f))]
    image_folder_list = images
    
    class_set = set()
    for _json in image_folder_list:
        src_json = os.path.join(json_folder, _json)
        if '.DS_Store' in src_json:
            continue
        # print(src_json)
        # angle_class = -1
        # tooth_id_list = []
        with open(src_json, 'r') as f_json:
            json_data = json.load(f_json)
        json_data['imageData'] = ''
        for _item in json_data['shapes']:
            if _item["shape_type"] == "rectangle":
                if _item['label'].startswith('mouth'):
                    if '_' not in _item['label']:
                        print(src_json)
                        print(_item['label'])
                    else:
                        if int(_item['label'].split('_')[1]) > 7:
                            _item['label'] = 'mouth_7'
                        # angle_class = _item['label'].split('_')[-1]
                    # print(src_json)
                    assert len(_item['label'].split('_')) == 2
                    assert _item['label'].split('_')[0] == 'mouth'
                    assert int(_item['label'].split('_')[1]) > 0 and int(_item['label'].split('_')[1]) <= 7
                    if int(_item['label'].split('_')[1]) != 7:
                        class_set.add(_item['label'].split('_')[1])
                    # for point convert cases
                    top_left = _item['points'][0]
                    bottom_right = _item['points'][1]
                    top_left, bottom_right = correct_coordinates(top_left, bottom_right)
                    _item['points'][0] = top_left
                    _item['points'][1] = bottom_right

                elif _item['label'] in class_name_to_idx_map.keys():
                    top_left = _item['points'][0]
                    bottom_right = _item['points'][1]
                    top_left, bottom_right = correct_coordinates(top_left, bottom_right)
                    _item['points'][0] = top_left
                    _item['points'][1] = bottom_right
                    # tooth_id_list.append(class_name_to_idx_map[_item['label']])
                elif 'crown' in _item['label']:
                    _item['label'] = 'crown'
                elif "_" in _item['label']:
                    class_name = _item['label'].split('_')[0]
                    if class_name in class_name_to_idx_map.keys():
                        _item['label'] = class_name
                    else:
                        print(src_json)
                        print(_item['label'] + " is bad class name !!!")
                elif '/' in _item['label'] or '\\' in _item['label']:
                    class_name = 'doubleteeth'
                    _item['label'] = class_name
                elif _item['label'].isdecimal() and len(_item['label']) == 2:
                    _item['label'] = str(int(_item['label']) + 40)
                else:
                    print(src_json)
                    print(_item['label'])
                    # import pdb; pdb.set_trace()

        # if angle_class == -1:
        #     print(src_json)
        #     print("The category of mouth is incorrect!!")
        # elif angle_class == '2' or angle_class == '1':
        #     pass
        # else:
        #     for tooth_id in tooth_id_list:
        #         if tooth_id not in angle_tooth_id_check_dict[angle_class]:
        #             print(src_json)
        #             print("The category of tooth_id is incorrct!!")

        with open(os.path.join(save_folder, _json) ,"w") as f_w:
            json.dump(json_data, f_w)
            
    if len(class_set) != 6:
        print(json_folder + ' is less than 6 classes !!! Pls fixed it !!!')
        # import pdb; pdb.set_trace()


def organize_data(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"路径不存在: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"提供的路径不是一个文件夹: {root_dir}")

    # 遍历根目录下的直接子文件夹
    child_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    if not child_folders:
        raise ValueError(f"根目录 `{root_dir}` 中没有子文件夹。")

    for child_folder in child_folders:
        # 遍历每个子文件夹的子文件夹
        nested_folders = [f.path for f in os.scandir(child_folder) if f.is_dir()]
        if not nested_folders:
            print(f"子文件夹 `{child_folder}` 中没有子文件夹，跳过检查。")
            continue
        if 'checked' in nested_folders:
            continue
        for nested_folder in nested_folders:
            # 检查子文件夹中的文件数量
            files = [f for f in os.listdir(nested_folder) if os.path.isfile(os.path.join(nested_folder, f))]
            file_count = len(files)
            if file_count != 6:
                print(f"子文件夹 `{nested_folder}` 中有 {file_count} 个文件，不符合要求！")
                # raise ValueError(f"子文件夹 `{nested_folder}` 中有 {file_count} 个文件，不符合要求！")

    folder_prefixes = []  # Collect folder prefixes for checking
    for folder in os.listdir(root_dir):
        if not 'post' in folder:
            continue
        if 'checked' in folder:
            continue
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            data_folders = os.listdir(folder_path)
            for data_folder in tqdm(data_folders):
                data_path = os.path.join(folder_path, data_folder)

                if os.path.isdir(data_path):
                    if contains_letters(data_folder):
                        folder_prefix = f"{folder}_{data_folder.split('.')[0]}"
                    else:
                        folder_prefix = f"{folder}_{data_folder}"
                        
                    folder_prefixes.append(folder_prefix)
                    json_folder = data_path
                    json_save_dir = os.path.join(root_dir, folder + '_checked', data_folder)
                    os.makedirs(json_save_dir, exist_ok=True)
                    check_and_resave_jsons(json_folder, json_save_dir)

# 输入多个根目录路径
root_directories = "/home/jinghao/projects/dental_plague_detection/dataset/testset/"

date = [ '9_26', '10_8/', '10_10', '10_24', '10_31', '11_3', '11_7', '11_12', '11_19', '11_20', '12_3', '12_5']
# date = [ '12_5']

# root_directories = ["/home/hust/haojing/dental_plague_dataset/10_24", ]
# root_directories = '/home/hust/haojing/dental_plague_dataset/raw_data'
# resaved_json_dir = '/home/hust/haojing/dental_plague_dataset/raw_data/resaved_json'

for root_dir in date:
    root_dir = os.path.join(root_directories, root_dir)
    # single_path = os.path.join(root_directories, root_dir)
    organize_data(root_dir)