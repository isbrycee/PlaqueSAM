import torch

def compare_model_weights(model_path_1, model_path_2):
    # 加载两个模型权重
    state_dict_1 = torch.load(model_path_1, map_location=torch.device('cpu'))['model']
    state_dict_2 = torch.load(model_path_2, map_location=torch.device('cpu'))['model']

    # 检查两个 state_dict 的 key 是否一致
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())

    # 找出两个模型中不一致的 key
    diff_keys_1 = keys_1 - keys_2
    diff_keys_2 = keys_2 - keys_1

    # if diff_keys_1:
    #     print("模型 1 中多出的参数键：")
    #     for key in diff_keys_1:
    #         print(key)
    # if diff_keys_2:
    #     print("模型 2 中多出的参数键：")
    #     for key in diff_keys_2:
    #         print(key)

    # 找出 key 相同但参数值不同的部分
    common_keys = keys_1 & keys_2
    mismatched_keys = []
    for key in common_keys:
        if not torch.equal(state_dict_1[key], state_dict_2[key]):
            mismatched_keys.append(key)
    mismatched_keys.sort()
    if mismatched_keys:
        print("\n参数值不一致的键：")
        for key in mismatched_keys:
            print(key)
    else:
        print("\n所有参数值都一致。")

# 示例调用
model_path_1 = "/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_512_100e_lr1e-4_4scales_twostage_100queries_test_512_ToI_3rd_9masklayer_wboxTemp/checkpoints/checkpoint.pt"  # 替换为实际模型权重路径
model_path_2 = "/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_512_100e_lr1e-4_4scales_twostage_100queries_test_512_ToI_3rd_2masklayer_wboxTemp/checkpoints/checkpoint.pt"  # 替换为实际模型权重路径

compare_model_weights(model_path_1, model_path_2)
