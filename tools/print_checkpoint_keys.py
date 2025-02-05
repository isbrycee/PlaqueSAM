import torch

DINO_pretrained_models = '/home/jinghao/projects/dental_plague_detection/DINO-main/logs/DINO/R50-MS4/checkpoint0025.pth'
Tri_P_models = '/home/jinghao/projects/dental_plague_detection/Self-PPD/checkpoints/sam2.1_hiera_tiny.pt'

# # 加载 .pt 文件
# weights = torch.load(Tri_P_models)['model']
# # # 打印模型参数名字
# for name, param in weights.items():
#     print(name)


# 加载权重文件 A 和 B
Tri_P_models = torch.load(Tri_P_models)['model']
DINO_pretrained_models = torch.load(DINO_pretrained_models)['model']

# 遍历 B 的权重
for name, param in DINO_pretrained_models.items():
    if 'backbone.'in name:
        continue
    elif 'transformer.encoder.' in name:
        continue
    elif 'level_embed' in name:
        continue
    elif 'enc_output' in name:
        continue
    elif 'enc_out_bbox_embed' in name:
        continue
    elif 'enc_out_class_embed' in name:
        continue
    elif 'transformer.' in name:
        name = name.replace('transformer.', 'box_decoder.')
    elif 'bbox_embed.' in name:
        name = 'box_decoder.' + name
    elif 'class_embed.' in name:
        name = 'box_decoder.' + name
    else:
        print(name)
        continue

    Tri_P_models[name] = param
    
    # 如果 A 中存在同名参数，并且形状一致，则用 B 的权重替换 A 的权重
    # if name in Tri_P_models and Tri_P_models[name].shape == param.shape:
    #     print(f"Updating {name} in A with weights from B")
    #     Tri_P_models[name] = param
    # else:
    #     continue
    #     print(f"Skipping {name} (not found in A or shape mismatch)")

# 保存更新后的权重 A
Tri_P_models = {'model': Tri_P_models}
torch.save(Tri_P_models, 'updated_weights_A.pt')
print("Updated weights saved to 'updated_weights_A.pt'")