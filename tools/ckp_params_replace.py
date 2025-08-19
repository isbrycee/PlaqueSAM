import torch

def replace_weights(source_file, target_file, save_file):
    """
    将 source_file 中 key 包含 'image_classify_decoder' 的权重
    替换到 target_file 中，并保存到 save_file。
    """
    # 加载权重文件
    source_weights = torch.load(source_file)
    target_weights = torch.load(target_file)

    first_stage_trained_params = ['image_classify_decoder'] 
    second_stage_trained_params = ['box_decoder', 'conv_s0', 'conv_s1', 'obj_ptr_proj', 'obj_ptr_tpos_proj']
    third_stage_trained_params = ['sam_prompt_encoder', 'sam_mask_decoder', 'no_mem_embed', 'no_mem_pos_enc', 'mask_downsample', 'memory_attention'] # 

    # 遍历 source_weights 的 keys
    for key, value in source_weights['model'].items():
        # print(key)
        if (any(s in key for s in third_stage_trained_params) and
                all(s not in key for s in second_stage_trained_params)) or ("image_classify_decoder" in key):
            if key in target_weights['model'].keys():
                print(f"替换权重: {key}")
                target_weights['model'][key] = value
            # else:
            #     print(f"新增权重: {key}")

    # 保存新权重文件
    torch.save(target_weights, save_file)
    print(f"新的权重文件已保存到: {save_file}")

    for key, value in target_weights['model'].items():
        print(key)

# 使用示例
source_file = "/home/jinghao/projects/dental_plague_detection/Self-PPD/exps_FINAL/FINAP_PlaqueSAM_logs_Eval_testset_revised_2025_July_512_ToI_3rd_9masklayer_wboxTemp/checkpoints/best_ins_seg_map50_imgAcc_0.986.pt"  # 替换的来源权重
target_file = "/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_512_100e_lr1e-4_4scales_twostage_100queries_test_512_ToI_2nd_9masklayer_woboxTemp_add_bn_relu/checkpoints/checkpoint_21.pt"  # 目标权重
save_file = "/home/jinghao/projects/dental_plague_detection/Self-PPD/logs_tiny_512_100e_lr1e-4_4scales_twostage_100queries_test_512_ToI_2nd_9masklayer_woboxTemp_add_bn_relu/checkpoints/checkpoint_21_replace_mask_decoder_with_ours.pt"       # 保存的新权重文件
replace_weights(source_file, target_file, save_file)
