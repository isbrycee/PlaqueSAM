import os
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from torchvision import transforms


class TemplateFeatureExtractor:
    def __init__(self, folder_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.templates = self._load_templates(folder_path)
        self.feature_net = self._build_network().to(self.device)
        self.template_features = self._process_templates()
        
    def _load_templates(self, folder_path):
        """加载并预处理模板图像"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        templates = []
        # 按文件名排序读取图像
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('.png'):
                img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
                templates.append(transform(img))
        assert len(templates) == 6, "需要6个jpg模板文件"
        return templates

    def _build_network(self):
        """构建轻量级特征提取网络"""
        net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),  # [16, 256, 256]
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [32, 128, 128]
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
        # 网络参数初始化
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        return net.eval()

    def _process_templates(self):
        """处理模板生成特征"""
        features = []
        for tensor in self.templates:
            # 添加batch维度并送入网络
            feat = self.feature_net(tensor.unsqueeze(0).to(self.device))
            features.append(feat.squeeze(0))  # 返回CPU减少显存占用
        return features

    def get_prior_loc_template_memory(self):
        """获取处理后的模板特征列表"""
        _, H, W = self.template_features[0].shape
        num_levels = 4 # Match n_levels in DeformableAttention

        # Repeat features and masks for 4 levels
        multi_level_features = [torch.cat([feat.unsqueeze(0)] * num_levels, dim=0) for feat in self.template_features]
        
        template_memory_spatial_shapes = torch.tensor([[H, W]] * num_levels).to(self.device)

        template_memory_level_start_index = torch.tensor([i * H * W for i in range(num_levels)]).to(self.device).long()

        template_memory_key_padding_mask = torch.zeros(len(self.template_features), num_levels * H * W, dtype=torch.bool).to(self.device)

        return multi_level_features, template_memory_spatial_shapes, template_memory_level_start_index, template_memory_key_padding_mask
    