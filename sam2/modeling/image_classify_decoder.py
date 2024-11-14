import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from sam2.modeling.sam2_utils import MLP, _get_activation_fn
from torch import Tensor
import random


class image_classify_decoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_mlp_layers: int = 3,
        num_classes: int = 3,
        num_frames: int = 6,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.num_classes = num_classes
        self.num_frames = num_frames
        assert (
            self.num_classes == 7
        ), f"The number of classes must be 7 ! Current setting is : {self.num_classes}. Pls change it to 7."

        self.activation = _get_activation_fn(activation, d_model=self.hidden_dim, batch_dim=-1)

        # prepare class & box embed
        self._class_head = nn.Linear(hidden_dim, self.num_classes)
        self._class_mlp = MLP(self.input_dim, hidden_dim, hidden_dim, self.num_mlp_layers)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._class_head.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self._class_mlp.layers[-1].weight.data, 0)
        nn.init.constant_(self._class_mlp.layers[-1].bias.data, 0)

    def forward(self, backbone_out: dict):

        img_feature = backbone_out['vision_features'] # [6, 256, 16, 16]
        img_pos = backbone_out['vision_pos_enc'][-1]
        # bs, fea_dim, h, w = img_feature.shape
        # device = img_feature.device

        x = F.adaptive_avg_pool2d(img_feature+img_pos, (1, 1)).squeeze(2,3)
        x = self._class_mlp(x)
        output_logits = self._class_head(x)
    
        return output_logits