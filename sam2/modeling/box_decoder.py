# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from sam2.modeling.sam2_utils import MLP, inverse_sigmoid, gen_sineembed_for_position, _get_activation_fn
from torch import Tensor
import random
from torchvision.ops.boxes import nms
from ..ops.modules import MSDeformAttn

from .template_encoder import TemplateFeatureExtractor

class BoxDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_classes: int = 3,
        num_frames: int = 6,
        num_queries: int = 100,
        dim_feedforward: int = 2048,
        num_decoder_layers: int = 6,
        dropout=0.0,
        activation="relu",
        num_feature_levels=1,
        nhead=8,
        enc_n_points=4,
        dec_n_points=4,
        key_aware_type=None,
        decoder_sa_type='sa',
        module_seq=['sa', 'plca', 'ca', 'ffn'],
        return_intermediate_dec=True,
        modulate_hw_attn=True,
        deformable_decoder=True,
        is_use_prior_loc_templates=False,
        prior_loc_templates_npy_input_path=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.random_refpoints_xy = True
        self.num_feature_levels = num_feature_levels
        # assert (
        #     self.num_classes == 30
        # ), f"The number of classes must be 30 ! Current setting is : {self.num_classes}. "

        # prepare queries
        self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        nn.init.normal_(self.tgt_embed.weight.data)

        # self.refpoint_embed = nn.Embedding(self.num_queries, 4)
        # if self.random_refpoints_xy:
        #     self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
        #     self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
        #     self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.query_dim = 4
        self.init_ref_points(self.num_queries)
        
        # prepare decoder model
        decoder_layer = DeformableTransformerDecoderLayer(self.hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=False, box_attn_type='roi_align',
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq,
                                                          is_use_prior_loc_templates=is_use_prior_loc_templates)
        
        decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=self.hidden_dim, query_dim=4, 
                                        modulate_hw_attn=modulate_hw_attn,
                                        num_feature_levels=num_feature_levels,
                                        deformable_decoder=deformable_decoder,
                                        decoder_query_perturber=None, 
                                        dec_layer_number=None, rm_dec_query_scale=True,
                                        dec_layer_share=False,
                                        use_detached_boxes_dec_out=False
                                        )
        
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, self.num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(num_decoder_layers)]
        class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)

        self.decoder.bbox_embed = self.bbox_embed
        self.decoder.class_embed = self.class_embed
        self.prior_loc_templates_npy_input_path = prior_loc_templates_npy_input_path
        self.is_use_prior_loc_templates = is_use_prior_loc_templates

        # for teo-stage
        use_two_stage = True
        if use_two_stage:
            self.enc_output = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.enc_output_norm = nn.LayerNorm(self.hidden_dim)      
            self.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

        # for generate 4 scale fests similarly in DINO
        self.fpn_channel_proj_32 = nn.Sequential(
            nn.Conv2d(
                256,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.fpn_channel_proj_64 = nn.Sequential(
            nn.Conv2d(
                256,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.fpn_channel_proj_256 = nn.Sequential(
            nn.Conv2d(
                256,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.fpn_channel_proj = nn.ModuleList([self.fpn_channel_proj_32, self.fpn_channel_proj_64, self.fpn_channel_proj_256])
        self.neck_generate_two_more_scales_feats = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),                                               
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim, track_running_stats=False),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),                                               
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim, track_running_stats=False),
                nn.ReLU(inplace=True),
            ),
        ])


        # if self.is_use_prior_loc_templates and self.prior_loc_templates_npy_input_path and len(os.listdir(self.prior_loc_templates_npy_input_path)) == 6:
        #     prior_loc_feas_list = []
        #     for prior_loc_item in os.listdir(self.prior_loc_templates_npy_input_path):
        #         prior_loc_feas_path = os.path.join(self.prior_loc_templates_npy_input_path, prior_loc_item)
        #         prior_loc_feas = np.load(prior_loc_feas_path)['arr_0'] # (256, 3200)
        #         prior_loc_feas_list.append(torch.from_numpy(prior_loc_feas).permute(1,0).contiguous())
        #     self.prior_loc_feas = torch.stack(prior_loc_feas_list, dim=0)
        # else:
        #     self.prior_loc_feas = None

        if self.is_use_prior_loc_templates and self.prior_loc_templates_npy_input_path and len(os.listdir(self.prior_loc_templates_npy_input_path)) == 6:
            # prior_loc_feas_list = []
            self.TemplateFeatureExtractor = TemplateFeatureExtractor(self.prior_loc_templates_npy_input_path)
            # for prior_loc_item in os.listdir(self.prior_loc_templates_npy_input_path):
            #     prior_loc_feas_path = os.path.join(self.prior_loc_templates_npy_input_path, prior_loc_item)
            #     prior_loc_feas = np.load(prior_loc_feas_path)['arr_0'] # (256, 3200)
            #     prior_loc_feas_list.append(torch.from_numpy(prior_loc_feas).permute(1,0).contiguous())
            # self.prior_loc_feas = torch.stack(prior_loc_feas_list, dim=0)
        else:
            self.TemplateFeatureExtractor = None
        
        
    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
        
    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        device = backbone_out["vision_pos_enc"][0].device
        
        # flatten NxCxHxW to HWxNxC
        vision_feats = []
        len_feat_list = []
        for id, (vision_feat, vision_pos_embed) in enumerate(zip(backbone_out["backbone_fpn"], backbone_out["vision_pos_enc"])):
            vision_feats.append(self.fpn_channel_proj[id].to(device)(vision_feat)+vision_pos_embed)
            len_feat_list.append(vision_feats[-1].shape[-2] * vision_feats[-1].shape[-1])
        
        for i, op_conv in enumerate(self.neck_generate_two_more_scales_feats):
            feats_more_scales = op_conv(vision_feats[-1])
            vision_feats.append(feats_more_scales)
            len_feat_list.append(vision_feats[-1].shape[-2] * vision_feats[-1].shape[-1])

        all_num_level_feats = len(vision_feats)
        level_feats_start_id = all_num_level_feats - self.num_feature_levels # drop the first (high-resolution) and second feature maps since GPU oom
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_feats][level_feats_start_id:]
        
        len_feat_list = [0] + len_feat_list[level_feats_start_id:]
        level_start_index = torch.tensor(np.cumsum(np.array(len_feat_list)).tolist()[:all_num_level_feats-level_feats_start_id]).to(device)
        spatial_shapes = torch.tensor(feat_sizes).to(device)
        bs, _, _, _ = backbone_out["backbone_fpn"][0].shape
        valid_ratios = torch.ones(bs, all_num_level_feats-level_feats_start_id, 2).to(device)

        # flatten NxCxHxW to HWxNxC
        memories = [x.flatten(2).permute(2, 0, 1) for x in vision_feats[level_feats_start_id:]]
        memories = torch.cat(memories, dim=0)
        return backbone_out, bs, memories, feat_sizes, level_start_index, spatial_shapes, valid_ratios

    def gen_encoder_output_proposals(self, memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
        r"""
        Input:
            - memory: bs, \sum{hw}, d_model
            - memory_padding_mask: bs, \sum{hw}
            - spatial_shapes: nlevel, 2
            - learnedwh: 2
        Output:
            - output_memory: bs, \sum{hw}, d_model
            - output_proposals: bs, \sum{hw}, 4
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            if learnedwh is not None:
                wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
            else:
                wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        return output_memory, output_proposals

    def forward(self, backbone_out: dict):
        (
            _, bs, memories, feat_sizes, level_start_index, spatial_shapes, valid_ratios
        ) = self._prepare_backbone_features(backbone_out)

        # img_feature = backbone_out['vision_features'] # [8, 256, 64, 64]
        # img_pos = backbone_out['vision_pos_enc'][-1]

        # bs, fea_dim, h, w = img_feature.shape
        # device = img_feature.device

        # img_fea_with_pos = []
        # for i in range (img_feature.shape[0]):
        #     img_fea_with_pos.append(img_feature[i]+img_pos[i])
        
        ################################# for two-stage from DINO #################################
        use_two_stage = True
        if use_two_stage:
            input_hw = None
            device = memories.device
            mask_flatten = torch.zeros(memories.shape[:2]).bool().transpose(0,1).contiguous().to(device)
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                                memories.transpose(0,1).contiguous(), 
                                mask_flatten, 
                                spatial_shapes, 
                                input_hw
                                )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq
            
            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid
            
            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            tgt_ = tgt_undetach.detach()
            refpoint_embed,tgt=refpoint_embed_,tgt_

            hs_enc = tgt_undetach.unsqueeze(0)
            interm_class =  self.enc_out_class_embed(hs_enc[-1])
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)

            output_for_two_stage = (interm_class, ref_enc, init_box_proposal)
        ################################# END two-stage from DINO #################################
        else:
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1).contiguous() # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1).contiguous() # nq, bs, 4
            # init_box_proposal = refpoint_embed_.sigmoid()
            refpoint_embed, tgt = refpoint_embed_, tgt_
            mask_flatten = None
            # spatial_shapes = torch.tensor((h,w)).to(device).unsqueeze(0)
            # valid_ratios = torch.ones(bs, 1, 2).to(device)
            # level_start_index = torch.tensor(0,).to(device)

        (
            prior_loc_template_feas, 
            prior_loc_template_spatial_shapes, 
            prior_loc_template_level_start_index, 
            prior_loc_template_key_padding_mask
        ) = backbone_out['prior_loc_template_memory']

        hs, references = self.decoder(
                tgt=tgt.transpose(0, 1).contiguous(), # query 
                memory=memories, # image feature, note here, the memory format is [hw, bs, d_model]
                memory_key_padding_mask=mask_flatten, 
                pos=None,
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1).contiguous(), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                tgt_mask=None, # torch.ones((self.num_queries, self.num_queries)).to(device),
                prior_loc_template_memory=prior_loc_template_feas, 
                prior_loc_template_spatial_shapes=prior_loc_template_spatial_shapes, 
                prior_loc_template_level_start_index=prior_loc_template_level_start_index, 
                prior_loc_template_key_padding_mask=prior_loc_template_key_padding_mask,
                prior_loc_template_order_id_in_img_batch=backbone_out['template_order_id_in_img_batch'])
        
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(references[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        
        outputs_class_list = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        return outputs_class_list, outputs_coord_list, output_for_two_stage # torch.Size([6, 6, 100, 30])
        # return output_box, output_cls


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, 
                    return_intermediate=False, 
                    d_model=256, query_dim=4, 
                    modulate_hw_attn=False,
                    num_feature_levels=1,
                    deformable_decoder=False,
                    decoder_query_perturber=None,
                    dec_layer_number=None, # number of queries each layer in decoder
                    rm_dec_query_scale=False,
                    dec_layer_share=False,
                    dec_layer_dropout_prob=None,
                    use_detached_boxes_dec_out=False
                    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # nq, bs, 2/4
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                # for prior loc feas
                prior_loc_template_memory: Optional[Tensor] = None,
                prior_loc_template_spatial_shapes: Optional[Tensor] = None,
                prior_loc_template_level_start_index: Optional[Tensor] = None,
                prior_loc_template_key_padding_mask: Optional[Tensor] = None,
                prior_loc_template_order_id_in_img_batch: List = None,
            ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
                reference_points_input = None
            
            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt = output,
                    tgt_query_pos = query_pos,
                    tgt_query_sine_embed = query_sine_embed,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    tgt_reference_points = reference_points_input,

                    memory = memory,
                    memory_key_padding_mask = memory_key_padding_mask,
                    memory_level_start_index = level_start_index,
                    memory_spatial_shapes = spatial_shapes,
                    memory_pos = pos,

                    self_attn_mask = tgt_mask,
                    cross_attn_mask = memory_mask,

                    prior_loc_template_memory = prior_loc_template_memory,
                    prior_loc_template_spatial_shapes = prior_loc_template_spatial_shapes,
                    prior_loc_template_level_start_index = prior_loc_template_level_start_index,
                    prior_loc_template_key_padding_mask = prior_loc_template_key_padding_mask,
                    prior_loc_template_order_id_in_img_batch = prior_loc_template_order_id_in_img_batch
                )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # select # ref points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output) # nq, bs, 91
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1] # new_nq, bs
                        new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = torch.gather(output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)) # unsigmoid

        return [
            [itm_out.transpose(0, 1).contiguous() for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1).contiguous() for itm_refpoint in ref_points]
        ]


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'plca', 'ca', 'ffn'],
                 is_use_prior_loc_templates=False,
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'plca', 'sa'] # changed by bryce; add 'plca'
        # cross attention
        if use_deformable_box_attn: # changed by bryce; note change back !!!
            # self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points) # changed by bryce; note change back !!!
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        
        if decoder_sa_type == 'ca_content': # changed by bryce; note change back !!!
            # self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.self_attn = nn.MultiheadAttention(d_model, n_heads)

        # add by bryce; for prior loc cross attention
        self.is_use_prior_loc_templates = is_use_prior_loc_templates
        if self.is_use_prior_loc_templates:
            if use_deformable_box_attn: # changed by bryce; note change back !!!
                # self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
                # self.prior_loc_cross_attn = nn.MultiheadAttention(d_model, n_heads)
                self.prior_loc_cross_attn = nn.Identity() 
            else:
                self.prior_loc_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout4 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)
            self.prior_loc_adapter = MLP(32, 64, d_model, 3)


    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1).contiguous(),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1).contiguous(), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1).contiguous()
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        
        # change by bryce; note change back !!!
        # import pdb; pdb.set_trace()
        # tgt2, tgt2_attn_weights  = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos), memory, memory)
        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
        #                        tgt_reference_points.transpose(0, 1).contiguous(),
        #                        memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask.bool()).transpose(0, 1)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1).contiguous(), # (6, 100, 256)
                                tgt_reference_points.transpose(0, 1).contiguous(),
                                memory.transpose(0, 1).contiguous(), # (6, 4096, 256)
                                memory_spatial_shapes, # (1, 2)
                                memory_level_start_index, # tensor(0, device='cuda:0')
                                memory_key_padding_mask # (6, 4096)
                                ).transpose(0, 1).contiguous()
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward_plca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                prior_loc_template_order_id_in_img_batch: List = None,

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        if memory is None or not self.is_use_prior_loc_templates:
            return tgt
        
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        
        device = tgt.device

        sorted_memory = [memory[i] for i in prior_loc_template_order_id_in_img_batch]
        sorted_memory = torch.stack(sorted_memory, dim=0)
        if sorted_memory.dim() == 5:
            B, L, C, H, W = sorted_memory.shape
            sorted_memory = sorted_memory.permute(0, 1, 3, 4, 2).reshape(B, L*H*W, C)
        else:
            B, C, H, W = sorted_memory.shape
            sorted_memory = sorted_memory.permute(0, 2, 3, 1).reshape(B, H*W, C)

        sorted_memory = self.prior_loc_adapter(sorted_memory.to(device)).contiguous() # .permute(1,0,2)
        # change by bryce; note change back !!!
        # tgt2, tgt2_attn_weights  = self.prior_loc_cross_attn(self.with_pos_embed(tgt, tgt_query_pos), memory, memory)
        tgt2 = self.prior_loc_cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1).contiguous(),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               sorted_memory, 
                               memory_spatial_shapes, 
                               memory_level_start_index, 
                               memory_key_padding_mask).transpose(0, 1).contiguous()
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)

        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention

                # for prior loc feas
                prior_loc_template_memory: Optional[Tensor] = None,
                prior_loc_template_spatial_shapes: Optional[Tensor] = None,
                prior_loc_template_level_start_index: Optional[Tensor] = None,
                prior_loc_template_key_padding_mask: Optional[Tensor] = None,
                prior_loc_template_order_id_in_img_batch: List = None,
            ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'plca':
                tgt = self.forward_plca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        prior_loc_template_memory, prior_loc_template_key_padding_mask, \
                            prior_loc_template_level_start_index, prior_loc_template_spatial_shapes, \
                                prior_loc_template_order_id_in_img_batch)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = self.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # add by bryce; clip the minor number to zero
        boxes[boxes < 0] = 0

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)