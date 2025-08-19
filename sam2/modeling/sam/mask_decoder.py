# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP
from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs #  + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2 # changed by bryce; remove ' // 4 '
                # transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            # LayerNorm2d(transformer_dim // 4),
            LayerNorm2d(transformer_dim),# changed by bryce; remove ' // 4 '
            activation(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2 # changed by bryce; remove ' // 8 ' ' // 4 '
                # transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2 
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim, kernel_size=1, stride=1 # changed by bryce; remove ' // 8 '
                # transformer_dim, transformer_dim // 8, kernel_size=1, stride=1 
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim, kernel_size=1, stride=1 # changed by bryce; remove ' // 4 '
                # transformer_dim, transformer_dim // 4 , kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                # MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                MLP(transformer_dim, transformer_dim, transformer_dim, 3) # changed by bryce; remove ' // 8 '
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        
        # add by bryce; for adding maskdino decoder into sam2; 
        # dim_feedforward = 2048
        # dropout = 0.0
        # activation_type = 'gelu'
        # nhead = 8
        # dec_n_points = 4
        # num_feature_levels = 1
        
        self.decoder_norm = nn.LayerNorm(transformer_dim)
        
        # self.num_layers = 9
        # return_intermediate_dec = True
        # query_dim = 4
        # dec_layer_share = True
        # self.initial_pred = True
        # self.num_classes = 3
        # mask_dim = 256
        # self.num_queries = 300
        # self.query_feat = nn.Embedding(self.num_queries, transformer_dim)
        # self.query_red_points = nn.Embedding(self.num_queries, 4)
        
        # self.class_embed = nn.Linear(transformer_dim, self.num_classes)
        # self.label_enc = nn.Embedding(self.num_classes,transformer_dim)
        # self.mask_embed = MLP(transformer_dim, transformer_dim, mask_dim, 3)

        # decoder_layer = DeformableTransformerDecoderLayer(transformer_dim, dim_feedforward,
        #                                                   dropout, activation_type,
        #                                                   num_feature_levels, nhead, dec_n_points)
        # self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
        #                             return_intermediate=return_intermediate_dec,
        #                             d_model=transformer_dim, query_dim=query_dim,
        #                             num_feature_levels=num_feature_levels,
        #                             dec_layer_share=dec_layer_share,
        #                             )
        
        # self._bbox_embed = _bbox_embed = MLP(transformer_dim, transformer_dim, 4, 3)
        # nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        # box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        # self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        # self.decoder.bbox_embed = self.bbox_embed
        
        # END; for adding maskdino decoder into sam2; 

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        mask_inputs: torch.Tensor = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, aux_masks_list, aux_object_score_logits_list = self.predict_masks(
            image_embeddings=image_embeddings, # (num_objs, 256, 32, 32)
            image_pe=image_pe, # (1, 256, 32, 32)
            sparse_prompt_embeddings=sparse_prompt_embeddings, # (4, 12 ,256)
            dense_prompt_embeddings=dense_prompt_embeddings, # (4, 256, 32, 32)
            repeat_image=repeat_image,# False 
            high_res_features=high_res_features, # False
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks # masks = masks[:, 1:, :, :]; changed by bryce
            iou_pred = iou_pred # iou_pred = iou_pred[:, 1:]; changed by bryce
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :] # (3, 1, 256, 256)
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, :]  # sam_tokens_out = mask_tokens_out[:, 1:] ; changed by bryce [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits, aux_masks_list, aux_object_score_logits_list

    def forward_prediction_heads(self, hs, src, s, high_res_features, feat_shape, is_aux_mask=False):
        if is_aux_mask:
            hs = self.decoder_norm(hs)
        iou_token_out = hs[:, s, :] # s=1
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :] # (3, 4, 256)
        b, c, h, w = feat_shape
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) # (3, 256, 16, 16)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) # (3, 32, 64, 64)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) # (4, 1, 256); 4 objs, 1 masks
            )
        hyper_in = torch.stack(hyper_in_list, dim=1) # (6, 4, 32) ; 6 objs, 4 masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # (3, 4, 256, 256); three classes; four masks for each class

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) # (3, 4) for quality 
        if self.pred_obj_scores:
            # assert s == 1
            # object_score_logits = self.pred_obj_score_head(hs[:, 0, :]) # (3, 1) for categories
            object_score_logits = self.pred_obj_score_head(mask_tokens_out) # (3, 1) for categories
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        del upscaled_embedding

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        mask_inputs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        # if self.pred_obj_scores:
        #     output_tokens = torch.cat( # [6, 256]
        #         [
        #             self.obj_score_token.weight,
        #             self.iou_token.weight,
        #             self.mask_tokens.weight,
        #         ],
        #         dim=0,
        #     )
        #     s = 1
        # else:
        #     output_tokens = torch.cat(
        #         [self.iou_token.weight, self.mask_tokens.weight], dim=0
        #     )

        output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [N, 4+2, 256]
        
        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings

        src = src + dense_prompt_embeddings

        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        feat_shape = src.shape
        # src: (6, 256, 64, 64)
        # pos_src: (6, 256, 64, 64)
        # tokens: (6, 7, 256)
        # Run the transformer
        hs, src, aux_queries_list, aux_keys_list = self.transformer(src, pos_src, tokens)
        # hs: (6, 7, 256)
        # src: (6, 4096, 256)
        
        masks, iou_pred, mask_tokens_out, object_score_logits = self.forward_prediction_heads(hs, src, s, high_res_features, feat_shape, is_aux_mask=False)

        aux_masks_list = []
        aux_object_score_logits_list = []
        for aux_hs, aux_src in zip(aux_queries_list, aux_keys_list):
            aux_masks, _, _, aux_object_score_logits = self.forward_prediction_heads(aux_hs, aux_src, s, high_res_features, feat_shape, is_aux_mask=True)
            aux_masks_list.append(aux_masks.float())
            aux_object_score_logits_list.append(aux_object_score_logits)
    
        return masks, iou_pred, mask_tokens_out, object_score_logits, aux_masks_list, aux_object_score_logits_list

    # def predict_masks(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_pe: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    #     repeat_image: bool,
    #     high_res_features: Optional[List[torch.Tensor]] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Predicts masks. See 'forward' for more details."""
    #     # Concatenate output tokens
    #     s = 0
    #     if self.pred_obj_scores:
    #         output_tokens = torch.cat( # [6, 256]
    #             [
    #                 self.obj_score_token.weight,
    #                 self.iou_token.weight,
    #                 self.mask_tokens.weight,
    #             ],
    #             dim=0,
    #         )
    #         s = 1
    #     else:
    #         output_tokens = torch.cat(
    #             [self.iou_token.weight, self.mask_tokens.weight], dim=0
    #         )
    #     output_tokens = output_tokens.unsqueeze(0).expand(
    #         sparse_prompt_embeddings.size(0), -1, -1
    #     )
    #     tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [3, 18(6+12), 256]
        
    #     # Expand per-image data in batch direction to be per-mask
    #     if repeat_image:
    #         src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    #     else:
    #         assert image_embeddings.shape[0] == tokens.shape[0]
    #         src = image_embeddings
    #     src = src + dense_prompt_embeddings
    #     assert (
    #         image_pe.size(0) == 1
    #     ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
    #     pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    #     b, c, h, w = src.shape

    #     # Run the transformer
    #     hs, src = self.transformer(src, pos_src, tokens) # (4, 18, 256); (4, 4096, 256)
        
    #     iou_token_out = hs[:, s, :] # s=1
    #     mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :] # (3, 4, 256)

    #     # Upscale mask embeddings and predict masks using the mask tokens
    #     src = src.transpose(1, 2).view(b, c, h, w) # (3, 256, 16, 16)
    #     if not self.use_high_res_features:
    #         upscaled_embedding = self.output_upscaling(src)
    #     else:
    #         dc1, ln1, act1, dc2, act2 = self.output_upscaling
    #         feat_s0, feat_s1 = high_res_features
    #         upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
    #         upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) # (3, 32, 64, 64)

    #     hyper_in_list: List[torch.Tensor] = []
    #     for i in range(self.num_mask_tokens):
    #         hyper_in_list.append(
    #             self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) # (4, 1, 256); 4 objs, 1 masks
    #         )
    #     hyper_in = torch.stack(hyper_in_list, dim=1) # (4, 4, 32) ; 4 objs, 4 masks
    #     b, c, h, w = upscaled_embedding.shape
    #     masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # (3, 4, 256, 256); three classes; four masks for each class

    #     # Generate mask quality predictions
    #     iou_pred = self.iou_prediction_head(iou_token_out) # (3, 4) for quality 
    #     if self.pred_obj_scores:
    #         assert s == 1
    #         object_score_logits = self.pred_obj_score_head(hs[:, 0, :]) # (3, 1) for categories
    #     else:
    #         # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
    #         object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

    #     return masks, iou_pred, mask_tokens_out, object_score_logits


    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out

################################## MaskDINO Decoder #################################
    def _forward_MaskDINO(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
            mask_inputs: torch.Tensor = None # 1 for selected areas
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict masks given image and prompt embeddings.

            Arguments:
            image_embeddings (torch.Tensor): the embeddings from the image encoder
            image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
            multimask_output (bool): Whether to return multiple masks or a single
                mask.

            Returns:
            torch.Tensor: batched predicted masks
            torch.Tensor: batched predictions of mask quality
            torch.Tensor: batched SAM token for mask output
            """
            out = self.predict_masks_MaskDINO(
                image_embeddings=image_embeddings, # (num_objs, 256, 32, 32)
                image_pe=image_pe, # (1, 256, 32, 32)
                sparse_prompt_embeddings=sparse_prompt_embeddings, # (4, 12 ,256)
                dense_prompt_embeddings=dense_prompt_embeddings, # (4, 256, 32, 32)
                repeat_image=repeat_image,# False 
                high_res_features=high_res_features, # False
                mask_inputs=mask_inputs
            )

            return out

            
    def predict_masks_MaskDINO(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
            mask_inputs: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Predicts masks. See 'forward' for more details."""


            bs = sparse_prompt_embeddings.shape[0]
            device = image_embeddings.device

            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_red_points.weight[None].repeat(bs, 1, 1)

            tgt = self.query_feat.weight[None].expand(bs, -1, -1)
            non_pe = torch.zeros_like(sparse_prompt_embeddings).to(device=device)
            # tgt = torch.cat((tgt, sparse_prompt_embeddings), dim=1) # [N, 4+2, 256]

            # Expand per-image data in batch direction to be per-mask
            if repeat_image:
                src = torch.repeat_interleave(image_embeddings, bs, dim=0)
            else:
                assert image_embeddings.shape[0] == bs
                src = image_embeddings

            # src = src + dense_prompt_embeddings # (N_prompts, dim, size, size)
            
            assert (
                image_pe.size(0) == 1
            ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
            pos_src = torch.repeat_interleave(image_pe, bs, dim=0)

            ########## prepare materials 
            src_flatten = src.view(bs, self.transformer_dim, -1).transpose(1, 2)
            
            mask_inputs = torch.nn.functional.interpolate(mask_inputs, size=image_pe.shape[2:], mode='nearest')
            mask_flatten = (1 - mask_inputs).view(bs, -1).bool()
            
            # mask_flatten = torch.zeros(src_flatten.shape[:2]).bool().to(device)
            level_start_index = torch.tensor([0], device=device)
            spatial_shapes = torch.tensor([(src.shape[-2], src.shape[-1])], device=device)
            valid_ratios = torch.ones(bs, 1, 2).to(device)

            predictions_class = []
            predictions_mask = []

            #TODO merge multi-scale feats
            mask_features = high_res_features[0]

            # tgt (bs, query, dim)
            if self.initial_pred:
                outputs_class, outputs_mask = self.forward_prediction_heads_MaskDINO(tgt[:, :self.num_queries, :].transpose(0, 1), mask_features, self.training)
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

            hs, references = self.decoder(
                tgt=tgt.transpose(0, 1),
                memory=src_flatten.transpose(0, 1),
                memory_key_padding_mask=mask_flatten,
                pos=None,
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                tgt_mask=None
            )

            # hs = [hss[:, :self.num_classes, :] for hss in hs]
            # if hs[0].shape != references[0].shape:
            #     references = [ref[:, :self.num_classes, :] for ref in references]

            for i, output in enumerate(hs):
                outputs_class, outputs_mask = self.forward_prediction_heads_MaskDINO(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

            # iteratively box prediction
            if self.initial_pred:
                out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
                assert len(predictions_class) == self.num_layers + 1
            else:
                out_boxes = self.pred_box(references, hs)
                
            if self.training:  # this is to insure self.label_enc participate in the model
                predictions_class[-1] += 0.0*self.label_enc.weight.sum()

            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'pred_boxes':out_boxes[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class, predictions_mask, out_boxes
                )
            }
            # if self.two_stage:
            #     out['interm_outputs'] = interm_outputs

            return out

    
    def forward_prediction_heads_MaskDINO(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # # (2, 396, 256)
        outputs_class = self.class_embed(decoder_output) # (2, 396, 80)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output) # (2, 396, 256)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) # (2, 396, 256, 256)

        return outputs_class, outputs_mask
    
    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + self.inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list
    
    def inverse_sigmoid(self, x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1/x2)
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]