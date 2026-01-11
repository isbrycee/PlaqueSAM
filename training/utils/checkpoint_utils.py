# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    """
    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern

    Args:
        model: the model that might have been updated
        patterns: for the parameters we want to observe
        allowed:
    """
    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location, weights_only=False)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    if isinstance(checkpoint, RecursiveScriptModule):
        # This is a torchscript JIT model
        return checkpoint.state_dict()
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        if (isinstance(pre_train_dict, Mapping) and key not in pre_train_dict) or (
            isinstance(pre_train_dict, Sequence) and key >= len(pre_train_dict)
        ):
            key_str = (
                '["' + '"]["'.join(list(map(ckpt_state_dict_keys[:i], str))) + '"]'
            )
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
        ckpt_state_dict_keys (str): Keys containing the model state dict.
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location, weights_only=False)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    # Not logging into info etc since it's a huge log
    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
        ignore_missing_keys: unix pattern of keys to ignore
    """
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    
    # add by bryce
    # unallowed_parameter_names = []
    # unallowed_parameter_names += ['maskmem_tpos_enc',  'obj_ptr_tpos_proj.weight', 'obj_ptr_tpos_proj.bias', 'sam_mask_decoder.mask_tokens.weight', 'sam_mask_decoder.iou_prediction_head.layers.2.weight', 'sam_mask_decoder.iou_prediction_head.layers.2.bias']
    # unallowed_parameter_names += ['sam_mask_decoder.output_upscaling.3.weight', 'sam_mask_decoder.output_upscaling.3.bias', 'sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.weight', 'sam_mask_decoder.output_hypernetworks_mlps.0.layers.2.bias', 'sam_mask_decoder.output_hypernetworks_mlps.1.layers.2.weight', 'sam_mask_decoder.output_hypernetworks_mlps.1.layers.2.bias', 'sam_mask_decoder.output_hypernetworks_mlps.2.layers.2.weight', 'sam_mask_decoder.output_hypernetworks_mlps.2.layers.2.bias']
    # unallowed_parameter_names += ['box_decoder.tgt_embed.weight', 'box_decoder.refpoint_embed.weight', 'box_decoder.decoder.layers.0.cross_attn.sampling_offsets.weight', 'box_decoder.decoder.layers.0.cross_attn.sampling_offsets.bias', 'box_decoder.decoder.layers.0.cross_attn.attention_weights.weight', 'box_decoder.decoder.layers.0.cross_attn.attention_weights.bias', 'box_decoder.decoder.layers.0.cross_attn.value_proj.weight', 'box_decoder.decoder.layers.0.cross_attn.value_proj.bias', 'box_decoder.decoder.layers.0.cross_attn.output_proj.weight', 'box_decoder.decoder.layers.0.cross_attn.output_proj.bias', 'box_decoder.decoder.layers.0.norm1.weight', 'box_decoder.decoder.layers.0.norm1.bias', 'box_decoder.decoder.layers.0.self_attn.in_proj_weight', 'box_decoder.decoder.layers.0.self_attn.in_proj_bias', 'box_decoder.decoder.layers.0.self_attn.out_proj.weight', 'box_decoder.decoder.layers.0.self_attn.out_proj.bias', 'box_decoder.decoder.layers.0.norm2.weight', 'box_decoder.decoder.layers.0.norm2.bias', 'box_decoder.decoder.layers.0.linear1.weight', 'box_decoder.decoder.layers.0.linear1.bias', 'box_decoder.decoder.layers.0.linear2.weight', 'box_decoder.decoder.layers.0.linear2.bias', 'box_decoder.decoder.layers.0.norm3.weight', 'box_decoder.decoder.layers.0.norm3.bias', 'box_decoder.decoder.layers.1.cross_attn.sampling_offsets.weight', 'box_decoder.decoder.layers.1.cross_attn.sampling_offsets.bias', 'box_decoder.decoder.layers.1.cross_attn.attention_weights.weight', 'box_decoder.decoder.layers.1.cross_attn.attention_weights.bias', 'box_decoder.decoder.layers.1.cross_attn.value_proj.weight', 'box_decoder.decoder.layers.1.cross_attn.value_proj.bias', 'box_decoder.decoder.layers.1.cross_attn.output_proj.weight', 'box_decoder.decoder.layers.1.cross_attn.output_proj.bias', 'box_decoder.decoder.layers.1.norm1.weight', 'box_decoder.decoder.layers.1.norm1.bias', 'box_decoder.decoder.layers.1.self_attn.in_proj_weight', 'box_decoder.decoder.layers.1.self_attn.in_proj_bias', 'box_decoder.decoder.layers.1.self_attn.out_proj.weight', 'box_decoder.decoder.layers.1.self_attn.out_proj.bias', 'box_decoder.decoder.layers.1.norm2.weight', 'box_decoder.decoder.layers.1.norm2.bias', 'box_decoder.decoder.layers.1.linear1.weight', 'box_decoder.decoder.layers.1.linear1.bias', 'box_decoder.decoder.layers.1.linear2.weight', 'box_decoder.decoder.layers.1.linear2.bias', 'box_decoder.decoder.layers.1.norm3.weight', 'box_decoder.decoder.layers.1.norm3.bias', 'box_decoder.decoder.norm.weight', 'box_decoder.decoder.norm.bias', 'box_decoder.decoder.ref_point_head.layers.0.weight', 'box_decoder.decoder.ref_point_head.layers.0.bias', 'box_decoder.decoder.ref_point_head.layers.1.weight', 'box_decoder.decoder.ref_point_head.layers.1.bias', 'box_decoder.decoder.bbox_embed.0.layers.0.weight', 'box_decoder.decoder.bbox_embed.0.layers.0.bias', 'box_decoder.decoder.bbox_embed.0.layers.1.weight', 'box_decoder.decoder.bbox_embed.0.layers.1.bias', 'box_decoder.decoder.bbox_embed.0.layers.2.weight', 'box_decoder.decoder.bbox_embed.0.layers.2.bias', 'box_decoder.decoder.bbox_embed.1.layers.0.weight', 'box_decoder.decoder.bbox_embed.1.layers.0.bias', 'box_decoder.decoder.bbox_embed.1.layers.1.weight', 'box_decoder.decoder.bbox_embed.1.layers.1.bias', 'box_decoder.decoder.bbox_embed.1.layers.2.weight', 'box_decoder.decoder.bbox_embed.1.layers.2.bias', 'box_decoder.decoder.class_embed.0.weight', 'box_decoder.decoder.class_embed.0.bias', 'box_decoder.decoder.class_embed.1.weight', 'box_decoder.decoder.class_embed.1.bias', 'box_decoder.bbox_embed.0.layers.0.weight', 'box_decoder.bbox_embed.0.layers.0.bias', 'box_decoder.bbox_embed.0.layers.1.weight', 'box_decoder.bbox_embed.0.layers.1.bias', 'box_decoder.bbox_embed.0.layers.2.weight', 'box_decoder.bbox_embed.0.layers.2.bias', 'box_decoder.bbox_embed.1.layers.0.weight', 'box_decoder.bbox_embed.1.layers.0.bias', 'box_decoder.bbox_embed.1.layers.1.weight', 'box_decoder.bbox_embed.1.layers.1.bias', 'box_decoder.bbox_embed.1.layers.2.weight', 'box_decoder.bbox_embed.1.layers.2.bias', 'box_decoder.class_embed.0.weight', 'box_decoder.class_embed.0.bias', 'box_decoder.class_embed.1.weight', 'box_decoder.class_embed.1.bias', 'box_decoder.enc_output.weight', 'box_decoder.enc_output.bias', 'box_decoder.enc_output_norm.weight', 'box_decoder.enc_output_norm.bias', 'box_decoder.enc_out_class_embed.weight', 'box_decoder.enc_out_class_embed.bias', 'box_decoder.enc_out_bbox_embed.layers.0.weight', 'box_decoder.enc_out_bbox_embed.layers.0.bias', 'box_decoder.enc_out_bbox_embed.layers.1.weight', 'box_decoder.enc_out_bbox_embed.layers.1.bias', 'box_decoder.enc_out_bbox_embed.layers.2.weight', 'box_decoder.enc_out_bbox_embed.layers.2.bias', 'box_decoder.fpn_channel_proj_32.0.weight', 'box_decoder.fpn_channel_proj_32.1.weight', 'box_decoder.fpn_channel_proj_32.1.bias', 'box_decoder.fpn_channel_proj_32.1.running_mean', 'box_decoder.fpn_channel_proj_32.1.running_var', 'box_decoder.fpn_channel_proj_32.2.weight', 'box_decoder.fpn_channel_proj_32.3.weight', 'box_decoder.fpn_channel_proj_32.3.bias', 'box_decoder.fpn_channel_proj_32.3.running_mean', 'box_decoder.fpn_channel_proj_32.3.running_var', 'box_decoder.fpn_channel_proj_64.0.weight', 'box_decoder.fpn_channel_proj_64.1.weight', 'box_decoder.fpn_channel_proj_64.1.bias', 'box_decoder.fpn_channel_proj_64.1.running_mean', 'box_decoder.fpn_channel_proj_64.1.running_var', 'box_decoder.fpn_channel_proj_64.2.weight', 'box_decoder.fpn_channel_proj_64.3.weight', 'box_decoder.fpn_channel_proj_64.3.bias', 'box_decoder.fpn_channel_proj_64.3.running_mean', 'box_decoder.fpn_channel_proj_64.3.running_var', 'box_decoder.fpn_channel_proj_256.0.weight', 'box_decoder.fpn_channel_proj_256.1.weight', 'box_decoder.fpn_channel_proj_256.1.bias', 'box_decoder.fpn_channel_proj_256.1.running_mean', 'box_decoder.fpn_channel_proj_256.1.running_var', 'box_decoder.fpn_channel_proj_256.2.weight', 'box_decoder.fpn_channel_proj_256.3.weight', 'box_decoder.fpn_channel_proj_256.3.bias', 'box_decoder.fpn_channel_proj_256.3.running_mean', 'box_decoder.fpn_channel_proj_256.3.running_var', 'box_decoder.fpn_channel_proj.0.0.weight', 'box_decoder.fpn_channel_proj.0.1.weight', 'box_decoder.fpn_channel_proj.0.1.bias', 'box_decoder.fpn_channel_proj.0.1.running_mean', 'box_decoder.fpn_channel_proj.0.1.running_var', 'box_decoder.fpn_channel_proj.0.2.weight', 'box_decoder.fpn_channel_proj.0.3.weight', 'box_decoder.fpn_channel_proj.0.3.bias', 'box_decoder.fpn_channel_proj.0.3.running_mean', 'box_decoder.fpn_channel_proj.0.3.running_var', 'box_decoder.fpn_channel_proj.1.0.weight', 'box_decoder.fpn_channel_proj.1.1.weight', 'box_decoder.fpn_channel_proj.1.1.bias', 'box_decoder.fpn_channel_proj.1.1.running_mean', 'box_decoder.fpn_channel_proj.1.1.running_var', 'box_decoder.fpn_channel_proj.1.2.weight', 'box_decoder.fpn_channel_proj.1.3.weight', 'box_decoder.fpn_channel_proj.1.3.bias', 'box_decoder.fpn_channel_proj.1.3.running_mean', 'box_decoder.fpn_channel_proj.1.3.running_var', 'box_decoder.fpn_channel_proj.2.0.weight', 'box_decoder.fpn_channel_proj.2.1.weight', 'box_decoder.fpn_channel_proj.2.1.bias', 'box_decoder.fpn_channel_proj.2.1.running_mean', 'box_decoder.fpn_channel_proj.2.1.running_var', 'box_decoder.fpn_channel_proj.2.2.weight', 'box_decoder.fpn_channel_proj.2.3.weight', 'box_decoder.fpn_channel_proj.2.3.bias', 'box_decoder.fpn_channel_proj.2.3.running_mean', 'box_decoder.fpn_channel_proj.2.3.running_var', 'box_decoder.neck_generate_two_more_scales_feats.0.1.weight', 'box_decoder.neck_generate_two_more_scales_feats.0.2.weight', 'box_decoder.neck_generate_two_more_scales_feats.0.2.bias', 'box_decoder.neck_generate_two_more_scales_feats.0.2.running_mean', 'box_decoder.neck_generate_two_more_scales_feats.0.2.running_var', 'box_decoder.neck_generate_two_more_scales_feats.1.1.weight', 'box_decoder.neck_generate_two_more_scales_feats.1.2.weight', 'box_decoder.neck_generate_two_more_scales_feats.1.2.bias', 'box_decoder.neck_generate_two_more_scales_feats.1.2.running_mean', 'box_decoder.neck_generate_two_more_scales_feats.1.2.running_var', 'image_classify_decoder._class_head.weight', 'image_classify_decoder._class_head.bias', 'image_classify_decoder._class_mlp.layers.0.weight', 'image_classify_decoder._class_mlp.layers.0.bias', 'image_classify_decoder._class_mlp.layers.1.weight', 'image_classify_decoder._class_mlp.layers.1.bias']
    # unallowed_parameter_names += ['sam_mask_decoder.output_upscaling.1.weight', 'sam_mask_decoder.output_upscaling.1.bias', 'sam_mask_decoder.output_upscaling.0.weight', 'sam_mask_decoder.output_upscaling.0.bias', 'sam_mask_decoder.conv_s0.weight', 'sam_mask_decoder.conv_s0.bias', 'sam_mask_decoder.conv_s1.weight', 'sam_mask_decoder.conv_s1.bias']
    # unallowed_parameter_names += ['sam_mask_decoder.class_embed.weight', 'sam_mask_decoder.class_embed.bias', 'sam_mask_decoder.label_enc.weight', 'sam_mask_decoder.mask_embed.layers.0.weight', 'sam_mask_decoder.mask_embed.layers.0.bias', 'sam_mask_decoder.mask_embed.layers.1.weight', 'sam_mask_decoder.mask_embed.layers.1.bias', 'sam_mask_decoder.mask_embed.layers.2.weight', 'sam_mask_decoder.mask_embed.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.0.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.0.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.0.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.0.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.0.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.0.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.1.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.1.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.1.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.1.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.1.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.1.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.2.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.2.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.2.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.2.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.2.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.2.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.3.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.3.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.3.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.3.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.3.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.3.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.4.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.4.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.4.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.4.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.4.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.4.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.5.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.5.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.5.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.5.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.5.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.5.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.6.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.6.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.6.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.6.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.6.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.6.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.7.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.7.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.7.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.7.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.7.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.7.layers.2.bias', 'sam_mask_decoder.decoder.bbox_embed.8.layers.0.weight', 'sam_mask_decoder.decoder.bbox_embed.8.layers.0.bias', 'sam_mask_decoder.decoder.bbox_embed.8.layers.1.weight', 'sam_mask_decoder.decoder.bbox_embed.8.layers.1.bias', 'sam_mask_decoder.decoder.bbox_embed.8.layers.2.weight', 'sam_mask_decoder.decoder.bbox_embed.8.layers.2.bias', 'sam_mask_decoder._bbox_embed.layers.0.weight', 'sam_mask_decoder._bbox_embed.layers.0.bias', 'sam_mask_decoder._bbox_embed.layers.1.weight', 'sam_mask_decoder._bbox_embed.layers.1.bias', 'sam_mask_decoder._bbox_embed.layers.2.weight', 'sam_mask_decoder._bbox_embed.layers.2.bias', 'sam_mask_decoder.bbox_embed.0.layers.0.weight', 'sam_mask_decoder.bbox_embed.0.layers.0.bias', 'sam_mask_decoder.bbox_embed.0.layers.1.weight', 'sam_mask_decoder.bbox_embed.0.layers.1.bias', 'sam_mask_decoder.bbox_embed.0.layers.2.weight', 'sam_mask_decoder.bbox_embed.0.layers.2.bias', 'sam_mask_decoder.bbox_embed.1.layers.0.weight', 'sam_mask_decoder.bbox_embed.1.layers.0.bias', 'sam_mask_decoder.bbox_embed.1.layers.1.weight', 'sam_mask_decoder.bbox_embed.1.layers.1.bias', 'sam_mask_decoder.bbox_embed.1.layers.2.weight', 'sam_mask_decoder.bbox_embed.1.layers.2.bias', 'sam_mask_decoder.bbox_embed.2.layers.0.weight', 'sam_mask_decoder.bbox_embed.2.layers.0.bias', 'sam_mask_decoder.bbox_embed.2.layers.1.weight', 'sam_mask_decoder.bbox_embed.2.layers.1.bias', 'sam_mask_decoder.bbox_embed.2.layers.2.weight', 'sam_mask_decoder.bbox_embed.2.layers.2.bias', 'sam_mask_decoder.bbox_embed.3.layers.0.weight', 'sam_mask_decoder.bbox_embed.3.layers.0.bias', 'sam_mask_decoder.bbox_embed.3.layers.1.weight', 'sam_mask_decoder.bbox_embed.3.layers.1.bias', 'sam_mask_decoder.bbox_embed.3.layers.2.weight', 'sam_mask_decoder.bbox_embed.3.layers.2.bias', 'sam_mask_decoder.bbox_embed.4.layers.0.weight', 'sam_mask_decoder.bbox_embed.4.layers.0.bias', 'sam_mask_decoder.bbox_embed.4.layers.1.weight', 'sam_mask_decoder.bbox_embed.4.layers.1.bias', 'sam_mask_decoder.bbox_embed.4.layers.2.weight', 'sam_mask_decoder.bbox_embed.4.layers.2.bias', 'sam_mask_decoder.bbox_embed.5.layers.0.weight', 'sam_mask_decoder.bbox_embed.5.layers.0.bias', 'sam_mask_decoder.bbox_embed.5.layers.1.weight', 'sam_mask_decoder.bbox_embed.5.layers.1.bias', 'sam_mask_decoder.bbox_embed.5.layers.2.weight', 'sam_mask_decoder.bbox_embed.5.layers.2.bias', 'sam_mask_decoder.bbox_embed.6.layers.0.weight', 'sam_mask_decoder.bbox_embed.6.layers.0.bias', 'sam_mask_decoder.bbox_embed.6.layers.1.weight', 'sam_mask_decoder.bbox_embed.6.layers.1.bias', 'sam_mask_decoder.bbox_embed.6.layers.2.weight', 'sam_mask_decoder.bbox_embed.6.layers.2.bias', 'sam_mask_decoder.bbox_embed.7.layers.0.weight', 'sam_mask_decoder.bbox_embed.7.layers.0.bias', 'sam_mask_decoder.bbox_embed.7.layers.1.weight', 'sam_mask_decoder.bbox_embed.7.layers.1.bias', 'sam_mask_decoder.bbox_embed.7.layers.2.weight', 'sam_mask_decoder.bbox_embed.7.layers.2.bias', 'sam_mask_decoder.bbox_embed.8.layers.0.weight', 'sam_mask_decoder.bbox_embed.8.layers.0.bias', 'sam_mask_decoder.bbox_embed.8.layers.1.weight', 'sam_mask_decoder.bbox_embed.8.layers.1.bias', 'sam_mask_decoder.bbox_embed.8.layers.2.weight', 'sam_mask_decoder.bbox_embed.8.layers.2.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.0.norm1.weight', 'sam_mask_decoder.decoder.layers.0.norm1.bias', 'sam_mask_decoder.decoder.layers.0.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.0.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.0.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.0.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.0.norm2.weight', 'sam_mask_decoder.decoder.layers.0.norm2.bias', 'sam_mask_decoder.decoder.layers.0.linear1.weight', 'sam_mask_decoder.decoder.layers.0.linear1.bias', 'sam_mask_decoder.decoder.layers.0.linear2.weight', 'sam_mask_decoder.decoder.layers.0.linear2.bias', 'sam_mask_decoder.decoder.layers.0.norm3.weight', 'sam_mask_decoder.decoder.layers.0.norm3.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.1.norm1.weight', 'sam_mask_decoder.decoder.layers.1.norm1.bias', 'sam_mask_decoder.decoder.layers.1.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.1.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.1.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.1.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.1.norm2.weight', 'sam_mask_decoder.decoder.layers.1.norm2.bias', 'sam_mask_decoder.decoder.layers.1.linear1.weight', 'sam_mask_decoder.decoder.layers.1.linear1.bias', 'sam_mask_decoder.decoder.layers.1.linear2.weight', 'sam_mask_decoder.decoder.layers.1.linear2.bias', 'sam_mask_decoder.decoder.layers.1.norm3.weight', 'sam_mask_decoder.decoder.layers.1.norm3.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.2.norm1.weight', 'sam_mask_decoder.decoder.layers.2.norm1.bias', 'sam_mask_decoder.decoder.layers.2.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.2.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.2.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.2.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.2.norm2.weight', 'sam_mask_decoder.decoder.layers.2.norm2.bias', 'sam_mask_decoder.decoder.layers.2.linear1.weight', 'sam_mask_decoder.decoder.layers.2.linear1.bias', 'sam_mask_decoder.decoder.layers.2.linear2.weight', 'sam_mask_decoder.decoder.layers.2.linear2.bias', 'sam_mask_decoder.decoder.layers.2.norm3.weight', 'sam_mask_decoder.decoder.layers.2.norm3.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.3.norm1.weight', 'sam_mask_decoder.decoder.layers.3.norm1.bias', 'sam_mask_decoder.decoder.layers.3.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.3.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.3.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.3.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.3.norm2.weight', 'sam_mask_decoder.decoder.layers.3.norm2.bias', 'sam_mask_decoder.decoder.layers.3.linear1.weight', 'sam_mask_decoder.decoder.layers.3.linear1.bias', 'sam_mask_decoder.decoder.layers.3.linear2.weight', 'sam_mask_decoder.decoder.layers.3.linear2.bias', 'sam_mask_decoder.decoder.layers.3.norm3.weight', 'sam_mask_decoder.decoder.layers.3.norm3.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.4.norm1.weight', 'sam_mask_decoder.decoder.layers.4.norm1.bias', 'sam_mask_decoder.decoder.layers.4.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.4.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.4.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.4.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.4.norm2.weight', 'sam_mask_decoder.decoder.layers.4.norm2.bias', 'sam_mask_decoder.decoder.layers.4.linear1.weight', 'sam_mask_decoder.decoder.layers.4.linear1.bias', 'sam_mask_decoder.decoder.layers.4.linear2.weight', 'sam_mask_decoder.decoder.layers.4.linear2.bias', 'sam_mask_decoder.decoder.layers.4.norm3.weight', 'sam_mask_decoder.decoder.layers.4.norm3.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.5.norm1.weight', 'sam_mask_decoder.decoder.layers.5.norm1.bias', 'sam_mask_decoder.decoder.layers.5.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.5.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.5.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.5.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.5.norm2.weight', 'sam_mask_decoder.decoder.layers.5.norm2.bias', 'sam_mask_decoder.decoder.layers.5.linear1.weight', 'sam_mask_decoder.decoder.layers.5.linear1.bias', 'sam_mask_decoder.decoder.layers.5.linear2.weight', 'sam_mask_decoder.decoder.layers.5.linear2.bias', 'sam_mask_decoder.decoder.layers.5.norm3.weight', 'sam_mask_decoder.decoder.layers.5.norm3.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.6.norm1.weight', 'sam_mask_decoder.decoder.layers.6.norm1.bias', 'sam_mask_decoder.decoder.layers.6.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.6.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.6.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.6.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.6.norm2.weight', 'sam_mask_decoder.decoder.layers.6.norm2.bias', 'sam_mask_decoder.decoder.layers.6.linear1.weight', 'sam_mask_decoder.decoder.layers.6.linear1.bias', 'sam_mask_decoder.decoder.layers.6.linear2.weight', 'sam_mask_decoder.decoder.layers.6.linear2.bias', 'sam_mask_decoder.decoder.layers.6.norm3.weight', 'sam_mask_decoder.decoder.layers.6.norm3.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.7.norm1.weight', 'sam_mask_decoder.decoder.layers.7.norm1.bias', 'sam_mask_decoder.decoder.layers.7.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.7.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.7.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.7.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.7.norm2.weight', 'sam_mask_decoder.decoder.layers.7.norm2.bias', 'sam_mask_decoder.decoder.layers.7.linear1.weight', 'sam_mask_decoder.decoder.layers.7.linear1.bias', 'sam_mask_decoder.decoder.layers.7.linear2.weight', 'sam_mask_decoder.decoder.layers.7.linear2.bias', 'sam_mask_decoder.decoder.layers.7.norm3.weight', 'sam_mask_decoder.decoder.layers.7.norm3.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn.sampling_offsets.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn.sampling_offsets.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn.attention_weights.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn.attention_weights.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn.value_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn.value_proj.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn.output_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn.output_proj.bias', 'sam_mask_decoder.decoder.layers.8.norm1.weight', 'sam_mask_decoder.decoder.layers.8.norm1.bias', 'sam_mask_decoder.decoder.layers.8.self_attn.in_proj_weight', 'sam_mask_decoder.decoder.layers.8.self_attn.in_proj_bias', 'sam_mask_decoder.decoder.layers.8.self_attn.out_proj.weight', 'sam_mask_decoder.decoder.layers.8.self_attn.out_proj.bias', 'sam_mask_decoder.decoder.layers.8.norm2.weight', 'sam_mask_decoder.decoder.layers.8.norm2.bias', 'sam_mask_decoder.decoder.layers.8.linear1.weight', 'sam_mask_decoder.decoder.layers.8.linear1.bias', 'sam_mask_decoder.decoder.layers.8.linear2.weight', 'sam_mask_decoder.decoder.layers.8.linear2.bias', 'sam_mask_decoder.decoder.layers.8.norm3.weight', 'sam_mask_decoder.decoder.layers.8.norm3.bias', 'sam_mask_decoder.decoder.norm.weight', 'sam_mask_decoder.decoder.norm.bias', 'sam_mask_decoder.decoder.ref_point_head.layers.0.weight', 'sam_mask_decoder.decoder.ref_point_head.layers.0.bias', 'sam_mask_decoder.decoder.ref_point_head.layers.1.weight', 'sam_mask_decoder.decoder.ref_point_head.layers.1.bias']
    # unallowed_parameter_names += ['sam_mask_decoder.decoder_norm.weight', 'sam_mask_decoder.decoder_norm.bias', 'sam_mask_decoder.query_feat.weight', 'sam_mask_decoder.query_red_points.weight']
    # unallowed_parameter_names += ['sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.0.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.0.norm4.weight', 'sam_mask_decoder.decoder.layers.0.norm4.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.1.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.1.norm4.weight', 'sam_mask_decoder.decoder.layers.1.norm4.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.2.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.2.norm4.weight', 'sam_mask_decoder.decoder.layers.2.norm4.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.3.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.3.norm4.weight', 'sam_mask_decoder.decoder.layers.3.norm4.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.4.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.4.norm4.weight', 'sam_mask_decoder.decoder.layers.4.norm4.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.5.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.5.norm4.weight', 'sam_mask_decoder.decoder.layers.5.norm4.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.6.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.6.norm4.weight', 'sam_mask_decoder.decoder.layers.6.norm4.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.7.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.7.norm4.weight', 'sam_mask_decoder.decoder.layers.7.norm4.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.q_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.q_proj.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.k_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.k_proj.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.v_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.v_proj.bias', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.out_proj.weight', 'sam_mask_decoder.decoder.layers.8.cross_attn_image_to_token.out_proj.bias', 'sam_mask_decoder.decoder.layers.8.norm4.weight', 'sam_mask_decoder.decoder.layers.8.norm4.bias']
    # unallowed_parameter_names += ['memory_attention.layers.0.cross_attn_image.k_proj.weight', 'memory_attention.layers.0.cross_attn_image.v_proj.weight', 'memory_attention.layers.1.cross_attn_image.k_proj.weight', 'memory_attention.layers.1.cross_attn_image.v_proj.weight', 'memory_attention.layers.2.cross_attn_image.k_proj.weight', 'memory_attention.layers.2.cross_attn_image.v_proj.weight', 'memory_attention.layers.3.cross_attn_image.k_proj.weight', 'memory_attention.layers.3.cross_attn_image.v_proj.weight',]
    
    if ignore_missing_keys:
        for item in ignore_missing_keys:
            if item in state_dict.keys():
                removed_value = state_dict.pop(item, None)
                print(f"The key {item} is removed.")

    # if ignore_missing_keys:
    #     ignore_missing_keys += unallowed_parameter_names
    # else:
    #     ignore_missing_keys = unallowed_parameter_names
    # end
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # ignore_unexpected_keys = ['memory_encoder.out_proj.weight', 'memory_encoder.out_proj.bias', 'no_obj_embed_spatial', 'no_obj_ptr', 'mask_downsample.weight', 'mask_downsample.bias', 'obj_ptr_proj.layers.0.weight', 'obj_ptr_proj.layers.0.bias', 'obj_ptr_proj.layers.1.weight', 'obj_ptr_proj.layers.1.bias', 'obj_ptr_proj.layers.2.weight', 'obj_ptr_proj.layers.2.bias']
    # ignore_unexpected_keys += ['sam_mask_decoder.output_hypernetworks_mlps.3.layers.0.weight', 'sam_mask_decoder.output_hypernetworks_mlps.3.layers.0.bias', 'sam_mask_decoder.output_hypernetworks_mlps.3.layers.1.weight', 'sam_mask_decoder.output_hypernetworks_mlps.3.layers.1.bias', 'sam_mask_decoder.output_hypernetworks_mlps.3.layers.2.weight', 'sam_mask_decoder.output_hypernetworks_mlps.3.layers.2.bias']

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model