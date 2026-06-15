"""Transformer backbone registry and feature extractor construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn

from dl_final.models.feature_extractors import FrozenFeatureExtractor


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    alias: str
    family: str
    model_id: str
    input_size: int
    pooling_policy: str
    timm_global_pool: str
    expected_feature_dim: int


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "vit_b16": BackboneSpec(
        name="vit_b16",
        alias="vit",
        family="vit",
        model_id="vit_base_patch16_224.augreg_in21k_ft_in1k",
        input_size=224,
        pooling_policy="cls_token",
        timm_global_pool="token",
        expected_feature_dim=768,
    ),
    "swin_tiny": BackboneSpec(
        name="swin_tiny",
        alias="swin",
        family="swin",
        model_id="swin_tiny_patch4_window7_224.ms_in1k",
        input_size=224,
        pooling_policy="avg_pooled_final_stage",
        timm_global_pool="avg",
        expected_feature_dim=768,
    ),
    "deit3_small": BackboneSpec(
        name="deit3_small",
        alias="deit3s",
        family="deit_iii",
        model_id="deit3_small_patch16_224.fb_in1k",
        input_size=224,
        pooling_policy="cls_token",
        timm_global_pool="token",
        expected_feature_dim=384,
    ),
    "beit_base": BackboneSpec(
        name="beit_base",
        alias="beit",
        family="beit",
        model_id="beit_base_patch16_224.in22k_ft_in22k_in1k",
        input_size=224,
        pooling_policy="avg_pooled_patch_tokens_with_fc_norm",
        timm_global_pool="avg",
        expected_feature_dim=768,
    ),
}

CANONICAL_BACKBONES = ("vit_b16", "swin_tiny", "beit_base")


def supported_backbones(*, include_candidates: bool = False) -> list[str]:
    if include_candidates:
        return list(BACKBONE_SPECS)
    return list(CANONICAL_BACKBONES)


def backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: "
            f"{', '.join(supported_backbones(include_candidates=True))}"
        ) from exc


def backbone_alias(name: str) -> str:
    return backbone_spec(name).alias


def expected_feature_dim(name: str) -> int:
    return backbone_spec(name).expected_feature_dim


def build_frozen_feature_extractor(
    name: str,
    *,
    pretrained: bool = True,
) -> FrozenFeatureExtractor:
    """Build a classifier-free frozen transformer feature extractor."""

    spec = backbone_spec(name)
    create_kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": 0,
        "global_pool": spec.timm_global_pool,
    }
    model = timm.create_model(spec.model_id, **create_kwargs)
    feature_dim = int(getattr(model, "num_features", spec.expected_feature_dim))
    if feature_dim != spec.expected_feature_dim:
        raise ValueError(
            f"{name} feature_dim={feature_dim}, expected {spec.expected_feature_dim}."
        )
    return FrozenFeatureExtractor(
        name=name,
        model=model,
        feature_dim=feature_dim,
        pooling_policy=spec.pooling_policy,
        model_id=spec.model_id,
    )


def build_classification_backbone(
    name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Build a timm classification model for image-level fine-tuning."""

    spec = backbone_spec(name)
    return timm.create_model(
        spec.model_id,
        pretrained=pretrained,
        num_classes=num_classes,
        global_pool=spec.timm_global_pool,
    )


def apply_finetuning_policy(
    model: nn.Module,
    name: str,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    """Freeze a transformer model, then unfreeze the policy-selected final blocks and head."""

    resolved_policy = policy or default_finetuning_policy(name)
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_prefixes = _trainable_prefixes_for_policy(model, name, resolved_policy)
    for parameter_name, parameter in model.named_parameters():
        if any(_matches_prefix(parameter_name, prefix) for prefix in trainable_prefixes):
            parameter.requires_grad = True

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_names = [
        parameter_name
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "backbone": name,
        "policy": resolved_policy,
        "trainable_prefixes": trainable_prefixes,
        "trainable_parameter_count": len(trainable_names),
        "trainable_parameter_examples": trainable_names[:20],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
    }


def default_finetuning_policy(name: str) -> str:
    if name == "swin_tiny":
        return "last_stage"
    if name in {"vit_b16", "beit_base"}:
        return "last_2_blocks"
    raise ValueError(f"No default fine-tuning policy for {name!r}.")


def build_finetuned_feature_extractor(
    name: str,
    *,
    checkpoint_path: str,
    num_classes: int,
    pretrained: bool = False,
) -> FrozenFeatureExtractor:
    """Build a classifier-free feature extractor from a selected fine-tuned checkpoint."""

    model = build_classification_backbone(name, num_classes=num_classes, pretrained=pretrained)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)

    spec = backbone_spec(name)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0, global_pool=spec.timm_global_pool)
    else:
        raise ValueError(f"{name} model does not support reset_classifier().")
    feature_dim = int(getattr(model, "num_features", spec.expected_feature_dim))
    if feature_dim != spec.expected_feature_dim:
        raise ValueError(
            f"{name} feature_dim={feature_dim}, expected {spec.expected_feature_dim}."
        )
    return FrozenFeatureExtractor(
        name=name,
        model=model,
        feature_dim=feature_dim,
        pooling_policy=spec.pooling_policy,
        model_id=spec.model_id,
    )


def _trainable_prefixes_for_policy(model: nn.Module, name: str, policy: str) -> list[str]:
    if name in {"vit_b16", "beit_base"} and policy == "last_1_block":
        block_count = len(model.blocks)
        prefixes = [f"blocks.{index}" for index in range(max(block_count - 1, 0), block_count)]
        prefixes.extend(_existing_prefixes(model, ["norm", "fc_norm", "head"]))
        return prefixes
    if name in {"vit_b16", "beit_base"} and policy == "last_2_blocks":
        block_count = len(model.blocks)
        prefixes = [f"blocks.{index}" for index in range(max(block_count - 2, 0), block_count)]
        prefixes.extend(_existing_prefixes(model, ["norm", "fc_norm", "head"]))
        return prefixes
    if name == "swin_tiny" and policy == "last_stage":
        layer_count = len(model.layers)
        prefixes = [f"layers.{layer_count - 1}"]
        prefixes.extend(_existing_prefixes(model, ["norm", "head"]))
        return prefixes
    raise ValueError(f"Unsupported fine-tuning policy {policy!r} for backbone {name!r}.")


def _existing_prefixes(model: nn.Module, candidates: list[str]) -> list[str]:
    parameter_names = [name for name, _ in model.named_parameters()]
    return [
        candidate
        for candidate in candidates
        if any(_matches_prefix(parameter_name, candidate) for parameter_name in parameter_names)
    ]


def _matches_prefix(parameter_name: str, prefix: str) -> bool:
    return parameter_name == prefix or parameter_name.startswith(f"{prefix}.")
