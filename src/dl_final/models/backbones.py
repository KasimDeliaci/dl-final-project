"""Transformer backbone registry and frozen feature extractor construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm

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

CANONICAL_BACKBONES = ("vit_b16", "swin_tiny", "deit3_small")


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
