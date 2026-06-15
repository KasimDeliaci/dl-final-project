"""Shape and stability tests for E3d metadata fusion models."""

from __future__ import annotations

import torch

from dl_final.models.metadata_fusion import (
    MetadataFiLMMLP,
    MetadataGatedBackboneMLP,
    MetadataTwoBranchMLP,
)


def test_metadata_gated_backbone_mlp_outputs_logits_and_valid_gates() -> None:
    model = MetadataGatedBackboneMLP(
        input_dims=[4, 3, 2],
        metadata_dim=5,
        num_classes=7,
        hidden_dims=[8],
        gate_hidden_dim=6,
    )
    features = torch.randn(6, 14)
    logits = model(features)
    gates = model.gate_values(features)

    assert logits.shape == (6, 7)
    assert gates.shape == (6, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(gates).all()
    assert torch.all(gates >= 0)
    assert torch.all(gates <= 1)


def test_metadata_film_mlp_outputs_logits_and_gradients() -> None:
    model = MetadataFiLMMLP(
        image_dim=9,
        metadata_dim=5,
        num_classes=7,
        hidden_dims=[8],
        film_hidden_dim=6,
    )
    features = torch.randn(6, 14)
    logits = model(features)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (6, 7)
    assert torch.isfinite(logits).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_metadata_two_branch_mlp_outputs_logits() -> None:
    model = MetadataTwoBranchMLP(
        image_dim=9,
        metadata_dim=5,
        num_classes=7,
        image_hidden_dim=8,
        metadata_hidden_dim=4,
        fusion_hidden_dims=[6],
    )
    logits = model(torch.randn(6, 14))

    assert logits.shape == (6, 7)
    assert torch.isfinite(logits).all()
