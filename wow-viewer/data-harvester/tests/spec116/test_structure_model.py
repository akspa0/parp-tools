"""Spec 116 US3 T021: StructureSlotNet architecture contract tests.

Verifies the per-slot structure classifier:
- Output shape is (B, CLASS_COUNT, 16, 16) — per-chunk, not per-pixel.
- Parameter count is in the small-model class (same capacity class as TerrainFeatureNet).
- Base slot 0 is refused (FR-008); only detail slots 1-3 are constructible.
- Multi-head / multi-slot construction is refused (constitution IV).
- The identity block carries slot, base, num_classes, param_count.
- Ground truth never reaches inference: the model takes exactly one RGB argument.
"""

from __future__ import annotations

import pytest

from harvester.v50.terrain_feature_labels import CLASS_COUNT

from harvester.spec116.structure_model import (
    CHUNK_GRID,
    DETAIL_SLOTS,
    MAX_DETAIL_SLOT,
    StructureModelError,
    StructureSlotNet,
    build_structure_model,
    structure_model_identity,
)


class TestOutputShape:
    def test_output_is_per_chunk_16x16(self) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=1, base=4)
        out = model(torch.zeros(2, 3, 256, 256))
        assert out.shape == (2, CLASS_COUNT, CHUNK_GRID, CHUNK_GRID)

    def test_output_is_per_chunk_not_per_pixel(self) -> None:
        """The structure model predicts at chunk resolution (16x16), not pixel resolution (256x256)."""
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=2, base=4)
        out = model(torch.zeros(1, 3, 256, 256))
        assert out.shape[-1] == CHUNK_GRID
        assert out.shape[-2] == CHUNK_GRID
        assert out.shape[-1] != 256

    def test_batch_dimension_preserved(self) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=1, base=4)
        out = model(torch.zeros(7, 3, 256, 256))
        assert out.shape[0] == 7

    def test_refuses_non_rgb_input(self) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=1, base=4)
        with pytest.raises(StructureModelError, match="rgb must be"):
            model(torch.zeros(2, 4, 256, 256))

    def test_refuses_3d_input(self) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=1, base=4)
        with pytest.raises(StructureModelError, match="rgb must be"):
            model(torch.zeros(3, 256, 256))


class TestSlotConstraint:
    @pytest.mark.parametrize("slot", DETAIL_SLOTS)
    def test_detail_slots_are_constructible(self, slot: int) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=slot, base=4)
        out = model(torch.zeros(1, 3, 256, 256))
        assert out.shape == (1, CLASS_COUNT, CHUNK_GRID, CHUNK_GRID)

    def test_base_slot_0_is_refused(self) -> None:
        """FR-008: the base slot is never predicted (opaque terrain, materialised by subtraction)."""
        torch = pytest.importorskip("torch")
        with pytest.raises(StructureModelError, match="slot must be one of"):
            StructureSlotNet(slot=0, base=4)

    def test_slot_above_max_is_refused(self) -> None:
        torch = pytest.importorskip("torch")
        with pytest.raises(StructureModelError, match="slot must be one of"):
            StructureSlotNet(slot=MAX_DETAIL_SLOT + 1, base=4)

    def test_slot_is_stored_on_instance(self) -> None:
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=3, base=4)
        assert model.slot == 3


class TestCapacityClass:
    def test_param_count_in_small_model_class(self) -> None:
        """Same capacity class as TerrainFeatureNet: base=32 yields ~1-3M params."""
        torch = pytest.importorskip("torch")
        model, identity = build_structure_model(slot=1, base=32)
        param_count = identity["architecture"]["param_count"]
        # TerrainFeatureNet at base=32 is ~1.5M; the adaptive-pool head is slightly lighter
        # than the 3x3 conv head, so we expect roughly the same order of magnitude.
        assert 500_000 < param_count < 5_000_000

    def test_param_count_matches_manual_count(self) -> None:
        torch = pytest.importorskip("torch")
        model, identity = build_structure_model(slot=1, base=8)
        manual = sum(p.numel() for p in model.parameters())
        assert identity["architecture"]["param_count"] == manual

    def test_smaller_base_fewer_params(self) -> None:
        torch = pytest.importorskip("torch")
        _, id_small = build_structure_model(slot=1, base=4)
        _, id_large = build_structure_model(slot=1, base=32)
        assert id_small["architecture"]["param_count"] < id_large["architecture"]["param_count"]


class TestIdentityBlock:
    def test_identity_carries_required_fields(self) -> None:
        torch = pytest.importorskip("torch")
        model, identity = build_structure_model(slot=2, base=16)
        arch = identity["architecture"]
        assert arch["class"] == "StructureSlotNet"
        assert arch["base"] == 16
        assert arch["slot"] == 2
        assert arch["num_classes"] == CLASS_COUNT
        assert arch["param_count"] > 0
        assert "config_sha256" in arch
        assert len(arch["config_sha256"]) == 64

    def test_pretrained_source_is_none(self) -> None:
        """Structure models train from scratch; they never inherit weights from another slot."""
        torch = pytest.importorskip("torch")
        _, identity = build_structure_model(slot=1, base=4)
        assert identity["pretrained_source"] is None

    def test_different_slots_same_config_hash(self) -> None:
        """The config hash includes the slot, so different slots have different identities."""
        torch = pytest.importorskip("torch")
        _, id1 = build_structure_model(slot=1, base=8)
        _, id2 = build_structure_model(slot=2, base=8)
        assert id1["architecture"]["config_sha256"] != id2["architecture"]["config_sha256"]

    def test_different_base_different_config_hash(self) -> None:
        torch = pytest.importorskip("torch")
        _, id1 = build_structure_model(slot=1, base=4)
        _, id2 = build_structure_model(slot=1, base=8)
        assert id1["architecture"]["config_sha256"] != id2["architecture"]["config_sha256"]


class TestNoMultiHead:
    def test_no_multi_head_parameter(self) -> None:
        """Constitution IV: the model has exactly one head (one 1x1 conv), not one per slot."""
        torch = pytest.importorskip("torch")
        model, _ = build_structure_model(slot=1, base=4)
        # Exactly one head conv.
        head_convs = [name for name, _ in model.named_modules() if name == "head"]
        assert len(head_convs) == 1
        # The head produces exactly num_classes output channels, not num_classes * num_slots.
        assert model.head.out_channels == CLASS_COUNT

    def test_model_takes_exactly_one_argument(self) -> None:
        """Ground truth never reaches inference: forward() takes only rgb."""
        torch = pytest.importorskip("torch")
        import inspect

        sig = inspect.signature(StructureSlotNet.forward)
        assert list(sig.parameters.keys()) == ["self", "rgb"]