from __future__ import annotations

import pytest
import torch

from harvester.v60.object_sieve_model import ObjectSieveNet, object_sieve_loss


@pytest.mark.parametrize("variant", ["clean_only", "auxiliary_mask_loss", "predicted_mask_guided"])
def test_object_sieve_variants_keep_ground_truth_out_of_forward(variant: str) -> None:
    model = ObjectSieveNet(variant=variant)
    input_signal = torch.full((2, 1, 32, 32), 0.5)
    clean_target = torch.full((2, 1, 32, 32), 0.4)
    mask_target = torch.zeros((2, 1, 32, 32))
    mask_target[:, :, 8:16, 8:16] = 1.0

    predictions = model(input_signal)
    losses = object_sieve_loss(predictions, clean_target, mask_target, variant)

    assert predictions.clean_terrain.shape == input_signal.shape
    assert predictions.contamination_logits.shape == input_signal.shape
    assert set(losses) == {"clean_loss", "mask_loss", "total_loss"}
    assert torch.isfinite(losses["total_loss"])
    if variant == "clean_only":
        assert float(losses["mask_loss"]) == 0.0
    else:
        assert float(losses["mask_loss"].detach()) > 0.0
