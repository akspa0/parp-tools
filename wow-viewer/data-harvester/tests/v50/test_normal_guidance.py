"""Normal-derived gradient supervision: physics, masking, and degenerate-input safety."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harvester.v50.normal_guidance import (  # noqa: E402
    GRID_SPACING,
    normal_gradient_loss,
)


def _plane_normals(dhdx: float, dhdy: float, size: int = 33) -> torch.Tensor:
    """Unit normals of the plane z = dhdx*x + dhdy*y, in ADT grid axes (channel 2 = up)."""
    n = np.zeros((1, size, size, 3), dtype=np.float32)
    vec = np.array([-dhdx, -dhdy, 1.0], dtype=np.float64)
    vec /= np.linalg.norm(vec)
    n[..., 0], n[..., 1], n[..., 2] = vec
    return torch.from_numpy(n)


def _plane_height(dhdx: float, dhdy: float, scale: float, size: int = 33) -> torch.Tensor:
    """The matching NORMALISED height field for that plane (v112.1 divides by `scale`)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    world = (dhdx * xx + dhdy * yy) * GRID_SPACING
    return torch.from_numpy((world / scale).astype(np.float32))[None]


class TestPhysics:
    @pytest.mark.parametrize("dhdx,dhdy", [(0.0, 0.0), (0.3, 0.0), (0.0, -0.25), (0.2, 0.15)])
    def test_matching_plane_gives_near_zero_loss(self, dhdx: float, dhdy: float) -> None:
        """A height field that exactly matches its normals must score ~0 — this pins the sign
        convention AND the GRID_SPACING/denominator scaling together."""
        scale = 40.0
        mask = torch.ones(1, 33, 33)
        loss = normal_gradient_loss(
            _plane_height(dhdx, dhdy, scale),
            _plane_normals(dhdx, dhdy),
            mask,
            torch.tensor([scale]),
        )
        assert float(loss) < 1e-4

    def test_wrong_sign_is_penalised(self) -> None:
        """A surface sloping the OPPOSITE way must score much worse than the correct one."""
        scale = 40.0
        mask = torch.ones(1, 33, 33)
        normals = _plane_normals(0.3, 0.0)
        good = normal_gradient_loss(_plane_height(0.3, 0.0, scale), normals, mask, torch.tensor([scale]))
        bad = normal_gradient_loss(_plane_height(-0.3, 0.0, scale), normals, mask, torch.tensor([scale]))
        assert float(bad) > float(good) * 10

    def test_flat_prediction_penalised_against_sloped_normals(self) -> None:
        scale = 40.0
        mask = torch.ones(1, 33, 33)
        normals = _plane_normals(0.4, 0.0)
        flat = normal_gradient_loss(torch.full((1, 33, 33), 0.5), normals, mask, torch.tensor([scale]))
        assert float(flat) > 1e-3

    def test_scale_is_honoured(self) -> None:
        """denominator is per-tile; the same normals against a different scale must differ."""
        mask = torch.ones(1, 33, 33)
        normals = _plane_normals(0.3, 0.0)
        height = _plane_height(0.3, 0.0, 40.0)
        matched = normal_gradient_loss(height, normals, mask, torch.tensor([40.0]))
        mismatched = normal_gradient_loss(height, normals, mask, torch.tensor([400.0]))
        assert float(mismatched) > float(matched)


class TestMaskingAndSafety:
    def test_only_masked_vertices_contribute(self) -> None:
        scale = 40.0
        normals = _plane_normals(0.3, 0.0)
        height = _plane_height(0.3, 0.0, scale)
        corrupted = height.clone()
        # Corrupt well clear of the masked region: a central difference at row r reads rows r±1,
        # so the corruption must start >1 row beyond the last masked row to be truly excluded.
        corrupted[:, 18:, :] += 5.0
        mask = torch.zeros(1, 33, 33)
        mask[:, :16, :] = 1.0
        loss = normal_gradient_loss(corrupted, normals, mask, torch.tensor([scale]))
        assert float(loss) < 1e-3

    def test_empty_mask_returns_finite_zero_not_nan(self) -> None:
        """An unlucky batch with no valid vertices must not poison training with NaN."""
        loss = normal_gradient_loss(
            torch.rand(1, 33, 33), _plane_normals(0.2, 0.2), torch.zeros(1, 33, 33), torch.tensor([40.0])
        )
        assert float(loss) == 0.0
        assert torch.isfinite(loss)

    def test_vertical_face_does_not_produce_inf(self) -> None:
        """nz -> 0 makes dh/dx diverge; clamping must keep the loss finite."""
        normals = torch.zeros(1, 33, 33, 3)
        normals[..., 0] = 1.0  # fully vertical: nz == 0
        loss = normal_gradient_loss(
            torch.rand(1, 33, 33), normals, torch.ones(1, 33, 33), torch.tensor([40.0])
        )
        assert torch.isfinite(loss)

    def test_gradient_flows_to_prediction(self) -> None:
        predicted = torch.rand(1, 33, 33, requires_grad=True)
        loss = normal_gradient_loss(
            predicted, _plane_normals(0.3, 0.1), torch.ones(1, 33, 33), torch.tensor([40.0])
        )
        loss.backward()
        assert predicted.grad is not None and torch.isfinite(predicted.grad).all()

    def test_malformed_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError):
            normal_gradient_loss(
                torch.rand(33, 33), _plane_normals(0.0, 0.0), torch.ones(1, 33, 33), torch.tensor([1.0])
            )


class TestTrainerWiring:
    def test_normal_loss_guard_precedes_plan_construction(self) -> None:
        """Regression: `use_normal_loss` was once defined AFTER the plan referenced it, which raised
        UnboundLocalError the moment --normal-weight was set."""
        import inspect

        from harvester.v50 import direct_geometry_train

        source = inspect.getsource(direct_geometry_train.main)
        definition = source.index("use_normal_loss = args.normal_weight > 0")
        first_use = source.index("if use_normal_loss:")
        assert definition < first_use
