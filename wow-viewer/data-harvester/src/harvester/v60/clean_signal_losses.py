"""Parity and independently ablatable structural losses for the clean-signal lane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from harvester.v60.clean_signal_model import CleanSignalPredictions

LOSS_SCHEMA = "v7-clean-signal-loss-v1"
PARITY_PROFILE = "parity"
STRUCTURAL_PROFILE = "v7_structural_v1"
CLEAN_SIGNAL_LOSS_PROFILES = (PARITY_PROFILE, STRUCTURAL_PROFILE)


class CleanSignalLossError(ValueError):
    """Raised when clean-signal loss inputs or profiles are invalid."""


@dataclass(frozen=True)
class V7GuidanceConfig:
    """Versioned weights for one independently ablatable loss profile."""

    name: str
    point: float = 1.0
    gradient: float = 0.1
    frequency: float = 0.0
    laplacian: float = 0.0
    edge: float = 0.0
    transition: float = 0.0
    border: float = 0.0
    low_frequency: float = 0.0
    high_frequency: float = 0.0
    schema: str = LOSS_SCHEMA

    def __post_init__(self) -> None:
        if not self.name:
            raise CleanSignalLossError("loss profile name must not be empty")
        if self.schema != LOSS_SCHEMA:
            raise CleanSignalLossError(f"loss schema must be {LOSS_SCHEMA!r}")
        if any(weight < 0.0 for weight in self.weights.values()):
            raise CleanSignalLossError("loss weights must be non-negative")
        if self.point == 0.0 and self.gradient == 0.0 and not any(
            weight > 0.0 for key, weight in self.weights.items() if key not in {"point", "gradient"}
        ):
            raise CleanSignalLossError("loss profile must contain at least one non-zero weight")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "point": self.point,
            "gradient": self.gradient,
            "frequency": self.frequency,
            "laplacian": self.laplacian,
            "edge": self.edge,
            "transition": self.transition,
            "border": self.border,
            "low_frequency": self.low_frequency,
            "high_frequency": self.high_frequency,
        }

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "name": self.name, "weights": self.weights}


PARITY_CONFIG = V7GuidanceConfig(name=PARITY_PROFILE)
STRUCTURAL_CONFIG = V7GuidanceConfig(
    name=STRUCTURAL_PROFILE,
    frequency=0.08,
    laplacian=0.12,
    edge=0.12,
    transition=0.10,
    border=0.12,
    low_frequency=0.08,
    high_frequency=0.08,
)
LOSS_CONFIGS = {
    PARITY_PROFILE: PARITY_CONFIG,
    STRUCTURAL_PROFILE: STRUCTURAL_CONFIG,
}


def get_clean_signal_loss_config(profile: str) -> V7GuidanceConfig:
    """Return a frozen, named loss configuration or fail closed."""

    try:
        return LOSS_CONFIGS[profile]
    except KeyError as exc:
        raise CleanSignalLossError(
            f"unknown loss profile {profile!r}; expected one of {list(CLEAN_SIGNAL_LOSS_PROFILES)}"
        ) from exc


def _as_bchw(value: Any, *, name: str, device: torch.device | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise CleanSignalLossError(f"{name} must have 2, 3, or 4 dimensions; got {tensor.ndim}")
    if tensor.shape[-2] < 3 or tensor.shape[-1] < 3:
        raise CleanSignalLossError(f"{name} spatial dimensions must be at least 3; got {tuple(tensor.shape[-2:])}")
    if not torch.isfinite(tensor).all():
        raise CleanSignalLossError(f"{name} contains non-finite values")
    return tensor


def _pair(predicted: Any, target: Any) -> tuple[Tensor, Tensor]:
    pred = _as_bchw(predicted, name="predicted")
    truth = _as_bchw(target, name="target", device=pred.device).to(dtype=pred.dtype)
    if pred.shape != truth.shape:
        raise CleanSignalLossError(f"predicted/target shapes differ: {tuple(pred.shape)} != {tuple(truth.shape)}")
    return pred, truth


def _zero_like(field: Tensor) -> Tensor:
    return torch.zeros((), dtype=field.dtype, device=field.device)


def point_loss(predicted: Any, target: Any) -> Tensor:
    """Pixelwise L1 parity term."""

    pred, truth = _pair(predicted, target)
    return functional.l1_loss(pred, truth)


def gradient_loss(predicted: Any, target: Any) -> Tensor:
    """First-derivative L1 parity term over horizontal and vertical differences."""

    pred, truth = _pair(predicted, target)
    pred_x = pred[..., 1:] - pred[..., :-1]
    truth_x = truth[..., 1:] - truth[..., :-1]
    pred_y = pred[..., 1:, :] - pred[..., :-1, :]
    truth_y = truth[..., 1:, :] - truth[..., :-1, :]
    return functional.l1_loss(pred_x, truth_x) + functional.l1_loss(pred_y, truth_y)


def _channelwise_conv(field: Tensor, kernel: Tensor) -> Tensor:
    channels = field.shape[1]
    return functional.conv2d(field, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels)


def _sobel_kernels(field: Tensor) -> tuple[Tensor, Tensor]:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)
    return sobel_x, sobel_x.transpose(2, 3)


def _log_spectrum(field: Tensor) -> Tensor:
    return torch.log1p(torch.fft.rfft2(field).abs())


def frequency_loss(predicted: Any, target: Any) -> Tensor:
    """Full 2D log-magnitude spectrum L1 term."""

    pred, truth = _pair(predicted, target)
    return functional.l1_loss(_log_spectrum(pred), _log_spectrum(truth))


def laplacian_loss(predicted: Any, target: Any) -> Tensor:
    """L1 curvature loss using a four-neighbour Laplacian."""

    pred, truth = _pair(predicted, target)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    return functional.l1_loss(_channelwise_conv(pred, kernel), _channelwise_conv(truth, kernel))


def edge_loss(predicted: Any, target: Any) -> Tensor:
    """L1 Sobel-edge magnitude loss."""

    pred, truth = _pair(predicted, target)
    sobel_x, sobel_y = _sobel_kernels(pred)
    pred_edge = _channelwise_conv(pred, sobel_x).abs() + _channelwise_conv(pred, sobel_y).abs()
    truth_edge = _channelwise_conv(truth, sobel_x).abs() + _channelwise_conv(truth, sobel_y).abs()
    return functional.l1_loss(pred_edge, truth_edge)


def transition_loss(predicted: Any, target: Any, *, gain: float = 3.0) -> Tensor:
    """Target-gradient-weighted L1 term that emphasizes relief transitions."""

    if gain < 0.0:
        raise CleanSignalLossError("transition gain must be non-negative")
    pred, truth = _pair(predicted, target)
    sobel_x, sobel_y = _sobel_kernels(truth)
    grad_x = _channelwise_conv(truth, sobel_x)
    grad_y = _channelwise_conv(truth, sobel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    normalized = magnitude / (magnitude.mean(dim=(-2, -1), keepdim=True) + 1e-6)
    weights = 1.0 + gain * normalized.clamp(0.0, 1.0)
    return (pred.sub(truth).abs() * weights).sum() / weights.sum().clamp_min(1e-6)


def border_loss(predicted: Any, target: Any, *, edge_width: int = 12) -> Tensor:
    """L1 error weighted on the published tile border."""

    if edge_width < 0:
        raise CleanSignalLossError("border edge_width must be non-negative")
    pred, truth = _pair(predicted, target)
    if edge_width == 0:
        return _zero_like(pred)
    height, width = pred.shape[-2:]
    border = torch.zeros((1, 1, height, width), dtype=pred.dtype, device=pred.device)
    border[..., :edge_width, :] = 1.0
    border[..., -edge_width:, :] = 1.0
    border[..., :, :edge_width] = 1.0
    border[..., :, -edge_width:] = 1.0
    weights = border.expand_as(pred)
    return (pred.sub(truth).abs() * weights).sum() / weights.sum().clamp_min(1e-6)


def _band_loss(pred: Tensor, truth: Tensor, *, low: bool, cutoff: float) -> Tensor:
    height, width = pred.shape[-2:]
    fy = torch.fft.fftfreq(height, device=pred.device, dtype=pred.dtype)
    fx = torch.fft.rfftfreq(width, device=pred.device, dtype=pred.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    mask = radius <= cutoff if low else radius >= cutoff
    pred_spectrum = _log_spectrum(pred)
    truth_spectrum = _log_spectrum(truth)
    if not bool(mask.any()):
        return _zero_like(pred)
    return functional.l1_loss(pred_spectrum[..., mask], truth_spectrum[..., mask])


def low_frequency_band_loss(predicted: Any, target: Any, *, cutoff: float = 0.15) -> Tensor:
    """L1 log-spectrum loss on low spatial frequencies."""

    if not 0.0 < cutoff < 1.0:
        raise CleanSignalLossError("frequency cutoff must be between 0 and 1")
    pred, truth = _pair(predicted, target)
    return _band_loss(pred, truth, low=True, cutoff=cutoff)


def high_frequency_band_loss(predicted: Any, target: Any, *, cutoff: float = 0.35) -> Tensor:
    """L1 log-spectrum loss on high spatial frequencies."""

    if not 0.0 < cutoff < 1.0:
        raise CleanSignalLossError("frequency cutoff must be between 0 and 1")
    pred, truth = _pair(predicted, target)
    return _band_loss(pred, truth, low=False, cutoff=cutoff)


def _target_value(targets: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in targets:
            return targets[name]
    raise CleanSignalLossError(f"targets missing one of {list(names)}")


def clean_signal_loss(
    predictions: CleanSignalPredictions,
    targets: Mapping[str, Any],
    *,
    profile: str = PARITY_PROFILE,
    config: V7GuidanceConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one profile and retain every component for independent reporting."""

    if not isinstance(predictions, CleanSignalPredictions):
        raise CleanSignalLossError("predictions must be CleanSignalPredictions")
    if not isinstance(targets, Mapping):
        raise CleanSignalLossError("targets must be a mapping of training-only target arrays")
    selected = config if config is not None else get_clean_signal_loss_config(profile)
    if config is not None and config.name != profile:
        raise CleanSignalLossError(f"config name {config.name!r} does not match profile {profile!r}")

    target_height = _target_value(targets, "relative_height_257", "height_prediction_257")
    target_coarse = _target_value(targets, "coarse_relief_257")
    target_detail = _target_value(targets, "detail_residual_257")
    components = {
        "point": point_loss(predictions.height_prediction_257, target_height),
        "coarse_point": point_loss(predictions.coarse_prediction_257, target_coarse),
        "detail_point": point_loss(predictions.detail_prediction_257, target_detail),
        "gradient": gradient_loss(predictions.height_prediction_257, target_height),
        "frequency": frequency_loss(predictions.height_prediction_257, target_height),
        "laplacian": laplacian_loss(predictions.height_prediction_257, target_height),
        "edge": edge_loss(predictions.height_prediction_257, target_height),
        "transition": transition_loss(predictions.height_prediction_257, target_height),
        "border": border_loss(predictions.height_prediction_257, target_height),
        "low_frequency": low_frequency_band_loss(predictions.height_prediction_257, target_height),
        "high_frequency": high_frequency_band_loss(predictions.height_prediction_257, target_height),
    }
    total = sum(
        (selected.weights[name] * components[name] for name in selected.weights),
        start=_zero_like(components["point"]),
    )
    components["total"] = total
    return total, components


def loss_metrics(components: Mapping[str, Tensor]) -> dict[str, float]:
    """Detach scalar components for JSON reports without changing the training graph."""

    return {name: float(value.detach().cpu().item()) for name, value in components.items()}


__all__ = [
    "CLEAN_SIGNAL_LOSS_PROFILES",
    "LOSS_CONFIGS",
    "LOSS_SCHEMA",
    "PARITY_CONFIG",
    "PARITY_PROFILE",
    "STRUCTURAL_CONFIG",
    "STRUCTURAL_PROFILE",
    "CleanSignalLossError",
    "V7GuidanceConfig",
    "border_loss",
    "clean_signal_loss",
    "edge_loss",
    "frequency_loss",
    "get_clean_signal_loss_config",
    "gradient_loss",
    "high_frequency_band_loss",
    "laplacian_loss",
    "loss_metrics",
    "low_frequency_band_loss",
    "point_loss",
    "transition_loss",
]
