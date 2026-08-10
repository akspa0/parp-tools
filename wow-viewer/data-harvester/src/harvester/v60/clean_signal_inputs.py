"""Deployment-safe four-channel observation contract for the v7-inspired v60 lane.

This module only derives values from an image observation and albedo-operation metadata.  It
never reads terrain targets.  Target-side arrays can be named in ``forbidden_signals`` so callers
can prove that a row was rejected before model input assembly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

INPUT_SCHEMA = "v7-clean-signal-input-v1"
IMAGE_SHAPE = (256, 256)
GRADIENT_SHAPE = (2, 256, 256)
INPUT_CHANNELS = 4
GRADIENT_VERSION = "finite-difference-edge-v1"
CONFIDENCE_STATUSES = frozenset({"measured", "absent_explicit", "rejected", "quarantined"})
OBSERVATION_STATUSES = frozenset({"accepted", "rejected", "quarantined"})
FORBIDDEN_INFERENCE_SIGNALS = frozenset(
    {
        "wdl",
        "height",
        "height_257",
        "normal",
        "mcnr_normal_xyz",
        "liquid",
        "liquid_mask",
        "object",
        "object_mask",
        "alpha",
        "material",
        "target",
    }
)


class CleanObservationError(ValueError):
    """Raised when an observation cannot enter the inference contract."""


def _as_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def finite_difference_gradient(luma: Any) -> np.ndarray:
    """Return deterministic ``[x, y]`` gradients using one-sided edge differences."""

    image = _as_float32(luma)
    if image.shape != IMAGE_SHAPE:
        raise CleanObservationError(f"luma shape {image.shape} != {IMAGE_SHAPE}")
    if not np.isfinite(image).all():
        raise CleanObservationError("luma contains non-finite values")

    gradient_x = np.empty_like(image)
    gradient_y = np.empty_like(image)
    gradient_x[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
    gradient_x[:, 0] = image[:, 1] - image[:, 0]
    gradient_x[:, -1] = image[:, -1] - image[:, -2]
    gradient_y[1:-1, :] = 0.5 * (image[2:, :] - image[:-2, :])
    gradient_y[0, :] = image[1, :] - image[0, :]
    gradient_y[-1, :] = image[-1, :] - image[-2, :]
    return np.ascontiguousarray(np.stack((gradient_x, gradient_y), axis=0), dtype=np.float32)


def _forbidden_names(forbidden_signals: Iterable[str] | Mapping[str, Any] | None) -> list[str]:
    if forbidden_signals is None:
        return []
    values = forbidden_signals.keys() if isinstance(forbidden_signals, Mapping) else forbidden_signals
    return sorted({str(value) for value in values if str(value)})


def validate_clean_observation(
    luma: Any,
    gradient: Any,
    confidence: Any,
    confidence_status: str,
    *,
    observation_status: str = "accepted",
    provenance: Mapping[str, Any] | None = None,
    forbidden_signals: Iterable[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an observation and return a JSON-safe report.

    ``absent_explicit`` confidence is accepted only when the materialized confidence channel is
    all zero.  Rejected, quarantined, stale, or target-contaminated rows fail closed.
    """

    failures: list[str] = []
    luma_array = _as_float32(luma)
    gradient_array = _as_float32(gradient)
    confidence_array = _as_float32(confidence)
    status = str(confidence_status)
    observation_state = str(observation_status)
    forbidden = _forbidden_names(forbidden_signals)

    if luma_array.shape != IMAGE_SHAPE:
        failures.append(f"luma shape {luma_array.shape} != {IMAGE_SHAPE}")
    elif not np.isfinite(luma_array).all():
        failures.append("luma contains non-finite values")
    elif luma_array.size and (float(luma_array.min()) < 0.0 or float(luma_array.max()) > 1.0):
        failures.append("luma is outside [0, 1]")

    if gradient_array.shape != GRADIENT_SHAPE:
        failures.append(f"gradient shape {gradient_array.shape} != {GRADIENT_SHAPE}")
    elif not np.isfinite(gradient_array).all():
        failures.append("gradient contains non-finite values")

    if confidence_array.shape != IMAGE_SHAPE:
        failures.append(f"confidence shape {confidence_array.shape} != {IMAGE_SHAPE}")
    elif not np.isfinite(confidence_array).all():
        failures.append("confidence contains non-finite values")
    elif confidence_array.size and (
        float(confidence_array.min()) < 0.0 or float(confidence_array.max()) > 1.0
    ):
        failures.append("confidence is outside [0, 1]")

    if status not in CONFIDENCE_STATUSES:
        failures.append(f"invalid confidence_status {status!r}")
    elif status in {"rejected", "quarantined"}:
        failures.append(f"confidence_status {status!r} is not admissible for inference")
    elif status == "absent_explicit" and confidence_array.shape == IMAGE_SHAPE:
        if not np.allclose(confidence_array, 0.0, atol=0.0):
            failures.append("absent_explicit confidence must be zero-filled")

    if observation_state not in OBSERVATION_STATUSES:
        failures.append(f"invalid observation_status {observation_state!r}")
    elif observation_state != "accepted":
        failures.append(f"observation_status {observation_state!r} is not admissible")

    provenance_dict = dict(provenance or {})
    for key in ("artifact_status", "gate_status"):
        value = provenance_dict.get(key)
        if value in {"stale", "rejected", "quarantined"}:
            failures.append(f"provenance {key}={value!r} is not admissible")

    if forbidden:
        failures.append(f"forbidden inference signals present: {forbidden}")

    return {
        "schema": "v7-clean-signal-input-validation-v1",
        "input_schema": INPUT_SCHEMA,
        "image_shape": list(IMAGE_SHAPE),
        "gradient_shape": list(GRADIENT_SHAPE),
        "channel_count": INPUT_CHANNELS,
        "gradient_version": GRADIENT_VERSION,
        "confidence_status": status,
        "observation_status": observation_state,
        "forbidden_signals": forbidden,
        "failures": failures,
        "valid": not failures,
    }


def _raise_if_invalid(report: dict[str, Any]) -> None:
    if not report["valid"]:
        raise CleanObservationError("; ".join(str(item) for item in report["failures"]))


@dataclass(frozen=True)
class CleanObservationPackage:
    """Validated observation package in named-array and concatenated-channel form."""

    luma: np.ndarray
    gradient: np.ndarray
    confidence: np.ndarray
    confidence_status: str
    observation_status: str = "accepted"
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        luma = _as_float32(self.luma)
        gradient = _as_float32(self.gradient)
        confidence = _as_float32(self.confidence)
        report = validate_clean_observation(
            luma,
            gradient,
            confidence,
            self.confidence_status,
            observation_status=self.observation_status,
            provenance=self.provenance,
        )
        _raise_if_invalid(report)
        object.__setattr__(self, "luma", np.ascontiguousarray(luma))
        object.__setattr__(self, "gradient", np.ascontiguousarray(gradient))
        object.__setattr__(self, "confidence", np.ascontiguousarray(confidence))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def channels(self) -> np.ndarray:
        """Return the exact model input order ``luma, gradient-x, gradient-y, confidence``."""

        return np.ascontiguousarray(
            np.concatenate((self.luma[None], self.gradient, self.confidence[None]), axis=0),
            dtype=np.float32,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "clean_observation_luma_256": self.luma,
            "clean_observation_gradient_256": self.gradient,
            "clean_observation_confidence_256": self.confidence,
        }


def build_clean_observation(
    luma: Any,
    confidence: Any | None,
    confidence_status: str = "measured",
    *,
    observation_status: str = "accepted",
    provenance: Mapping[str, Any] | None = None,
    forbidden_signals: Iterable[str] | Mapping[str, Any] | None = None,
) -> CleanObservationPackage:
    """Derive gradients and build one validated four-channel package."""

    image = _as_float32(luma)
    if confidence is None:
        if confidence_status != "absent_explicit":
            raise CleanObservationError("confidence is required unless confidence is absent_explicit")
        confidence_array = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    else:
        confidence_array = _as_float32(confidence)
    gradient = finite_difference_gradient(image)
    report = validate_clean_observation(
        image,
        gradient,
        confidence_array,
        confidence_status,
        observation_status=observation_status,
        provenance=provenance,
        forbidden_signals=forbidden_signals,
    )
    _raise_if_invalid(report)
    return CleanObservationPackage(
        image,
        gradient,
        confidence_array,
        confidence_status,
        observation_status,
        dict(provenance or {}),
    )


__all__ = [
    "CleanObservationError",
    "CleanObservationPackage",
    "CONFIDENCE_STATUSES",
    "FORBIDDEN_INFERENCE_SIGNALS",
    "GRADIENT_VERSION",
    "INPUT_CHANNELS",
    "INPUT_SCHEMA",
    "build_clean_observation",
    "finite_difference_gradient",
    "validate_clean_observation",
]
