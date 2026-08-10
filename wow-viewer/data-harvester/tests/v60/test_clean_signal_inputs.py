from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.clean_signal_inputs import (
    CleanObservationError,
    build_clean_observation,
    finite_difference_gradient,
    validate_clean_observation,
)


def _luma() -> np.ndarray:
    y, x = np.mgrid[0:256, 0:256]
    return ((x + y) / 510.0).astype(np.float32)


def test_build_clean_observation_uses_exact_four_channel_order() -> None:
    package = build_clean_observation(
        _luma(),
        np.ones((256, 256), dtype=np.float32),
        provenance={"operation": "synthetic-albedo-v1", "gate_status": "accepted"},
    )

    assert package.gradient.shape == (2, 256, 256)
    assert package.channels.shape == (4, 256, 256)
    np.testing.assert_array_equal(package.channels[0], package.luma)
    np.testing.assert_array_equal(package.channels[1:3], package.gradient)
    np.testing.assert_array_equal(package.channels[3], package.confidence)


def test_absent_confidence_requires_explicit_zero_fill() -> None:
    package = build_clean_observation(_luma(), None, "absent_explicit")
    assert package.confidence_status == "absent_explicit"
    assert float(package.confidence.sum()) == 0.0

    report = validate_clean_observation(
        _luma(),
        finite_difference_gradient(_luma()),
        np.ones((256, 256), dtype=np.float32),
        "absent_explicit",
    )
    assert report["valid"] is False
    assert any("zero-filled" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"confidence": None, "confidence_status": "measured"}, "confidence is required"),
        ({"confidence": np.ones((255, 256), dtype=np.float32)}, "confidence shape"),
        ({"confidence": np.ones((256, 256), dtype=np.float32), "observation_status": "rejected"}, "not admissible"),
        ({"confidence": np.ones((256, 256), dtype=np.float32), "provenance": {"artifact_status": "stale"}}, "stale"),
        ({"confidence": np.ones((256, 256), dtype=np.float32), "forbidden_signals": ["height_257"]}, "forbidden"),
    ],
)
def test_observation_gate_rejects_invalid_rows(kwargs: dict, expected: str) -> None:
    with pytest.raises(CleanObservationError, match=expected):
        build_clean_observation(_luma(), **kwargs)


def test_gradient_is_deterministic_and_finite() -> None:
    first = finite_difference_gradient(_luma())
    second = finite_difference_gradient(_luma())
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
