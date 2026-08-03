"""The DXT1 approximation must reproduce the authored tiles' measured degradation class.

The reference measurements (0.5.3.3368 / Azeroth, 16 tiles) are: Blp2/Dxtc/Dxt1, 256x256, one mip,
no palette, 1196-5269 unique colours per tile with a ~3200 median. Those numbers are what this
transform has to land near — bit-exactness with Blizzard's encoder is explicitly not the bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v50.dxt1_approx import (
    block_edge_ratio,
    dxt1_round_trip,
    unique_colour_count,
)


def _synthetic_minimap(seed: int = 125) -> np.ndarray:
    """A smooth full-colour render: many unique colours, no block structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256] / 255.0
    base = np.stack(
        [
            0.45 + 0.35 * np.sin(6.0 * xx + 0.7) * np.cos(4.0 * yy),
            0.50 + 0.30 * np.cos(5.0 * yy - 0.3) * np.sin(3.0 * xx),
            0.40 + 0.25 * np.sin(4.0 * xx + 2.0 * yy),
        ],
        axis=-1,
    )
    base += rng.normal(0.0, 0.01, base.shape)  # mild sensor-ish variation
    return np.clip(base * 255.0, 0, 255).astype(np.uint8)


def test_round_trip_collapses_each_block_to_at_most_four_colours():
    """The defining DXT1 constraint: two endpoints plus two interpolants per 4x4 block."""
    out = dxt1_round_trip(_synthetic_minimap())
    blocks = out.reshape(64, 4, 64, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3)
    for block in blocks[::97]:  # stride-sample; checking all 4096 is redundant
        assert len(np.unique(block, axis=0)) <= 4


def test_round_trip_collapses_the_palette_toward_the_authored_scale():
    """A pristine render carries tens of thousands of colours; DXT1 caps a 256x256 tile at
    4096 blocks x 4 = 16384 and in practice lands far below that.

    The authored reference band (1196-5269 per tile, ~3200 median) is the target for REAL synthetic
    minimaps; it is not asserted here because this fixture is a synthetic gradient with added noise,
    which is more colourful than any real tile. Checking that band belongs on real corpus output.
    """
    source = _synthetic_minimap()
    assert unique_colour_count(source) > 20000
    degraded = unique_colour_count(dxt1_round_trip(source))
    assert degraded <= 4 * (256 // 4) ** 2  # the codec's structural ceiling
    assert degraded < unique_colour_count(source) / 5


def test_round_trip_introduces_block_edge_discontinuity():
    """The right detector for a block codec: steps across 4-aligned boundaries grow relative to
    interior steps. A smooth render sits near 1."""
    source = _synthetic_minimap()
    assert block_edge_ratio(source) == pytest.approx(1.0, abs=0.15)
    assert block_edge_ratio(dxt1_round_trip(source)) > 1.3


def test_flat_blocks_stay_flat_within_rgb565_error():
    """A flat block yields c0 == c1 and decodes through the 3-colour branch, so it stays a single
    colour — but that colour is snapped to the RGB565 lattice, so it is NOT bit-identical unless the
    input already sits on the lattice. This matches Dxt1TileCodec's own flat-colour test, which
    asserts uniformity rather than equality.
    """
    flat = np.full((16, 16, 3), (37, 211, 96), dtype=np.uint8)
    out = dxt1_round_trip(flat)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1
    # Red/blue quantise in steps of 8, green in steps of 4.
    assert np.abs(out.astype(int) - flat.astype(int)).max() <= 8

    on_lattice = np.full((16, 16, 3), (33, 211, 99), dtype=np.uint8)
    assert np.array_equal(dxt1_round_trip(on_lattice), on_lattice)


def test_round_trip_is_deterministic_and_idempotent_in_shape_and_dtype():
    source = _synthetic_minimap()
    first = dxt1_round_trip(source)
    assert np.array_equal(first, dxt1_round_trip(source))
    assert first.shape == source.shape and first.dtype == np.uint8


def test_round_trip_is_near_idempotent():
    """Re-encoding already-DXT1 pixels should barely move them — the property RoundTripAgreement
    relies on to validate the encoder against authored bytes."""
    source = _synthetic_minimap()
    once = dxt1_round_trip(source)
    twice = dxt1_round_trip(once)
    first_pass = float(np.abs(once.astype(int) - source.astype(int)).mean())
    second_pass = float(np.abs(twice.astype(int) - once.astype(int)).mean())
    # Not exactly idempotent: refitting the bounding box on already-quantised colours can move an
    # endpoint. It must still be much smaller than the first pass's damage.
    assert second_pass < first_pass / 2.0


def test_green_survives_more_finely_than_red_and_blue():
    """RGB565 gives green 6 bits and red/blue 5, so a grey ramp quantises coarsest in R and B.
    This is pure codec, and it biases any channel-ratio comparison against authored tiles."""
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 256, axis=0)
    grey = np.stack([ramp, ramp, ramp], axis=-1)
    out = dxt1_round_trip(grey).astype(int)
    error = np.abs(out - grey.astype(int)).mean(axis=(0, 1))
    assert error[1] < error[0] and error[1] < error[2]


def test_matches_the_csharp_codec_bit_for_bit():
    """Locks this codec to ``WowViewer.Core.IO.Blp.Dxt1TileCodec``.

    The dataset degrades minimaps here in Python; the harvester's ``--dxt1-parity`` companion does it
    in C#. If the two diverge, training data and scoring data carry different codec noise. The
    mirrored assertion is ``Dxt1PythonParityTests`` in tests/WowViewer.Core.Tests — the fixture is
    pure integer arithmetic so both languages build identical input with no shared binary asset.
    If this fails, fix both codecs together; never update one hash alone.
    """
    import hashlib

    size = 64
    x = np.arange(size, dtype=np.int64)[None, :]
    y = np.arange(size, dtype=np.int64)[:, None]
    fixture = np.stack(
        [(x * 7 + y * 3) % 256, (x * x + y * 5) % 256, (x * 11 + y * y) % 256], axis=-1
    ).astype(np.uint8)
    fixture = np.broadcast_to(fixture, (size, size, 3)).copy()

    assert hashlib.sha256(fixture.tobytes()).hexdigest() == (
        "67306dac32c0c3d9db29f092aa35e15902885953f7f03311011a84299928aba4"
    )
    assert hashlib.sha256(dxt1_round_trip(fixture).tobytes()).hexdigest() == (
        "cdacc08445ba0e00ca5bb71bb04a5abea09fb9e71a0bf776cc51ff7fe6d3e451"
    )


def test_rejects_dimensions_that_are_not_block_aligned():
    with pytest.raises(ValueError, match="multiples of 4"):
        dxt1_round_trip(np.zeros((10, 8, 3), dtype=np.uint8))
