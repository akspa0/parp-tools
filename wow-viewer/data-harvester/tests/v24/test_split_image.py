"""Tests for v24_split_image.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v24_split_image.py"


def _write_composite(path: Path, width: int, height: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 3)) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))


@pytest.mark.v24
def test_split_writes_correct_number_of_tiles(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    composite = src_dir / "composite.png"
    _write_composite(composite, 4 * 256, 2 * 256)  # 4 cols x 2 rows = 8 tiles
    out = tmp_path / "out"  # sibling of src/, not inside it
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--image", str(composite),
            "--output-dir", str(out),
            "--grid-cols", "4",
            "--grid-rows", "2",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    tiles = sorted(out.glob("*.png"))
    assert len(tiles) == 8


@pytest.mark.v24
def test_split_uses_correct_naming_for_xy_and_yx(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    composite = src_dir / "composite.png"
    _write_composite(composite, 2 * 256, 2 * 256)
    for naming, expected in [("xy", "tile_1_0"), ("yx", "tile_0_1")]:
        out = tmp_path / naming  # sibling of src/, not inside it
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--image", str(composite),
                "--output-dir", str(out),
                "--grid-cols", "2",
                "--grid-rows", "2",
                "--naming", naming,
            ],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
        assert (out / f"{expected}.png").exists()


@pytest.mark.v24
def test_split_refuses_to_write_inside_image_directory(tmp_path: Path) -> None:
    composite_dir = tmp_path / "source"
    composite_dir.mkdir()
    composite = composite_dir / "composite.png"
    _write_composite(composite, 256, 256)
    out = composite_dir / "tiles"  # inside the source dir -> should refuse
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--image", str(composite),
            "--output-dir", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    combined = (proc.stderr or "") + (proc.stdout or "")
    assert "refusing to run" in combined


@pytest.mark.v24
def test_split_upsamples_undersized_composite_to_fit(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    composite = src_dir / "composite.png"
    _write_composite(composite, 256, 256)  # 1x1 only
    out = tmp_path / "out"  # sibling, not inside src/
    # Request a 2x2 grid of 256-pixel tiles = 512x512 total. The 256x256
    # input is upsampled with nearest-neighbor to 512x512.
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--image", str(composite),
            "--output-dir", str(out),
            "--grid-cols", "2",
            "--grid-rows", "2",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    # 4 tiles written, all 256x256.
    tiles = sorted(out.glob("*.png"))
    assert len(tiles) == 4
    for t in tiles:
        from PIL import Image
        with Image.open(t) as img:
            assert img.size == (256, 256)


@pytest.mark.v24
def test_split_skips_mostly_black_tiles(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    composite = src_dir / "composite.png"
    # Build a 2x2 = 512x512 composite. Tile (0,0) all black;
    # the other three all white. With --max-black-fraction 0.5 the
    # black tile is skipped (it is 100% black, > 0.5).
    arr = np.full((512, 512, 3), 255, dtype=np.uint8)
    arr[0:256, 0:256, :] = 0
    Image.fromarray(arr, mode="RGB").save(str(composite))
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--image", str(composite),
            "--output-dir", str(out),
            "--grid-cols", "2",
            "--grid-rows", "2",
            "--max-black-fraction", "0.5",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    tiles = sorted(out.glob("*.png"))
    # 3 tiles written (the all-black one was skipped).
    assert len(tiles) == 3
    combined = (proc.stderr or "") + (proc.stdout or "")
    assert "skipped 1" in combined


@pytest.mark.v24
def test_split_no_skip_by_default(tmp_path: Path) -> None:
    """The default --max-black-fraction 1.0 means no skip happens, even
    for an all-black tile (since the fraction must exceed 1.0)."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    composite = src_dir / "composite.png"
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(composite))
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--image", str(composite),
            "--output-dir", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    tiles = sorted(out.glob("*.png"))
    # 1 tile written, no skip (default 1.0 means never skip).
    assert len(tiles) == 1
    combined = (proc.stderr or "") + (proc.stdout or "")
    assert "skipped" not in combined
