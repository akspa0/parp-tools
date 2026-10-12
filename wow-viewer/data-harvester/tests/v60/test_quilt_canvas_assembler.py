"""Unit tests for QuiltCanvasAssembler and Seam Boundary Solver (Spec 268 AC-001)."""

import numpy as np
import pytest

from harvester.v60.quilt_canvas_assembler import QuiltBounds, QuiltCanvasAssembler


def test_quilt_coordinate_transforms():
    """Verify local <-> global coordinate mappings across a bounding box."""
    assembler = QuiltCanvasAssembler([(16, 32), (16, 33), (17, 32), (17, 33)])
    bounds = assembler.compute_bounds()

    assert bounds.min_tx == 16
    assert bounds.min_ty == 32
    assert bounds.max_tx == 17
    assert bounds.max_ty == 33
    assert bounds.width_tiles == 2
    assert bounds.height_tiles == 2
    assert bounds.pixel_width == 512
    assert bounds.pixel_height == 512

    # Local (0, 0) on tile (16, 32) -> global (0, 0)
    gu, gv = assembler.to_global_pixel_coords(16, 32, 0.0, 0.0, bounds)
    assert gu == 0.0
    assert gv == 0.0

    # Local (128, 64) on tile (17, 33) -> global (256 + 128, 256 + 64)
    gu, gv = assembler.to_global_pixel_coords(17, 33, 128.0, 64.0, bounds)
    assert gu == 384.0
    assert gv == 320.0

    # Inverse mapping
    tx, ty, lu, lv = assembler.to_local_tile_coords(384.0, 320.0, bounds)
    assert tx == 17
    assert ty == 33
    assert lu == 128.0
    assert lv == 64.0


def test_quilt_minimap_stitching():
    """Verify assembling minimap RGB crops into a contiguous canvas."""
    assembler = QuiltCanvasAssembler()
    tile_a = np.full((256, 256, 3), 100, dtype=np.uint8)
    tile_b = np.full((256, 256, 3), 200, dtype=np.uint8)

    assembler.add_tile(0, 0, minimap_rgb=tile_a)
    assembler.add_tile(1, 0, minimap_rgb=tile_b)

    stitched, bounds = assembler.stitch_minimap_quilt()
    assert stitched.shape == (256, 512, 3)
    assert np.all(stitched[:, :256] == 100)
    assert np.all(stitched[:, 256:] == 200)


def test_seam_boundary_solver_horizontal_ac001():
    """Verify horizontal seam solving eliminates height cliffs and aligns normals (AC-001)."""
    # Create two adjacent tiles with a deliberate 10-yard vertical step mismatch along the seam
    y_coords, x_coords = np.mgrid[0:257, 0:257].astype(np.float32)

    # Tile 0: slope rising towards 100 yards at east border
    tile_left = 50.0 + (x_coords / 256.0) * 50.0  # left east border = 100.0

    # Tile 1: slope starting at 110 yards at west border (10 yard cliff)
    tile_right = 110.0 + (x_coords / 256.0) * 40.0  # right west border = 110.0

    assembler = QuiltCanvasAssembler([(0, 0), (1, 0)])

    # Initial mismatch
    init_metrics = assembler.verify_seam_continuity({(0, 0): tile_left, (1, 0): tile_right})
    assert init_metrics["max_height_step_yards"] == pytest.approx(10.0, abs=0.1)

    # Solve seams
    stitched_elev = assembler.solve_seam_boundaries(
        {(0, 0): tile_left, (1, 0): tile_right},
        margin=6,
    )

    metrics = assembler.verify_seam_continuity(stitched_elev)

    # Acceptance Criteria AC-001: Max height step <= 0.05 yards, normal cosine >= 0.98
    assert metrics["max_height_step_yards"] <= 0.05, f"Height step too large: {metrics['max_height_step_yards']}"
    assert metrics["mean_normal_cosine_similarity"] >= 0.98, f"Normal alignment too low: {metrics['mean_normal_cosine_similarity']}"


def test_seam_boundary_solver_2x2_cluster_ac001():
    """Verify 2x2 grid of tiles with horizontal, vertical, and 4-way corner intersections."""
    assembler = QuiltCanvasAssembler()
    np.random.seed(42)

    # Generate 4 random smooth terrains
    tiles = {}
    for tx in [0, 1]:
        for ty in [0, 1]:
            base = 100.0 + 30.0 * np.sin(np.linspace(0, 3.14, 257)[:, None]) * np.cos(np.linspace(0, 3.14, 257)[None, :])
            # Add random offset per tile to create seams
            noise = np.random.uniform(-5.0, 5.0)
            tiles[(tx, ty)] = (base + noise).astype(np.float32)

    stitched = assembler.solve_seam_boundaries(tiles, margin=8)
    metrics = assembler.verify_seam_continuity(stitched)

    assert metrics["seams_evaluated"] == 4.0  # 2 horizontal + 2 vertical seams
    assert metrics["max_height_step_yards"] <= 0.05
    assert metrics["mean_normal_cosine_similarity"] >= 0.98

    # Check 4-way corner intersection
    c_00 = stitched[(0, 0)][256, 256]
    c_10 = stitched[(1, 0)][256, 0]
    c_01 = stitched[(0, 1)][0, 256]
    c_11 = stitched[(1, 1)][0, 0]

    assert abs(c_00 - c_10) <= 0.001
    assert abs(c_00 - c_01) <= 0.001
    assert abs(c_00 - c_11) <= 0.001

    # Verify global elevation canvas assembly
    global_canvas, bounds = assembler.assemble_global_elevation_canvas(stitched)
    assert global_canvas.shape == (513, 513)
    assert global_canvas[256, 256] == pytest.approx(c_00, abs=1e-4)
