"""Operator-facing visual review for the Spec 139 clean-signal corpus."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from harvester.v60.clean_signal_corpus import (
    COARSE_SIGNAL,
    CONFIDENCE_SIGNAL,
    DETAIL_SIGNAL,
    LUMA_SIGNAL,
    load_clean_signal_manifest,
    validate_clean_signal_corpus,
)

VISUAL_REVIEW_SCHEMA = "v7-clean-signal-visual-review-v1"
CELL_SIZE = 256
CELL_LABEL_HEIGHT = 24
PANEL_NAMES = ("luma", "confidence", "coarse", "detail", "height")


def _to_image(array: np.ndarray, *, detail: bool = False) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if detail:
        values = 0.5 + (values * 2.0)
    values = np.clip(values, 0.0, 1.0)
    pixels = np.rint(values * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def _read_panels(root: Path, row: dict[str, Any]) -> tuple[dict[str, Image.Image], dict[str, float]]:
    with np.load(root / str(row["npz"]), allow_pickle=False) as payload:
        arrays = {
            "luma": np.asarray(payload[LUMA_SIGNAL], dtype=np.float32),
            "confidence": np.asarray(payload[CONFIDENCE_SIGNAL], dtype=np.float32),
            "coarse": np.asarray(payload[COARSE_SIGNAL], dtype=np.float32),
            "detail": np.asarray(payload[DETAIL_SIGNAL], dtype=np.float32),
            "height": np.asarray(payload["relative_height_257"], dtype=np.float32),
        }
    images = {
        name: _to_image(array, detail=name == "detail")
        for name, array in arrays.items()
    }
    metrics = {
        "detail_abs_max": float(np.abs(arrays["detail"]).max()),
        "confidence_min": float(arrays["confidence"].min()),
        "confidence_max": float(arrays["confidence"].max()),
    }
    return images, metrics


def _draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (CELL_SIZE, CELL_SIZE + CELL_LABEL_HEIGHT), (24, 24, 24))
    canvas.paste(image.resize((CELL_SIZE, CELL_SIZE), Image.Resampling.NEAREST), (0, CELL_LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(245, 245, 245))
    return canvas


def _render_contact_sheet(
    root: Path,
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"cannot render empty visual sheet: {title}")
    cell_width = CELL_SIZE
    cell_height = CELL_SIZE + CELL_LABEL_HEIGHT
    sheet = Image.new("RGB", (cell_width * len(PANEL_NAMES), cell_height * len(rows)), (12, 12, 12))
    detail_ranges: list[float] = []
    for row_index, row in enumerate(rows):
        panels, metrics = _read_panels(root, row)
        detail_ranges.append(metrics["detail_abs_max"])
        for panel_index, panel_name in enumerate(PANEL_NAMES):
            cell = _draw_label(panels[panel_name], panel_name)
            sheet.paste(cell, (panel_index * cell_width, row_index * cell_height))
    sheet.save(path)
    return {
        "path": path.as_posix(),
        "title": title,
        "row_ids": [str(row["row_id"]) for row in rows],
        "panel_names": list(PANEL_NAMES),
        "detail_abs_max": detail_ranges,
    }


def _render_cross_tile_sheet(root: Path, groups: dict[str, list[dict[str, Any]]], path: Path) -> dict[str, Any]:
    complete: list[tuple[str, list[dict[str, Any]]]] = []
    expected_positions = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for pattern_id, rows in sorted(groups.items()):
        by_position = {
            (int(row.get("pattern_tile_x", -1)), int(row.get("pattern_tile_y", -1))): row
            for row in rows
        }
        if set(by_position) == expected_positions:
            complete.append((pattern_id, [by_position[position] for position in ((0, 0), (1, 0), (0, 1), (1, 1))]))
    if not complete:
        return {"available": False, "complete_pattern_count": 0}

    tile_size = CELL_SIZE
    label_height = CELL_LABEL_HEIGHT
    block_height = tile_size * 2 + label_height * 2
    sheet = Image.new("RGB", (tile_size * 2, block_height * len(complete)), (12, 12, 12))
    for block_index, (pattern_id, rows) in enumerate(complete):
        for row_index, row in enumerate(rows):
            with np.load(root / str(row["npz"]), allow_pickle=False) as payload:
                image = _to_image(np.asarray(payload[LUMA_SIGNAL], dtype=np.float32))
            cell = _draw_label(image, f"{pattern_id} ({row.get('pattern_tile_x')},{row.get('pattern_tile_y')})")
            x = (row_index % 2) * tile_size
            y = block_index * block_height + (row_index // 2) * (tile_size + label_height)
            sheet.paste(cell, (x, y))
    sheet.save(path)
    return {
        "available": True,
        "complete_pattern_count": len(complete),
        "path": path.as_posix(),
        "pattern_ids": [pattern_id for pattern_id, _ in complete],
    }


def render_clean_signal_review(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    rows_per_family: int = 4,
) -> dict[str, Any]:
    """Validate and render family, variant, and complete cross-tile review sheets."""

    if rows_per_family < 1:
        raise ValueError("rows_per_family must be positive")
    root = Path(corpus_root)
    output = Path(output_dir)
    manifest = load_clean_signal_manifest(root)
    validation = validate_clean_signal_corpus(root)
    if not validation["valid"]:
        failures = "; ".join(str(value) for value in validation["failures"][:8])
        raise ValueError(f"refusing visual review of invalid corpus: {failures}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite visual review output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = sorted(manifest["rows"], key=lambda row: str(row["row_id"]))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_tile_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
        pattern_id = row.get("pattern_id")
        if pattern_id:
            cross_tile_groups[str(pattern_id)].append(row)
    family_rows = [row for family in sorted(by_family) for row in by_family[family][:rows_per_family]]
    variant_rows = sorted(rows, key=lambda row: (int(row.get("variant", 0)), str(row["row_id"])))
    reports = {
        "family": _render_contact_sheet(
            root,
            family_rows,
            output / "clean-signal-family-atlas.png",
            title="family coverage",
        ),
        "variant": _render_contact_sheet(
            root,
            variant_rows[: max(rows_per_family * 4, 4)],
            output / "clean-signal-variant-atlas.png",
            title="variant coverage",
        ),
        "cross_tile": _render_cross_tile_sheet(
            root,
            cross_tile_groups,
            output / "clean-signal-cross-tile-atlas.png",
        ),
    }
    report = {
        "schema": VISUAL_REVIEW_SCHEMA,
        "corpus_root": str(root.resolve()),
        "row_count": len(rows),
        "family_count": len(by_family),
        "family_row_count": len(family_rows),
        "rows_per_family": rows_per_family,
        "outputs": reports,
        "validation": validation,
    }
    (output / "clean-signal-visual-review.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


__all__ = ["VISUAL_REVIEW_SCHEMA", "render_clean_signal_review"]
