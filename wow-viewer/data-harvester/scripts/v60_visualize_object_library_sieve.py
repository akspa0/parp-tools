#!/usr/bin/env python3
"""Render a compact visual review of real-library object overlays and exact masks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_library_sieve import (  # noqa: E402
    INPUT_SIGNAL,
    INSTANCE_SIGNAL,
    MASK_SIGNAL,
    PLACEMENT_REGIMES,
    load_object_library_sieve_manifest,
    validate_object_library_sieve_corpus,
)


def _instance_preview(ids: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0, 0, 0],
            [238, 85, 85],
            [85, 190, 95],
            [70, 120, 235],
            [235, 185, 55],
            [180, 85, 220],
        ],
        dtype=np.uint8,
    )
    return palette[np.asarray(ids, dtype=np.uint16) % len(palette)]


def render_review(corpus: Path, output_dir: Path, *, rows_per_regime: int = 3) -> dict:
    from PIL import Image, ImageDraw

    manifest = load_object_library_sieve_manifest(corpus)
    validation = validate_object_library_sieve_corpus(corpus)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for row in manifest["rows"]:
        by_regime[str(row["placement_regime"])].append(row)
    for regime in PLACEMENT_REGIMES:
        selected.extend(by_regime.get(regime, [])[:rows_per_regime])
    tile = 256
    label_height = 34
    sheet = Image.new("RGB", (tile * 4, (tile + label_height) * max(1, len(selected))), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with np.load(corpus / row["npz"], allow_pickle=False) as payload:
            objectified = np.asarray(payload[INPUT_SIGNAL], dtype=np.float32)
            mask = np.asarray(payload[MASK_SIGNAL], dtype=np.float32)
            instance_ids = np.asarray(payload[INSTANCE_SIGNAL], dtype=np.uint16)
        panels = [
            np.repeat((np.clip(objectified, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=2),
            np.repeat((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=2),
            _instance_preview(instance_ids),
        ]
        with np.load(corpus / row["npz"], allow_pickle=False) as payload:
            clean = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
        panels.insert(1, np.repeat((np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=2))
        y = index * (tile + label_height)
        draw.text((4, y + 2), f"{row['placement_regime']} | instances={row['object_instance_count']} | coverage={row['object_coverage']:.3f}", fill="black")
        for panel_index, panel in enumerate(panels):
            image = Image.fromarray(panel, mode="RGB")
            sheet.paste(image, (panel_index * tile, y + label_height))
    sheet_path = output_dir / "object-library-sieve-atlas.png"
    sheet.save(sheet_path)
    report = {
        "schema": "v60-object-library-sieve-visual-review-v1",
        "corpus": str(corpus.resolve()),
        "validation": validation,
        "rows_rendered": len(selected),
        "panels": ["objectified_input", "clean_target", "union_mask", "instance_id_map"],
        "atlas": str(sheet_path),
    }
    (output_dir / "object-library-sieve-visual-review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Render v60 real-library object-sieve visual review")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-regime", type=int, default=3)
    args = parser.parse_args()
    report = render_review(args.corpus, args.output_dir, rows_per_regime=args.rows_per_regime)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
