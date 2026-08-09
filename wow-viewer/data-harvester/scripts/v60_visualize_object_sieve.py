"""Render visual proof for the v60 synthetic object-sieve corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.object_sieve import (  # noqa: E402
    CLEAN_SIGNAL,
    INPUT_SIGNAL,
    MASK_SIGNAL,
    PLACEMENT_REGIMES,
    load_object_sieve_manifest,
    validate_object_sieve_corpus,
)

PANEL = 160
HEADER = 52
ROW_LABEL = 28
GAP = 4


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _shadow_rgb(values: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _mask_overlay(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = _shadow_rgb(clean).astype(np.float32)
    active = np.asarray(mask, dtype=np.float32) >= 0.5
    base[active, 0] = 255.0
    base[active, 1] *= 0.25
    base[active, 2] *= 0.25
    return np.clip(base, 0, 255).astype(np.uint8)


def _resize(values: np.ndarray) -> Image.Image:
    return Image.fromarray(values, mode="RGB").resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _load_item(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    with np.load(root / str(row["npz"]), allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name], dtype=np.float32) for name in (INPUT_SIGNAL, CLEAN_SIGNAL, MASK_SIGNAL)}
    if any(value.shape != (256, 256) for value in values.values()):
        raise ValueError(f"{row['row_id']}: object-sieve signal shape mismatch")
    return {"row": row, **values}


def _paste_label(sheet: Image.Image, image: Image.Image, x: int, y: int, label: str) -> None:
    sheet.paste(image, (x, y))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((x, y + PANEL, x + PANEL, y + PANEL + ROW_LABEL), fill=(28, 31, 38))
    draw.text((x + 4, y + PANEL + 8), label, fill=(235, 235, 235), font=_font())


def render_object_sieve_review(corpus: Path, output_dir: Path) -> dict[str, Any]:
    validation = validate_object_sieve_corpus(corpus)
    if not validation["valid"]:
        raise ValueError("object-sieve corpus is invalid: " + "; ".join(validation["failures"][:5]))
    manifest = load_object_sieve_manifest(corpus)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in manifest["rows"]:
        grouped[str(row["terrain_control_family"])][str(row["placement_regime"])].append(row)
    families = sorted(grouped)
    output_dir.mkdir(parents=True, exist_ok=True)

    width = len(PLACEMENT_REGIMES) * (PANEL + GAP)
    height = HEADER + (len(families) * (PANEL + ROW_LABEL))
    input_sheet = Image.new("RGB", (width, height), (18, 21, 27))
    mask_sheet = Image.new("RGB", (width, height), (18, 21, 27))
    input_draw = ImageDraw.Draw(input_sheet)
    mask_draw = ImageDraw.Draw(mask_sheet)
    input_draw.text((8, 8), "v60 object sieve: contaminated input", fill=(245, 245, 245), font=_font())
    input_draw.text((8, 27), "Columns: none / sparse / dense / overlap / boundary-crossing", fill=(180, 186, 194), font=_font())
    mask_draw.text((8, 8), "v60 object sieve: contamination overlay", fill=(245, 245, 245), font=_font())
    mask_draw.text((8, 27), "Red pixels are the exact synthetic removal/inpainting target", fill=(180, 186, 194), font=_font())

    selected_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        y = HEADER + (family_index * (PANEL + ROW_LABEL))
        input_draw.text((4, y + 6), family, fill=(235, 235, 235), font=_font())
        mask_draw.text((4, y + 6), family, fill=(235, 235, 235), font=_font())
        for regime_index, regime in enumerate(PLACEMENT_REGIMES):
            candidates = grouped[family].get(regime, [])
            if not candidates:
                continue
            # Rotate the selected base variant by row so the atlas visibly covers all four object families.
            selected = sorted(candidates, key=lambda row: int(row.get("terrain_control_row_id", "-v00").split("-v")[-1]))[family_index % len(candidates)]
            item = _load_item(corpus, selected)
            x = regime_index * (PANEL + GAP)
            _paste_label(input_sheet, _resize(_shadow_rgb(item[INPUT_SIGNAL])), x, y, regime)
            _paste_label(mask_sheet, _resize(_mask_overlay(item[CLEAN_SIGNAL], item[MASK_SIGNAL])), x, y, regime)
            selected_rows.append(selected)
        input_draw.rectangle((0, y, width, y + PANEL + ROW_LABEL), outline=(92, 122, 160), width=1)
        mask_draw.rectangle((0, y, width, y + PANEL + ROW_LABEL), outline=(160, 92, 92), width=1)

    input_path = output_dir / "object-sieve-input-atlas.png"
    mask_path = output_dir / "object-sieve-mask-atlas.png"
    input_sheet.save(input_path)
    mask_sheet.save(mask_path)
    report = {
        "schema": "v60-object-sieve-visual-review-v1",
        "corpus": str(corpus.resolve()),
        "outputs": [str(input_path.resolve()), str(mask_path.resolve())],
        "row_count": len(manifest["rows"]),
        "family_count": len(families),
        "placement_regimes": list(PLACEMENT_REGIMES),
        "validation": validation,
        "selected_review_rows": [str(row["row_id"]) for row in selected_rows],
    }
    report_path = output_dir / "object-sieve-visual-review.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {input_path}")
    print(f"Wrote {mask_path}")
    print(f"Wrote {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_object_sieve_review(args.corpus, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
