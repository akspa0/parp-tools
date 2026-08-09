"""Create a small, validation-only authored/synthetic v50 pair report and atlas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.real_synthetic_pairs import (  # noqa: E402
    PAIR_SCHEMA,
    load_pair_rows,
    pair_domain_report,
    pair_validation_rows,
)


def _pair_atlas(group: zarr.Group, pairs: list, output: Path, shadow_npz_dir: Path | None) -> None:
    font = ImageFont.load_default()
    atlas_panels: list[Image.Image] = []
    for pair in pair_validation_rows(pairs):
        authored = np.asarray(group["minimap_rgb"][pair.authored_row_index], dtype=np.uint8)
        synthetic = np.asarray(group["minimap_rgb"][pair.synthetic_row_index], dtype=np.uint8)
        difference = np.abs(authored.astype(np.int16) - synthetic.astype(np.int16))
        difference = np.clip(difference * 3, 0, 255).astype(np.uint8)
        image_panels = [authored, synthetic, difference]
        if shadow_npz_dir is not None:
            shadow_path = shadow_npz_dir / f"{pair.map_name}_{pair.tile_x}_{pair.tile_y}_harvest.npz"
            with np.load(shadow_path, allow_pickle=False) as payload:
                shadow = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
            image_panels.append(np.repeat((np.clip(shadow, 0.0, 1.0) * 255).astype(np.uint8)[..., None], 3, axis=2))
        row_image = Image.fromarray(np.concatenate(tuple(image_panels), axis=1))
        draw = ImageDraw.Draw(row_image)
        draw.rectangle((0, 0, row_image.width, 18), fill=(0, 0, 0))
        draw.text(
            (4, 3),
            f"{pair.map_name} {pair.tile_x}_{pair.tile_y} | authored | flat synthetic | abs diff x3"
            + (" | fixed terrain shadow" if shadow_npz_dir is not None else ""),
            fill=(255, 255, 0),
            font=font,
        )
        atlas_panels.append(row_image)
    if not atlas_panels:
        raise ValueError("cannot render an empty pair atlas")
    columns = 2
    rows = (len(atlas_panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * atlas_panels[0].width, rows * 256), (0, 0, 0))
    for index, panel in enumerate(atlas_panels):
        canvas.paste(panel, ((index % columns) * panel.width, (index // columns) * 256))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", dest="split_policy", choices=("manifest", "map_holdout"), default="map_holdout")
    parser.add_argument("--val-map", default="Azeroth")
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument(
        "--shadow-npz-dir",
        type=Path,
        help="optional user-harvested NPZ directory containing post-fix terrain_shadow_256",
    )
    args = parser.parse_args(argv)
    if not args.store.is_dir():
        raise SystemExit(f"store does not exist: {args.store}")
    if args.validation_rows < 1:
        raise SystemExit("--validation-rows must be positive")

    pairs, selection = load_pair_rows(
        args.store,
        split_policy=args.split_policy,
        val_map=args.val_map,
        validation_limit=args.validation_rows,
    )
    group = zarr.open_group(str(args.store), mode="r")
    domain = pair_domain_report(group, pairs, args.shadow_npz_dir)
    validation_pairs = pair_validation_rows(pairs)
    report = {
        "schema": PAIR_SCHEMA,
        "store": str(args.store.resolve()),
        "input_contract": "authored_minimap_rgb_plus_legacy_flat_synthetic_rgb_absdiff_diagnostic",
        "labels": ["object_precise_mask", "object_mask"],
        "labels_used_as_inputs": False,
        "legacy_synthetic_is_flat_fake_maptexture": True,
        "legacy_synthetic_is_terrain_shadow_target": False,
        "fixed_shadow_npz_dir": str(args.shadow_npz_dir.resolve()) if args.shadow_npz_dir else None,
        "selection": {
            **selection,
            "split_policy": args.split_policy,
            "val_map": args.val_map,
            "train_pairs": sum(pair.split == "train" for pair in pairs),
            "validation_pairs": len(validation_pairs),
        },
        "domain_report": domain,
        "validation_pairs": [
            {
                "authored_row_index": pair.authored_row_index,
                "synthetic_row_index": pair.synthetic_row_index,
                "source_group_id": pair.source_group_id,
                "map": pair.map_name,
                "tile_x": pair.tile_x,
                "tile_y": pair.tile_y,
                "split": pair.split,
            }
            for pair in validation_pairs
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real-synthetic-pair-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _pair_atlas(group, validation_pairs, args.output / "real-synthetic-pair-atlas.png", args.shadow_npz_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {args.output / 'real-synthetic-pair-report.json'}")
    print(f"Wrote {args.output / 'real-synthetic-pair-atlas.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
