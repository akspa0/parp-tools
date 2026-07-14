"""Spec 103 — batch v7 inference: store rows -> predicted height + paired WDL lattice.

Per tile this writes, under <output>/<tile_name>/:
  predicted_height_257.npy   world-unit vertex grid (align_corners=True resample of the
                             predicted global channel)
  wdl_lattice.npz            outer (17×17, ::16) + inner (16×16, 8::16) — the real paired
                             lattice (FR-005); never a 33×33 raster
  inference_summary.json     terrain-patch-adt-compatible, so predictions can be patched
                             straight back into ADTs for eyeball review

`--drop-prior` runs prior-absent inference (deployment-shaped: the model sees only image +
flat prior fill). GT arrays are read only to build input channels the checkpoint was trained
with; nothing here scores against GT — that is validate_spec103_labelfree.py's job (and its
optional dev-only diagnostic).

Run from wow-viewer/data-harvester/ (GPU; USER runs):

    uv run python scripts/infer_spec103_v7.py \
        --store ../output/datasets/spec103/synthetic_v1.zarr \
        --checkpoint ../output/spec103_v7_synth_v1/checkpoint_best.pt \
        --val-key pattern --val-value crater --output ../output/spec103_v7_synth_v1/predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.spec103.v7_inputs import (  # noqa: E402
    WORKING_SIZE,
    assemble_v7_input,
    prediction_to_height257,
    wdl_lattice_from_height257,
)
from harvester.spec103.v7_model import MultiChannelUNetV7  # noqa: E402
from harvester.spec103.v8_model import V8LeanUNet  # noqa: E402

OPTIONAL_ARRAYS = ("normal_xyz", "liquid_mask", "liquid_height", "object_precise_mask")


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 v7 batch inference")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default=None, help="restrict to index rows where this column ...")
    ap.add_argument("--val-value", default=None, help="... equals this value (default: all rows)")
    ap.add_argument("--drop-prior", action="store_true", help="prior-absent (deployment-shaped) inference")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    use_detail = bool(ck.get("use_detail_head", False))
    height_hints = str(ck.get("height_hints", "gt"))
    arch = str(ck.get("arch", "v7"))  # pre-v8 checkpoints carry no arch key
    model_cls = V8LeanUNet if arch == "v8" else MultiChannelUNetV7
    model = model_cls(
        out_channels=3 if use_detail else 2,
        use_wdl_global_trestle=bool(ck.get("use_wdl_global_trestle", True)),
        use_detail_head=use_detail,
        output_size=int(ck.get("output_size", WORKING_SIZE)),
        output_head_mode=str(ck.get("output_head_mode", "legacy_clamped")),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    group = zarr.open_group(str(args.store), mode="r")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    rows = [
        (i, row) for i, row in enumerate(index)
        if args.val_key is None or str(row.get(args.val_key)) == str(args.val_value)
    ]
    if not rows:
        raise SystemExit(f"no rows matched {args.val_key}={args.val_value!r}")
    present = {name: name in group for name in OPTIONAL_ARRAYS}

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, row in rows:
        tile_name = f"{row['map']}_{row['tile_x']}_{row['tile_y']}"
        height = np.asarray(group["height_257"][i], dtype=np.float32)
        x = assemble_v7_input(
            minimap_rgb=np.asarray(group["minimap_rgb"][i]),
            height_257=None if args.drop_prior else height,
            normal_xyz=np.asarray(group["normal_xyz"][i]) if present["normal_xyz"] else None,
            liquid_mask=np.asarray(group["liquid_mask"][i]) if present["liquid_mask"] else None,
            liquid_height=np.asarray(group["liquid_height"][i]) if present["liquid_height"] else None,
            object_mask=np.asarray(group["object_precise_mask"][i]) if present["object_precise_mask"] else None,
            height_hints="none" if args.drop_prior else height_hints,
            drop_wdl_prior=args.drop_prior,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs, _bounds = model(x)
        predicted = prediction_to_height257(outputs[0, 0].float().cpu().numpy())
        outer, inner = wdl_lattice_from_height257(predicted)

        tile_dir = args.output / tile_name
        tile_dir.mkdir(parents=True, exist_ok=True)
        np.save(tile_dir / "predicted_height_257.npy", predicted)
        np.savez(tile_dir / "wdl_lattice.npz", outer_17=outer, inner_16=inner)
        (tile_dir / "inference_summary.json").write_text(json.dumps({
            "tile_name": tile_name,
            "predicted_height_257_path": "predicted_height_257.npy",
            "checkpoint": str(args.checkpoint.resolve()),
            "drop_prior": args.drop_prior,
        }, indent=2), encoding="utf-8")
        manifest.append({
            "tile_name": tile_name, "map": str(row["map"]),
            "tile_x": int(row["tile_x"]), "tile_y": int(row["tile_y"]),
            "prediction_dir": tile_name,
        })
        print(f"[infer] {tile_name}: height [{predicted.min():.1f}, {predicted.max():.1f}]", flush=True)

    (args.output / "predictions_manifest.json").write_text(json.dumps({
        "schema": "spec103-predictions-v1",
        "store": str(args.store.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "drop_prior": args.drop_prior,
        "tiles": manifest,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] {len(manifest)} tiles -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
