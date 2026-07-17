"""Canonical v50 owner for terrain-refiner batch inference: store rows -> predicted height + WDL lattice.

Per tile this writes, under <output>/<tile_name>/:
  predicted_height_257.npy   world-unit vertex grid (align_corners=True resample of the
                             predicted global channel)
  wdl_lattice.npz            outer (17x17, ::16) + inner (16x16, 8::16) -- the real paired
                             lattice (FR-005); never a 33x33 raster
  inference_summary.json     terrain-patch-adt-compatible, so predictions can be patched
                             straight back into ADTs for eyeball review

--drop-prior runs prior-absent inference (deployment-shaped: the model sees only image +
flat prior fill). GT arrays are read only to build input channels the checkpoint was trained
with; nothing here scores against GT -- that is validate_spec103_labelfree.py's job (and its
optional dev-only diagnostic).

Run from wow-viewer/data-harvester/ (GPU; USER runs):

    uv run python scripts/v50_infer_terrain.py \
        --store ../output/v50/v50.1/dataset/mixed_053_synthetic.zarr \
        --checkpoint ../output/v50/v50.1/terrain/checkpoint_best.pt \
        --val-key split --val-value val --output ../output/v50/v50.1/terrain/predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

from harvester.spec103.v7_inputs import (
    WORKING_SIZE,
    assemble_v7_input,
    brush_mask_from_alpha,
    prediction_to_height257,
    wdl_lattice_from_height257,
)
from harvester.spec103.v7_model import MultiChannelUNetV7
from harvester.spec103.v8_model import V50TerrainRefiner
from harvester.spec103.wdl_prior_io import read_prediction_archive
from harvester.v50.contracts import release_identity, require_metadata_release, require_store_release, validate_release

OPTIONAL_ARRAYS = ("normal_xyz", "liquid_mask", "liquid_height", "object_precise_mask", "alpha_256")


def main() -> int:
    ap = argparse.ArgumentParser(description="v50 terrain-refiner batch inference")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default=None, help="restrict to index rows where this column ...")
    ap.add_argument("--val-value", default=None, help="... equals this value (default: all rows)")
    ap.add_argument("--drop-prior", action="store_true", help="prior-absent (deployment-shaped) inference")
    ap.add_argument("--generated-wdl-priors", type=Path, default=None,
                    help="Spec 108 row-addressed generated-WDL archive. Its outer grid is used for ch6; "
                         "ground-truth WDL is not read for this path.")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--release", default="v50.1", type=validate_release,
                    help="must match the v50 store/checkpoint/archive release (default: v50.1)")
    args = ap.parse_args()
    if args.drop_prior and args.generated_wdl_priors is not None:
        raise SystemExit("--drop-prior and --generated-wdl-priors are mutually exclusive")

    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    try:
        require_metadata_release(ck, args.release, artifact="terrain checkpoint")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    use_detail = bool(ck.get("use_detail_head", False))
    height_hints = str(ck.get("height_hints", "gt"))
    arch = str(ck.get("arch", ""))
    if arch not in ("v50", "v7"):
        raise SystemExit(f"terrain checkpoint has unsupported arch {arch!r}; v50 will not load legacy v8 artifacts")
    model_cls = V50TerrainRefiner if arch == "v50" else MultiChannelUNetV7
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
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    rows = [
        (i, row) for i, row in enumerate(index)
        if args.val_key is None or str(row.get(args.val_key)) == str(args.val_value)
    ]
    if not rows:
        raise SystemExit(f"no rows matched {args.val_key}={args.val_value!r}")
    generated_outer: dict[int, np.ndarray] | None = None
    generated_metadata: dict | None = None
    if args.generated_wdl_priors is not None:
        generated_outer, generated_metadata = read_prediction_archive(args.generated_wdl_priors)
        try:
            require_metadata_release(generated_metadata, args.release, artifact="generated WDL archive")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        archive_store = Path(str(generated_metadata.get("store") or "")).resolve()
        if archive_store != args.store.resolve():
            raise SystemExit(f"generated WDL archive store does not match --store: {archive_store} != {args.store.resolve()}")
        missing = [i for i, _row in rows if i not in generated_outer]
        if missing:
            raise SystemExit(f"generated WDL archive lacks {len(missing)} requested store rows (first {missing[0]})")
    present = {name: name in group for name in OPTIONAL_ARRAYS}

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, row in rows:
        tile_name = f"{row['map']}_{row['tile_x']}_{row['tile_y']}"
        height = np.asarray(group["height_257"][i], dtype=np.float32)
        supplied_outer = generated_outer[i] if generated_outer is not None else None
        x = assemble_v7_input(
            minimap_rgb=np.asarray(group["minimap_rgb"][i]),
            height_257=None if args.drop_prior or supplied_outer is not None else height,
            normal_xyz=np.asarray(group["normal_xyz"][i]) if present["normal_xyz"] else None,
            liquid_mask=np.asarray(group["liquid_mask"][i]) if present["liquid_mask"] else None,
            liquid_height=np.asarray(group["liquid_height"][i]) if present["liquid_height"] else None,
            object_mask=np.asarray(group["object_precise_mask"][i]) if present["object_precise_mask"] else None,
            brush_mask=brush_mask_from_alpha(np.asarray(group["alpha_256"][i]) if present["alpha_256"] else None),
            wdl_outer_17=supplied_outer,
            height_hints="none" if args.drop_prior else ("wdl" if supplied_outer is not None else height_hints),
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
            **release_identity(args.release),
            "tile_name": tile_name,
            "predicted_height_257_path": "predicted_height_257.npy",
            "checkpoint": str(args.checkpoint.resolve()),
            "drop_prior": args.drop_prior,
            "generated_wdl_priors": str(args.generated_wdl_priors.resolve()) if args.generated_wdl_priors else None,
        }, indent=2), encoding="utf-8")
        manifest.append({
            "tile_name": tile_name, "map": str(row["map"]),
            "tile_x": int(row["tile_x"]), "tile_y": int(row["tile_y"]),
            "prediction_dir": tile_name,
        })
        print(f"[infer] {tile_name}: height [{predicted.min():.1f}, {predicted.max():.1f}]", flush=True)

    (args.output / "predictions_manifest.json").write_text(json.dumps({
        "schema": "v50-terrain-predictions-v1", **release_identity(args.release),
        "store": str(args.store.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "drop_prior": args.drop_prior,
        "generated_wdl_priors": str(args.generated_wdl_priors.resolve()) if args.generated_wdl_priors else None,
        "generated_wdl_metadata": generated_metadata,
        "tiles": manifest,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] {len(manifest)} tiles -> {args.output}", flush=True)
    return 0
