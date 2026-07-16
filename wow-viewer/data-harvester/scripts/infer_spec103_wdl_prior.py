"""Spec 108 RGB-only inference from a paired-store row set or a standalone minimap PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.spec103.wdl_prior_io import write_prediction_archive
from harvester.spec103.wdl_prior_model import INPUT_CONTRACT, TARGET_CONTRACT, WdlPriorNet, decode_wdl_target, normalize_minimap_rgb
from harvester.v50_contract import WDL_ARCHIVE_SCHEMA, require_metadata_release, require_store_release, release_identity, validate_release


def load_model(checkpoint_path: Path, device: torch.device, release: str = "v50.1") -> WdlPriorNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require_metadata_release(checkpoint, release, artifact="WDL checkpoint")
    if checkpoint.get("input_contract") != INPUT_CONTRACT or checkpoint.get("target_contract") != TARGET_CONTRACT:
        raise ValueError("checkpoint is not a compatible Spec 108 RGB/WDL prior model")
    model = WdlPriorNet().to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def predict_rgb(model: WdlPriorNet, rgb: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """The deployment path: RGB pixels in, paired lattice out; no WDL/height input exists here."""
    with torch.no_grad():
        values = model(normalize_minimap_rgb(rgb).unsqueeze(0).to(device))[0].cpu().numpy()
    return decode_wdl_target(values)


def main() -> int:
    ap = argparse.ArgumentParser(description="v50 RGB-only spatial WDL prior inference")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--store", type=Path, help="paired store; RGB is the only model input")
    source.add_argument("--image", type=Path, help="standalone minimap PNG/JPEG; no store or WDL required")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default=None); ap.add_argument("--val-value", default=None)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--release", default="v50.1", type=validate_release,
                    help="must match the v50 checkpoint/store release (default: v50.1)")
    args = ap.parse_args()
    if (args.val_key is None) != (args.val_value is None):
        ap.error("--val-key and --val-value must be supplied together")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    try:
        model = load_model(args.checkpoint, device, args.release)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.image is not None:
        with Image.open(args.image) as image:
            rgb = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.uint8)
        outer, inner = predict_rgb(model, rgb, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, outer_17=outer, inner_16=inner, metadata_json=json.dumps({
            "schema": "v50-standalone-wdl-v1", **release_identity(args.release), "image": str(args.image.resolve()),
            "checkpoint": str(args.checkpoint.resolve()), "input_contract": INPUT_CONTRACT,
            "target_contract": TARGET_CONTRACT,
        }, sort_keys=True))
        print(f"[DONE] RGB image only -> paired WDL lattice {args.output}", flush=True)
        return 0

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "minimap_rgb" not in group:
        raise SystemExit("store lacks minimap_rgb")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    rows = [i for i, row in enumerate(index) if args.val_key is None or str(row.get(args.val_key)) == str(args.val_value)]
    if not rows:
        raise SystemExit("no store rows matched")
    outer, inner = [], []
    for row in rows:
        o, inn = predict_rgb(model, np.asarray(group["minimap_rgb"][row]), device)
        outer.append(o); inner.append(inn)
    write_prediction_archive(args.output, np.asarray(rows), np.stack(outer), np.stack(inner), {
        "schema": WDL_ARCHIVE_SCHEMA, **release_identity(args.release), "store": str(args.store.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "input_contract": INPUT_CONTRACT,
        "target_contract": TARGET_CONTRACT,
    })
    print(f"[DONE] {len(rows)} RGB-only WDL priors -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
