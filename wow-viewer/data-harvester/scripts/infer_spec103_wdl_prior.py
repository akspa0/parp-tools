"""RGB-only Spec 108 inference; writes a row-addressed generated-WDL archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.spec103.wdl_prior_io import write_prediction_archive
from harvester.spec103.wdl_prior_model import INPUT_CONTRACT, TARGET_CONTRACT, WdlPriorNet, decode_wdl_target, normalize_minimap_rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 108 RGB-only WDL prior inference")
    ap.add_argument("--store", required=True, type=Path); ap.add_argument("--checkpoint", required=True, type=Path); ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default=None); ap.add_argument("--val-value", default=None); ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args(); device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("input_contract") != INPUT_CONTRACT or checkpoint.get("target_contract") != TARGET_CONTRACT:
        raise SystemExit("checkpoint is not a compatible Spec 108 RGB/WDL prior model")
    model = WdlPriorNet().to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    group = zarr.open_group(str(args.store), mode="r")
    if "minimap_rgb" not in group:
        raise SystemExit("store lacks minimap_rgb")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    rows = [i for i, row in enumerate(index) if args.val_key is None or str(row.get(args.val_key)) == str(args.val_value)]
    if not rows: raise SystemExit("no store rows matched")
    outer, inner = [], []
    with torch.no_grad():
        for row in rows:
            values = model(normalize_minimap_rgb(group["minimap_rgb"][row]).unsqueeze(0).to(device))[0].cpu().numpy()
            o, inn = decode_wdl_target(values); outer.append(o); inner.append(inn)
    write_prediction_archive(args.output, np.asarray(rows), np.stack(outer), np.stack(inner), {"schema": "spec108-generated-wdl-v1", "store": str(args.store.resolve()), "checkpoint": str(args.checkpoint.resolve()), "input_contract": INPUT_CONTRACT, "target_contract": TARGET_CONTRACT})
    print(f"[DONE] {len(rows)} RGB-only WDL priors -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
