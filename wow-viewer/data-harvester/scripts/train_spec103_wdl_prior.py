"""User-run CUDA trainer for Spec 108's independent RGB-to-WDL prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.spec103.prefab_curation import validate_source_group_split
from harvester.spec103.wdl_prior_model import INPUT_CONTRACT, MODEL_VARIANT_WDL_PRIOR, TARGET_CONTRACT, WdlPriorNet, build_wdl_target, normalize_minimap_rgb


class PriorDataset(Dataset):
    def __init__(self, group, rows: list[int]) -> None:
        self.group, self.rows = group, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return normalize_minimap_rgb(self.group["minimap_rgb"][row]), torch.from_numpy(build_wdl_target(self.group["height_257"][row]))


def evaluate(model, loader, device) -> float:
    model.eval(); losses = []
    with torch.no_grad():
        for x, y in loader:
            losses.append(float(torch.nn.functional.l1_loss(model(x.to(device)), y.to(device)).item()))
    return float(np.mean(losses)) if losses else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 108 RGB-only WDL prior trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="compact representative paired store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", required=True, help="complete source-group column, normally pattern")
    ap.add_argument("--val-value", required=True)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    group = zarr.open_group(str(args.store), mode="r")
    if "minimap_rgb" not in group or "height_257" not in group:
        raise SystemExit("store must contain minimap_rgb and height_257")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    if not index or args.val_key not in index[0]:
        raise SystemExit(f"index lacks requested holdout key {args.val_key!r}")
    train_rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) != str(args.val_value)]
    val_rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) == str(args.val_value)]
    validate_source_group_split(index, train_rows, val_rows)
    if not train_rows or not val_rows:
        raise SystemExit(f"invalid complete-group split: train={len(train_rows)} val={len(val_rows)}")
    device = torch.device("cuda"); model = WdlPriorNet().to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(PriorDataset(group, train_rows), batch_size=min(args.batch, len(train_rows)), shuffle=True, num_workers=args.workers, pin_memory=True)
    val = DataLoader(PriorDataset(group, val_rows), batch_size=min(args.batch, len(val_rows)), num_workers=args.workers, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True); best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            opt.zero_grad(set_to_none=True); loss = torch.nn.functional.smooth_l1_loss(model(x.to(device)), y.to(device)); loss.backward(); opt.step()
        val_l1 = evaluate(model, val, device)
        checkpoint = {"model_variant": MODEL_VARIANT_WDL_PRIOR, "input_contract": INPUT_CONTRACT, "target_contract": TARGET_CONTRACT, "model": model.state_dict(), "epoch": epoch, "val_l1": val_l1, "store": str(args.store.resolve()), "split": {"key": args.val_key, "value": args.val_value}}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_l1 < best:
            best = val_l1; torch.save(checkpoint, args.output / "checkpoint_best.pt")
        print(f"[epoch {epoch:03d}] val_l1={val_l1:.6f} best={best:.6f}", flush=True)
    (args.output / "training_summary.json").write_text(json.dumps({"best_val_l1": best, "train_rows": len(train_rows), "val_rows": len(val_rows), "split": {"key": args.val_key, "value": args.val_value}}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
