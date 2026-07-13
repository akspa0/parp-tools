"""Spec 102 M0 — simple object-mask training (Route A).

RGB minimap -> one object-mask signal. Trains directly on the existing
`object_precise_mask_257` numeric store, with a complete-map holdout. No strict
fragment-trace target, no coverage gate, no reharvest. Reuses the committed M0
model + loss so nothing new is invented.

Run from wow-viewer/data-harvester/:

    uv run python scripts/train_spec102_m0_simple.py \
        --store ../output/datasets/spec102/numeric_3_3_5_full_raw_v1.zarr \
        --output ../output/spec102_m0_precise_simple_v1 \
        --val-map Expansion01 --epochs 80 --batch 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvester.spec102.m0 import M0ObjectMask, precise_object_target_256, segmentation_loss


class TileDataset(Dataset):
    def __init__(self, rgb: np.ndarray, mask257: np.ndarray, rows: list[int]) -> None:
        self.rgb = rgb
        self.mask257 = mask257
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        rgb = self.rgb[r].astype(np.float32).transpose(2, 0, 1) / 255.0
        target = (precise_object_target_256(self.mask257[r]) > 0.5).astype(np.float32)[None]
        return torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(target)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    model.eval()
    inter = union = pred_pos = true_pos = loss_sum = n = 0.0
    for rgb, target in loader:
        rgb = rgb.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(rgb)
        loss, _ = segmentation_loss(logits, target)
        p = torch.sigmoid(logits) >= 0.5
        t = target >= 0.5
        inter += float((p & t).sum())
        union += float((p | t).sum())
        pred_pos += float(p.sum())
        true_pos += float(t.sum())
        loss_sum += float(loss) * rgb.size(0)
        n += rgb.size(0)
    return {"loss": loss_sum / max(n, 1), "iou": inter / max(union, 1.0), "dice": (2.0 * inter) / max(pred_pos + true_pos, 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 102 M0 simple object-mask trainer (Route A)")
    ap.add_argument("--store", required=True, type=Path, help="numeric store with minimap_rgb + object_precise_mask_257")
    ap.add_argument("--output", required=True, type=Path, help="output dir for checkpoints + history")
    ap.add_argument("--val-map", default="Expansion01", help="complete map held out for validation")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to train on CPU.")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.store}", flush=True)
    g = zarr.open_group(str(args.store), mode="r")
    rgb = np.asarray(g["minimap_rgb"][:])
    mask257 = np.asarray(g["object_precise_mask_257"][:])
    maps = [r["map"] for r in pq.read_table(args.store / "index.parquet", columns=["map"]).to_pylist()]
    val_rows = [i for i, m in enumerate(maps) if m == args.val_map]
    train_rows = [i for i, m in enumerate(maps) if m != args.val_map]
    if not val_rows or not train_rows:
        raise SystemExit(f"bad holdout: val={len(val_rows)} train={len(train_rows)} for map {args.val_map!r}")
    print(f"[split] train={len(train_rows)} val={len(val_rows)} holdout={args.val_map}", flush=True)

    train_loader = DataLoader(TileDataset(rgb, mask257, train_rows), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(TileDataset(rgb, mask257, val_rows), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    model = M0ObjectMask().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] M0ObjectMask params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_iou = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, _ = segmentation_loss(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss)
        sched.step()
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": run / max(len(train_loader), 1), "val": val, "secs": round(time.time() - t0, 1)}
        history.append(rec)
        star = ""
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "params": n_params}, args.output / "checkpoint_best.pt")
            star = " *best"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "params": n_params}, args.output / "checkpoint_last.pt")
        (args.output / "history.json").write_text(json.dumps({"store": str(args.store), "val_map": args.val_map, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "history": history, "best_val_iou": best_iou}, indent=2), encoding="utf-8")
        print(f"[EPOCH {epoch}/{args.epochs}] train_loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} val_iou={val['iou']:.4f} val_dice={val['dice']:.4f} ({rec['secs']}s){star}", flush=True)
    print(f"[DONE] best_val_iou={best_iou:.4f} -> {args.output/'checkpoint_best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
