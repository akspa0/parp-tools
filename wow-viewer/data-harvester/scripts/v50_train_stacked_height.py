"""Train a simple height model (U-Net-lite) with the frozen residual extractor as a preprocessing step.

Chains: minimap RGB → frozen residual extractor → residual → concat [RGB, residual] → height model → height

The height model is a 4-channel U-Net-lite (same architecture as direct_cnn_v112 but with
the residual as a 4th input channel). The residual extractor is frozen — only the height
model trains.

Usage (from wow-viewer/data-harvester):

    uv run python scripts/v50_train_stacked_height.py ^
        --store "I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curriculum-0_5_3_3368-dual_v3.zarr" ^
        --residual-checkpoint "I:\parp\parp-tools\wow-viewer\output\runs\residual-extractor-v2\checkpoint_best.pt" ^
        --output "I:\parp\parp-tools\wow-viewer\output\runs\stacked-height-v1" ^
        --epochs 100 --batch 16 --lr 2e-4 --seed 114 --confirm-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.height_relative_model import HEIGHT_GRID, TARGET_CONTRACT_VERSION, encode_relative_height, height_loss
from harvester.v50.residual_extractor_infer import load_model as load_extractor
from harvester.v50.residual_extractor_model import ResidualExtractorNet


class StackedHeightNet(nn.Module):
    """4-channel U-Net-lite: [R, G, B, residual] -> 257x257 relative height.

    Same architecture as HeightRelativeNet/direct_cnn_v112 but with the
    residual as a 4th input channel. The residual is the terrain shading,
    which the height model can use to infer slope direction.
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _block(4, base)
        self.enc2 = _block(base, base * 2, stride=2)
        self.enc3 = _block(base * 2, base * 4, stride=2)
        self.enc4 = _block(base * 4, base * 8, stride=2)
        self.mid = _block(base * 8, base * 8)
        self.up3 = _block(base * 8 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        m = self.mid(e4)
        u3 = self.up3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.up2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.up1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1))
        out = torch.sigmoid(self.head(u1))
        return nn.functional.interpolate(out, size=(HEIGHT_GRID, HEIGHT_GRID), mode="bilinear", align_corners=True).squeeze(1)


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train stacked height model (RGB + frozen residual extractor)")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--residual-checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import zarr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; refusing CPU training.")

    # Load frozen residual extractor.
    print("loading residual extractor...", flush=True)
    extractor = load_extractor(args.residual_checkpoint, device, args.release)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)

    # Open store.
    group = zarr.open_group(str(args.store), mode="r")
    require_store_release(group, args.release, store=args.store)
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    REQUIRED = {"minimap_rgb", "height_257"}
    missing = sorted(REQUIRED - set(group.array_keys()))
    if missing:
        raise SystemExit(f"store missing {missing}")

    # Split: authored only.
    import hashlib
    rng = np.random.default_rng(args.seed)
    all_rows = list(range(len(index)))
    rng.shuffle(all_rows)
    split = int(len(all_rows) * 0.85)
    train_rows, val_rows = all_rows[:split], all_rows[split:]
    print(f"rows: {len(train_rows)} train, {len(val_rows)} val", flush=True)

    # Model.
    torch.manual_seed(args.seed)
    model = StackedHeightNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"StackedHeightNet: {param_count:,} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            # Run frozen extractor on CPU/GPU
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                residual = extractor(rgb_t)[0].cpu().numpy()  # (H, W)
            # Concatenate: [R, G, B, residual]
            rgb_chw = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
            residual_chw = torch.from_numpy(residual).unsqueeze(0)  # (1, H, W)
            channels = torch.cat([rgb_chw, residual_chw], dim=0)  # (4, H, W)
            raw_h = np.asarray(group["height_257"][row], dtype=np.float32)
            target, _, _ = encode_relative_height(raw_h)
            return channels, torch.from_numpy(target)

    train_loader = DataLoader(RowDataset(train_rows), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(RowDataset(val_rows), batch_size=args.batch, num_workers=0, pin_memory=True)

    val_targets = {r: encode_relative_height(np.asarray(group["height_257"][r], dtype=np.float32))[0] for r in val_rows}
    baseline_mae = float(np.mean([np.abs(t - t.mean()).mean() for t in val_targets.values()]))
    print(f"tile-mean baseline: {baseline_mae:.6f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    run_id = {
        **release_identity(args.release),
        "model": "StackedHeightNet",
        "residual_checkpoint": str(args.residual_checkpoint),
        "store": str(args.store.resolve()),
        "param_count": param_count,
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": 1e-4,
        "loss": "height_loss (smooth_l1 + gradient_l1)",
        "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
    }
    (args.output / "run_id.json").write_text(json.dumps(run_id, indent=2), encoding="utf-8")

    best = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = height_loss(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        val_mae = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                y_d = y.to(device)
                err = torch.abs(model(x.to(device)) - y_d)
                val_mae += float(err.sum())
                val_n += err.numel()
        val_mae /= val_n
        beats = val_mae < baseline_mae
        print(f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} val_mae={val_mae:.6f} baseline={baseline_mae:.6f} beats={beats}", flush=True)
        if val_mae < best:
            best = val_mae
            best_epoch = epoch
            stale = 0
            torch.save({**run_id, "model": model.state_dict(), "epoch": epoch, "val_mae": val_mae}, args.output / "checkpoint_best.pt")
        else:
            stale += 1
        torch.save({**run_id, "model": model.state_dict(), "epoch": epoch, "val_mae": val_mae}, args.output / "checkpoint_last.pt")
        if args.patience > 0 and stale >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    print(f"best_epoch={best_epoch} best_val_mae={best:.6f} baseline={baseline_mae:.6f} beats_baseline={best < baseline_mae}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())