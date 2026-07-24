"""Spec 120 OBB Detector Trainer (T006).

Dry-run-first trainer for MinimapOBBDetector with OBB loss (Confidence BCE + Location L1 + Angle L1 + Class CE)
and OneCycle LR schedule.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from harvester.spec120.obb_contract import STAGE_OBB_DETECTOR
from harvester.spec120.obb_detector_model import MinimapOBBDetector
from harvester.v50.lr_schedule import make_onecycle_scheduler


class OBBDetectorLoss(nn.Module):
    """Loss module for OBB Detector."""

    def __init__(self, loc_weight: float = 2.0, angle_weight: float = 1.0, cls_weight: float = 1.0):
        super().__init__()
        self.loc_weight = loc_weight
        self.angle_weight = angle_weight
        self.cls_weight = cls_weight

    def forward(self, pred: torch.Tensor, targets_array: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute loss between raw predictions and target grid.

        pred: (N, 16, 16, 11)
        targets_array: (N, 64, 6) [class_id, cx_norm, cy_norm, w_norm, h_norm, angle_deg]
        """
        device = pred.device
        batch_size, grid_h, grid_w, _ = pred.shape

        # Build ground-truth target grid (N, 16, 16, 11)
        target_grid = torch.zeros_like(pred)
        pos_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)

        for b in range(batch_size):
            t_set = targets_array[b]  # (64, 6)
            for t_idx in range(t_set.shape[0]):
                cls_id = float(t_set[t_idx, 0])
                if cls_id < 0:
                    continue  # Padded slot

                cx_norm = float(t_set[t_idx, 1])
                cy_norm = float(t_set[t_idx, 2])
                w_norm = float(t_set[t_idx, 3])
                h_norm = float(t_set[t_idx, 4])
                angle_deg = float(t_set[t_idx, 5])

                gx = int(clamp(cx_norm * grid_w, 0, grid_w - 1))
                gy = int(clamp(cy_norm * grid_h, 0, grid_h - 1))

                dx = cx_norm * grid_w - gx
                dy = cy_norm * grid_h - gy

                angle_rad = math.radians(angle_deg)
                sin_a = math.sin(angle_rad)
                cos_a = math.cos(angle_rad)

                pos_mask[b, gy, gx] = True
                target_grid[b, gy, gx, 0] = 1.0  # Confidence
                target_grid[b, gy, gx, 1] = dx
                target_grid[b, gy, gx, 2] = dy
                target_grid[b, gy, gx, 3] = w_norm
                target_grid[b, gy, gx, 4] = h_norm
                target_grid[b, gy, gx, 5] = sin_a
                target_grid[b, gy, gx, 6] = cos_a
                target_grid[b, gy, gx, 7 + int(cls_id)] = 1.0

        # Confidence BCE Loss with positive class weighting (object vs background imbalance)
        conf_pred = pred[..., 0]
        conf_target = target_grid[..., 0]
        pos_weight = torch.tensor([10.0], device=device)
        conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_target, pos_weight=pos_weight)

        # Location Loss (on positive cells only)
        if pos_mask.sum() > 0:
            loc_pred = torch.sigmoid(pred[pos_mask, 1:3])
            loc_target = target_grid[pos_mask, 1:3]
            loc_loss = F.smooth_l1_loss(loc_pred, loc_target)

            size_pred = F.softplus(pred[pos_mask, 3:5])
            size_target = target_grid[pos_mask, 3:5]
            size_loss = F.smooth_l1_loss(size_pred, size_target)

            angle_pred = pred[pos_mask, 5:7]
            angle_target = target_grid[pos_mask, 5:7]
            angle_loss = F.smooth_l1_loss(angle_pred, angle_target)

            cls_pred = pred[pos_mask, 7:]
            cls_target = target_grid[pos_mask, 7:].argmax(dim=-1)
            cls_loss = F.cross_entropy(cls_pred, cls_target)
        else:
            loc_loss = torch.tensor(0.0, device=device)
            size_loss = torch.tensor(0.0, device=device)
            angle_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = conf_loss + self.loc_weight * (loc_loss + size_loss) + self.angle_weight * angle_loss + self.cls_weight * cls_loss

        return {
            "loss": total_loss,
            "conf_loss": conf_loss,
            "loc_loss": loc_loss + size_loss,
            "angle_loss": angle_loss,
            "cls_loss": cls_loss,
        }


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def generate_dry_run_report(
    model: MinimapOBBDetector, epochs: int, batch_size: int, lr: float
) -> dict[str, Any]:
    """Generate dry-run training plan report for inspection without CUDA training."""
    num_params = sum(p.numel() for p in model.parameters())
    return {
        "stage": STAGE_OBB_DETECTOR,
        "arch": model.model_config()["arch"],
        "base": model.base,
        "num_params": num_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_lr": lr,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dry_run": True,
    }


def train_obb_detector_loop(
    dataset_dir: Path,
    output_dir: Path,
    base: int = 16,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> Path:
    """Full PyTorch training loop for MinimapOBBDetector."""
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    npz_path = dataset_dir / "obb_dataset.npz"
    split_path = dataset_dir / "split.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {npz_path.resolve()}. Run dataset builder first.")

    npz_data = np.load(npz_path)
    images = npz_data["images"]  # (N, 256, 256, 3) uint8
    targets = npz_data["targets"]  # (N, 64, 6) float32

    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    train_indices = split_info["train_indices"]
    val_indices = split_info["val_indices"]

    class OBBDataset(Dataset):
        def __init__(self, idxs: list[int]) -> None:
            self.idxs = idxs

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, idx: int):
            real_idx = self.idxs[idx]
            img = images[real_idx]  # (256, 256, 3) uint8
            tgt = targets[real_idx]  # (64, 6) float32
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tgt_tensor = torch.from_numpy(tgt).float()
            return img_tensor, tgt_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    torch.manual_seed(seed)
    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=base).to(device)
    loss_fn = OBBDetectorLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(OBBDataset(train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OBBDataset(val_indices), batch_size=batch_size, shuffle=False)

    scheduler, warmup_epochs = make_onecycle_scheduler(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(train_loader))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "obb_detector_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            losses = loss_fn(pred, y_b)
            loss = losses["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                losses = loss_fn(pred, y_b)
                val_loss += losses["loss"].item()

        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": model.model_config(),
                },
                best_ckpt_path,
            )

    print(f"\n[TRAINING COMPLETE] Best checkpoint saved to {best_ckpt_path.resolve()} (Val Loss: {best_val_loss:.4f})")
    return best_ckpt_path

