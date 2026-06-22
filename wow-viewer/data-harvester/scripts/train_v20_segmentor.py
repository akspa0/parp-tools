"""Train V20 Minimap Semantic Segmentor (V20-MSS).

Trains Model 1 to predict:
- liquid_type_256 (5 classes: none, ocean, river, magma, slime)
- object_precise_mask_256 (1ch, binary footprint)
- alpha_256 (4ch, texture weights)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw

# Add src to python path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v20_dataset import V20Dataset
from harvester.v20_models import V20SemanticSegmentor


def compute_f1_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute binary F1 score for object presence mask."""
    p_bin = (preds > threshold).float()
    t_bin = (targets > threshold).float()
    tp = (p_bin * t_bin).sum().item()
    fp = (p_bin * (1.0 - t_bin)).sum().item()
    fn = ((1.0 - p_bin) * t_bin).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2.0 * (precision * recall) / (precision + recall + 1e-8)


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy for liquid type classes."""
    # preds: [B, 5, H, W], targets: [B, 1, H, W]
    p_cls = torch.argmax(preds, dim=1, keepdim=True)
    correct = (p_cls == targets).float().sum().item()
    total = targets.numel()
    return correct / (total + 1e-8)


_PANEL_SIZE = 256
_PANEL_LABEL_HEIGHT = 18
_ROW_LABEL_HEIGHT = 18


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


def _draw_text_strip(img: Image.Image, text: str, height: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), color=(0, 0, 0))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (canvas.width, height - 1)], fill=(14, 14, 14))
    draw.text((4, 3), str(text), fill=(240, 240, 240))
    return canvas


def _compose_horizontal_panel(panels: list[tuple[str, torch.Tensor]]) -> Image.Image:
    images: list[Image.Image] = []
    for label, tensor in panels:
        arr = _to_uint8_hwc(tensor)
        img = Image.fromarray(arr)
        if img.size != (_PANEL_SIZE, _PANEL_SIZE):
            img = img.resize((_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR)
        img = _draw_text_strip(img, label, _PANEL_LABEL_HEIGHT)
        images.append(img)
    canvas = Image.new("RGB", (_PANEL_SIZE * len(images), _PANEL_SIZE + _PANEL_LABEL_HEIGHT), color=(0, 0, 0))
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * _PANEL_SIZE, 0))
    return canvas


def _save_preview_grid(rows: list[list[tuple[str, torch.Tensor]]], out_path: Path, row_titles: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError("Cannot save preview grid with no rows.")
    row_images = []
    for idx, row in enumerate(rows):
        row_img = _compose_horizontal_panel(row)
        if row_titles is not None:
            row_img = _draw_text_strip(row_img, row_titles[idx], _ROW_LABEL_HEIGHT)
        row_images.append(row_img)
    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for img in row_images:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(out_path)


def _meta_value(x: Any, idx: int) -> Any:
    if isinstance(x, torch.Tensor):
        return x[idx].item()
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, (list, tuple)):
        return x[idx]
    return x


def _preview_row_title(batch: dict[str, Any], idx: int) -> str:
    build = _meta_value(batch.get("meta_build", "unknown"), idx)
    map_name = _meta_value(batch.get("meta_map", ""), idx)
    tile_id = _meta_value(batch.get("meta_tile_id", -1), idx)
    tile_x = _meta_value(batch.get("meta_tile_x", -1), idx)
    tile_y = _meta_value(batch.get("meta_tile_y", -1), idx)
    return f"{build} | {map_name} | tile={tile_id} | ({tile_x},{tile_y})"


def _coarse_type_to_rgb(type_grid: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.45, 0.9],
            [0.1, 0.7, 0.95],
            [0.9, 0.35, 0.1],
            [0.25, 0.85, 0.2],
        ],
        dtype=torch.float32,
        device=type_grid.device,
    )
    rgb = palette[type_grid.long().clamp(0, 4)]
    if rgb.ndim == 3:
        return rgb.permute(2, 0, 1)
    return rgb


def _alpha_to_rgb(alpha_grid: torch.Tensor) -> torch.Tensor:
    return alpha_grid[1:4]


def save_validation_previews(batch: dict[str, Any], model: nn.Module, device: torch.device, out_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        inputs = batch["input"].to(device)
        targets_liquid = batch["liquid_type_256"].to(device).squeeze(1)
        targets_object = batch["object_precise_mask_256"].to(device)
        targets_alpha = batch["alpha"].to(device)

        liquid_logits, object_mask, alpha_weights = model(inputs)
        pred_liquid_cls = torch.argmax(liquid_logits, dim=1)

        n = min(int(inputs.shape[0]), 8)
        rows = []
        row_titles = []
        for idx in range(n):
            row_titles.append(_preview_row_title(batch, idx))
            liq_gt_rgb = _coarse_type_to_rgb(targets_liquid[idx])
            liq_pred_rgb = _coarse_type_to_rgb(pred_liquid_cls[idx])
            alpha_gt_rgb = _alpha_to_rgb(targets_alpha[idx])
            alpha_pred_rgb = _alpha_to_rgb(alpha_weights[idx])

            rows.append(
                [
                    ("minimap", inputs[idx][:3]),
                    ("liq_gt", liq_gt_rgb),
                    ("liq_pred", liq_pred_rgb),
                    ("obj_gt", targets_object[idx]),
                    ("obj_pred", object_mask[idx]),
                    ("alpha_gt", alpha_gt_rgb),
                    ("alpha_pred", alpha_pred_rgb),
                ]
            )
        _save_preview_grid(rows, out_path, row_titles=row_titles)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V20 Semantic Segmentor (Model 1)")
    parser.add_argument("--dataset-dir", type=str, default="../output/datasets/v18")
    parser.add_argument("--builds", nargs="+", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument(
        "--curation-manifest",
        type=str,
        default=None,
        help="Optional curation manifest directory or kept_tiles.parquet file"
    )
    parser.add_argument(
        "--curation-min-terrain-validity",
        type=float,
        default=0.20,
        help="Drop tiles below this terrain-validity score."
    )
    parser.add_argument(
        "--curation-min-minimap-usefulness",
        type=float,
        default=0.10,
        help="Drop tiles below this minimap usefulness score."
    )
    parser.add_argument(
        "--curation-reject-what-plate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject manifest tiles flagged as whiteplate/noise tiles."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="../output/ml-training/v20_segmentor")
    parser.add_argument("--early-stop", type=int, default=7, help="Early stopping patience epochs")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limit the number of train dataset samples to load (default: 1000)."
    )
    args = parser.parse_args()

    # Seed reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve wow-viewer root (parent of data-harvester)
    script_dir = Path(__file__).resolve().parent
    wow_viewer_root = None
    for p in [script_dir] + list(script_dir.parents):
        if p.name == "wow-viewer":
            wow_viewer_root = p
            break
        elif (p / "wow-viewer").exists():
            wow_viewer_root = p / "wow-viewer"
            break

    # Resolve directories
    dataset_root = Path(args.dataset_dir)
    if not dataset_root.exists():
        resolved = False
        if wow_viewer_root is not None:
            # Try relative to wow-viewer root directly
            p_alt = wow_viewer_root / args.dataset_dir
            if p_alt.exists():
                dataset_root = p_alt
                resolved = True
            else:
                # Try stripping 'wow-viewer/' prefix from args.dataset_dir if present
                clean_dir = args.dataset_dir
                if clean_dir.startswith("wow-viewer/"):
                    clean_dir = clean_dir[len("wow-viewer/"):]
                elif clean_dir.startswith("wow-viewer\\"):
                    clean_dir = clean_dir[len("wow-viewer\\"):]
                
                p_alt2 = wow_viewer_root / clean_dir
                if p_alt2.exists():
                    dataset_root = p_alt2
                    resolved = True
                else:
                    # Try output/datasets/v18 directly under wow-viewer root
                    p_direct = wow_viewer_root / "output" / "datasets" / "v18"
                    if p_direct.exists():
                        dataset_root = p_direct
                        resolved = True

        if not resolved:
            # Try absolute project roots
            alt_absolute = Path("i:/parp/parp-tools/wow-viewer") / args.dataset_dir
            if alt_absolute.exists():
                dataset_root = alt_absolute
            else:
                alt_absolute_2 = Path("i:/parp/parp-tools") / args.dataset_dir
                if alt_absolute_2.exists():
                    dataset_root = alt_absolute_2
                else:
                    # Fallback to direct absolute paths
                    fb1 = Path("i:/parp/parp-tools/wow-viewer/output/datasets/v18")
                    if fb1.exists():
                        dataset_root = fb1
                    else:
                        print(f"Error: dataset dir not found at {args.dataset_dir} or relative to {wow_viewer_root}")
                        sys.exit(1)

    # Resolve curation manifest
    curation_manifest_path = None
    if args.curation_manifest is not None:
        p_cur = Path(args.curation_manifest)
        if p_cur.exists():
            curation_manifest_path = p_cur
        elif wow_viewer_root is not None:
            p_alt = wow_viewer_root / args.curation_manifest
            if p_alt.exists():
                curation_manifest_path = p_alt
            else:
                # Strip wow-viewer/ prefix
                clean_cur = args.curation_manifest
                if clean_cur.startswith("wow-viewer/"):
                    clean_cur = clean_cur[len("wow-viewer/"):]
                elif clean_cur.startswith("wow-viewer\\"):
                    clean_cur = clean_cur[len("wow-viewer\\"):]
                p_alt2 = wow_viewer_root / clean_cur
                if p_alt2.exists():
                    curation_manifest_path = p_alt2
        
        if curation_manifest_path is None:
            # Check relative to dataset_root
            p_ds_rel = dataset_root / args.curation_manifest
            if p_ds_rel.exists():
                curation_manifest_path = p_ds_rel
            else:
                # Check under dataset_root / curation / name
                p_ds_cur = dataset_root / "curation" / args.curation_manifest
                if p_ds_cur.exists():
                    curation_manifest_path = p_ds_cur
                else:
                    # Check under wow_viewer_root / output / datasets / v18 / curation / name
                    if wow_viewer_root is not None:
                        p_full_cur = wow_viewer_root / "output" / "datasets" / "v18" / "curation" / args.curation_manifest
                        if p_full_cur.exists():
                            curation_manifest_path = p_full_cur
                    
                    if curation_manifest_path is None:
                        print(f"Warning: curation manifest '{args.curation_manifest}' not found. Initializing without curation filtering.")
                        curation_manifest_path = Path(args.curation_manifest)

    if curation_manifest_path is not None:
        print(f"Resolved curation manifest: {curation_manifest_path}")
    else:
        print("No curation manifest specified.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load datasets
    print("Initializing datasets...")
    train_ds = V20Dataset(
        dataset_root=dataset_root,
        builds=args.builds,
        split="train",
        val_fraction=args.val_fraction,
        augment=True,
        seed=args.seed,
        curation_manifest=curation_manifest_path,
        curation_min_terrain_validity=args.curation_min_terrain_validity,
        curation_min_minimap_usefulness=args.curation_min_minimap_usefulness,
        curation_reject_what_plate=args.curation_reject_what_plate,
        limit=args.limit,
    )
    # Validation dataset limit scales proportionally
    val_limit = max(1, int(args.limit * args.val_fraction)) if args.limit is not None else None
    val_ds = V20Dataset(
        dataset_root=dataset_root,
        builds=args.builds,
        split="val",
        val_fraction=args.val_fraction,
        augment=False,
        seed=args.seed,
        curation_manifest=curation_manifest_path,
        curation_min_terrain_validity=args.curation_min_terrain_validity,
        curation_min_minimap_usefulness=args.curation_min_minimap_usefulness,
        curation_reject_what_plate=args.curation_reject_what_plate,
        limit=val_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # Grab a fixed batch from validation loader for preview visualization
    preview_batch = None
    try:
        preview_batch = next(iter(val_loader))
    except StopIteration:
        pass

    val_dir = out_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Instantiate Model 1
    model = V20SemanticSegmentor(in_channels=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss criteria
    criterion_liquid = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Track metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_liquid_acc": [],
        "val_object_f1": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(device)
            targets_liquid = batch["liquid_type_256"].to(device).squeeze(1)  # [B, H, W] long
            targets_object = batch["object_precise_mask_256"].to(device)  # [B, 1, H, W] float
            targets_alpha = batch["alpha"].to(device)  # [B, 4, H, W] float

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    liquid_logits, object_mask, alpha_weights = model(inputs)
                    loss_liq = criterion_liquid(liquid_logits, targets_liquid)
                    # Temporarily disable autocast to compute BCELoss and L1Loss stably in float32
                    with torch.amp.autocast("cuda", enabled=False):
                        loss_obj = F.binary_cross_entropy(object_mask.float(), targets_object.float())
                        loss_alp = F.l1_loss(alpha_weights.float(), targets_alpha.float())
                    loss = 1.0 * loss_liq + 1.0 * loss_obj + 0.5 * loss_alp

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                liquid_logits, object_mask, alpha_weights = model(inputs)
                loss_liq = criterion_liquid(liquid_logits, targets_liquid)
                loss_obj = F.binary_cross_entropy(object_mask, targets_object)
                loss_alp = F.l1_loss(alpha_weights, targets_alpha)
                loss = 1.0 * loss_liq + 1.0 * loss_obj + 0.5 * loss_alp
                loss.backward()
                optimizer.step()

            train_loss_total += loss.item()

        scheduler.step()
        train_loss_avg = train_loss_total / len(train_loader)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_liq_acc_total = 0.0
        val_obj_f1_total = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets_liquid = batch["liquid_type_256"].to(device).squeeze(1)
                targets_object = batch["object_precise_mask_256"].to(device)
                targets_alpha = batch["alpha"].to(device)

                liquid_logits, object_mask, alpha_weights = model(inputs)
                loss_liq = criterion_liquid(liquid_logits, targets_liquid)
                loss_obj = F.binary_cross_entropy(object_mask.float(), targets_object.float())
                loss_alp = F.l1_loss(alpha_weights.float(), targets_alpha.float())
                loss = 1.0 * loss_liq + 1.0 * loss_obj + 0.5 * loss_alp

                val_loss_total += loss.item()
                val_liq_acc_total += compute_accuracy(liquid_logits, targets_liquid.unsqueeze(1))
                val_obj_f1_total += compute_f1_score(object_mask, targets_object)
                val_count += 1

        val_loss_avg = val_loss_total / val_count
        val_liq_acc = val_liq_acc_total / val_count
        val_obj_f1 = val_obj_f1_total / val_count

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss={train_loss_avg:.4f} | "
            f"Val Loss={val_loss_avg:.4f} | Liq Acc={val_liq_acc:.4f} | Obj F1={val_obj_f1:.4f}"
        )

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_liquid_acc"].append(val_liq_acc)
        history["val_object_f1"].append(val_obj_f1)

        # Checkpoint save
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                },
                out_dir / "best_model.pth",
            )
            print("  *** New best validation loss, saved model.pth")
            if preview_batch is not None:
                save_validation_previews(preview_batch, model, device, val_dir / f"best_epoch_{epoch:02d}.png")
                print(f"  *** Saved validation preview to {val_dir / f'best_epoch_{epoch:02d}.png'}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training finished.")


if __name__ == "__main__":
    main()
