"""Train V21 minimap -> alpha-scar-mask segmentation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v21_scar_dataset import V21ScarMaskDataset  # noqa: E402
from harvester.v21_scar_crop_dataset import V21ScarFilteredDataset  # noqa: E402
from harvester.v21_scar_model import V21ScarMaskModel  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "models" / "v21" / "scar-mask" / "runs"


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V21 scar-mask segmentation.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--builds", nargs="*", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--layers", default="1,2,3")
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-max-steps", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--preview-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from latest.pt/best.pt checkpoint.")
    parser.add_argument("--scar-dir", type=Path, default=None, help="Path to pre-mined scar index (enables crop-based training).")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scar_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target = target.float()
    pos_frac = target.mean() + 1e-6
    scale = (1.0 - pos_frac) / pos_frac
    bce = torch_f.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor([scale], device=logits.device)
    )
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
    loss = bce + (1.0 - dice)
    pred = (probs >= 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice_loss": float((1.0 - dice).detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "f1": float(f1.detach().cpu()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "no metrics"
    return (
        f"loss={metrics.get('loss', 0.0):.4f} "
        f"bce={metrics.get('bce', 0.0):.4f} "
        f"dice={metrics.get('dice_loss', 0.0):.4f} "
        f"iou={metrics.get('iou', 0.0):.4f} "
        f"f1={metrics.get('f1', 0.0):.4f}"
    )


def _load_history_from_metrics(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history", [])
    return history if isinstance(history, list) else []


def _best_loss_from_history(history: list[dict]) -> float:
    losses = [float(row["val"]["loss"]) for row in history if isinstance(row.get("val"), dict) and "loss" in row["val"]]
    return min(losses) if losses else float("inf")


def _save_preview(batch: dict, logits: torch.Tensor, out_path: Path) -> None:
    inp = batch["input"][:4].detach().cpu().numpy()
    target = batch["scar_mask"][:4].detach().cpu().numpy()
    probs = torch.sigmoid(logits[:4]).detach().cpu().numpy()
    rows = []
    for idx in range(inp.shape[0]):
        minimap = np.transpose(inp[idx], (1, 2, 0))
        tgt = np.repeat(target[idx, 0, :, :, None], 3, axis=2)
        pred = np.repeat(probs[idx, 0, :, :, None], 3, axis=2)
        err = np.zeros((256, 256, 3), dtype=np.float32)
        err[:, :, 0] = np.maximum(target[idx, 0] - probs[idx, 0], 0.0)
        err[:, :, 1] = np.maximum(probs[idx, 0] - target[idx, 0], 0.0)
        rows.append([minimap, tgt, pred, err])
    panel_w, panel_h = 256, 256
    label_h = 18
    canvas = Image.new("RGB", (panel_w * 4, (panel_h + label_h) * len(rows)), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    labels = ["minimap", "target", "pred", "err red=miss green=false"]
    for row_idx, row in enumerate(rows):
        y0 = row_idx * (panel_h + label_h)
        for col_idx, arr in enumerate(row):
            x0 = col_idx * panel_w
            draw.text((x0 + 4, y0 + 2), labels[col_idx], fill=(255, 255, 255))
            img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
            canvas.paste(img, (x0, y0 + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    max_steps: int | None,
    *,
    phase: str,
    epoch: int,
    epochs: int,
    log_interval: int,
) -> dict[str, float]:
    model.train(optimizer is not None)
    rows: list[dict[str, float]] = []
    step_limit = len(loader) if max_steps is None else min(len(loader), int(max_steps))
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        x = batch["input"].to(device)
        y = batch["scar_mask"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, metrics = scar_loss(logits, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        rows.append(metrics)
        if step == 1 or step % max(1, int(log_interval)) == 0 or step == step_limit:
            elapsed = time.perf_counter() - started
            print(
                f"[{phase}] epoch {epoch}/{epochs} step {step}/{step_limit} "
                f"{_format_metrics(_mean_metrics(rows))} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if max_steps is not None and step >= int(max_steps):
            break
    return _mean_metrics(rows)


def _preview_from_loader(model, val_loader, device, out_path: Path) -> None:
    batch = next(iter(val_loader))
    with torch.inference_mode():
        logits = model(batch["input"].to(device))
    _save_preview(batch, logits, out_path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = _device(args.device)
    layers = _parse_layers(args.layers)
    out_dir = Path(args.output_dir) if args.output_dir is not None else _DEFAULT_MODEL_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "checkpoints": {
                    "latest": str(out_dir / "latest.pt"),
                    "best": str(out_dir / "best.pt"),
                },
                "metrics": str(out_dir / "metrics.json"),
                "preview_latest": str(out_dir / "preview_latest.png"),
                "preview_best": str(out_dir / "preview_best.png"),
                "preview_epochs_dir": str(out_dir / "previews"),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("V21 scar-mask training", flush=True)
    print(f"  dataset_dir: {args.dataset_dir}", flush=True)
    print(f"  builds: {', '.join(args.builds)}", flush=True)
    print(f"  output_dir: {out_dir}", flush=True)
    print(f"  checkpoints: {out_dir / 'best.pt'} and {out_dir / 'latest.pt'}", flush=True)
    print(f"  previews: {out_dir / 'preview_best.png'}, {out_dir / 'preview_latest.png'}, {out_dir / 'previews'}", flush=True)
    print(f"  metrics: {out_dir / 'metrics.json'}", flush=True)
    print(f"  target: alpha layers {','.join(str(layer) for layer in layers)} > {float(args.threshold):.3f}", flush=True)
    print(f"  mode: {'filtered' if args.scar_dir else 'full-tile'}", flush=True)
    if args.scar_dir:
        print(f"  scar_dir: {args.scar_dir}", flush=True)
    print(f"  device: {device}", flush=True)
    print("Building datasets...", flush=True)
    if args.scar_dir:
        train_ds = V21ScarFilteredDataset(args.dataset_dir, args.scar_dir, args.builds, "train", augment=True, seed=args.seed, alpha_threshold=args.threshold)
        val_ds = V21ScarFilteredDataset(args.dataset_dir, args.scar_dir, args.builds, "val", augment=False, seed=args.seed, alpha_threshold=args.threshold)
    else:
        train_ds = V21ScarMaskDataset(args.dataset_dir, args.builds, "train", threshold=args.threshold, layers=layers, max_tiles=args.max_tiles, augment=True, seed=args.seed)
        val_ds = V21ScarMaskDataset(args.dataset_dir, args.builds, "val", threshold=args.threshold, layers=layers, max_tiles=args.max_tiles, augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    model = V21ScarMaskModel(base_channels=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    start_epoch = 1
    best_loss = float("inf")
    history = []

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Resume checkpoint has no optimizer state; continuing with a fresh optimizer.", flush=True)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = list(checkpoint.get("history", []))
        if not history:
            history = _load_history_from_metrics(resume_path.parent / "metrics.json")
        best_loss = float(checkpoint.get("best_loss", _best_loss_from_history(history)))
        print(f"Resumed from {resume_path}: next_epoch={start_epoch} best_loss={best_loss:.4f}", flush=True)
    print(
        f"Datasets ready: train_tiles={len(train_ds)} val_tiles={len(val_ds)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} batch_size={int(args.batch_size)}",
        flush=True,
    )
    print(f"Model parameters: {model.count_parameters():,}", flush=True)

    previews_dir = out_dir / "previews"
    if start_epoch > int(args.epochs):
        print(f"Nothing to do: resume epoch {start_epoch} is beyond --epochs {int(args.epochs)}", flush=True)
        return
    for epoch in range(start_epoch, int(args.epochs) + 1):
        print(f"--- epoch {epoch}/{int(args.epochs)} ---", flush=True)
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.max_steps,
            phase="train",
            epoch=epoch,
            epochs=int(args.epochs),
            log_interval=int(args.log_interval),
        )
        with torch.inference_mode():
            val_metrics = _run_epoch(
                model,
                val_loader,
                None,
                device,
                args.val_max_steps,
                phase="val",
                epoch=epoch,
                epochs=int(args.epochs),
                log_interval=int(args.log_interval),
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "epoch": epoch,
            "best_loss": best_loss,
            "history": history,
        }
        torch.save(checkpoint_payload, out_dir / "latest.pt")
        if bool(args.preview_every_epoch):
            epoch_preview = previews_dir / f"epoch_{epoch:04d}.png"
            _preview_from_loader(model, val_loader, device, epoch_preview)
            _preview_from_loader(model, val_loader, device, out_dir / "preview_latest.png")
            print(f"  wrote validation preview: {epoch_preview}", flush=True)
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_loss:
            best_loss = val_metrics["loss"]
            checkpoint_payload["best_loss"] = best_loss
            torch.save(checkpoint_payload, out_dir / "best.pt")
            if bool(args.preview_every_epoch):
                best_preview = out_dir / f"best_epoch_{epoch:04d}.png"
                _preview_from_loader(model, val_loader, device, best_preview)
                _preview_from_loader(model, val_loader, device, out_dir / "preview_best.png")
                print(f"  new best: val_loss={best_loss:.4f}; wrote best.pt, {best_preview.name}, and preview_best.png", flush=True)
            else:
                print(f"  new best: val_loss={best_loss:.4f}; wrote best.pt", flush=True)
        else:
            print(f"  no improvement: val_loss={val_loss:.4f} best={best_loss:.4f}", flush=True)
        print(
            f"epoch {epoch}/{int(args.epochs)} summary | train {_format_metrics(train_metrics)} | val {_format_metrics(val_metrics)}",
            flush=True,
        )

    (out_dir / "metrics.json").write_text(json.dumps({"history": history, "best_loss": best_loss}, indent=2), encoding="utf-8")
    if not (out_dir / "preview_latest.png").exists():
        _preview_from_loader(model, val_loader, device, out_dir / "preview_latest.png")
    print(f"Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
