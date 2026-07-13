"""Spec 102 M0 — object-mask trainer (Route A), full guidance/robustness stack.

RGB minimap -> one object mask, trained on the existing object_precise_mask store.
No strict fragment-trace target, no coverage gate, no reharvest.

Every mechanism below is here to get the most out of each (energy-costly) run and
to remove guesswork — no speculative loss terms, no untuned knobs left to chance:

- D4 augmentation (8 flips/rotations): minimaps are top-down, so orientation is a
  free 8x expansion of the data with no label ambiguity.
- Auto threshold sweep: report + checkpoint the IoU-optimal threshold instead of a
  hardcoded 0.5, so a sparse-mask metric stops lying.
- EMA weights: evaluate and deploy an exponential moving average (steadier than the
  raw last-step weights).
- AMP mixed precision: faster, more VRAM headroom.
- LR warmup + cosine: removes the early-epoch instability.
- Gradient clipping: stability.
- Early stopping (patience): stop the moment val stops improving; do not burn dead epochs.
- Deterministic seed + fully resumable checkpoints (--resume) and warm start
  (--init-weights) so a good run can be continued (SGDR-style) instead of restarted.
- Optional edge channel (--edge-channel): gradient-magnitude of the minimap as a 4th
  input, encoding "objects are sharp / terrain is blurry". Off by default (it makes a
  4-channel model, so it cannot warm-start a 3-channel checkpoint).

Loss is unchanged: the committed segmentation_loss (weighted BCE + Dice).

Run from wow-viewer/data-harvester/:

    uv run python scripts/train_spec102_m0_simple.py \
        --store ../output/datasets/spec102/numeric_3_3_5_full_raw_v1.zarr \
        --output ../output/spec102_m0_precise_full_v1.3 --val-map Azeroth --epochs 80 \
        --init-weights ../output/spec102_m0_precise_full_v1.2/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvester.spec102.m0 import M0ObjectMask, precise_object_target_256, segmentation_loss

THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2)


def d4(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply one of the 8 dihedral transforms (rot90*r, optional horizontal flip)."""
    out = np.rot90(arr, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def edge_channel(rgb_hw3: np.ndarray) -> np.ndarray:
    """Normalized gradient magnitude of the minimap: high on sharp object edges."""
    gray = rgb_hw3.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    return np.clip(np.hypot(gx, gy) / 128.0, 0.0, 1.0).astype(np.float32)


class TileDataset(Dataset):
    def __init__(self, rgb, mask257, rows, *, augment: bool, edge: bool) -> None:
        self.rgb, self.mask257, self.rows, self.augment, self.edge = rgb, mask257, rows, augment, edge

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        rgb = self.rgb[r].astype(np.uint8)                       # (256,256,3)
        target = (precise_object_target_256(self.mask257[r]) > 0.5).astype(np.float32)  # (256,256)
        if self.augment:
            k = int(np.random.randint(0, 8))
            rgb, target = d4(rgb, k), d4(target, k)
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0    # (3,256,256)
        if self.edge:
            x = np.concatenate([x, edge_channel(rgb)[None]], axis=0)  # (4,256,256)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(target[None])


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> dict:
    """Sweep thresholds; return metrics at the IoU-optimal one plus loss and iou@0.5."""
    model.eval()
    inter = np.zeros(len(THRESHOLDS)); union = np.zeros(len(THRESHOLDS)); predpos = np.zeros(len(THRESHOLDS))
    truepos = 0.0; loss_sum = 0.0; n = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss, _ = segmentation_loss(logits, y)
        prob = torch.sigmoid(logits.float())
        t = y >= 0.5
        truepos += float(t.sum()); loss_sum += float(loss) * x.size(0); n += x.size(0)
        for j, th in enumerate(THRESHOLDS):
            p = prob >= float(th)
            inter[j] += float((p & t).sum()); union[j] += float((p | t).sum()); predpos[j] += float(p.sum())
    iou = inter / np.maximum(union, 1.0)
    dice = 2.0 * inter / np.maximum(predpos + truepos, 1.0)
    j = int(np.argmax(iou))
    half = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
    return {
        "loss": loss_sum / max(n, 1), "best_thr": float(THRESHOLDS[j]), "iou": float(iou[j]),
        "dice": float(dice[j]), "iou_at_0.5": float(iou[half]),
        "precision": float(inter[j] / max(predpos[j], 1.0)), "recall": float(inter[j] / max(truepos, 1.0)),
    }


def rgb_std_per_tile(rgb: np.ndarray) -> np.ndarray:
    """Per-tile RGB std (chunked). Blank-minimap tiles score ~0 and are dropped."""
    n = rgb.shape[0]; out = np.zeros(n, dtype=np.float32)
    for a in range(0, n, 512):
        b = min(n, a + 512)
        out[a:b] = rgb[a:b].reshape(b - a, -1).astype(np.float32).std(axis=1)
    return out


@torch.no_grad()
def render_val_preview(model, rgb, mask257, preview_rows, identities, device, out_path: Path, epoch: int, threshold: float, val: dict, edge: bool) -> None:
    """RGB | pred prob | truth | TP/FP/FN agreement, for the object-densest val tiles."""
    model.eval()
    sep, label_h, header_h = 2, 16, 22
    rows_imgs = []
    for r in preview_rows:
        x = rgb[r].astype(np.float32).transpose(2, 0, 1) / 255.0
        if edge:
            x = np.concatenate([x, edge_channel(rgb[r].astype(np.uint8))[None]], axis=0)
        xt = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
        prob = torch.sigmoid(model(xt).float())[0, 0].cpu().numpy()
        truth = precise_object_target_256(mask257[r]) > 0.5
        pred = prob >= threshold
        inter, union = float((pred & truth).sum()), float((pred | truth).sum())
        iou = inter / union if union else 1.0
        rgb_img = rgb[r].astype(np.uint8)
        prob_img = np.repeat((np.clip(prob, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        truth_img = np.zeros((256, 256, 3), np.uint8); truth_img[truth] = (255, 255, 255)
        agree = np.zeros((256, 256, 3), np.uint8)
        agree[pred & truth] = (0, 200, 0); agree[pred & ~truth] = (220, 0, 0); agree[~pred & truth] = (0, 90, 255)
        gap = np.full((256, sep, 3), 255, np.uint8)
        row = np.concatenate([rgb_img, gap, prob_img, gap, truth_img, gap, agree], axis=1)
        m, tx, ty = identities[r]
        rows_imgs.append((row, f"{m} {tx}_{ty}   IoU={iou:.3f}"))
    width = rows_imgs[0][0].shape[1]
    canvas = np.zeros((header_h + len(rows_imgs) * (256 + label_h + sep), width, 3), np.uint8)
    y = header_h
    for row, _ in rows_imgs:
        canvas[y:y + 256, :row.shape[1]] = row; y += 256 + label_h + sep
    img = Image.fromarray(canvas); draw = ImageDraw.Draw(img); font = ImageFont.load_default()
    draw.text((4, 5), f"epoch {epoch}  thr={threshold:.2f}  val_IoU={val['iou']:.3f} Dice={val['dice']:.3f} P={val['precision']:.2f} R={val['recall']:.2f}   RGB|pred|truth|TP(grn)/FP(red)/FN(blu)", fill=(255, 255, 0), font=font)
    y = header_h
    for _, label in rows_imgs:
        draw.text((4, y + 256 + 2), label, fill=(255, 255, 0), font=font); y += 256 + label_h + sep
    out_path.parent.mkdir(parents=True, exist_ok=True); img.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 102 M0 object-mask trainer (Route A)")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-map", default="Azeroth")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=3, help="linear LR warmup epochs before cosine")
    ap.add_argument("--patience", type=int, default=20, help="early-stop after N epochs with no val improvement (0 disables)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rgb-std", type=float, default=1.0, help="drop blank-minimap tiles below this RGB std (0 keeps all)")
    ap.add_argument("--resume", action="store_true", help="continue exactly from <output>/checkpoint_last.pt")
    ap.add_argument("--init-weights", type=Path, default=None, help="warm start: load model weights, then train fresh (new optimizer + cosine cycle)")
    ap.add_argument("--edge-channel", action="store_true", help="add gradient-magnitude 4th input channel (makes a 4-ch model)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--no-aug", action="store_true", help="disable D4 augmentation")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to train on CPU.")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    use_amp = not args.no_amp
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.store}", flush=True)
    g = zarr.open_group(str(args.store), mode="r")
    rgb = np.asarray(g["minimap_rgb"][:]); mask257 = np.asarray(g["object_precise_mask_257"][:])
    idx = pq.read_table(args.store / "index.parquet", columns=["map", "tile_x", "tile_y"]).to_pylist()
    maps = [r["map"] for r in idx]; identities = [(r["map"], r["tile_x"], r["tile_y"]) for r in idx]
    keep = rgb_std_per_tile(rgb) >= args.min_rgb_std
    n_blank = int((~keep).sum())
    val_rows = [i for i, m in enumerate(maps) if m == args.val_map and keep[i]]
    train_rows = [i for i, m in enumerate(maps) if m != args.val_map and keep[i]]
    if not val_rows or not train_rows:
        raise SystemExit(f"bad holdout: val={len(val_rows)} train={len(train_rows)} for map {args.val_map!r}")
    obj_count = {r: int((precise_object_target_256(mask257[r]) > 0.5).sum()) for r in val_rows}
    preview_rows = sorted(val_rows, key=lambda r: -obj_count[r])[:6]
    print(f"[split] train={len(train_rows)} val={len(val_rows)} holdout={args.val_map} dropped_blank={n_blank} edge={args.edge_channel} amp={use_amp} aug={not args.no_aug}", flush=True)

    train_loader = DataLoader(TileDataset(rgb, mask257, train_rows, augment=not args.no_aug, edge=args.edge_channel), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(TileDataset(rgb, mask257, val_rows, augment=False, edge=args.edge_channel), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    in_ch = 4 if args.edge_channel else 3
    model = M0ObjectMask(in_channels=in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] M0ObjectMask in_ch={in_ch} params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(e: int) -> float:
        if e < args.warmup:
            return (e + 1) / max(1, args.warmup)
        prog = (e - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    history, best_iou, start_epoch = [], -1.0, 1
    resume_path = args.output / "checkpoint_last.pt"
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck.get("ema", ck["model"]))
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "sched" in ck: sched.load_state_dict(ck["sched"])
        if "scaler" in ck: scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_iou = float(ck.get("best_iou", -1.0))
        history = ck.get("history", []) or []
        print(f"[resume] {resume_path} -> continue at epoch {start_epoch}, best_iou={best_iou:.4f}", flush=True)
    elif args.resume:
        print(f"[resume] no checkpoint at {resume_path}; starting fresh", flush=True)
    elif args.init_weights is not None:
        ck = torch.load(args.init_weights, map_location=device)
        try:
            model.load_state_dict(ck["model"])
            loaded = "full"
        except RuntimeError:
            missing = model.load_state_dict(ck["model"], strict=False)
            loaded = f"partial (unmatched: {list(missing.missing_keys) + list(missing.unexpected_keys)})"
        ema_model.load_state_dict(model.state_dict())
        print(f"[init] warm-start {loaded} from {args.init_weights} (prior epoch {ck.get('epoch','?')}, val_iou {ck.get('val',{}).get('iou','?')}); fresh optimizer + {args.epochs}-epoch cosine", flush=True)

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); t0, run = time.time(), 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _ = segmentation_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                for e, p in zip(ema_model.parameters(), model.parameters()):
                    e.mul_(args.ema_decay).add_(p.detach(), alpha=1.0 - args.ema_decay)
                for eb, b in zip(ema_model.buffers(), model.buffers()):
                    eb.copy_(b)
            run += float(loss)
        sched.step()

        val = evaluate(ema_model, val_loader, device, use_amp)   # EMA is the deploy model
        raw = evaluate(model, val_loader, device, use_amp)
        rec = {"epoch": epoch, "train_loss": run / max(len(train_loader), 1), "lr": opt.param_groups[0]["lr"],
               "val_ema": val, "val_raw_iou": raw["iou"], "secs": round(time.time() - t0, 1)}
        history.append(rec)
        star = ""
        if val["iou"] > best_iou:
            best_iou = val["iou"]; epochs_no_improve = 0; star = " *best"
            torch.save({"model": ema_model.state_dict(), "threshold": val["best_thr"], "val": val,
                        "epoch": epoch, "params": n_params, "in_channels": in_ch, "edge": args.edge_channel},
                       args.output / "checkpoint_best.pt")
            render_val_preview(ema_model, rgb, mask257, preview_rows, identities, device,
                               args.output / "val_previews" / f"best_epoch_{epoch:03d}.png", epoch, val["best_thr"], val, args.edge_channel)
        else:
            epochs_no_improve += 1
        torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_iou": best_iou, "history": history, "in_channels": in_ch, "edge": args.edge_channel},
                   args.output / "checkpoint_last.pt")
        (args.output / "history.json").write_text(json.dumps({"store": str(args.store), "val_map": args.val_map, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "edge": args.edge_channel, "history": history, "best_val_iou": best_iou}, indent=2), encoding="utf-8")
        print(f"[EPOCH {epoch}/{args.epochs}] loss {rec['train_loss']:.4f}/{val['loss']:.4f}  iou {val['iou']:.4f}@{val['best_thr']:.2f} (raw {raw['iou']:.4f}, @0.5 {val['iou_at_0.5']:.3f})  P={val['precision']:.3f} R={val['recall']:.3f}  lr={rec['lr']:.2e} ({rec['secs']}s){star}", flush=True)
        if args.patience and epochs_no_improve >= args.patience:
            print(f"[early-stop] no val improvement for {args.patience} epochs; stopping at epoch {epoch}", flush=True)
            break
    print(f"[DONE] best_val_iou={best_iou:.4f} -> {args.output / 'checkpoint_best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
