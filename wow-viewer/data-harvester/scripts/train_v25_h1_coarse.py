"""Train Spec 102 H1 only: RGB + frozen H0 -> one 33x33 relief residual."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harvester.v24.train_common import RunLogger, configure_perf, peak_vram_gb, set_determinism
from harvester.v25.h0_offset import H0OffsetModel
from harvester.v25.h1_coarse import H1CoarseReliefModel, NEIGHBOR_SLOTS, RELIEF_SCALE, parameter_count

# (slot_name, tile_x offset, tile_y offset). Axis correspondence with the
# flip augmentation below (x = rgb dim 3 = h33 dim 2) matches the convention
# the trainer already used for flip_augment before this change.
NEIGHBOR_OFFSETS = (("x_minus", -1, 0), ("x_plus", 1, 0), ("y_minus", 0, -1), ("y_plus", 0, 1))
assert tuple(name for name, _, _ in NEIGHBOR_OFFSETS) == NEIGHBOR_SLOTS


def validate_contract(epochs: int, device: str, h0_report: dict) -> None:
    if not h0_report.get("gate_pass", False):
        raise RuntimeError("H1 blocked: H0 gate did not pass")
    if not 1 <= epochs <= 3:
        raise ValueError("H1 decision runs are hard-capped at 3 epochs")
    if device != "cuda":
        raise ValueError("H1 is CUDA-only; CPU fallback is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def block_mean(rgb_uint8: np.ndarray, size: int) -> np.ndarray:
    """(N, 256, 256, 3) uint8 -> (N, size, size, 3) float32 block-mean in [0, 1]."""
    if 256 % size != 0:
        raise ValueError(f"size must evenly divide 256, got {size}")
    factor = 256 // size
    n = rgb_uint8.shape[0]
    small = rgb_uint8.reshape(n, size, factor, size, factor, 3).mean(axis=(2, 4))
    return small.astype(np.float32) / 255.0


def load_tile_metadata(store: Path) -> tuple[dict[int, tuple[str, str, int, int]], dict[tuple[str, str, int, int], int]]:
    """Row -> (build, map, tile_x, tile_y) and the reverse lookup, over the
    WHOLE store (adjacency must resolve regardless of split). Complete-map
    holdout (FR-102-R003) means any resolvable neighbor of a validation/
    test-era tile is always in that same map, hence that same split -- this
    never leaks a train-split label into a held-out sample's input.
    """
    table = pq.read_table(store / "index.parquet", columns=["row", "build", "map", "tile_x", "tile_y"])
    by_row: dict[int, tuple[str, str, int, int]] = {}
    lookup: dict[tuple[str, str, int, int], int] = {}
    for b, m, x, y, r in zip(
        table["build"].to_pylist(), table["map"].to_pylist(),
        table["tile_x"].to_pylist(), table["tile_y"].to_pylist(), table["row"].to_pylist(),
    ):
        key = (str(b), str(m), int(x), int(y))
        by_row[int(r)] = key
        lookup[key] = int(r)
    return by_row, lookup


def neighbor_rows_for(row: int, by_row: dict, lookup: dict) -> dict[str, int | None]:
    build, map_name, tx, ty = by_row[row]
    return {
        name: lookup.get((build, map_name, tx + dx, ty + dy))
        for name, dx, dy in NEIGHBOR_OFFSETS
    }


def preload_context(
    store: Path, rows: list[int], by_row: dict, lookup: dict, context_size: int, batch: int = 256,
) -> dict[int, np.ndarray]:
    """Coarse (context_size, context_size) block-mean RGB for the four
    neighbors of every row in ``rows``, stacked as (12, ctx, ctx) in
    ``NEIGHBOR_SLOTS`` order. Missing neighbors (map edge, or curated out of
    this corpus) replicate the center tile's own image rather than a
    zero/black tile -- the model should never have to special-case an
    artificial edge that doesn't correspond to anything in-world.
    """
    group = zarr.open_group(str(store), mode="r")
    rgb_array = group["minimap_rgb"]

    needed = set(rows)
    for row in rows:
        needed.update(r for r in neighbor_rows_for(row, by_row, lookup).values() if r is not None)
    needed_sorted = sorted(needed)

    small_by_row: dict[int, np.ndarray] = {}
    for start in range(0, len(needed_sorted), batch):
        for row in needed_sorted[start:start + batch]:
            raw = np.asarray(rgb_array[row])[None, ...]
            small_by_row[row] = block_mean(raw, context_size)[0]

    context_by_row: dict[int, np.ndarray] = {}
    for row in rows:
        own = small_by_row[row]
        neighbors = neighbor_rows_for(row, by_row, lookup)
        stacked = [
            small_by_row[neighbors[name]] if neighbors[name] is not None else own
            for name in NEIGHBOR_SLOTS
        ]
        context_by_row[row] = np.concatenate([np.moveaxis(s, -1, 0) for s in stacked], axis=0)
    return context_by_row


def preload(
    store: Path, split_by_row: dict[int, str], batch: int = 64, input_size: int = 128, context_size: int = 32,
):
    """Downsample the 256x256 minimap to ``input_size`` (default 128, half the
    256->64 factor H0 uses). H0's target is a single scalar tile mean, so 4x
    mean-pooling barely matters; H1 predicts a 33x33 *spatial* relief field,
    where 4x pooling washes out exactly the cliff/rock-texture edges most
    likely to correlate with local relief. Halving the pooling factor keeps
    the model tiny while giving it a real chance to see that signal.

    Also loads coarse neighboring-tile context (see ``preload_context``) --
    the center tile alone has a hard, arbitrary 533-yard boundary; relief
    structure (ridgelines, valleys) does not respect that boundary.
    """
    by_row, lookup = load_tile_metadata(store)
    all_rows = sorted(split_by_row)
    context_by_row = preload_context(store, all_rows, by_row, lookup, context_size)

    group = zarr.open_group(str(store), mode="r")
    rgb_array, h33_array, liquid_array = group["minimap_rgb"], group["wdl_height_33"], group["liquid_mask_256"]
    values = {name: {"rgb": [], "h33": [], "mask": [], "context": []} for name in set(split_by_row.values())}

    for start in range(0, len(all_rows), batch):
        chunk = all_rows[start:start + batch]
        lo, hi = chunk[0], chunk[-1] + 1
        rgb_block = np.asarray(rgb_array[lo:hi])
        h33_block = np.asarray(h33_array[lo:hi], dtype=np.float32)
        liquid_block = np.asarray(liquid_array[lo:hi]) > 127
        for row in chunk:
            i = row - lo
            rgb_small = block_mean(rgb_block[i:i + 1], input_size)[0]
            liquid = liquid_block[i]
            covered = np.zeros((257, 257), dtype=bool)
            covered[:-1, :-1] |= liquid
            covered[1:, :-1] |= liquid
            covered[:-1, 1:] |= liquid
            covered[1:, 1:] |= liquid
            mask33 = ~covered[::8, ::8]
            split = split_by_row[row]
            values[split]["rgb"].append(np.moveaxis(rgb_small, -1, 0).astype(np.float32))
            values[split]["h33"].append(h33_block[i])
            values[split]["mask"].append(mask33)
            values[split]["context"].append(context_by_row[row])
    return {
        name: (
            torch.from_numpy(np.stack(v["rgb"])),
            torch.from_numpy(np.stack(v["h33"])),
            torch.from_numpy(np.stack(v["mask"])),
            torch.from_numpy(np.stack(v["context"])),
        )
        for name, v in values.items()
    }


def frozen_h0_means(rgb: torch.Tensor, checkpoint: dict, device: torch.device, batch_size: int) -> torch.Tensor:
    """Evaluate the frozen H0 model at exactly the 64x64 resolution it was
    trained/validated on, regardless of H1's own input resolution.

    H1 may downsample less aggressively than H0 (to preserve spatial detail
    for its relief field), so when ``rgb`` is finer than 64x64 it is
    block-mean-pooled down first. This composes exactly with H0's original
    256->64 block-mean preprocessing (equal-size non-overlapping blocks), so
    it reproduces H0's frozen behavior bit-for-bit rather than approximating
    it — evaluating a frozen checkpoint off its trained resolution would
    silently break the freeze contract.
    """
    config = checkpoint["config"]
    model = H0OffsetModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size = rgb.shape[-1]
    if size % 64 != 0:
        raise ValueError(f"H1 input size {size} must be a multiple of 64 to reproduce H0's frozen preprocessing")
    factor = size // 64
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rgb), batch_size):
            xb = rgb[start:start + batch_size].to(device)
            xb64 = F.avg_pool2d(xb, kernel_size=factor, stride=factor) if factor > 1 else xb
            rgb_mean = xb64.float().mean(dim=(1, 2, 3))
            baseline = float(config["rgb_flat_slope"]) * rgb_mean + float(config["rgb_flat_intercept"])
            outputs.append((baseline + model(xb64).float()).cpu())
    return torch.cat(outputs)


def make_loader(rgb, h0, h33, mask, context, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(rgb, h0, h33, mask, context), batch_size=batch_size,
        shuffle=shuffle, pin_memory=True, num_workers=0,
    )


def run_epoch(model, loader, device, optimizer=None, scheduler=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    error_sum = 0.0
    valid_sum = 0
    context_mgr = torch.enable_grad() if training else torch.no_grad()
    with context_mgr:
        for rgb, h0, h33, mask, neighbor_context in loader:
            rgb, h0, h33, mask, neighbor_context = (
                value.to(device, non_blocking=True) for value in (rgb, h0, h33, mask, neighbor_context)
            )
            target = h33 - h0[:, None, None]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(rgb, h0, neighbor_context)
                loss = ((prediction.float() - target).abs() * mask).sum() / mask.sum().clamp(min=1)
                normalized_loss = loss / RELIEF_SCALE
            if training:
                normalized_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            error_sum += float(((prediction.float() - target).abs() * mask).sum().item())
            valid_sum += int(mask.sum().item())
    return error_sum / max(valid_sum, 1)


def flip_context(context: torch.Tensor) -> torch.Tensor:
    """Mirror all four neighbor images, then swap the x_minus/x_plus SLOTS
    (whichever tile was to the -x side is now the +x side under a global
    world mirror; y_minus/y_plus keep their slot, only their own content
    flips). Channel layout is fixed by ``NEIGHBOR_SLOTS`` order: 3 channels
    each for x_minus, x_plus, y_minus, y_plus.
    """
    flipped = torch.flip(context, dims=[3])
    x_minus, x_plus, y_minus, y_plus = flipped[:, 0:3], flipped[:, 3:6], flipped[:, 6:9], flipped[:, 9:12]
    return torch.cat([x_plus, x_minus, y_minus, y_plus], dim=1)


def flip_augment(
    rgb: torch.Tensor, h0: torch.Tensor, h33: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
):
    """Double the training set with a horizontal (x-axis) mirror.

    RGB width and the h33 grid's last axis are the same world x-axis at
    different resolutions, so this is a label-preserving symmetry, not a
    approximation — it directly targets the train/val generalization gap
    (1,089-way spatial target overfits fast on ~2.4k tiles in 3 epochs) by
    forcing the tiny conv net to be x-mirror invariant instead of memorizing
    absolute pixel positions. The neighbor context mirrors consistently via
    ``flip_context``.
    """
    return (
        torch.cat([rgb, torch.flip(rgb, dims=[3])], dim=0),
        torch.cat([h0, h0], dim=0),
        torch.cat([h33, torch.flip(h33, dims=[2])], dim=0),
        torch.cat([mask, torch.flip(mask, dims=[2])], dim=0),
        torch.cat([context, flip_context(context)], dim=0),
    )


def baseline_mae(h0: torch.Tensor, h33: torch.Tensor, mask: torch.Tensor) -> float:
    error = (h33 - h0[:, None, None]).abs() * mask
    return float(error.sum().item() / max(int(mask.sum().item()), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Spec 102 H1 coarse-relief residual")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--h0-checkpoint", required=True, type=Path)
    parser.add_argument("--h0-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                         help="0 = warm up over exactly one training epoch's steps")
    parser.add_argument("--flip-augment", action="store_true", default=True)
    parser.add_argument("--no-flip-augment", dest="flip_augment", action="store_false")
    parser.add_argument("--input-size", type=int, default=128,
                         help="minimap downsample resolution fed to H1 (must divide 256)")
    parser.add_argument("--pretrained-texture", action="store_true", default=True,
                         help="fuse frozen timm mobilenetv3_small_050 stage-1 features (default on)")
    parser.add_argument("--no-pretrained-texture", dest="pretrained_texture", action="store_false")
    parser.add_argument("--neighbor-context", action="store_true", default=True,
                         help="fuse coarse x_minus/x_plus/y_minus/y_plus neighbor-tile context (default on)")
    parser.add_argument("--no-neighbor-context", dest="neighbor_context", action="store_false")
    parser.add_argument("--context-size", type=int, default=32,
                         help="per-neighbor block-mean resolution fed to the context encoder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=102)
    args = parser.parse_args()

    h0_report = json.loads(args.h0_report.read_text(encoding="utf-8"))
    validate_contract(args.epochs, args.device, h0_report)
    set_determinism(args.seed, strict=False)
    configure_perf(True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_by_row = {int(record["row"]): str(record["split"]) for record in manifest["rows"]}
    started = time.time()
    samples = preload(
        args.v25_store, split_by_row, input_size=args.input_size, context_size=args.context_size,
    )
    h0_checkpoint = torch.load(args.h0_checkpoint, map_location="cpu", weights_only=False)
    prepared = {}
    for split, (rgb, h33, mask, ctx) in samples.items():
        prepared[split] = (rgb, frozen_h0_means(rgb, h0_checkpoint, device, args.batch_size), h33, mask, ctx)

    model = H1CoarseReliefModel(
        use_pretrained_texture=args.pretrained_texture, use_neighbor_context=args.neighbor_context,
    ).to(device)
    inputs = list(inspect.signature(model.forward).parameters)
    if inputs != ["minimap_rgb", "h0_tile_mean", "neighbor_context"]:
        raise RuntimeError(f"H1 input contract drift: {inputs}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    logger = RunLogger(args.output_dir)
    logger.write_json("input_manifest.json", {
        "stage": "H1",
        "deployment_inputs": [
            "minimap_rgb (center tile)", "h0_tile_mean (frozen H0 output)",
            "neighbor_context: minimap_rgb of x_minus/x_plus/y_minus/y_plus adjacent tiles "
            "(each exactly as available as the center tile at deployment; Input Invariant "
            "explicitly permits adjacent-tile RGB)",
        ],
        "output_signal": "coarse_relief_residual_33", "upstream_checkpoint": str(args.h0_checkpoint.resolve()),
        "target_only": ["wdl_height_33", "liquid_mask_256"],
    })

    train_rgb, train_h0, train_h33, train_mask, train_ctx = prepared["train"]
    if args.flip_augment:
        train_rgb, train_h0, train_h33, train_mask, train_ctx = flip_augment(
            train_rgb, train_h0, train_h33, train_mask, train_ctx
        )

    config = {
        "stage": "H1", "output_signal": "coarse_relief_residual_33", "parameters": parameter_count(model),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "flip_augment": args.flip_augment,
        "input_size": args.input_size, "train_samples": int(train_rgb.shape[0]),
        "pretrained_texture": args.pretrained_texture,
        "pretrained_texture_backbone": "timm/mobilenetv3_small_050.lamb_in1k stage1 (frozen)" if args.pretrained_texture else None,
        "pretrained_texture_weights_loaded": bool(model.texture.pretrained) if model.texture is not None else None,
        "neighbor_context": args.neighbor_context, "context_size": args.context_size,
        "frozen_params": frozen_params,
        "h0_checkpoint": str(args.h0_checkpoint.resolve()), "h0_checkpoint_sha256": sha256_file(args.h0_checkpoint),
        "split_manifest": str(args.split_manifest.resolve()), "preload_seconds": round(time.time() - started, 3),
        "git_revision": git_revision(),
    }
    logger.write_json("config.json", config)
    train_loader = make_loader(train_rgb, train_h0, train_h33, train_mask, train_ctx, args.batch_size, True)
    val_loader = make_loader(*prepared["validation_map"], args.batch_size, False)
    era_loader = make_loader(*prepared["test_era"], args.batch_size, False)
    val_baseline = baseline_mae(prepared["validation_map"][1], prepared["validation_map"][2], prepared["validation_map"][3])
    required = 0.8 * val_baseline

    warmup_steps = args.warmup_steps or len(train_loader)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1)
    )

    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_mae = run_epoch(model, train_loader, device, optimizer, scheduler, args.grad_clip)
        val_mae = run_epoch(model, val_loader, device)
        logger.log_epoch(epoch, train_coarse_mae=train_mae, val_coarse_mae=val_mae,
                         epoch_seconds=round(time.time() - epoch_start, 3),
                         peak_vram_gb=round(peak_vram_gb() or 0.0, 3))
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_coarse_mae": val_mae}
        torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            torch.save(checkpoint, args.output_dir / "checkpoint_best.pt")

    best_checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    report = {
        "stage": "H1", "best_epoch": best_epoch, "best_val_coarse_mae": best_val,
        "era_test_coarse_mae": run_epoch(model, era_loader, device), "h0_plane_baseline_mae": val_baseline,
        "required_mae": required, "gate_pass": best_val <= required, "peak_vram_gb": peak_vram_gb(),
    }
    logger.write_json("report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
