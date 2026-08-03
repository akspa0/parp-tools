"""Spec 126 US7: iteratively refine height from a terrain-shadow residual using the known forward model.

The forward model (hillshade) is deterministic and differentiable.  The refinement
loop exploits this: instead of training a feed-forward neural network to approximate
the inverse, we *optimize the height pixels directly* to match the target residual
when rendered through the known lighting model.

The loss is a hybrid:
- **shape_loss** (mean-centered, variance-normalised L1) drives the shading pattern.
- **affine-fit L1** (recomputed each iteration) drives the actual brightness amplitude.
- **TV smoothness** (very light) keeps the height clean.

When the store carries ``height_257`` (the residual-height curriculum), the report
includes the Pearson correlation between refined height and ground-truth relative
height, providing an honest gate on whether the refinement actually recovers relief.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image, ImageDraw, ImageFont

from harvester.v50.contracts import (
    release_identity,
    require_store_release,
    validate_release,
)
from harvester.v50.residual_extractor_infer import (
    load_model as load_extractor,
    predict_residual,
)
from harvester.v50.residual_extractor_model import RESIDUAL_GRID
from harvester.v50.residual_extractor_train import (
    CURRICULUM_SCHEMA,
    REQUIRED_ARRAYS,
    validate_curriculum_contract,
    TrainerContractError,
)
from harvester.v50.residual_height_model import (
    TARGET_CONTRACT_VERSION as HEIGHT_CONTRACT_VERSION,
    encode_relative_height,
    decode_relative_height,
)
from harvester.v50.terrain_lighting_torch import (
    fit_affine_shading_torch,
    render_hillshade_torch,
    shape_loss,
    total_variation_loss,
    sun_vector_torch,
)

CELL_SIZE = 256
HEADER_HEIGHT = 86
ROW_TITLE_HEIGHT = 28
CELL_LABEL_HEIGHT = 32
COLUMN_TITLES = (
    "Target residual",
    "Initial hillshade (flat)",
    "Refined hillshade",
    "Refined heightmap",
)

HEIGHT_CURRICULUM_SCHEMA = "v125-residual-curriculum-v1"


class RefinementError(ValueError):
    """Raised when the refinement loop cannot produce a valid result."""


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _grayscale_rgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _relative_height_rgb(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float32)
    lo = float(values.min())
    hi = float(values.max())
    scale = max(hi - lo, 1.0)
    t = np.clip((values - lo) / scale, 0.0, 1.0)
    stops = np.asarray(
        [[18, 34, 70], [42, 112, 80], [156, 145, 86], [238, 236, 220]],
        dtype=np.float32,
    )
    pos = t * (len(stops) - 1)
    lower = np.floor(pos).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    blend = (pos - lower)[..., None]
    return np.clip(stops[lower] * (1.0 - blend) + stops[upper] * blend, 0, 255).astype(np.uint8)


def _rgb_image(array: np.ndarray, *, size: int = CELL_SIZE) -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def refine_height(
    target_residual: torch.Tensor,
    *,
    iterations: int = 500,
    lr: float = 0.1,
    tv_weight: float = 1e-6,
    shape_weight: float = 0.5,
    spacing: float = 533.333 / 256.0,
    azimuth_deg: float = 45.0,
    elevation_deg: float = 90.0,
    progress_callback=None,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize height to match target residual under the known differentiable forward model.

    Hybrid loss::
        loss = shape_weight * shape_loss(shade, target)
             + (1 - shape_weight) * L1(gain*shade+ambient, target)
             + tv_weight * TV(height)

    The affine fit (gain, ambient) is recomputed each iteration from the current
    rendered shading, so the L1 term drives both the shading pattern AND the
    brightness amplitude.  The shape_loss seeds the pattern early when starting
    from flat ground.

    Returns (height, losses) where height is (1, 1, H, W) in [0, 1].
    """
    device = target_residual.device
    grid = target_residual.shape[-1]

    height = torch.full((1, 1, grid, grid), 0.5, device=device) + (
        torch.rand(1, 1, grid, grid, device=device) - 0.5
    ) * 0.01
    height.requires_grad_(True)

    light_dir = sun_vector_torch(azimuth_deg, elevation_deg, device=device)
    losses: list[float] = []
    optimizer = torch.optim.Adam([height], lr=lr)

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        shade = render_hillshade_torch(height, light_dir, spacing=spacing)
        # Shape loss: scale-and-shift invariant, drives pattern.
        data_shape = shape_loss(shade, target_residual)
        # Affine-fit L1: recomputed each iteration, drives amplitude.
        gain, ambient = fit_affine_shading_torch(shade, target_residual)
        shade_affine = gain * shade + ambient
        data_affine = torch.nn.functional.l1_loss(shade_affine, target_residual)
        data_loss = shape_weight * data_shape + (1.0 - shape_weight) * data_affine
        smoothness = total_variation_loss(height)
        loss = data_loss + tv_weight * smoothness
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_val = float(loss.item())
            losses.append(loss_val)
            height.data.clamp_(0.0, 1.0)
        if progress_callback is not None:
            progress_callback(iteration, loss_val,
                              shape=float(data_shape), affine=float(data_affine),
                              tv=float(smoothness))

    return height.detach(), losses


def render_review_sheet(
    *,
    target_residual: np.ndarray,
    initial_shade: np.ndarray,
    refined_shade: np.ndarray,
    refined_height: np.ndarray,
    title: str,
    output: Path,
) -> None:
    cells = (
        _rgb_image(_grayscale_rgb(target_residual)),
        _rgb_image(_grayscale_rgb(initial_shade)),
        _rgb_image(_grayscale_rgb(refined_shade)),
        _rgb_image(_relative_height_rgb(refined_height)),
    )
    width = len(COLUMN_TITLES) * CELL_SIZE
    block_height = ROW_TITLE_HEIGHT + CELL_SIZE + CELL_LABEL_HEIGHT
    sheet = Image.new("RGB", (width, HEADER_HEIGHT + block_height), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    draw.text((10, 10), "v50 residual-to-height refinement (Spec 126 US7)", fill=(245, 245, 245), font=font)
    draw.text((10, 30), str(output.resolve()), fill=(178, 184, 194), font=font)
    draw.text((10, 50), "Height optimised to match residual through the KNOWN differentiable hillshade.", fill=(255, 210, 110), font=font)
    draw.rectangle((0, HEADER_HEIGHT, width, HEADER_HEIGHT + ROW_TITLE_HEIGHT), fill=(34, 38, 45))
    draw.text((8, HEADER_HEIGHT + 8), title, fill=(230, 232, 236), font=font)
    image_y = HEADER_HEIGHT + ROW_TITLE_HEIGHT
    for column, (cell, label) in enumerate(zip(cells, COLUMN_TITLES)):
        x = column * CELL_SIZE
        sheet.paste(cell, (x, image_y))
        draw.rectangle((x, image_y + CELL_SIZE, x + CELL_SIZE, image_y + CELL_SIZE + CELL_LABEL_HEIGHT), fill=(28, 31, 37))
        draw.text((x + 6, image_y + CELL_SIZE + 10), label, fill=(220, 222, 226), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    ap = argparse.ArgumentParser(description="v50 residual-to-height refinement (Spec 126 US7)")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--input-array", default="minimap_rgb_dxt1")
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tv-weight", type=float, default=1e-6)
    ap.add_argument("--shape-weight", type=float, default=0.3,
                    help="Blend: 1.0 = pure shape_loss, 0.0 = pure affine-fit L1")
    ap.add_argument("--azimuth", type=float, default=45.0)
    ap.add_argument("--elevation", type=float, default=90.0)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")

    try:
        extractor = load_extractor(args.checkpoint, device, args.release)
    except (ValueError, RefinementError) as exc:
        raise SystemExit(str(exc)) from exc

    group = zarr.open_group(str(args.store), mode="r")
    # Accept either the extractor curriculum schema or the height curriculum schema.
    actual_schema = str(group.attrs.get("schema", ""))
    if actual_schema not in (CURRICULUM_SCHEMA, HEIGHT_CURRICULUM_SCHEMA):
        raise SystemExit(
            f"store schema {actual_schema!r} not recognised; expected {CURRICULUM_SCHEMA!r} or "
            f"{HEIGHT_CURRICULUM_SCHEMA!r}"
        )
    try:
        require_store_release(group, args.release, store=args.store, expected_schema=actual_schema)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
    # Required arrays differ by schema: extractor curriculum carries minimap_rgb;
    # height curriculum carries residual_256 + height_257. The shared requirement is residual_256.
    if actual_schema == CURRICULUM_SCHEMA:
        required = {"minimap_rgb", "residual_256"}
    else:
        required = {"residual_256", "height_257"}
    missing = sorted(required - array_lengths.keys())
    if missing:
        raise SystemExit(f"store is missing required arrays {missing}")

    has_height_gt = "height_257" in group
    has_minimap = args.input_array in group or "minimap_rgb" in group
    if not has_minimap and args.input_array not in group and "minimap_rgb" not in group:
        print(f"store has no minimap array — using stored residual_256 directly (no extractor step)", flush=True)
    if has_height_gt:
        print(f"store has height_257 — will report height correlation", flush=True)

    rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) == str(args.val_value)]
    if not rows:
        raise SystemExit(f"no store rows matched {args.val_key}={args.val_value!r}")
    rows = rows[: args.samples]

    spacing = 533.333 / RESIDUAL_GRID
    per_row = []
    # Height curriculum carries residual_256 directly (no minimap_rgb); extractor
    # curriculum carries minimap_rgb we must run the extractor on.
    store_residual_np = np.asarray(group["residual_256"][0], dtype=np.float32) / 255.0
    if store_residual_np.ndim == 3:
        store_residual_np = store_residual_np[..., 0]
    has_minimap = args.input_array in group
    if not has_minimap:
        print("store has no minimap_rgb — using stored residual_256 directly (no extractor step)", flush=True)

    for row in rows:
        metadata = index[row]
        target_residual_np = np.asarray(group["residual_256"][row], dtype=np.float32) / 255.0
        if target_residual_np.ndim == 3:
            target_residual_np = target_residual_np[..., 0]

        if has_minimap:
            rgb = np.asarray(group[args.input_array][row], dtype=np.uint8)
            predicted_residual_np = predict_residual(extractor, rgb, device)
            use_residual = predicted_residual_np
        else:
            use_residual = target_residual_np
        target_t = torch.from_numpy(use_residual[None, None]).to(device)

        # Load ground-truth height if available.
        height_gt_np = None
        height_gt_relative = None
        if has_height_gt:
            raw_h = np.asarray(group["height_257"][row], dtype=np.float32)
            height_gt_relative, _, _ = encode_relative_height(raw_h)

        # Initial hillshade from flat height, with affine fit.
        with torch.no_grad():
            height_flat = torch.full((1, 1, RESIDUAL_GRID, RESIDUAL_GRID), 0.5, device=device)
            flat_shade = render_hillshade_torch(height_flat, spacing=spacing,
                                                azimuth_deg=args.azimuth, elevation_deg=args.elevation)
            flat_gain, flat_ambient = fit_affine_shading_torch(flat_shade, target_t)
            initial_shade_affine = (flat_gain * flat_shade + flat_ambient).clamp(0, 1)
            initial_shade_np = initial_shade_affine[0, 0].cpu().numpy()
            initial_mae = float(np.abs(initial_shade_np - use_residual).mean())

        # Refine.
        print(f"[row {row:04d}] init_mae={initial_mae:.4f} refining...", end=" ", flush=True)

        def _log(i: int, loss: float, **kwargs) -> None:
            if i % 100 == 0 or i == args.iterations - 1:
                parts = [f"iter={i} loss={loss:.6f}"]
                for k, v in kwargs.items():
                    parts.append(f"{k}={v:.6f}")
                print(" ".join(parts), end=" ", flush=True)

        refined_height, losses = refine_height(
            target_t,
            iterations=args.iterations, lr=args.lr, tv_weight=args.tv_weight,
            shape_weight=args.shape_weight,
            spacing=spacing, azimuth_deg=args.azimuth, elevation_deg=args.elevation,
            progress_callback=_log,
        )

        # Render refined with affine fit.
        with torch.no_grad():
            refined_shade_raw = render_hillshade_torch(
                refined_height, spacing=spacing,
                azimuth_deg=args.azimuth, elevation_deg=args.elevation,
            )
            ref_gain, ref_ambient = fit_affine_shading_torch(refined_shade_raw, target_t)
            refined_shade_affine = (ref_gain * refined_shade_raw + ref_ambient).clamp(0, 1)
            refined_shade_np = refined_shade_affine[0, 0].cpu().numpy()
            refined_height_np = refined_height[0, 0].cpu().numpy()

        final_mae = float(np.abs(refined_shade_np - use_residual).mean())
        mae_change = (final_mae - initial_mae) / max(initial_mae, 1e-8) * 100
        direction = "BETTER" if mae_change < 0 else "WORSE"

        # Height correlation against ground truth.
        height_r = None
        height_mae = None
        if height_gt_relative is not None:
            height_r = _pearson(refined_height_np, height_gt_relative)
            height_mae = float(np.abs(refined_height_np - height_gt_relative).mean())
            print(f"final_mae={final_mae:.4f} ({direction} {abs(mae_change):.1f}%) "
                  f"height_r={height_r:.4f} height_mae={height_mae:.4f}", flush=True)
        else:
            print(f"final_mae={final_mae:.4f} ({direction} {abs(mae_change):.1f}%)", flush=True)

        title = (
            f"row={row} map={metadata.get('map')} tile={metadata.get('tile_x')},{metadata.get('tile_y')} "
            f"regime={metadata.get('height_regime','?')}  "
            f"init_MAE={initial_mae:.4f} final_MAE={final_mae:.4f} ({direction})"
            + (f"  height_r={height_r:.4f}" if height_r is not None else "")
        )
        output = args.output_dir / f"row-{row:04d}-refine.png"
        render_review_sheet(
            target_residual=use_residual,
            initial_shade=initial_shade_np,
            refined_shade=refined_shade_np,
            refined_height=refined_height_np,
            title=title, output=output,
        )
        entry = {
            "row": row, "map": str(metadata.get("map", "")),
            "tile_x": int(metadata.get("tile_x", -1)),
            "tile_y": int(metadata.get("tile_y", -1)),
            "regime": str(metadata.get("height_regime", "")),
            "initial_mae": initial_mae, "final_mae": final_mae,
            "mae_change_pct": mae_change, "iterations": len(losses),
            "image": str(output.resolve()),
        }
        if height_r is not None:
            entry["height_pearson_r"] = height_r
            entry["height_mae"] = height_mae
        per_row.append(entry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v50-residual-height-refinement-v1",
        **release_identity(args.release),
        "checkpoint": str(args.checkpoint.resolve()),
        "store": str(args.store.resolve()),
        "input_array": args.input_array,
        "refinement": {"iterations": args.iterations, "lr": args.lr, "tv_weight": args.tv_weight,
                        "shape_weight": args.shape_weight,
                        "azimuth_deg": args.azimuth, "elevation_deg": args.elevation},
        "rows": per_row,
        "height_ground_truth_available": has_height_gt,
    }
    (args.output_dir / "refine-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] {len(per_row)} tiles -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())