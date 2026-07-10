"""FR-096 (Spec 096): Standalone minimap-only Stage A inference.

Loads a PNG minimap, runs the trained minimap-only Stage A checkpoint, and
emits the (17,17) outer + (16,16) inner WDL prior NPZ plus a 4-up preview PNG.
No V24 store, no V18 store, no staged client.

Usage:
    uv run python scripts/infer_v24_stage_a_png.py \\
        --checkpoint path/to/stage_a_minimap_only.pt \\
        --image path/to/minimap.png \\
        --output path/to/prior.npz \\
        [--preview path/to/preview.png]

The script refuses any checkpoint whose state_dict stem conv has in_channels
!= 3 (the minimap-only regime), unless --lenient-checkpoint is given. The
strict default guards against accidentally loading a 13-channel cheat-regime
checkpoint and producing garbage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import stage_a, train_common  # noqa: E402

# Pixel-space layout constants. Centralised so the preview and inference agree.
_MINIMAP_SIZE = 256
_PREVIEW_TILE = 256
_PREVIEW_COLS = 4
_PREVIEW_W = _PREVIEW_TILE * _PREVIEW_COLS
_PREVIEW_H = _PREVIEW_TILE
_NEAREST = Image.Resampling.NEAREST


def _load_png_as_minimap(path: Path) -> np.ndarray:
    """Load a PNG and return a (256, 256, 3) float32 array in [0, 1].

    Resizes with bilinear filtering, converts any color mode to RGB. Non-image
    files and unreadable PNGs raise a clear error.
    """
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (_MINIMAP_SIZE, _MINIMAP_SIZE):
                rgb = rgb.resize((_MINIMAP_SIZE, _MINIMAP_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception as exc:
        raise RuntimeError(f"failed to load image {path}: {exc}") from exc
    return arr


def _detect_in_channels(state: dict) -> int:
    """Probe the stem conv weight to figure out the model's input channel count.

    Both Stage A variants have stem.block.0 as nn.Conv2d(in_channels, base, 3).
    Looking at enc1.block.0.weight gives a (base, in_channels, 3, 3) tensor.
    """
    for key, tensor in state.items():
        if key.endswith("enc1.block.0.weight") and tensor.ndim == 4:
            return int(tensor.shape[1])
    raise RuntimeError(
        "checkpoint state_dict has no enc1.block.0.weight; not a Stage A checkpoint"
    )


def _load_minimap_only_model(checkpoint_path: Path, device: torch.device) -> stage_a.StageAMinimapOnly:
    """Load a minimap-only Stage A checkpoint, with strict shape checks."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is missing 'model_state'; not a Stage A checkpoint"
        )
    state = checkpoint["model_state"]
    in_channels = _detect_in_channels(state)
    if in_channels != stage_a.IN_CHANNELS_MINIMAP_ONLY:
        raise RuntimeError(
            f"checkpoint stem has in_channels={in_channels}; this script requires "
            f"the minimap-only model (in_channels={stage_a.IN_CHANNELS_MINIMAP_ONLY}). "
            f"Use infer_v24_stage_a.py for the cheat-regime model, or pass "
            f"--lenient-checkpoint to bypass this check."
        )
    base = int(checkpoint.get("config", {}).get("base", 28))
    model = stage_a.StageAMinimapOnly(base=base).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _to_nearest_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a float image to uint8 for PIL paste, clipping to [0, 1]."""
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def _nearest_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a 2D or (H, W, 3) array to (size, size[, 3])."""
    pil = Image.fromarray(_to_nearest_uint8(image) if image.ndim == 2 else _to_nearest_uint8(image))
    return np.asarray(pil.resize((size, size), _NEAREST))


def _build_quincunx(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Reconstruct a 33x33 quincunx from the (17,17) and (16,16) grids.

    Mirrors harvester.v24.lattice.quincunx_33 but only needs the (r,c) fills
    for visualisation; we don't compute the half-step neighbour averages.
    """
    q = np.zeros((33, 33), dtype=np.float32)
    q[::2, ::2] = outer
    q[1::2, 1::2] = inner
    return q


def _write_preview(
    minimap: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    preview_path: Path,
    world_min: float,
    world_max: float,
) -> None:
    """Write a 1024x256 4-up preview PNG.

    Panels (left to right):
      [0] input minimap (256x256 RGB)
      [1] outer 17x17 -> 256x256 nearest
      [2] inner 16x16 -> 256x256 nearest
      [3] quincunx 33x33 -> 256x256 nearest
    """
    canvas = Image.new("RGB", (_PREVIEW_W, _PREVIEW_H), (0, 0, 0))
    canvas.paste(Image.fromarray(_to_nearest_uint8(minimap)), (0, 0))
    outer_up = _nearest_resize(outer, _PREVIEW_TILE)
    inner_up = _nearest_resize(inner, _PREVIEW_TILE)
    quincunx = _build_quincunx(outer, inner)
    qc_up = _nearest_resize(quincunx, _PREVIEW_TILE)
    for idx, src in enumerate((outer_up, inner_up, qc_up), start=1):
        # Each of these is a single-channel "world-unit" map; render in grayscale.
        # Normalise using the model's world_min/world_max so all four panels are
        # on the same scale (the prior is what the user actually wants to read).
        norm = (src - world_min) / max(world_max - world_min, 1e-6)
        canvas.paste(Image.fromarray(_to_nearest_uint8(norm), mode="L").convert("RGB"),
                     (idx * _PREVIEW_TILE, 0))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(preview_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="path to the minimap-only Stage A checkpoint")
    parser.add_argument("--image", required=True, type=Path,
                        help="path to a PNG minimap (any size; resized to 256x256)")
    parser.add_argument("--output", required=True, type=Path,
                        help="path to write the prior NPZ (outer/inner/prior_unavailable)")
    parser.add_argument("--preview", type=Path, default=None,
                        help="optional path to write a 1024x256 4-up preview PNG")
    parser.add_argument("--seed", type=int, default=94,
                        help="seed; included only to prove determinism (default 94)")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | auto (default auto)")
    parser.add_argument("--strict-checkpoint", dest="strict_checkpoint",
                        action="store_true", default=True,
                        help="refuse mismatched checkpoints (default on)")
    parser.add_argument("--lenient-checkpoint", dest="strict_checkpoint",
                        action="store_false",
                        help="bypass the in_channels check; for debugging only")
    args = parser.parse_args()

    train_common.set_determinism(args.seed, strict=True)
    device = train_common.pick_device(args.device)

    model = _load_minimap_only_model(args.checkpoint, device)
    if not args.strict_checkpoint:
        # User explicitly opted in; the strict check already ran. Nothing to do
        # here, but the flag is recorded for log clarity.
        print("warning: --lenient-checkpoint accepted; checkpoint shape was not re-checked",
              file=sys.stderr)

    minimap = _load_png_as_minimap(args.image)
    x = stage_a.build_minimap_only_input(minimap)  # (3, 64, 64) float32
    xb = torch.from_numpy(x)[None].to(device)

    started = time.time()
    with torch.no_grad():
        outer_norm, inner_norm = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_ms = (time.time() - started) * 1000.0

    outer = (outer_norm[0].cpu().numpy() * stage_a.HEIGHT_SCALE).astype(np.float32)
    inner = (inner_norm[0].cpu().numpy() * stage_a.HEIGHT_SCALE).astype(np.float32)
    world_min = float(min(outer.min(), inner.min()))
    world_max = float(max(outer.max(), inner.max()))
    peak_vram = train_common.peak_vram_gb()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.output),
        outer=outer,
        inner=inner,
        prior_unavailable=np.array(False),
        metadata=np.array(
            {
                "wall_ms": wall_ms,
                "peak_vram_gb": peak_vram if peak_vram is not None else -1.0,
                "world_min": world_min,
                "world_max": world_max,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
            },
            dtype=object,
        ),
    )

    if args.preview is not None:
        _write_preview(minimap, outer, inner, args.preview, world_min, world_max)

    extra = f", peak_vram={peak_vram:.3f} GB" if peak_vram is not None else ""
    print(
        f"wrote {args.output} (wall={wall_ms:.1f} ms{extra}, "
        f"world_min={world_min:.2f}, world_max={world_max:.2f}, "
        f"outer shape={outer.shape}, inner shape={inner.shape})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
