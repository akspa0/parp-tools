"""Test D1 model — show predicted alphas vs MCAL ground truth + layer signatures.

Usage:
    uv run python scripts/test_d1.py
    uv run python scripts/test_d1.py --npz-path <path_with_texture_pixels>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from harvester.d1_model import D1UNet  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "d1_best.pt"
DEFAULT_NPZ = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "datasets" / "d1_test" / "tex_test.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _save_layer(arr: np.ndarray, path: Path) -> None:
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 2:
        img = (arr * 255).astype(np.uint8)
        Image.fromarray(img, "L").save(path)
    else:
        img = (arr * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = D1UNet()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with np.load(args.npz_path, allow_pickle=False) as data:
        minimap = data["minimap_rgb_256"].astype(np.float32) / 255.0
        alpha_gt_pack = data["mcal_alpha_pack_256"].astype(np.float32)
        if alpha_gt_pack.max() > 1.5:
            alpha_gt_pack /= 255.0
        alpha_gt_pack = alpha_gt_pack.clip(0, 1)
        tex_ids = data.get("mcly_texture_ids", None)

        # Load texture swatches if present
        tex_pixels = {}
        tex_names = []
        meta = json.loads(bytes(data["metadata.json"]).decode("utf-8"))
        for entry in meta.get("mcly_texture_name_table", []):
            tex_names.append(entry["path"])
        for i in range(len(tex_names)):
            key = f"mcly_texture_pixels_{i}"
            if key in data:
                tex_pixels[i] = (data[key].astype(np.float32) / 255.0).clip(0, 1)

    inp = torch.from_numpy(minimap.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_t1, pred_t2, pred_a1, pred_a2 = model(inp)

    tileset_1 = pred_t1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tileset_2 = pred_t2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    alpha_1 = pred_a1.squeeze(0).squeeze(0).cpu().numpy()
    alpha_2 = pred_a2.squeeze(0).squeeze(0).cpu().numpy()

    tile_name = args.npz_path.stem.replace("_harvest", "")
    out_dir = args.output_dir or args.npz_path.parent / f"{tile_name}_d1_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Row A: input + alpha comparison
    _save_layer(minimap, out_dir / "A_input_minimap.png")
    _save_layer(alpha_gt_pack[:, :, 0], out_dir / "B_alpha1_gt.png")
    _save_layer(alpha_1, out_dir / "C_alpha1_pred.png")
    _save_layer(alpha_gt_pack[:, :, 1], out_dir / "D_alpha2_gt.png")
    _save_layer(alpha_2, out_dir / "E_alpha2_pred.png")

    # Row B: predicted tileset layer signatures
    _save_layer(tileset_1, out_dir / "F_tileset_layer1.png")
    _save_layer(tileset_2, out_dir / "G_tileset_layer2.png")

    # Row C: texture swatches from the shard (what the layers SHOULD match)
    for idx, pixels in tex_pixels.items():
        name = Path(tex_names[idx]).stem
        _save_layer(pixels, out_dir / f"tex_{idx:02d}_{name}.png")

    # Metrics
    l1_a1 = float(np.abs(alpha_1 - alpha_gt_pack[:, :, 0]).mean())
    l1_a2 = float(np.abs(alpha_2 - alpha_gt_pack[:, :, 1]).mean())
    metrics = {
        "tile": tile_name,
        "epoch": ckpt["epoch"],
        "alpha_l1": {"a1": l1_a1, "a2": l1_a2},
        "num_textures": len(tex_pixels),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {len(list(out_dir.iterdir()))} files -> {out_dir}")
    print(f"alpha L1: a1={l1_a1:.4f}  a2={l1_a2:.4f}")
    print(f"texture swatches: {len(tex_pixels)}")


if __name__ == "__main__":
    main()
