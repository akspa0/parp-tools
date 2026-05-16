"""Test V15 terrain model — run inference on one tile, save per-layer PNGs.

Usage:
    uv run python scripts/test_v15.py
    uv run python scripts/test_v15.py --npz-path <path> --checkpoint checkpoints/v15_best.pt
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

from harvester.v15_model import V15Model  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "v15_best.pt"
DEFAULT_NPZ = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "datasets" / "d1_reharvest" / "shards"
    / "3_0_1_8303" / "Azeroth" / "Azeroth_32_32_harvest.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test V15 terrain model")
    p.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _save(arr: np.ndarray, path: Path) -> None:
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 2:
        img = (arr * 255).astype(np.uint8)
        Image.fromarray(img, "L").save(path)
    else:
        img = (arr * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(path)


def _export_obj(height_257: np.ndarray, path: Path) -> None:
    """Export a 257×257 heightmap as a triangulated OBJ mesh."""
    h, w = height_257.shape
    assert h == 257 and w == 257, f"Expected 257×257, got {h}×{w}"
    verts = []
    faces = []
    for y in range(h):
        for x in range(w):
            verts.append(f"v {x * 1.0:.4f} {y * 1.0:.4f} {height_257[y, x]:.4f}")
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            faces.append(f"f {i + 1} {i + 2} {i + w + 2}")
            faces.append(f"f {i + 1} {i + w + 2} {i + w + 1}")
    with open(path, "w") as f:
        f.write("# V15 terrain mesh\n")
        f.write("\n".join(verts))
        f.write("\n")
        f.write("\n".join(faces))
        f.write("\n")
    print(f"OBJ: {path} ({len(verts)} verts, {len(faces)} faces)")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = V15Model()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with np.load(args.npz_path, allow_pickle=False) as data:
        minimap = data["minimap_rgb_256"].astype(np.float32) / 255.0
        height_gt = data["height_257"].astype(np.float32)
        has_n = "mcnr_normal_xyz" in data
        nrm_gt = data["mcnr_normal_xyz"].astype(np.float32) if has_n else np.zeros((257, 257, 3))
        has_a = "mcal_alpha_pack_256" in data or "mcal_alpha_pack" in data
        a_key = "mcal_alpha_pack_256" if "mcal_alpha_pack_256" in data else "mcal_alpha_pack"
        alp_gt = data[a_key].astype(np.float32) if has_a else np.zeros((256, 256, 4))
        if has_a and alp_gt.shape[0] != 256:
            factor = alp_gt.shape[0] // 256
            nshape = (256, factor, 256, factor, 4)
            alp_gt = alp_gt.reshape(nshape).mean(axis=1).mean(axis=2)
        if has_a and alp_gt.max() > 1.5:
            alp_gt /= 255.0
        has_ho = "hole_mask_16" in data
        hol_gt = data["hole_mask_16"].astype(np.float32) if has_ho else np.zeros((16, 16))
        has_obj = "object_mask_257" in data
        obj_gt = data["object_mask_257"][:256, :256] if has_obj else np.zeros((256, 256))

    # Normalise height (match training)
    h_mean = height_gt.mean()
    h_std = height_gt.std() + 1e-8

    inp = torch.from_numpy(minimap.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_h_raw, pred_n, pred_a, pred_ho = model(inp)

    # Denormalise height to world-space meters
    pred_h = (pred_h_raw.squeeze().cpu().numpy() * h_std) + h_mean
    pred_n = pred_n.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_a = pred_a.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_ho = pred_ho.squeeze().cpu().numpy()

    # Normalise for visualisation
    h_min, h_max = pred_h.min(), pred_h.max()
    h_viz = (pred_h - h_min) / (h_max - h_min + 1e-8)
    n_viz = (pred_n + 1) / 2  # [-1,1] → [0,1]
    gt_h_min, gt_h_max = height_gt.min(), height_gt.max()
    gt_h_viz = (height_gt - gt_h_min) / (gt_h_max - gt_h_min + 1e-8)
    gt_n_viz = (nrm_gt + 1) / 2

    tile = args.npz_path.stem.replace("_harvest", "")
    out = args.output_dir or args.npz_path.parent / f"{tile}_v15_test"
    out.mkdir(parents=True, exist_ok=True)

    _save(minimap, out / "input_minimap.png")
    _save(gt_h_viz, out / "height_gt.png")
    _save(h_viz, out / "height_pred.png")
    _save(gt_n_viz, out / "normals_gt.png")
    _save(n_viz, out / "normals_pred.png")
    for ch in range(4):
        _save(alp_gt[:, :, ch], out / f"alpha_gt_ch{ch}.png")
        _save(pred_a[:, :, ch], out / f"alpha_pred_ch{ch}.png")
    _save(hol_gt, out / "holes_gt.png")
    _save(pred_ho, out / "holes_pred.png")
    _save(obj_gt, out / "object_mask.png")

    metrics = {
        "tile": tile,
        "epoch": ckpt["epoch"],
        "height_l1_m": float(np.abs(pred_h - height_gt).mean()),
        "normals_l1": float(np.abs(pred_n - nrm_gt).mean()) if has_n else None,
        "alpha_l1": float(np.abs(pred_a - alp_gt).mean()) if has_a else None,
        "holes_l1": float(np.abs(pred_ho - hol_gt).mean()) if has_ho else None,
        "height_stats": {"mean": float(h_mean), "std": float(h_std)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Export OBJ mesh from predicted heightmap
    _export_obj(pred_h, out / f"{tile}_terrain.obj")

    print(f"Output: {out}")
    print(f"Height L1: {metrics['height_l1_m']:.2f}m")
    if has_n:
        print(f"Normals L1: {metrics['normals_l1']:.4f}")
    if has_a:
        print(f"Alpha L1: {metrics['alpha_l1']:.4f}")
    if has_ho:
        print(f"Holes L1: {metrics['holes_l1']:.4f}")


if __name__ == "__main__":
    main()
