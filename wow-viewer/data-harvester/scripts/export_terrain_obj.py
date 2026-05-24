"""Export a minimap image through the normal/height model and produce OBJ + MTL + texture PNG.

Usage:
  python export_terrain_obj.py --image path/to/minimap.png --height-checkpoint path/to/best.pt --output-dir ./out
  python export_terrain_obj.py --image path/to/minimap.png --normal-checkpoint path/to/best.pt --output-dir ./out --height-channel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v16_1_models import V161HeightModel, V161NormalHeightCombinedModel, V161NormalHeightModel, V161NormalModel  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_cls: type[torch.nn.Module], path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = model_cls().to(device)
    model_state = model.state_dict()
    ckpt_has = any(k.startswith("_orig_mod.") for k in state)
    model_has = any(k.startswith("_orig_mod.") for k in model_state)
    if ckpt_has and not model_has:
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    elif model_has and not ckpt_has:
        state = {f"_orig_mod.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _height_to_obj(
    height: np.ndarray,
    texture_path: Path,
    obj_path: Path,
    tile_size: float = 533.333,
    height_scale: float = 1.0,
) -> None:
    """Write an OBJ file from a 257x257 heightmap with texture coordinates."""
    h, w = height.shape
    rows = h - 1
    cols = w - 1

    mtl_name = obj_path.stem + ".mtl"
    obj_lines: list[str] = []
    obj_lines.append(f"mtllib {mtl_name}")
    obj_lines.append(f"usemtl terrain")
    obj_lines.append("")

    # vertices
    for y in range(h):
        for x in range(w):
            wx = (x / cols) * tile_size
            wy = (y / rows) * tile_size
            wz = float(height[y, x]) * height_scale
            obj_lines.append(f"v {wx:.6f} {wy:.6f} {wz:.6f}")

    obj_lines.append("")

    # texture coordinates
    for y in range(h):
        for x in range(w):
            u = x / cols
            v = 1.0 - (y / rows)
            obj_lines.append(f"vt {u:.6f} {v:.6f}")

    obj_lines.append("")

    # faces (two triangles per grid cell)
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            obj_lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
            obj_lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")

    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")

    # MTL file
    mtl_path = obj_path.parent / mtl_name
    mtl_lines = [
        "newmtl terrain",
        f"map_Kd {texture_path.name}",
        "Ka 1.0 1.0 1.0",
        "Kd 1.0 1.0 1.0",
        "Ns 0.0",
        "d 1.0",
    ]
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export minimap image through model to OBJ + MTL + texture")
    p.add_argument("--image", type=Path, required=True, help="Path to minimap image (PNG/JPG, square preferred)")
    p.add_argument("--normal-checkpoint", type=Path, default=None)
    p.add_argument("--height-checkpoint", type=Path, default=None)
    p.add_argument("--combined-checkpoint", type=Path, default=None, help="V16.1.4 combined model checkpoint")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for OBJ/MTL/PNG")
    p.add_argument("--height-scale", type=float, default=1.0, help="Height multiplier for vertex Z")
    p.add_argument("--tile-size", type=float, default=533.333, help="World-space tile size for X/Y")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--height-channel", action="store_true", help="Use height-channel normal model (4ch input)")
    p.add_argument("--height-mean", type=float, default=0.0, help="Height normalization mean (if using height-channel model)")
    p.add_argument("--height-std", type=float, default=1.0, help="Height normalization std (if using height-channel model)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    inp = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    height_pred = None

    if args.height_checkpoint:
        h_model = _load_model(V161HeightModel, args.height_checkpoint.resolve(), device)
        with torch.no_grad():
            h_norm = h_model(inp).squeeze().cpu().numpy()
        # denormalize to world-space
        height_pred = h_norm * (args.height_std + 1e-8) + args.height_mean

    if args.combined_checkpoint:
        c_model = _load_model(V161NormalHeightCombinedModel, args.combined_checkpoint.resolve(), device)
        gray = img_np.mean(axis=2)
        h_norm_ch = ((gray - args.height_mean) / (args.height_std + 1e-8))[:256, :256].astype(np.float32)
        inp_4ch = torch.cat([inp, torch.from_numpy(h_norm_ch).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
        with torch.no_grad():
            _, h_raw = c_model(inp_4ch)
            h_norm = h_raw.squeeze().cpu().numpy()
        height_pred = h_norm * (args.height_std + 1e-8) + args.height_mean

    if args.normal_checkpoint:
        if args.height_channel:
            n_model = _load_model(V161NormalHeightModel, args.normal_checkpoint.resolve(), device)
            # build 4th channel: height_norm from the grayscale minimap as rough height proxy
            gray = img_np.mean(axis=2)
            h_norm_ch = ((gray - args.height_mean) / (args.height_std + 1e-8))[:256, :256].astype(np.float32)
            inp_4ch = torch.cat([inp, torch.from_numpy(h_norm_ch).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
            with torch.no_grad():
                n = n_model(inp_4ch).squeeze().cpu().numpy()
        else:
            n_model = _load_model(V161NormalModel, args.normal_checkpoint.resolve(), device)
            with torch.no_grad():
                n = n_model(inp).squeeze().cpu().numpy()

        if height_pred is None:
            # derive height from normals via cumulative integration
            nx = n[0]
            ny = n[1]
            nz = n[2].clip(min=0.1)
            cell = args.tile_size / 256.0
            # slopes: dh/dx = -nx/nz, dh/dy = -ny/nz
            slope_x = (-nx / nz).astype(np.float64)
            slope_y = (-ny / nz).astype(np.float64)
            # integrate rows (x direction)
            row_int = np.cumsum(slope_x, axis=1) * cell
            # integrate columns (y direction) from the row-integrated base
            col_int = np.cumsum(slope_y, axis=0) * cell
            # combine: average the two independent integrations
            h257 = ((row_int + col_int) * 0.5).astype(np.float32)
            # center around zero
            h257 -= h257.mean()
            height_pred = h257

    if height_pred is None:
        print("Error: provide at least --height-checkpoint or --normal-checkpoint")
        sys.exit(1)

    # resize height to 257x257 if needed
    if height_pred.shape != (257, 257):
        from PIL import Image as PILImage
        h_img = PILImage.fromarray(height_pred).resize((257, 257), PILImage.Resampling.BILINEAR)
        height_pred = np.array(h_img).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # save texture
    tex_path = args.output_dir / "texture.png"
    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_resized.save(tex_path)

    # write OBJ
    obj_path = args.output_dir / "terrain.obj"
    tile_size = args.tile_size
    _height_to_obj(height_pred, tex_path.relative_to(args.output_dir), obj_path, tile_size=tile_size, height_scale=args.height_scale)

    print(f"Exported to {args.output_dir}")
    print(f"  terrain.obj ({(257*257)} vertices, {(256*256*2)} faces)")
    print(f"  terrain.mtl")
    print(f"  texture.png (256x256)")


if __name__ == "__main__":
    main()
