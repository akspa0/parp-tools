#!/usr/bin/env python3
"""
WoW V7.5 Lite - Data Cacher
Pre-processes dataset into tensors for rapid CPU training.
Features:
- Resolution: 256x256 (Downscaled from 512)
- Rotates Inputs -90 Degrees (Fix)
- Ignores Water Mask
- Computes Synthetic WDL (16x16) for Residual Learning
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[4]  # scripts/→WoWMapConverter/→src/→gillijimproject_refactor/→parp-tools/
_DATASETS_ROOT = _WORKSPACE_ROOT / "datasets"

OUTPUT_DIR_DEFAULT = _WORKSPACE_ROOT / "output" / "cache_v751"
CACHE_FILE_TRAIN = "train_cache.pt"
CACHE_FILE_VAL = "val_cache.pt"

TILE_SIZE = 533.33333

INPUT_SIZE = 256
OUTPUT_SIZE = 256
VAL_SPLIT = 0.1

# Global Range used for normalization
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN


# ---------------------------------------------------------------------------
# Dataset discovery (mirrors cache_v7_6_data.py)
# ---------------------------------------------------------------------------

def discover_dataset_roots(search_roots: list) -> list:
    discovered = []
    seen = set()
    for root_text in search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        if (root / "dataset").exists() and (root / "images").exists():
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(root)
        for manifest_path in sorted(root.rglob("ml_dataset_manifest.json")):
            candidate = manifest_path.parent
            if not (candidate / "dataset").exists():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)
    return discovered


def parse_tile_coords(tile_name: str):
    import re
    m = re.search(r"_(\d+)_(\d+)$", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def load_heightmap(path, size):
    img = Image.open(path)
    if img.mode == 'I;16':
        arr = np.array(img, dtype=np.float32) / 65535.0
    elif img.mode == 'I':
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0

    if arr.shape[0] != size:
        from scipy.ndimage import zoom
        scale = size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
    return torch.from_numpy(arr).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def scan_samples(dataset_roots: list) -> list:
    samples = []
    print("Scanning datasets …")
    for root in dataset_roots:
        root = Path(root)
        if not root.exists():
            continue
        ds_dir = root / "dataset"
        for json_path in ds_dir.glob("*.json"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                td = data.get("terrain_data", {})

                hm_g = td.get("heightmap_global") or td.get("heightmap")
                hm_l = td.get("heightmap_local") or td.get("heightmap")
                nm = td.get("normalmap")

                if not (hm_g and hm_l and nm):
                    continue

                hm_g_path = root / hm_g
                hm_l_path = root / hm_l
                nm_path = root / nm
                mm_path = root / "images" / f"{json_path.stem}.png"

                if not (hm_g_path.exists() and hm_l_path.exists() and nm_path.exists() and mm_path.exists()):
                    continue

                samples.append({
                    "root": root,
                    "json": json_path,
                    "mm": mm_path,
                    "nm": nm_path,
                    "hm_g": hm_g_path,
                    "hm_l": hm_l_path,
                    "td": td,
                })
            except Exception:
                continue
    return samples


def process_batch(sample_list, output_path):
    print(f"Processing {len(sample_list)} samples → {output_path} …")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_list = []
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for s in tqdm(sample_list):
        try:
            # 1. Inputs (Minimap, Normal)
            mm = Image.open(s["mm"]).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            nm = Image.open(s["nm"]).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

            # FIX: Rotate Inputs -90
            mm = mm.rotate(-90)
            nm = nm.rotate(-90)

            mm_t = norm(to_tensor(mm))
            nm_t = norm(to_tensor(nm))

            # 2. Targets (Height Global, Local)
            hm_g = load_heightmap(s["hm_g"], OUTPUT_SIZE)
            hm_l = load_heightmap(s["hm_l"], OUTPUT_SIZE)

            # 3. Synthetic WDL (Downsample → 16×16, Upsample)
            wdl_low = torch.nn.functional.interpolate(hm_g.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            wdl_upscaled = torch.nn.functional.interpolate(wdl_low, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False).squeeze(0)

            # RESIDUAL: Target = Height − WDL
            target_residual = hm_g - wdl_upscaled

            # 4. Metadata Channels (H_Min, H_Max)
            td = s["td"]
            g_min = td.get("height_global_min", HEIGHT_GLOBAL_MIN)
            g_max = td.get("height_global_max", HEIGHT_GLOBAL_MAX)
            min_n = (g_min - HEIGHT_GLOBAL_MIN) / RANGE
            max_n = (g_max - HEIGHT_GLOBAL_MIN) / RANGE

            h_min_mask = torch.full((1, INPUT_SIZE, INPUT_SIZE), min_n, dtype=torch.float32)
            h_max_mask = torch.full((1, INPUT_SIZE, INPUT_SIZE), max_n, dtype=torch.float32)

            # 5. Object Mask — prefer generated mask PNG, fall back to simple dot rendering
            obj_mask = torch.zeros((1, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
            obj_mask_rel = td.get("object_visibility_mask") or ""
            if obj_mask_rel:
                mask_path = s["root"] / str(obj_mask_rel)
                if mask_path.exists():
                    try:
                        with Image.open(mask_path).convert("L") as mask_img:
                            if mask_img.size != (INPUT_SIZE, INPUT_SIZE):
                                mask_img = mask_img.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
                            obj_mask = torch.from_numpy(
                                (np.asarray(mask_img, dtype=np.float32) / 255.0)
                            ).unsqueeze(0)
                    except Exception:
                        pass

            if obj_mask.sum() == 0:
                # Fallback: simple dot per object
                objects = td.get("objects") or []
                for obj in objects:
                    px = float(obj.get("x") or obj.get("pos_x") or 0)
                    pz = float(obj.get("z") or obj.get("pos_z") or 0)
                    coords = parse_tile_coords(s["json"].stem)
                    if coords:
                        tx, ty = coords
                        nu = (px / TILE_SIZE - tx)
                        nv = (pz / TILE_SIZE - ty)
                        nx = int(nu * INPUT_SIZE) % INPUT_SIZE
                        ny = int(nv * INPUT_SIZE) % INPUT_SIZE
                        y1, y2 = max(0, ny - 1), min(INPUT_SIZE, ny + 2)
                        x1, x2 = max(0, nx - 1), min(INPUT_SIZE, nx + 2)
                        obj_mask[0, y1:y2, x1:x2] = 1.0

            # Pack Input: [MM(3), NM(3), WDL(1), Min(1), Max(1), Obj(1)] = 10 channels
            input_t = torch.cat([mm_t, nm_t, wdl_upscaled, h_min_mask, h_max_mask, obj_mask], dim=0)

            # Pack Target: [Height_Global(1), Height_Local(1), Residual(1)]
            target_t = torch.cat([hm_g, hm_l, target_residual], dim=0)

            data_list.append((input_t.half(), target_t.half()))

        except Exception:
            continue

    print(f"Saving {len(data_list)} samples to {output_path} …")
    torch.save(data_list, output_path)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Cache V7.5 dataset tiles into pre-computed tensors.")
    p.add_argument("dataset_roots", nargs="*",
                   help="Explicit dataset roots to process. Omit to auto-discover from --search-root.")
    p.add_argument("--search-root", action="append", default=None,
                   help="Discovery root. Default: datasets/. Repeat to add more.")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT),
                   help=f"Output directory. Default: {OUTPUT_DIR_DEFAULT}")
    p.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_roots = args.search_root or [str(_DATASETS_ROOT)]
    if args.dataset_roots:
        roots = [Path(r) for r in args.dataset_roots]
    else:
        roots = discover_dataset_roots(search_roots)
        if not roots:
            raise SystemExit("No dataset roots found. Use --search-root or pass explicit roots.")

    print(f"Found {len(roots)} dataset root(s).")
    samples = scan_samples(roots)
    print(f"Found {len(samples)} valid samples.")

    if args.limit is not None:
        samples = samples[:args.limit]

    np.random.shuffle(samples)
    split_idx = int(len(samples) * (1 - VAL_SPLIT))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    process_batch(train_samples, output_dir / CACHE_FILE_TRAIN)
    process_batch(val_samples, output_dir / CACHE_FILE_VAL)


if __name__ == "__main__":
    main()

