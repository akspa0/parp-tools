
from __future__ import annotations

import argparse
import json
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Target Resolution (V7++ Native)
TARGET_RES = 512
CHUNK_RES = 32  # 512 / 16 chunks = 32 pixels per chunk
DEFAULT_OUTPUT_DIR = Path("cached_v7_6")
MANIFEST_FILE = "v76_cache_manifest.json"

def load_image_tensor(path, size=None, grayscale=False):
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to Tensor (C, H, W) normalized 0-1
        tensor = TF.to_tensor(img) 
        
        if grayscale:
             # Force 1 channel
            if tensor.shape[0] > 1:
                tensor = tensor[0:1]
        elif tensor.shape[0] == 4:
            # Drop Alpha for regular images (Minimap/Texture) -> Force RGB
            tensor = tensor[:3]
            
        return tensor
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a V7.6 paired cache from harvested dataset roots.")
    parser.add_argument("dataset_roots", nargs="*", help="Optional dataset roots to process. When omitted, roots are discovered under --search-root.")
    parser.add_argument("--search-root", action="append", default=["datasets"], help="Search root used when dataset roots are omitted. Repeat to add more roots.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output cache directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional global sample limit.")
    return parser.parse_args()


def discover_dataset_roots(search_roots: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
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


def resolve_dataset_roots(args) -> list[Path]:
    if args.dataset_roots:
        return [Path(value) for value in args.dataset_roots]
    roots = discover_dataset_roots(args.search_root)
    if not roots:
        raise SystemExit("No dataset roots were found. Pass explicit dataset roots or use a valid --search-root.")
    return roots


def sanitize_sample_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "sample"


def dataset_root_key(dataset_root: Path) -> str:
    parts = [part for part in dataset_root.parts if part and part != dataset_root.anchor]
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    if parts:
        return parts[-1]
    return dataset_root.name or "dataset"


def resolve_dataset_path(dataset_root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    candidate = dataset_root / relative_path
    if candidate.exists():
        return candidate
    return None


def load_alpha_mask(layer_info, dataset_root: Path):
    alpha_rel_path = layer_info.get("alpha_path")
    if alpha_rel_path:
        alpha_path = resolve_dataset_path(dataset_root, alpha_rel_path)
        if alpha_path is not None:
            tensor = load_image_tensor(alpha_path, size=(CHUNK_RES, CHUNK_RES), grayscale=True)
            if tensor is not None:
                return tensor, "alpha_path"

    alpha_bits = layer_info.get("alpha_bits")
    if isinstance(alpha_bits, list) and alpha_bits:
        count = len(alpha_bits)
        size = int(round(count ** 0.5))
        if size * size == count:
            array = np.asarray(alpha_bits, dtype=np.uint8).reshape(size, size)
            image = Image.fromarray(array, mode="L").resize((CHUNK_RES, CHUNK_RES), Image.Resampling.NEAREST)
            return TF.to_tensor(image), "alpha_bits"

    return None, None


def synthesize_albedo(tile_data, dataset_root: Path):
    # Blank Canvas (3, 512, 512)
    full_albedo = torch.zeros((3, TARGET_RES, TARGET_RES), dtype=torch.float32)
    stats = {
        "chunks_with_layers": 0,
        "textures_missing": 0,
        "layers_missing_alpha": 0,
        "alpha_path_layers": 0,
        "alpha_bits_layers": 0,
        "chunks_written": 0,
    }
    
    # Iterate over chunks (0 to 255)
    # The JSON structure for layers: data['terrain_data']['chunk_layers'] -> list of dicts with 'idx' and 'layers'

    chunk_layers_map = {item['idx']: item['layers'] for item in tile_data.get('terrain_data', {}).get('chunk_layers', [])}
    tilesets_dir = dataset_root / "tilesets"

    for r in range(16):
        for c in range(16):
            chunk_idx = r * 16 + c
            
            # Define pixel window for this chunk in the global 512 map
            y_start, y_end = r * CHUNK_RES, (r + 1) * CHUNK_RES
            x_start, x_end = c * CHUNK_RES, (c + 1) * CHUNK_RES
            
            # Start with Black (or maybe a default texture if layer 0 is missing?)
            # Usually Layer 0 is the base.
            chunk_albedo = torch.zeros((3, CHUNK_RES, CHUNK_RES), dtype=torch.float32)
            
            layers = chunk_layers_map.get(chunk_idx, [])
            
            if not layers:
                # Fill with black or debug color if no layers
                continue
            stats["chunks_with_layers"] += 1

            for i, layer_info in enumerate(layers):
                tex_path_raw = layer_info.get('texture_path', '')
                # Clean path: "Tileset\\Foo\\Bar.blp" -> "Bar.png" check in tilesets dir
                tex_name_blp = Path(tex_path_raw).name
                tex_name_png = tex_name_blp.replace('.blp', '.png').replace('.BLP', '.png')
                
                tex_file = tilesets_dir / tex_name_png
                
                # Load Texture and resize to CHUNK_RES (32x32)
                # We simply load it once. Optimization: Cache loaded textures in memory if they repeat often?
                # For now, just load.
                texture_tensor = load_image_tensor(tex_file, size=(CHUNK_RES, CHUNK_RES))
                
                if texture_tensor is None:
                    # Missing texture, skip or use pink placeholder?
                    # Let's skip to avoid crashing, maybe log
                    stats["textures_missing"] += 1
                    continue
                
                # Handling Alpha
                # Layer 0 usually has no alpha (it's base), effectively alpha=1
                # Subsequent layers have alpha masks
                
                if i == 0:
                    # Base layer, fully opaque override (usually)
                    # But blending logic in WoW is: Lerp(Current, New, Alpha)
                    chunk_albedo = texture_tensor
                else:
                    mask_tensor, alpha_source = load_alpha_mask(layer_info, dataset_root)
                    if mask_tensor is not None:
                        if alpha_source == "alpha_path":
                            stats["alpha_path_layers"] += 1
                        elif alpha_source == "alpha_bits":
                            stats["alpha_bits_layers"] += 1
                        chunk_albedo = chunk_albedo * (1 - mask_tensor) + texture_tensor * mask_tensor
                    else:
                        stats["layers_missing_alpha"] += 1
            
            # Place chunk into full map
            full_albedo[:, y_start:y_end, x_start:x_end] = chunk_albedo
            stats["chunks_written"] += 1

    return full_albedo, stats


def iter_json_files(dataset_root: Path):
    dataset_dir = dataset_root / "dataset"
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*.json"))


def process_dataset_roots(dataset_roots: list[Path], output_dir: Path, limit: int | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    samples_written = 0
    samples_skipped_missing_input = 0
    samples_skipped_missing_height = 0
    used_sample_ids: set[str] = set()

    for dataset_root in dataset_roots:
        json_files = iter_json_files(dataset_root)
        if not json_files:
            print(f"Skipping {dataset_root}: no dataset JSON files found")
            continue

        print(f"Processing {dataset_root} ({len(json_files)} tiles)")
        for json_file in tqdm(json_files, desc=f"cache {dataset_root.name}"):
            if limit is not None and samples_written >= limit:
                break

            try:
                with open(json_file, "r", encoding="utf-8") as handle:
                    tile_data = json.load(handle)
            except Exception as exc:
                print(f"Failed to read {json_file}: {exc}")
                continue

            tile_name = tile_data.get("terrain_data", {}).get("adt_tile") or json_file.stem
            dataset_key = dataset_root_key(dataset_root)
            sample_id = sanitize_sample_id(f"{dataset_key}__{tile_name}")
            if sample_id in used_sample_ids:
                index = 2
                while f"{sample_id}_{index}" in used_sample_ids:
                    index += 1
                sample_id = f"{sample_id}_{index}"
            used_sample_ids.add(sample_id)

            image_rel = tile_data.get("image")
            height_rel = tile_data.get("terrain_data", {}).get("heightmap_global") or tile_data.get("terrain_data", {}).get("heightmap")
            minimap_path = resolve_dataset_path(dataset_root, image_rel)
            height_path = resolve_dataset_path(dataset_root, height_rel)

            if minimap_path is None:
                samples_skipped_missing_input += 1
                continue
            if height_path is None:
                samples_skipped_missing_height += 1
                continue

            minimap = load_image_tensor(minimap_path)
            height = load_image_tensor(height_path, grayscale=True)
            if minimap is None or height is None:
                continue

            if minimap.shape[1:] != (TARGET_RES, TARGET_RES):
                minimap = TF.resize(minimap, [TARGET_RES, TARGET_RES])
            if height.shape[1:] != (TARGET_RES, TARGET_RES):
                height = TF.resize(height, [TARGET_RES, TARGET_RES])

            albedo, albedo_stats = synthesize_albedo(tile_data, dataset_root)

            input_path = output_dir / f"input_{sample_id}.pt"
            height_tensor_path = output_dir / f"target_height_{sample_id}.pt"
            albedo_tensor_path = output_dir / f"target_albedo_{sample_id}.pt"
            torch.save(minimap.half(), input_path)
            torch.save(height.half(), height_tensor_path)
            torch.save(albedo.half(), albedo_tensor_path)

            samples_written += 1
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "dataset_root": str(dataset_root.resolve()),
                    "dataset_key": dataset_key,
                    "tile_name": tile_name,
                    "map_name": tile_name.rsplit("_", 2)[0] if tile_name.count("_") >= 2 else tile_name,
                    "source_tile_json_path": str(json_file.resolve()),
                    "source_minimap_path": str(minimap_path.resolve()),
                    "source_heightmap_path": str(height_path.resolve()),
                    "input_tensor_path": input_path.name,
                    "target_height_tensor_path": height_tensor_path.name,
                    "target_albedo_tensor_path": albedo_tensor_path.name,
                    "albedo_stats": albedo_stats,
                }
            )

        if limit is not None and samples_written >= limit:
            break

    manifest = {
        "schema_version": "wowterrain-v76-cache.v1",
        "generated_at_utc": utc_now_iso(),
        "output_dir": str(output_dir.resolve()),
        "target_resolution": TARGET_RES,
        "dataset_roots": [str(root.resolve()) for root in dataset_roots],
        "coverage": {
            "samples_written": samples_written,
            "samples_skipped_missing_input": samples_skipped_missing_input,
            "samples_skipped_missing_height": samples_skipped_missing_height,
        },
        "entries": manifest_entries,
    }
    manifest_path = output_dir / MANIFEST_FILE
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote cache manifest: {manifest_path}")
    print(f"  samples_written: {samples_written}")
    print(f"  samples_skipped_missing_input: {samples_skipped_missing_input}")
    print(f"  samples_skipped_missing_height: {samples_skipped_missing_height}")

if __name__ == "__main__":
    parsed = parse_args()
    process_dataset_roots(resolve_dataset_roots(parsed), Path(parsed.output_dir), parsed.limit)
