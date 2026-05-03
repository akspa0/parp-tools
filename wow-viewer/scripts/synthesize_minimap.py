"""
Synthetic minimap compositing from MCAL/MCLY + tileset textures.

For a tile with MCAL alpha weights and MCLY texture references:
  1. Load the tileset BLPs for each referenced texture
  2. For each chunk (16x16), blend up to 4 texture layers by MCAL alpha
  3. Downsample to 256x256 → synthetic minimap
  4. Compute residual: real_minimap - synthetic = objects + shadows + detail

Output: NPZ with additional arrays:
  synthetic_minimap_256  - composited texture-only minimap
  texture_residual_256   - what the model should learn to add (detail/objects)
"""
import argparse, json, os, sys, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
# Tileset texture cache
# ═══════════════════════════════════════════════════════════════════════════

class TilesetCache:
    """Loads and caches tileset textures by filename (no path/extension)."""

    def __init__(self, harvest_dir: str):
        self._harvest_dir = Path(harvest_dir)
        self._cache: dict[str, np.ndarray] = {}

    def get(self, texture_name: str) -> np.ndarray | None:
        """Return texture as HWC uint8 array, or None."""
        key = texture_name.replace('\\', '/').strip().lower()
        if key in self._cache:
            return self._cache[key]

        stem = Path(key).stem
        for ext in ('.png', '.blp', '.PNG', '.BLP'):
            for candidate in self._harvest_dir.rglob(f'{stem}{ext}'):
                try:
                    img = Image.open(candidate).convert('RGB')
                    arr = np.asarray(img)
                    self._cache[key] = arr
                    return arr
                except Exception:
                    continue
        self._cache[key] = None
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Compositing
# ═══════════════════════════════════════════════════════════════════════════

def composite_tile_texture(
    mcal_alpha: np.ndarray,
    mcly_ids: np.ndarray,
    texture_names: list[str],
    tileset_cache: TilesetCache,
) -> np.ndarray | None:
    """
    Composite a synthetic terrain texture from MCAL alpha + MCLY + tilesets.

    Args:
        mcal_alpha: (1024, 1024, 4) or (256, 256, 4) float32 in [0,1]
        mcly_ids: (16, 16, 4) int32, texture IDs per layer, -1 = inactive
        texture_names: list of texture paths indexed by texture ID

    Returns:
        (1024, 1024, 3) uint8 synthetic minimap, or None if no textures available
    """
    tile_chunks = 16
    chunk_alpha = 64 if mcal_alpha.shape[0] >= 512 else 16
    out_size = chunk_alpha * tile_chunks

    # Upsample alpha if it's at 256x256
    if mcal_alpha.shape[0] < 512:
        from PIL import Image
        alpha_1024 = np.zeros((1024, 1024, 4), dtype=np.float32)
        for l in range(4):
            a = Image.fromarray(mcal_alpha[:, :, l])
            a = np.asarray(a.resize((1024, 1024), Image.NEAREST)).astype(np.float32) / 255.0
            alpha_1024[:, :, l] = a
        mcal_alpha = alpha_1024
        chunk_alpha = 64
        out_size = 1024

    synthetic = np.zeros((out_size, out_size, 3), dtype=np.float32)
    any_texture = False

    for cy in range(tile_chunks):
        for cx in range(tile_chunks):
            for layer in range(4):
                tid = mcly_ids[cy, cx, layer]
                if tid < 0 or tid >= len(texture_names):
                    continue
                tex_name = texture_names[tid]
                if not tex_name:
                    continue

                tex = tileset_cache.get(tex_name)
                if tex is None:
                    continue

                any_texture = True
                tex_h, tex_w = tex.shape[:2]
                alpha_chunk = mcal_alpha[
                    cy * chunk_alpha : (cy + 1) * chunk_alpha,
                    cx * chunk_alpha : (cx + 1) * chunk_alpha,
                    layer,
                ]

                for ly in range(chunk_alpha):
                    ty = int(ly * tex_h / chunk_alpha) % tex_h
                    for lx in range(chunk_alpha):
                        tx = int(lx * tex_w / chunk_alpha) % tex_w
                        a = alpha_chunk[ly, lx]
                        if a <= 0.005:
                            continue
                        py = cy * chunk_alpha + ly
                        px = cx * chunk_alpha + lx
                        synthetic[py, px] += tex[ty, tx].astype(np.float32) * a

    if not any_texture:
        return None

    synthetic = synthetic.clip(0, 255).astype(np.uint8)
    return synthetic


def downsample_1024_to_256(arr: np.ndarray) -> np.ndarray:
    """Average-pool 1024x1024 → 256x256."""
    if arr.shape[0] != 1024:
        return arr
    from skimage.measure import block_reduce
    return block_reduce(arr, (4, 4, 1), np.mean).astype(arr.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def process_shard(shard_path: str, harvest_dir: str, output_dir: str, cache: TilesetCache | None = None):
    """Process a single NPZ shard: composite + residual, write new NPZ."""
    if cache is None:
        cache = TilesetCache(harvest_dir)

    try:
        with np.load(shard_path) as npz:
            raw = {k: v.copy() for k, v in npz.items() if isinstance(v, np.ndarray)}
    except Exception as e:
        return f"SKIP {shard_path}: load failed {e}"

    if 'mcal_alpha_pack_256' not in raw or 'mcly_texture_ids' not in raw:
        return f"SKIP {shard_path}: no MCAL/MCLY"

    # Read metadata for texture names
    meta_path = Path(shard_path).with_name('metadata.json')
    if not meta_path.exists():
        # Try inside NPZ
        zf = np.load(shard_path)
        if 'metadata.json' not in zf.files and hasattr(zf, 'files'):
            pass
        texture_names = []
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            texture_names = meta.get('mcly_texture_names', [])
        except Exception:
            texture_names = []

    # Also try from a sidecar _metadata.json
    if not texture_names:
        sidecar = Path(shard_path).parent / f'{Path(shard_path).stem}_metadata.json'
        if sidecar.exists():
            with open(sidecar) as f:
                texture_names = json.load(f).get('mcly_texture_names', [])

    mcal = raw['mcal_alpha_pack_256']
    mcly = raw['mcly_texture_ids']

    # Ensure MCAL is HWC
    if mcal.ndim == 3 and mcal.shape[0] in (4, 1):
        mcal = mcal.transpose(1, 2, 0)
    if mcal.ndim == 3 and mcal.shape[-1] not in (1, 4):
        mcal = mcal.transpose(1, 2, 0)

    # Ensure MCLY is HWC
    if mcly.ndim == 3 and mcly.shape[0] == 4:
        mcly = mcly.transpose(1, 2, 0)

    synthetic = composite_tile_texture(mcal, mcly, texture_names, cache)
    if synthetic is None:
        return f"SKIP {shard_path}: compositing produced no output"

    synthetic_256 = downsample_1024_to_256(synthetic)

    # Compute residual from real minimap if available
    residual = None
    if 'minimap_rgb_256' in raw:
        real = raw['minimap_rgb_256']
        if real.ndim == 3 and real.shape[0] == 3:
            real = real.transpose(1, 2, 0)
        residual = real.astype(np.float32) - synthetic_256.astype(np.float32)

    # Write output
    out_path = Path(output_dir) / f'{Path(shard_path).stem}_synth.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_data = dict(raw)
    out_data['synthetic_minimap_256'] = synthetic_256
    if residual is not None:
        out_data['texture_residual_256'] = residual

    np.savez_compressed(out_path, **out_data)
    return f"OK {Path(shard_path).name}: synth={synthetic_256.shape} residual={residual.shape if residual is not None else 'none'}"


def main():
    p = argparse.ArgumentParser(description='Synthesize minimaps from MCAL/MCLY + tilesets')
    p.add_argument('input', nargs='+', help='NPZ shards, manifest JSON, or directory')
    p.add_argument('--harvest-dir', required=True, help='Directory with harvested tileset PNGs')
    p.add_argument('--output-dir', '-o', default='synthetic_minimaps')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()

    harvest_dir = Path(args.harvest_dir)
    if not harvest_dir.exists():
        print(f"ERROR: harvest dir not found: {harvest_dir}")
        sys.exit(1)

    # Discover shards
    shard_paths = []
    for inp in args.input:
        p = Path(inp)
        if p.suffix == '.npz':
            shard_paths.append(str(p))
        elif p.suffix == '.json':
            with open(p) as f:
                manifest = json.load(f)
            for entry in manifest.get('entries', []):
                if isinstance(entry, dict):
                    sp = entry.get('shard_path', entry.get('path', ''))
                    if sp:
                        shard_paths.append(sp)
        elif p.is_dir():
            shard_paths.extend([str(f) for f in p.rglob('*.npz')])

    if args.limit > 0:
        shard_paths = shard_paths[:args.limit]

    print(f"Processing {len(shard_paths)} shards with {args.num_workers} workers...")
    cache = TilesetCache(str(harvest_dir))

    ok = 0
    skip = 0
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(process_shard, s, str(harvest_dir), args.output_dir, cache)
                       for s in shard_paths]
            for f in as_completed(futures):
                result = f.result()
                print(result)
                if result.startswith('OK'):
                    ok += 1
                else:
                    skip += 1
    else:
        for s in shard_paths:
            result = process_shard(s, str(harvest_dir), args.output_dir, cache)
            print(result)
            if result.startswith('OK'):
                ok += 1
            else:
                skip += 1

    print(f"\nDone: {ok} OK, {skip} skipped")


if __name__ == '__main__':
    main()
