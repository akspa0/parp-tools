"""Pre-composite MapTextures from ground-truth MCAL/MCLY + tilesets.
Uses WoW's normalized weighted sum formula: w[0]=1-clamp(Σα[1..3],0,1), w[i]=α[i].
Run before training — writes sidecar NPZ files with composited textures + residual."""
import argparse, json, os, sys, numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

TILE_CHUNKS = 16


def load_tileset_cache(harvest_dir):
    cache = {}
    harvest = Path(harvest_dir)
    if not harvest.exists(): return cache
    for f in harvest.rglob('*.png'):
        cache[f.stem.lower()] = f
    return cache


def get_texture(cache, tex_name):
    stem = Path(tex_name).stem.lower()
    path = cache.get(stem)
    if path is None:
        for k in cache:
            if stem in k:
                path = cache[k]; break
    if path and path.exists():
        return np.asarray(Image.open(path).convert('RGB'))
    return None


def composite_one(shard_path, tileset_cache, out_dir):
    try:
        z = np.load(shard_path)
        mcal = z.get('mcal_alpha_pack_256')
        mcly = z.get('mcly_texture_ids')
        mm = z.get('minimap_rgb_256')
        z.close()
        if mcal is None or mcly is None or mm is None: return None

        sidecar = Path(shard_path).with_name(Path(shard_path).stem + '_metadata.json')
        texture_names = []
        if sidecar.exists():
            with open(sidecar) as f:
                texture_names = json.load(f).get('mcly_texture_names', [])

        mcal = mcal.astype(np.float32)
        if mcal.ndim == 3 and mcal.shape[0] in (1, 4): mcal = mcal.transpose(1, 2, 0)
        if mcly.ndim == 3 and mcly.shape[0] == 4: mcly = mcly.transpose(1, 2, 0)

        out_size = mcal.shape[0]
        chunk_alpha = out_size // TILE_CHUNKS

        synthetic = np.zeros((out_size, out_size, 3), dtype=np.float32)
        any_tex = False

        for cy in range(TILE_CHUNKS):
            for cx in range(TILE_CHUNKS):
                # Get textures for each layer in this chunk
                texs = []
                for layer in range(4):
                    if layer >= mcly.shape[-1]: texs.append(None); continue
                    tid = int(mcly[cy, cx, layer])
                    if tid < 0 or tid >= len(texture_names):
                        texs.append(None); continue
                    tex = get_texture(tileset_cache, texture_names[tid])
                    texs.append(tex)
                    if tex is not None: any_tex = True

                for ly in range(chunk_alpha):
                    py = cy * chunk_alpha + ly
                    for lx in range(chunk_alpha):
                        px = cx * chunk_alpha + lx

                        al = mcal[py, px, :]  # (4,) — α[0]=unused, α[1..3]=layer overlays

                        # Sample the texture colors at this pixel
                        def sample_tex(tex):
                            if tex is None: return None
                            th, tw = tex.shape[:2]
                            tx = int(lx * tw / chunk_alpha) % tw
                            ty = int(ly * th / chunk_alpha) % th
                            return tex[ty, tx].astype(np.float32)

                        cols = [sample_tex(t) for t in texs]

                        # Sequential mix() chain — matches MdxViewer terrain shader & WoW client
                        # result = tex0                                          (base)
                        # result = mix(result, tex1, α1)   → result*(1-α1) + tex1*α1
                        # result = mix(result, tex2, α2)
                        # result = mix(result, tex3, α3)
                        color = cols[0].copy() if cols[0] is not None else np.zeros(3)
                        for li in range(1, 4):
                            if cols[li] is not None and al[li] > 0.005:
                                color = color * (1.0 - al[li]) + cols[li] * al[li]

                        synthetic[py, px] = color

        if not any_tex:
            return None

        synthetic = synthetic.clip(0, 255).astype(np.uint8)
        synth_256 = np.asarray(Image.fromarray(synthetic).resize((256, 256), Image.BILINEAR))

        mm_256 = np.asarray(Image.fromarray(mm).resize((256, 256), Image.BILINEAR))

        residual = (mm_256.astype(np.float32) - synth_256.astype(np.float32))

        out_path = Path(out_dir) / (Path(shard_path).stem + '_composited.npz')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path,
            synthetic_minimap_256=synth_256,
            texture_residual_256=residual,
        )
        return shard_path
    except Exception as e:
        print(f"  ERROR {shard_path}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description='Pre-composite MapTextures (normalized weighted sum blending)')
    p.add_argument('input', nargs='+', help='NPZ shard dir or manifest')
    p.add_argument('--harvest-dir', required=True, help='Tileset PNG directory')
    p.add_argument('--output-dir', '-o', default='composited_maptextures')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    paths = []
    for inp in args.input:
        p = Path(inp)
        if p.suffix == '.npz': paths.append(str(p))
        elif p.suffix == '.json':
            with open(p) as f:
                for e in json.load(f).get('entries', []):
                    if e.get('shard_path'): paths.append(e['shard_path'])
        elif p.is_dir(): paths.extend(str(f) for f in p.rglob('*.npz'))

    if args.limit: paths = paths[:args.limit]
    print(f'Processing {len(paths)} shards...')

    cache = load_tileset_cache(args.harvest_dir)
    print(f'Tileset cache: {len(cache)} textures')

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    ok, skip = 0, 0

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            futures = {ex.submit(composite_one, sp, cache, out_dir): sp for sp in paths}
            for f in as_completed(futures):
                if f.result(): ok += 1
                else: skip += 1
    else:
        for sp in paths:
            if composite_one(sp, cache, out_dir): ok += 1
            else: skip += 1

    print(f'Done: {ok} composited, {skip} skipped')


if __name__ == '__main__':
    main()
