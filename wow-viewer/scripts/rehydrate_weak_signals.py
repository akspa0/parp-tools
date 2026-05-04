"""
Weak-signal detector and rehydrator for WoW terrain tiles.

Blizzard's erase tool doesn't delete data — it squishes it by 33.334×
in amplitude, reducing terrain to a thin band (−2.778 to +2.778 Z).
This "weak signal" still contains the original terrain shape at ~3% scale.

Detect these tiles and re-amplify them back to full strength.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

WEAK_SIGNAL_MAX_RANGE = 6.0  # absolute max range that might be squished
AMPLIFY_FACTOR = 33.334


def detect_weak_signal(heightmap: np.ndarray, neighbor_range: float | None = None) -> bool:
    """Returns True if this heightmap shows signs of being squished.
    
    Primary indicator: very narrow height range.
    Secondary: range is tiny compared to a neighbor tile.
    """
    if heightmap.ndim != 2:
        return False
    hr = heightmap.max() - heightmap.min()
    # Very narrow range is the primary signal
    if hr < WEAK_SIGNAL_MAX_RANGE:
        return True
    # If we have neighbor context: tile range is < 5% of neighbor range
    if neighbor_range is not None and neighbor_range > 0:
        if hr / neighbor_range < 0.05:
            return True
    return False


def rehydrate(heightmap: np.ndarray) -> np.ndarray:
    """Re-amplify squished terrain back to original scale."""
    return heightmap * AMPLIFY_FACTOR


def process_shard(npz_path: str, output_dir: str, rehydrate_mode: bool = False) -> dict | None:
    try:
        z = np.load(npz_path)
        if 'height_257' not in z.files:
            return None
        h = z['height_257']
        if not detect_weak_signal(h):
            return None
        
        info = {
            'path': npz_path,
            'height_min': float(h.min()),
            'height_max': float(h.max()),
            'height_range': float(h.max() - h.min()),
        }
        
        if rehydrate_mode:
            h_new = rehydrate(h)
            info['rehydrated_min'] = float(h_new.min())
            info['rehydrated_max'] = float(h_new.max())
            info['rehydrated_range'] = float(h_new.max() - h_new.min())
            
            out_path = Path(output_dir) / Path(npz_path).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for key in z.files:
                if isinstance(z[key], np.ndarray):
                    data[key] = z[key].copy() if key != 'height_257' else h_new
                else:
                    data[key] = z[key]
            # Also copy sidecar metadata
            sidecar = Path(npz_path).with_name(Path(npz_path).stem + '_metadata.json')
            if sidecar.exists():
                import shutil
                shutil.copy(sidecar, out_path.parent / sidecar.name)
            np.savez_compressed(out_path, **data)
            info['output'] = str(out_path)
        
        z.close()
        return info
    except Exception as e:
        return {'path': npz_path, 'error': str(e)}


def main():
    p = argparse.ArgumentParser(description='Detect weak-signal terrain tiles')
    p.add_argument('input', nargs='+', help='NPZ shards, manifest, or directory')
    p.add_argument('--output-dir', '-o', default='rehydrated_tiles')
    p.add_argument('--rehydrate', action='store_true', help='Re-amplify squished tiles')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    shard_paths = []
    for inp in args.input:
        p = Path(inp)
        if p.suffix == '.npz':
            shard_paths.append(str(p))
        elif p.suffix == '.json':
            with open(p) as f:
                manifest = json.load(f)
            for entry in manifest.get('entries', []):
                sp = entry.get('shard_path', '')
                if sp:
                    shard_paths.append(sp)
        elif p.is_dir():
            shard_paths.extend([str(f) for f in p.rglob('*.npz')])

    if args.limit > 0:
        shard_paths = shard_paths[:args.limit]

    # Phase 1: collect all tiles by (map, x, y)
    tile_grid: dict[tuple, float] = {}  # (map_name, x, y) -> height_range
    for sp in shard_paths:
        try:
            z = np.load(sp)
            h = z['height_257']
            z.close()
            # Parse map/x/y from path name
            stem = Path(sp).stem.replace('_v10', '').replace('_v11', '')
            parts = stem.rsplit('_', 2)
            if len(parts) == 3:
                map_name, ty, tx = parts[0], int(parts[1]), int(parts[2])  # name is Map_Y_X
                tile_grid[(map_name, tx, ty)] = float(h.max() - h.min())
        except:
            pass

    # Phase 2: detect weak signals using neighbor context
    detected = []
    for sp in shard_paths:
        stem = Path(sp).stem.replace('_v10', '').replace('_v11', '')
        parts = stem.rsplit('_', 2)
        if len(parts) != 3:
            continue
        map_name, ty, tx = parts[0], int(parts[1]), int(parts[2])  # name is Map_Y_X
        
        hr = tile_grid.get((map_name, tx, ty), 0)
        neighbor_ranges = [tile_grid.get((map_name, tx+dx, ty+dy), 0) 
                          for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx==0 and dy==0)]
        best_neighbor = max(neighbor_ranges) if neighbor_ranges else hr
        
        if hr < 10.0 or (best_neighbor > 0 and hr / best_neighbor < 0.05):
            info = {'tile': stem, 'path': sp, 'range': hr, 'neighbor_range': best_neighbor}
            detected.append(info)
            
            if args.rehydrate:
                z = np.load(sp)
                h = z['height_257']
                h_new = h * AMPLIFY_FACTOR
                info['before'] = f'{h.min():.2f}..{h.max():.2f}'
                info['after'] = f'{h_new.min():.2f}..{h_new.max():.2f}'
                
                out_path = Path(args.output_dir) / Path(sp).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                data = {k: z[k].copy() if isinstance(z[k], np.ndarray) and k != 'height_257' else h_new 
                        if k == 'height_257' else z[k] for k in z.files}
                z.close()
                np.savez_compressed(out_path, **data)
                info['output'] = str(out_path)
                print(f"[REHYDRATED] {stem}: {info['before']} -> {info['after']}")
            elif args.limit and len(detected) > args.limit:
                break

    print(f"\nDetected: {len(detected)} weak-signal tiles")
    if args.rehydrate:
        print(f"Rehydrated to: {args.output_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save stripped report (no path field — too long for JSON)
    simple = [{'tile': d['tile'], 'range': d['range'], 'neighbor_range': d.get('neighbor_range', 0)} for d in detected]
    with open(out_dir / 'weak_signal_report.json', 'w') as f:
        json.dump({'count': len(detected), 'tiles': simple}, f, indent=2)


if __name__ == '__main__':
    main()
