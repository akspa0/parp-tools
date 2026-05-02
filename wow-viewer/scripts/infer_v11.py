import argparse, json, re, struct, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from train_v11 import V11TerrainModel, N_CHANNELS, discover_npz_paths, V11Dataset


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def write_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    if clipped.ndim == 2:
        clipped = np.stack([clipped] * 3, axis=-1)
    Image.fromarray(clipped).save(path)


def parse_tile_name(name):
    m = re.match(r'^(.+)_(\d+)_(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def export_obj(heightmap, obj_path, texture_name, tile_world_size=533.333, center_mesh=False, height_offset=0):
    h, w = heightmap.shape
    spacing_x = tile_world_size / max(w - 1, 1)
    spacing_y = tile_world_size / max(h - 1, 1)
    ox = tile_world_size * 0.5 if center_mesh else 0.0
    oy = tile_world_size * 0.5 if center_mesh else 0.0
    mtl_path = obj_path.with_suffix('.mtl')
    mat_name = obj_path.stem + '_mat'

    with open(mtl_path, 'w') as f:
        f.write(f'newmtl {mat_name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {texture_name}\n')

    with open(obj_path, 'w') as f:
        f.write(f'# v11 terrain mesh {w}x{h}\nmtllib {mtl_path.name}\nusemtl {mat_name}\n')
        for row in range(h):
            for col in range(w):
                wx = col * spacing_x - ox
                wz = row * spacing_y - oy
                wy = float(heightmap[row, col] + height_offset)
                f.write(f'v {wx:.6f} {wy:.6f} {wz:.6f}\n')
        tw, th = max(w - 1, 1), max(h - 1, 1)
        hu, hv = 0.5 / max(w, 1), 0.5 / max(h, 1)
        for row in range(h):
            for col in range(w):
                u = hu + (col / tw) * (1 - 2 * hu)
                v = hv + (1 - row / th) * (1 - 2 * hv)
                f.write(f'vt {u:.6f} {v:.6f}\n')
        for row in range(h - 1):
            for col in range(w - 1):
                vi = row * w + col + 1
                f.write(f'f {vi}/{vi} {vi + w}/{vi + w} {vi + w + 1}/{vi + w + 1}\n')
                f.write(f'f {vi}/{vi} {vi + w + 1}/{vi + w + 1} {vi + 1}/{vi + 1}\n')
    return obj_path, mtl_path


def normalize_state_dict(sd):
    prefix = '_orig_mod.'
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


@torch.no_grad()
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = normalize_state_dict(ckpt['model'])
    height_mean = ckpt.get('height_mean', 0.0)
    height_std = max(ckpt.get('height_std', 1.0), 0.01)
    mcly_vocab = ckpt.get('mcly_vocab', {})
    mcly_unk_idx = ckpt.get('mcly_unk_idx', len(mcly_vocab))

    in_channels = sd['stem.0.weight'].shape[1]
    mcly_weight = sd.get('head_mcly.3.weight')
    num_tex_classes = mcly_weight.shape[0] if mcly_weight is not None else len(mcly_vocab) + 1
    model = V11TerrainModel(in_channels=in_channels, decoder_dim=256, num_texture_classes=num_tex_classes)
    model.mcly_unk_idx = mcly_unk_idx
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model: {total_params:,} params, {in_channels} input channels, {num_tex_classes} MCLY classes')

    shard_paths = discover_npz_paths(args.input)
    print(f'Discovered {len(shard_paths)} shards')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for shard_idx, shard_path_str in enumerate(shard_paths[:args.limit] if args.limit > 0 else shard_paths):
        shard_path = Path(shard_path_str)
        print(f'[{shard_idx + 1}/{len(shard_paths)}] {shard_path}')

        with np.load(shard_path) as npz:
            raw = {k: v.copy() for k, v in npz.items() if isinstance(v, np.ndarray)}

        ds = V11Dataset([shard_path], mcly_vocab, mcly_unk_idx, signal_dropout=0,
                        height_mean=height_mean, height_std=height_std)
        inp_tensor, targets = ds[0]
        inp_tensor = inp_tensor.unsqueeze(0).to(device)

        pred = model(inp_tensor)

        tile_name = str(shard_path.stem)
        if tile_name.endswith('_v10'):
            tile_name = tile_name[:-4]
        parsed = parse_tile_name(tile_name)
        map_name = parsed[0] if parsed else 'map'
        tile_stem = f'{map_name}_{parsed[1]}_{parsed[2]}' if parsed else tile_name

        tile_dir = output_dir / tile_stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        height_257 = pred['height_257'].squeeze().float().cpu().numpy()
        height_257 = height_257 * height_std + height_mean
        height_65 = pred['height_65'].squeeze().float().cpu().numpy()
        height_65 = height_65 * height_std + height_mean
        height_17 = pred['height_17'].squeeze().float().cpu().numpy()
        height_17 = height_17 * height_std + height_mean

        mcal = pred['mcal_alpha'].squeeze().float().cpu().numpy()
        mcal_rgb = (mcal.transpose(1, 2, 0) * 255).astype(np.uint8) if mcal.ndim == 3 else None

        mcly_logits = pred['mcly_logits'].squeeze().float().cpu().numpy()
        mcly_labels = np.argmax(mcly_logits, axis=0) if mcly_logits.ndim == 3 else None

        hole_logits = pred['hole_logits'].squeeze().float().cpu().numpy()
        hole_mask = (hole_logits > 0).astype(np.uint8) * 255 if hole_logits.ndim == 2 else None

        export_items = {}

        if args.export_obj:
            tex_rgb = mcal_rgb if mcal_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            tex_path = tile_dir / f'{tile_stem}_texture.png'
            write_png(tex_path, tex_rgb)
            obj, mtl = export_obj(
                height_257, tile_dir / f'{tile_stem}.obj', tex_path.name,
                tile_world_size=args.tile_world_size, center_mesh=args.center_mesh,
                height_offset=args.height_offset)
            export_items.update({'obj': str(obj), 'mtl': str(mtl), 'texture': str(tex_path)})

        npz_path = tile_dir / f'{tile_stem}_pred.npz'
        npz_data = {'height_257': height_257, 'height_65': height_65, 'height_17': height_17}
        if mcal_rgb is not None:
            npz_data['mcal_alpha_pack_256'] = mcal_rgb
        if mcly_labels is not None:
            npz_data['mcly_labels_16'] = mcly_labels
        if hole_mask is not None:
            npz_data['hole_mask_16'] = hole_mask
        np.savez_compressed(npz_path, **npz_data)
        export_items['npz'] = str(npz_path)

        gt_height = targets.get('height_257')
        mae = float(np.abs(height_257 - (gt_height.squeeze().numpy() * height_std + height_mean)).mean()) if gt_height is not None else -1

        result = {
            'tile': tile_stem,
            'shard': str(shard_path),
            'mae': mae,
            'height_min': float(height_257.min()),
            'height_max': float(height_257.max()),
            'exports': export_items,
        }
        results.append(result)

        print(f'  height: {height_257.shape} [{height_257.min():.1f}, {height_257.max():.1f}] '
              f'mae={mae:.2f} mcal={mcal_rgb.shape if mcal_rgb is not None else "none"}')

    write_json(output_dir / 'inference_report.json', {
        'checkpoint': str(args.checkpoint),
        'shards_processed': len(results),
        'output_dir': str(output_dir),
        'results': results,
    })
    print(f'\nDone. Processed {len(results)} shards. Report: {output_dir / "inference_report.json"}')


def main():
    p = argparse.ArgumentParser(description='V11 Terrain Model Inference')
    p.add_argument('checkpoint', help='Path to best_ema.pt or best.pt')
    p.add_argument('input', nargs='+', help='NPZ shard(s), manifest, or directory')
    p.add_argument('--output-dir', '-o', default='v11_inference')
    p.add_argument('--limit', type=int, default=0, help='Max shards to process')
    p.add_argument('--export-obj', action='store_true', help='Export OBJ + texture meshes')
    p.add_argument('--tile-world-size', type=float, default=533.333)
    p.add_argument('--center-mesh', action='store_true')
    p.add_argument('--height-offset', type=float, default=0.0)
    args = p.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
