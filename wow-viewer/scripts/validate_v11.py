"""Run validation + preview on latest checkpoint without training."""
import argparse, json, sys, gc
from pathlib import Path
import torch
import numpy as np

from train_v11 import (V11TerrainModel, V11Dataset, discover_npz_paths,
                        build_mcly_vocab, N_CHANNELS, save_preview_from_batch)


@torch.no_grad()
def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    height_mean = ckpt.get('height_mean', 0.0)
    height_std = max(ckpt.get('height_std', 1.0), 0.01)
    mcly_vocab = ckpt.get('mcly_vocab', {})
    mcly_unk_idx = ckpt.get('mcly_unk_idx', len(mcly_vocab))

    in_ch = sd['stem.0.weight'].shape[1]
    mcly_w = sd.get('head_mcly.3.weight')
    num_tex = mcly_w.shape[0] if mcly_w is not None else len(mcly_vocab) + 1

    model = V11TerrainModel(in_channels=in_ch, decoder_dim=args.decoder_dim, num_texture_classes=num_tex)
    model.mcly_unk_idx = mcly_unk_idx
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print(f'Loaded checkopt {args.checkpoint} ({sum(p.numel() for p in model.parameters()):,} params)')

    shards = discover_npz_paths(args.input)
    print(f'Found {len(shards)} shards')

    # Use a small subset for validation
    n = min(args.limit or 8, len(shards))
    val_paths = shards[:n]
    ds = V11Dataset(val_paths, mcly_vocab, mcly_unk_idx, signal_dropout=0,
                    height_mean=height_mean, height_std=height_std)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(4, n), shuffle=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / 'previews'
    preview_dir.mkdir(exist_ok=True)

    # Compute validation loss
    total_loss = 0.0
    count = 0
    for inp, targets in loader:
        inp = inp.to(device)
        t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        pred = model(inp)
        loss = F.l1_loss(pred['height_257'], t['height_257'])
        total_loss += loss.item()
        count += 1
    print(f'Val L1 loss: {total_loss / max(count, 1):.4f}')

    # Save preview
    save_preview_from_batch(model, inp, targets, device, height_mean, height_std,
                            preview_dir / 'validation.png', num_rows=min(4, n))
    print(f'Preview saved: {preview_dir / "validation.png"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='V11 validation')
    p.add_argument('checkpoint')
    p.add_argument('input', nargs='+')
    p.add_argument('--output-dir', '-o', default='v11_validation')
    p.add_argument('--decoder-dim', type=int, default=96)
    p.add_argument('--limit', type=int, default=8)
    args = p.parse_args()
    validate(args)
