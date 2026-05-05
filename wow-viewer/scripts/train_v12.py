"""V12 Texture Decomposer (Stage 1) — minimap → MCAL + MCLY + residual.
SegFormer B2 backbone, 3-channel RGB input (no tileset channels).
Stage 1 output (residual) becomes the clean input for Stage 2 height model."""
import argparse, json, math, os, random, sys, time, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import SegformerModel, SegformerConfig
except ImportError:
    SegformerModel = None


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def find_paths(inputs):
    paths = []
    for p in inputs:
        p = str(p)
        if p.endswith('.npz'): paths.append(p)
        elif p.endswith('.json'):
            with open(p) as f:
                for e in json.load(f).get('entries', []):
                    sp = e.get('shard_path', '') or e.get('path', '')
                    if sp: paths.append(sp)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith('.npz') and 'composited' not in f: paths.append(os.path.join(root, f))
    return list(set(paths))


# ImageNet normalization (SegFormer was pretrained with these)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class V12Dataset(Dataset):
    """3ch minimap in, MCAL/MCLY/residual out. No tileset channels."""
    def __init__(self, paths, mcly_vocab, augment=False, residual_dir=None):
        self.paths = paths
        self.vocab = mcly_vocab
        self.augment = augment
        self.residual_dir = Path(residual_dir) if residual_dir else None

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            data = np.load(path)
        except Exception:
            # Corrupted NPZ — return zeros, loss masks will skip it
            return torch.zeros(3, 256, 256), {
                'mcal': torch.zeros(4, 256, 256),
                'mcly': torch.full((16, 16), -100, dtype=torch.long),
                'residual': torch.zeros(3, 256, 256),
                'has_mcal': torch.tensor(0.0),
                'has_mcly': torch.tensor(0.0),
                'has_residual': torch.tensor(0.0),
            }

        # Input: minimap RGB (3, 256, 256) with ImageNet normalization
        mm = torch.from_numpy(data['minimap_rgb_256'].astype(np.float32))
        if mm.dim() == 3 and mm.shape[-1] == 3: mm = mm.permute(2, 0, 1)
        mm = (mm / 255.0 - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)

        # MCAL target (4, 256, 256)  in [0, 1]
        has_mcal = 'mcal_alpha_pack_256' in data
        if has_mcal:
            mcal = torch.from_numpy(data['mcal_alpha_pack_256'].astype(np.float32))
            if mcal.dim() == 3 and mcal.shape[-1] == 4: mcal = mcal.permute(2, 0, 1)
        else:
            mcal = torch.zeros(4, 256, 256)

        # MCLY target (16, 16) — vocab-mapped texture class per chunk, -100 = ignore
        has_mcly = 'mcly_texture_ids' in data
        if has_mcly:
            mcly = torch.from_numpy(data['mcly_texture_ids'].astype(np.int64))
            if mcly.dim() == 3 and mcly.shape[-1] == 4: mcly = mcly[..., 0]
            if mcly.dim() == 3 and mcly.shape[0] == 4: mcly = mcly[0]
            mapped = torch.full((16, 16), -100, dtype=torch.long)
            for old_id, new_id in self.vocab.items():
                mapped[mcly == old_id] = new_id
        else:
            mapped = torch.full((16, 16), -100, dtype=torch.long)

        # Residual target (3, 256, 256) in [-1, 1] from pre-computed MapTexture
        stem = Path(path).stem
        cp_paths = [Path(path).with_name(stem + '_composited.npz')]
        if self.residual_dir:
            cp_paths.append(self.residual_dir / (stem + '_composited.npz'))
        composited_path = next((p for p in cp_paths if p.exists()), None)
        has_residual = False
        if composited_path:
            try:
                cz = np.load(composited_path)
                residual = torch.from_numpy(cz['texture_residual_256'].astype(np.float32)) / 255.0
                cz.close()
                has_residual = True
                if residual.dim() == 3 and residual.shape[-1] == 3: residual = residual.permute(2, 0, 1)
            except Exception:
                has_residual = False
        if not has_residual:
            residual = torch.zeros(3, 256, 256)

        data.close()

        if self.augment:
            if random.random() > 0.5:
                mm = torch.flip(mm, [-1]); mcal = torch.flip(mcal, [-1])
                mapped = torch.flip(mapped, [0]); residual = torch.flip(residual, [-1])
            if random.random() > 0.5:
                mm = torch.flip(mm, [-2]); mcal = torch.flip(mcal, [-2])
                mapped = torch.flip(mapped, [1]); residual = torch.flip(residual, [-2])

        return mm, {
            'mcal': mcal,
            'mcly': mapped,
            'residual': residual,
            'has_mcal': torch.tensor(1.0 if has_mcal else 0.0),
            'has_mcly': torch.tensor(1.0 if has_mcly else 0.0),
            'has_residual': torch.tensor(1.0 if has_residual else 0.0),
        }


def build_vocab(paths, min_n=3):
    from collections import Counter
    cnt = Counter()
    for p in paths:
        try:
            z = np.load(p)
            if 'mcly_texture_ids' in z:
                for i in z['mcly_texture_ids'].flat:
                    if int(i) >= 0: cnt[int(i)] += 1
            z.close()
        except: pass
    return {tid: idx for idx, (tid, c) in enumerate(cnt.items()) if c >= min_n}


# ═══════════════════════════════════════════════════════════════════════════
# Model — SegFormer B2 → U-Net decoder → 3 heads
# ═══════════════════════════════════════════════════════════════════════════

class UpBlock(nn.Module):
    """Upsample 2× → concat skip → Conv → GELU."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class V12Model(nn.Module):
    """SegFormer B2 backbone → MCLY head (16×16) + MCAL + residual (256×256)."""
    def __init__(self, num_tex_classes, backbone_name='nvidia/segformer-b2-finetuned-ade-512-512'):
        super().__init__()
        if SegformerModel is None: raise RuntimeError("pip install transformers")
        self.backbone = SegformerModel.from_pretrained(backbone_name)

        # SegFormer B2 channel dims at each stage
        self.stage_dims = [64, 128, 320, 512]
        decoder_dim = 256

        # Project each stage to decoder_dim (1×1 conv)
        self.stage_proj = nn.ModuleList([
            nn.Conv2d(d, decoder_dim, 1) for d in self.stage_dims
        ])

        # MCLY head on Stage 2 features (1/16 scale, 16×16)
        self.mcly_head = nn.Sequential(
            nn.Conv2d(decoder_dim, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, num_tex_classes, 1),
        )

        # Decoder: 8×8 → 16×16 → 32×32 → 64×64 → 256×256
        self.up3 = UpBlock(decoder_dim, decoder_dim, decoder_dim)
        self.up2 = UpBlock(decoder_dim, decoder_dim, decoder_dim)
        self.up1 = UpBlock(decoder_dim, decoder_dim, decoder_dim)
        self.to_full = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, 2),  # 64→128
            nn.GELU(),
            nn.ConvTranspose2d(decoder_dim, 64, 2, 2),           # 128→256
            nn.GELU(),
        )

        # MCAL head: 4 channels, sigmoid for [0,1] alpha
        self.mcal_head = nn.Sequential(
            nn.Conv2d(64 + decoder_dim, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 4, 3, 1, 1),
        )

        # Residual head: 3 channels, linear (residual can be negative)
        self.residual_head = nn.Sequential(
            nn.Conv2d(64 + decoder_dim, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, x):
        # x: (B, 3, 256, 256)
        hs = self.backbone(x, output_hidden_states=True).hidden_states
        # hs: 4 tensors at (B, C, H/4, W/4) ... (B, C, H/32, W/32)

        # Project each stage to decoder_dim
        p = [proj(h) for proj, h in zip(self.stage_proj, hs)]
        # p[0]: (B, 256, 64, 64)  @ 1/4
        # p[1]: (B, 256, 32, 32)  @ 1/8
        # p[2]: (B, 256, 16, 16)  @ 1/16
        # p[3]: (B, 256, 8, 8)    @ 1/32

        # MCLY from Stage 2 (1/16 scale)
        mcly = self.mcly_head(p[2])  # (B, num_classes, 16, 16)

        # Decoder with skip connections
        d = self.up3(p[3], p[2])   # 8→16
        d = self.up2(d, p[1])      # 16→32
        d = self.up1(d, p[0])      # 32→64
        d_full = self.to_full(d)   # 64→128→256

        # Merge decoder features with projected stage 0 for fine detail
        p0_up = F.interpolate(p[0], size=(256, 256), mode='bilinear', align_corners=False)
        f = torch.cat([d_full, p0_up], dim=1)

        mcal = torch.sigmoid(self.mcal_head(f))
        residual = self.residual_head(f)

        return {
            'mcal': mcal,        # (B, 4, 256, 256) in [0, 1]
            'mcly': mcly,        # (B, num_classes, 16, 16) logits
            'residual': residual, # (B, 3, 256, 256) unconstrained
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_v12(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}'); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    paths = find_paths(args.input)
    has_mcly_paths = []
    for p in paths:
        try:
            z = np.load(p)
            if 'mcly_texture_ids' in z.files and 'minimap_rgb_256' in z.files:
                has_mcly_paths.append(p)
            z.close()
        except: pass
    print(f'Total: {len(paths)}, with MCLY: {len(has_mcly_paths)}')
    paths = has_mcly_paths

    if args.max_samples and len(paths) > args.max_samples:
        random.seed(1337); paths = random.sample(paths, args.max_samples)
    random.shuffle(paths)
    nv = max(1, int(len(paths) * 0.12))
    train_p, val_p = paths[nv:], paths[:nv]
    print(f'Train: {len(train_p)}, Val: {len(val_p)}')

    vocab = build_vocab(train_p)
    ntex = len(vocab)
    print(f'MCLY vocab: {ntex} classes')

    train_ds = V12Dataset(train_p, vocab, augment=True, residual_dir=args.residual_dir)
    val_ds = V12Dataset(val_p, vocab, augment=False, residual_dir=args.residual_dir)
    nw = args.num_workers
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    model = V12Model(ntex).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f'Params: {npar:,} ({npar*4/1e6:.1f}MB)')

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda') if dev.type == 'cuda' else None

    best_val = float('inf')
    for ep in range(args.epochs):
        if ep < 5: lrv = args.lr * (0.01 + 0.99 * ep / 5)
        else:
            f = (ep - 5) / max(args.epochs - 5, 1)
            lrv = max(args.lr * 0.5 * (1 + math.cos(math.pi * f)), args.lr * 0.01)
        for pg in opt.param_groups: pg['lr'] = lrv

        model.train(); tl = 0.0; nb = 0
        for inp, tgt in train_ld:
            inp = inp.to(dev, non_blocking=True)
            tm = tgt['mcal'].to(dev, non_blocking=True)
            tlbl = tgt['mcly'].to(dev, non_blocking=True)
            tr = tgt['residual'].to(dev, non_blocking=True)
            hm = tgt['has_mcal'].to(dev)
            hl = tgt['has_mcly'].to(dev)
            hr = tgt['has_residual'].to(dev)

            if scaler:
                with torch.amp.autocast('cuda'): p = model(inp)
            else: p = model(inp)

            B = inp.shape[0]
            loss = 0.0
            # MCAL L1 (5× weight — alpha is sparse)
            if hm.sum() > 0:
                m = F.l1_loss(p['mcal'], tm, reduction='none').mean(dim=(1,2,3))
                loss = loss + 5.0 * ((m * hm).sum() / hm.sum())
            # MCLY CE (-100 = ignore)
            if hl.sum() > 0:
                c = F.cross_entropy(p['mcly'], tlbl, ignore_index=-100, reduction='none')
                loss = loss + 0.2 * ((c.view(B,-1).mean(1) * hl).sum() / hl.sum())
            # Residual L1
            if hr.sum() > 0:
                r = F.l1_loss(p['residual'], tr, reduction='none').mean(dim=(1,2,3))
                loss = loss + 0.5 * ((r * hr).sum() / hr.sum())

            if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
            else: loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad()
            tl += loss.item(); nb += 1

        avg = tl / nb
        if ep % 10 == 0:
            vl = ' —'
            if val_ld:
                model.eval(); vloss = 0.0; vc = 0
                with torch.no_grad():
                    for iv, tv in val_ld:
                        iv = iv[:4].to(dev)
                        pv = model(iv)
                        if tv['has_mcal'].sum() > 0:
                            vloss += F.l1_loss(pv['mcal'][:4], tv['mcal'][:4].to(dev)).item(); vc += 1
                            if vc >= 2: break
                vl = f' val={vloss/max(vc,1):.4f}'
                if vc and vloss/vc < best_val:
                    best_val = vloss/vc; torch.save({'model': model.state_dict(), 'epoch': ep, 'vocab': vocab}, out / 'best.pt')
            print(f'E{ep:4d} loss={avg:.4f} lr={lrv:.2e}{vl}')
            model.train()

        torch.save({'model': model.state_dict(), 'epoch': ep, 'optimizer': opt.state_dict(), 'vocab': vocab}, out / 'last.pt')

    print(f'\nDone. Best val: {best_val:.4f}. {out}')


def main():
    p = argparse.ArgumentParser(description='V12 Texture Decomposer (Stage 1) — SegFormer backbone, 3ch RGB input')
    p.add_argument('input', nargs='+')
    p.add_argument('--output-dir', '-o', default='runs/v12_stage1')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-samples', type=int, default=2000)
    p.add_argument('--num-workers', type=int, default=2,
                   help='DataLoader workers for parallel data loading (default: 2)')
    p.add_argument('--residual-dir', type=str, default=None,
                   help='Directory with pre-computed _composited.npz files (e.g. output/tmp/maptextures)')
    args = p.parse_args()
    train_v12(args)

if __name__ == '__main__':
    main()
