"""V12 Texture Decomposer — minimap → MCAL + MCLY + residual.
Train on minimap + tileset references (17ch) with ground-truth MCAL/MCLY + pre-computed MapTexture residual."""
import argparse, json, math, os, random, sys, time, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except ImportError:
    timm = None


# ═══════════════════════════════════════════════════════════════════════════
# Tileset Cache
# ═══════════════════════════════════════════════════════════════════════════

class TilesetCache:
    """Load tileset PNGs from harvested tileset directory, hold thumbnails for model input."""
    def __init__(self, tileset_dir, max_cache_mb=500):
        self.thumbnails = {}  # normalized texture path → (16,16,3) uint8
        idx_path = os.path.join(tileset_dir, 'tileset_index.json')
        if not os.path.exists(idx_path):
            print(f'WARNING: tileset index not found at {idx_path}')
            return
        with open(idx_path) as f:
            idx = json.load(f)
        limit = max_cache_mb * 1024 * 1024
        total = 0
        n = 0
        for tex_path, file_path in idx.get('textures', {}).items():
            if total >= limit:
                break
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path).convert('RGB')
                    thumb = np.asarray(img.resize((16, 16), Image.BILINEAR))
                    self.thumbnails[tex_path.lower()] = thumb
                    total += thumb.nbytes
                    n += 1
                except Exception:
                    pass
        print(f'TilesetCache: {n} textures loaded ({total/1e6:.0f}MB)')

    def get_thumbnail(self, texture_path):
        return self.thumbnails.get(texture_path.lower())

    def __len__(self):
        return len(self.thumbnails)


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


class V12Dataset(Dataset):
    def __init__(self, paths, mcly_vocab, mcly_unk, augment=False, tileset_cache=None):
        self.paths = paths
        self.vocab = mcly_vocab
        self.unk = mcly_unk
        self.augment = augment
        self.tileset_cache = tileset_cache

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(path)

        # Minimap input (5ch)
        mm = torch.from_numpy(data['minimap_rgb_256'].astype(np.float32))
        if mm.dim() == 3 and mm.shape[-1] == 3: mm = mm.permute(2, 0, 1)
        mm = mm / 255.0
        luma = mm.mean(dim=0, keepdim=True)
        gy = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)
        gx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32)
        grad = (F.conv2d(luma.unsqueeze(0), gx.unsqueeze(0), padding=1).abs() +
                F.conv2d(luma.unsqueeze(0), gy.unsqueeze(0), padding=1).abs()).squeeze(0)
        inp = torch.cat([mm, luma, grad], dim=0)

        # Tileset reference channels (12ch: RGB × 4 layers)
        has_tileset = self.tileset_cache is not None
        if has_tileset and 'mcly_texture_ids' in data:
            mcly_raw = data['mcly_texture_ids']
            mcly_t = torch.from_numpy(mcly_raw.astype(np.int64))
            if mcly_t.dim() == 3 and mcly_t.shape[0] == 4:
                mcly_t = mcly_t.permute(1, 2, 0)

            sidecar = Path(path).with_name(Path(path).stem + '_metadata.json')
            texture_names = []
            if sidecar.exists():
                with open(sidecar) as f:
                    texture_names = json.load(f).get('mcly_texture_names', [])

            tileset_ch = torch.zeros(12, 256, 256, dtype=torch.float32)
            for cy in range(16):
                for cx in range(16):
                    for layer in range(min(4, mcly_t.shape[-1])):
                        tid = int(mcly_t[cy, cx, layer])
                        if tid < 0 or tid >= len(texture_names):
                            continue
                        thumb = self.tileset_cache.get_thumbnail(texture_names[tid])
                        if thumb is None:
                            continue
                        thumb_t = torch.from_numpy(thumb.astype(np.float32)) / 255.0
                        ch_base = layer * 3
                        tileset_ch[ch_base:ch_base+3,
                                   cy*16:(cy+1)*16,
                                   cx*16:(cx+1)*16] = thumb_t.permute(2, 0, 1)
            inp = torch.cat([inp, tileset_ch], dim=0)
        elif has_tileset:
            inp = torch.cat([inp, torch.zeros(12, 256, 256)], dim=0)

        # MCAL target
        has_mcal = 'mcal_alpha_pack_256' in data
        if has_mcal:
            mcal = torch.from_numpy(data['mcal_alpha_pack_256'].astype(np.float32))
            if mcal.dim() == 3 and mcal.shape[0] == 4: mcal = mcal.permute(1, 2, 0)
        else:
            mcal = torch.zeros(256, 256, 4)

        # MCLY target
        has_mcly = 'mcly_texture_ids' in data
        if has_mcly:
            mcly = torch.from_numpy(data['mcly_texture_ids'].astype(np.int64))
            if mcly.dim() == 3 and mcly.shape[-1] == 4: mcly = mcly[..., 0]
            if mcly.dim() == 3 and mcly.shape[0] == 4: mcly = mcly[0]
            mapped = mcly.clone()
            for old_id, new_id in self.vocab.items():
                mapped[mcly == old_id] = new_id
            mapped[mcly < 0] = self.unk
        else:
            mapped = torch.full((16, 16), self.unk, dtype=torch.long)

        # Residual target (from pre-computed MapTexture)
        composited_path = Path(path).with_name(Path(path).stem + '_composited.npz')
        has_residual = composited_path.exists()
        if has_residual:
            cz = np.load(composited_path)
            residual = torch.from_numpy(cz['texture_residual_256'].astype(np.float32)) / 255.0
            cz.close()
            if residual.dim() == 3 and residual.shape[0] == 3: residual = residual.permute(1, 2, 0)
        else:
            residual = torch.zeros(3, 256, 256)

        data.close()

        if self.augment:
            if random.random() > 0.5:
                inp = torch.flip(inp, [-1]); mcal = torch.flip(mcal, [0]); mapped = torch.flip(mapped, [0])
                has_mcly = has_mcly and torch.flip(torch.ones(1), [0]).item() > 0
                residual = torch.flip(residual, [-1])
            if random.random() > 0.5:
                inp = torch.flip(inp, [-2]); mcal = torch.flip(mcal, [1]); mapped = torch.flip(mapped, [1])
                residual = torch.flip(residual, [-2])

        return inp, {
            'mcal': mcal.permute(2, 0, 1),
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
    return {tid: idx for idx, (tid, c) in enumerate(cnt.items()) if c >= min_n}, len(cnt)


# ═══════════════════════════════════════════════════════════════════════════
# Model — ConvNeXt V2 Nano → 3 task heads
# ═══════════════════════════════════════════════════════════════════════════

class LN2d(nn.Module):
    def __init__(self, d): super().__init__(); self.n = nn.LayerNorm(d, 1e-6)
    def forward(self, x): return self.n(x.permute(0,2,3,1)).permute(0,3,1,2)

class DecBlock(nn.Module):
    def __init__(self, i, s, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, o, 2, 2)
        self.f = nn.Sequential(nn.Conv2d(o+s, o, 3, 1), LN2d(o), nn.GELU(), nn.Conv2d(o, o, 3, 1), LN2d(o), nn.GELU())
    def forward(self, x, s):
        x = self.up(x)
        if x.shape[-1] != s.shape[-1]: x = F.interpolate(x, s.shape[-2:], mode='bilinear', align_corners=False)
        return self.f(torch.cat([x, s], 1))

class V12Model(nn.Module):
    def __init__(self, num_tex):
        super().__init__()
        if timm is None: raise RuntimeError("pip install timm")
        bb = timm.create_model('convnextv2_nano', pretrained=False)
        self.stages = nn.ModuleList(bb.stages)
        sc = bb.stem[0]
        self.stem = nn.Sequential(nn.Conv2d(17, sc.out_channels, kernel_size=sc.kernel_size, stride=sc.stride, padding=sc.padding), bb.stem[1])
        with torch.no_grad():
            self.stem[0].weight[:, :3] = sc.weight; self.stem[0].weight[:, 3:] = 0

        ch = [80, 160, 320, 640]
        self.d3 = DecBlock(ch[3], ch[2], 128)
        self.d2 = DecBlock(128, ch[1], 128)
        self.d1 = DecBlock(128, ch[0], 128)
        self.d0 = nn.Sequential(nn.Conv2d(128, 64, 3, 1), LN2d(64), nn.GELU())

        self.mcal = nn.Sequential(nn.Conv2d(64, 32, 3, 1), nn.GELU(), nn.Conv2d(32, 4, 3, 1))
        self.mcly = nn.Sequential(nn.Conv2d(64, 64, 1), nn.GELU(), nn.Conv2d(64, num_tex, 3, 1))
        self.resid = nn.Sequential(nn.Conv2d(64, 32, 3, 1), nn.GELU(), nn.Conv2d(32, 3, 3, 1))

    def forward(self, x):
        x = self.stem(x)
        fs = []; y = x
        for s in self.stages: y = s(y); fs.append(y)
        f0, f1, f2, f3 = fs
        d = self.d3(f3, f2); d = self.d2(d, f1); d = self.d1(d, f0)
        d = self.d0(d)
        d = F.interpolate(d, size=(256,256), mode='bilinear', align_corners=False)
        return {'mcal': torch.sigmoid(F.interpolate(self.mcal(d), size=(256,256), mode='bilinear', align_corners=False)),
                'mcly': F.interpolate(self.mcly(d), size=16, mode='bilinear', align_corners=False),
                'residual': torch.sigmoid(F.interpolate(self.resid(d), size=(256,256), mode='bilinear', align_corners=False)),
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

    vocab, unk = build_vocab(train_p)
    ntex = len(vocab) + 1
    print(f'MCLY vocab: {ntex} classes')

    tileset_cache = None
    if args.tileset_dir:
        tileset_cache = TilesetCache(args.tileset_dir)
    train_ds = V12Dataset(train_p, vocab, unk, augment=True, tileset_cache=tileset_cache)
    val_ds = V12Dataset(val_p, vocab, unk, augment=False, tileset_cache=tileset_cache)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

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
            # MCAL L1 (5x weight — alpha is sparse)
            if hm.sum() > 0:
                m = F.l1_loss(p['mcal'], tm, reduction='none').mean(dim=(1,2,3))
                loss = loss + 5.0 * ((m * hm).sum() / hm.sum())
            # MCLY CE
            if hl.sum() > 0:
                c = F.cross_entropy(p['mcly'], tlbl, ignore_index=unk, reduction='none')
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
                    best_val = vloss/vc; torch.save({'model': model.state_dict(), 'epoch': ep, 'vocab': vocab, 'unk': unk}, out / 'best.pt')
            print(f'E{ep:4d} loss={avg:.4f} lr={lrv:.2e}{vl}')
            model.train()

        torch.save({'model': model.state_dict(), 'epoch': ep, 'optimizer': opt.state_dict(), 'vocab': vocab, 'unk': unk}, out / 'last.pt')

    print(f'\nDone. Best val: {best_val:.4f}. {out}')


def main():
    p = argparse.ArgumentParser(description='V12 Texture Decomposer')
    p.add_argument('input', nargs='+')
    p.add_argument('--output-dir', '-o', default='runs/v12')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-samples', type=int, default=2000)
    p.add_argument('--tileset-dir', type=str, default=None,
                   help='Harvested tileset PNG directory (with tileset_index.json)')
    args = p.parse_args()
    train_v12(args)

if __name__ == '__main__':
    main()
