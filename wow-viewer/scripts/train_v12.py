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
# Tileset Cache (for reconstruction loss texture grid)
# ═══════════════════════════════════════════════════════════════════════════

class TilesetCache:
    """Load tileset PNGs from harvested tileset directory, hold 16×16 thumbnails for reconstruction loss."""
    def __init__(self, tileset_dir):
        self.thumbnails = {}  # normalized texture path → (16,16,3) float32 [0,1]
        idx_path = os.path.join(tileset_dir, 'tileset_index.json')
        if not os.path.exists(idx_path): return
        with open(idx_path) as f:
            idx = json.load(f)
        from PIL import Image
        for tex_path, file_path in idx.get('textures', {}).items():
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path).convert('RGB')
                    thumb = np.asarray(img.resize((16, 16), Image.BILINEAR), dtype=np.float32) / 255.0
                    self.thumbnails[tex_path.lower()] = thumb
                except Exception:
                    pass
        print(f'TilesetCache: {len(self.thumbnails)} textures')

    def get_thumbnail(self, texture_path):
        return self.thumbnails.get(texture_path.lower())


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
    """3ch minimap in, MCAL/MCLY/residual + texture grid out."""
    def __init__(self, paths, mcly_vocab, augment=False, residual_dir=None, tileset_cache=None):
        self.paths = paths
        self.vocab = mcly_vocab
        self.augment = augment
        self.residual_dir = Path(residual_dir) if residual_dir else None
        self.tileset_cache = tileset_cache

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
                'mm_orig': torch.zeros(3, 256, 256),
                'tex_grid': torch.zeros(4, 3, 256, 256),
                'has_mcal': torch.tensor(0.0),
                'has_mcly': torch.tensor(0.0),
                'has_residual': torch.tensor(0.0),
                'has_tex_grid': torch.tensor(0.0),
            }

        # Input and original minimap for reconstruction loss
        mm_raw = torch.from_numpy(data['minimap_rgb_256'].astype(np.float32))
        if mm_raw.dim() == 3 and mm_raw.shape[-1] == 3: mm_raw = mm_raw.permute(2, 0, 1)
        mm = (mm_raw / 255.0 - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        mm_orig = mm_raw / 255.0  # (3, 256, 256) in [0, 1] for reconstruction loss

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

        # Texture grid (4, 256, 256, 3) — per-layer RGB for reconstruction loss
        has_tex_grid = self.tileset_cache is not None and 'mcly_texture_ids' in data
        if has_tex_grid:
            mcly_raw = data['mcly_texture_ids']
            if mcly_raw.ndim == 3 and mcly_raw.shape[0] == 4:
                mcly_raw = mcly_raw.transpose(1, 2, 0)

            sidecar = Path(path).with_name(Path(path).stem + '_metadata.json')
            texture_names = []
            if sidecar.exists():
                with open(sidecar) as f:
                    texture_names = json.load(f).get('mcly_texture_names', [])

            tex_grid = np.zeros((4, 256, 256, 3), dtype=np.float32)
            for cy in range(16):
                for cx in range(16):
                    for layer in range(min(4, mcly_raw.shape[-1])):
                        tid = int(mcly_raw[cy, cx, layer])
                        if tid < 0 or tid >= len(texture_names): continue
                        thumb = self.tileset_cache.get_thumbnail(texture_names[tid])
                        if thumb is None: continue
                        tex_grid[layer, cy*16:(cy+1)*16, cx*16:(cx+1)*16] = thumb
            tex_grid = torch.from_numpy(tex_grid).permute(0, 3, 1, 2)  # (4,3,256,256)
        else:
            tex_grid = torch.zeros(4, 3, 256, 256)

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
                mm_orig = torch.flip(mm_orig, [-1]); tex_grid = torch.flip(tex_grid, [-1])
            if random.random() > 0.5:
                mm = torch.flip(mm, [-2]); mcal = torch.flip(mcal, [-2])
                mapped = torch.flip(mapped, [1]); residual = torch.flip(residual, [-2])
                mm_orig = torch.flip(mm_orig, [-2]); tex_grid = torch.flip(tex_grid, [-2])

        return mm, {
            'mcal': mcal,
            'mcly': mapped,
            'residual': residual,
            'mm_orig': mm_orig,
            'tex_grid': tex_grid,
            'has_mcal': torch.tensor(1.0 if has_mcal else 0.0),
            'has_mcly': torch.tensor(1.0 if has_mcly else 0.0),
            'has_residual': torch.tensor(1.0 if has_residual else 0.0),
            'has_tex_grid': torch.tensor(1.0 if has_tex_grid else 0.0),
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
    """SegFormer B2 backbone → MCLY head (16×16) + hierarchical MCAL + residual (256×256).

    MCAL is predicted sequentially: L0 (base) first, then L1 conditioned on L0,
    then L2 conditioned on L0+L1, then L3 (detail) conditioned on L0+L1+L2.
    This encodes the terrain blending hierarchy directly into the architecture.
    """
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

        # Hierarchical MCAL heads: predict α0→α1→α2→α3 sequentially.
        # Each head sees shared features + all previous layer predictions.
        feat_dim = 64 + decoder_dim  # 320 channels: d_full (64) + p0_up (256)
        head_ch = 32
        mcal_conv = lambda in_ch: nn.Sequential(
            nn.Conv2d(in_ch, head_ch * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(head_ch * 2, head_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(head_ch, 1, 3, 1, 1),
        )
        self.mcal_head_l0 = mcal_conv(feat_dim)        # features → α0
        self.mcal_head_l1 = mcal_conv(feat_dim + 1)    # features + α0 → α1
        self.mcal_head_l2 = mcal_conv(feat_dim + 2)    # features + α0+α1 → α2
        self.mcal_head_l3 = mcal_conv(feat_dim + 3)    # features + α0+α1+α2 → α3

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

        a0 = torch.sigmoid(self.mcal_head_l0(f))                          # L0 base
        a1 = torch.sigmoid(self.mcal_head_l1(torch.cat([f, a0], dim=1)))   # L1 overlay on L0
        a2 = torch.sigmoid(self.mcal_head_l2(torch.cat([f, a0, a1], dim=1)))  # L2 features
        a3 = torch.sigmoid(self.mcal_head_l3(torch.cat([f, a0, a1, a2], dim=1)))  # L3 detail
        mcal = torch.cat([a0, a1, a2, a3], dim=1)
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

    tileset_cache = None
    if args.tileset_dir:
        tileset_cache = TilesetCache(args.tileset_dir)
    train_ds = V12Dataset(train_p, vocab, augment=True, residual_dir=args.residual_dir, tileset_cache=tileset_cache)
    val_ds = V12Dataset(val_p, vocab, augment=False, residual_dir=args.residual_dir, tileset_cache=tileset_cache)
    nw = args.num_workers
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    model = V12Model(ntex).to(dev)
    if args.compile and dev.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('torch.compile: ON (reduce-overhead)')
        except Exception as e:
            print(f'torch.compile: SKIPPED ({e})')
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

        model.train()
        tl = 0.0; nb = 0
        # Per-component accumulators (raw, unweighted)
        lm_acc = 0.0; lc_acc = 0.0; lr_acc = 0.0
        lm_n = 0; lc_n = 0; lr_n = 0
        pmax_acc = 0.0; pnz_acc = 0.0; gtz_acc = 0.0; gtnz_acc = 0.0; stat_n = 0
        for inp, tgt in train_ld:
            inp = inp.to(dev, non_blocking=True)
            tm = tgt['mcal'].to(dev, non_blocking=True)
            tlbl = tgt['mcly'].to(dev, non_blocking=True)
            tr = tgt['residual'].to(dev, non_blocking=True)
            hm = tgt['has_mcal'].to(dev)
            hl = tgt['has_mcly'].to(dev)
            hr = tgt['has_residual'].to(dev)
            hm_orig = tgt.get('mm_orig', torch.zeros_like(inp[:,:3])).to(dev)
            htg = tgt.get('tex_grid', torch.zeros(B, 4, 3, 256, 256)).to(dev)
            htg_flag = tgt.get('has_tex_grid', torch.zeros(B)).to(dev)

            if scaler:
                with torch.amp.autocast('cuda'): p = model(inp)
            else: p = model(inp)

            B = inp.shape[0]
            loss = 0.0
            # MCAL L1 (5× weight — alpha is sparse)
            if hm.sum() > 0:
                m = F.l1_loss(p['mcal'], tm, reduction='none').mean(dim=(1,2,3))
                loss = loss + 5.0 * ((m * hm).sum() / hm.sum())
                lm_acc = lm_acc + (m * hm).sum().item(); lm_n += hm.sum().item()
            # MCLY CE (-100 = ignore)
            if hl.sum() > 0:
                c = F.cross_entropy(p['mcly'], tlbl, ignore_index=-100, reduction='none')
                loss = loss + 0.2 * ((c.view(B,-1).mean(1) * hl).sum() / hl.sum())
                lc_acc = lc_acc + (c.view(B,-1).mean(1) * hl).sum().item(); lc_n += hl.sum().item()
            # Residual L1
            if hr.sum() > 0:
                r = F.l1_loss(p['residual'], tr, reduction='none').mean(dim=(1,2,3))
                loss = loss + 0.5 * ((r * hr).sum() / hr.sum())
                lr_acc = lr_acc + (r * hr).sum().item(); lr_n += hr.sum().item()
            # Reconstruction loss: composite MCAL × textures + residual → match minimap
            if htg_flag.sum() > 0:
                # composite = Σ_l α_l × Texture_l  at each pixel
                composite = (p['mcal'].unsqueeze(2) * htg).sum(dim=1)  # (B,3,256,256)
                reconstructed = composite + p['residual']
                recon = F.l1_loss(reconstructed, hm_orig, reduction='none').mean(dim=(1,2,3))
                loss = loss + 1.0 * ((recon * htg_flag).sum() / htg_flag.sum())

            # MCAL prediction stats (detect all-zeros shortcut)
            if hm.sum() > 0:
                pm = p['mcal'].detach()
                gm = tm.detach()
                pmax_acc += pm.amax(dim=(1,2,3)).mean().item()
                pnz_acc += (pm > 0.01).float().mean().item()
                gtz_acc += gm.amax(dim=(1,2,3)).mean().item()
                gtnz_acc += (gm > 0.01).float().mean().item()
                stat_n += 1

            if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
            else: loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad()
            tl += loss.item(); nb += 1

        avg = tl / nb
        lm_avg = lm_acc / max(lm_n, 1)
        lc_avg = lc_acc / max(lc_n, 1)
        lr_avg = lr_acc / max(lr_n, 1)
        pm_avg = pmax_acc / max(stat_n, 1)
        pnz_avg = pnz_acc / max(stat_n, 1)
        gtz_avg = gtz_acc / max(stat_n, 1)
        gtnz_avg = gtnz_acc / max(stat_n, 1)
        if ep % 10 == 0:
            vl = ' —'
            if val_ld:
                model.eval(); vloss = 0.0; vc = 0
                vm_avg = 0.0; vp_avg = 0.0; vnz_avg = 0.0; vstat_n = 0
                with torch.no_grad():
                    for iv, tv in val_ld:
                        iv = iv[:4].to(dev)
                        pv = model(iv)
                        if tv['has_mcal'].sum() > 0:
                            tmv = tv['mcal'][:4].to(dev)
                            l1v = F.l1_loss(pv['mcal'][:4], tmv).item()
                            vloss += l1v; vc += 1
                            vm_avg += pv['mcal'][:4].amax(dim=(1,2,3)).mean().item()
                            vnz_avg += (pv['mcal'][:4] > 0.01).float().mean().item()
                            vstat_n += 1
                            if vc >= 2: break
                vloss /= max(vc, 1)
                vm_avg /= max(vstat_n, 1); vnz_avg /= max(vstat_n, 1)
                vl = f' v_mcal={vloss:.4f} v_max={vm_avg:.3f} v_nz={vnz_avg:.3f}'
                if vc and vloss < best_val:
                    best_val = vloss
                    sd = model.state_dict()
                    if any(k.startswith('_orig_mod.') for k in sd):
                        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
                    torch.save({'model': sd, 'epoch': ep, 'vocab': vocab}, out / 'best.pt')
            print(f'E{ep:4d} L={avg:.4f} mc={lm_avg:.4f} my={lc_avg:.3f} rs={lr_avg:.4f} '
                  f'pm={pm_avg:.3f} pnz={pnz_avg:.3f} gt={gtz_avg:.3f} gnz={gtnz_avg:.3f}{vl}')
            model.train()

        sd = model.state_dict()
        if any(k.startswith('_orig_mod.') for k in sd):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        torch.save({'model': sd, 'epoch': ep, 'optimizer': opt.state_dict(), 'vocab': vocab}, out / 'last.pt')

    print(f'\nDone. Best val: {best_val:.4f}. {out}')


def main():
    p = argparse.ArgumentParser(description='V12 Texture Decomposer (Stage 1) — SegFormer backbone, 3ch RGB input')
    p.add_argument('input', nargs='*')
    p.add_argument('--output-dir', '-o', default='runs/v12_stage1')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-samples', type=int, default=2000)
    p.add_argument('--num-workers', type=int, default=2,
                   help='DataLoader workers for parallel data loading (default: 2)')
    p.add_argument('--residual-dir', type=str, default=None,
                   help='Directory with pre-computed _composited.npz files (e.g. output/tmp/maptextures)')
    p.add_argument('--tileset-dir', type=str, default=None,
                   help='Harvested tileset PNG directory for reconstruction loss texture grid (e.g. output/tmp/tilesets)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint .pt file for visualization mode')
    p.add_argument('--tile', type=str, default=None,
                   help='Single .npz tile path for visualization mode')
    p.add_argument('--compile', action='store_true',
                   help='Use torch.compile on the model (CUDA only, first epoch is slow)')
    args = p.parse_args()
    if args.checkpoint:
        visualize(args)
    else:
        if not args.input:
            p.error('input manifest(s) required for training mode')
        train_v12(args)


def visualize(args):
    """Run inference on a single tile and save PNG visualization."""
    try:
        from PIL import Image
    except ImportError:
        print('pip install Pillow'); return

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # Strip _orig_mod. prefix from compiled checkpoints
    sd = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model = V12Model(len(ckpt['vocab'])).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f'Loaded checkpoint: epoch={ckpt.get("epoch", "?")}')

    tile_path = args.tile
    if not tile_path:
        print('--tile required for visualization')
        return
    if not os.path.exists(tile_path):
        print(f'Tile not found: {tile_path}')
        return

    data = np.load(tile_path)
    # Input
    mm = torch.from_numpy(data['minimap_rgb_256'].astype(np.float32))
    if mm.dim() == 3 and mm.shape[-1] == 3: mm = mm.permute(2, 0, 1)
    inp = (mm / 255.0 - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)

    with torch.no_grad():
        out = model(inp.unsqueeze(0).to(dev))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(tile_path).stem
    base = out_dir / stem
    print(f'Saving visualizations to {out_dir}/')

    # Minimap (denormalize)
    mm_vis = (mm / 255.0).clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((mm_vis * 255).astype(np.uint8)).save(f'{base}_minimap.png')

    # MCAL predicted vs ground truth
    mc = out['mcal'][0].cpu()
    gt_mc = torch.from_numpy(data['mcal_alpha_pack_256'].astype(np.float32))
    if gt_mc.dim() == 3 and gt_mc.shape[-1] == 4: gt_mc = gt_mc.permute(2, 0, 1)

    for li in range(4):
        pred = mc[li].numpy()
        gt = gt_mc[li].numpy()
        Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).save(f'{base}_mcal_pred_l{li}.png')
        Image.fromarray((gt * 255).clip(0, 255).astype(np.uint8)).save(f'{base}_mcal_gt_l{li}.png')

    # MCAL stats
    mc_max = mc.amax(dim=(1,2)).numpy()
    mc_nz = (mc > 0.01).float().mean(dim=(1,2)).numpy()
    gt_max = gt_mc.amax(dim=(1,2)).numpy()
    gt_nz = (gt_mc > 0.01).float().mean(dim=(1,2)).numpy()
    for li in range(4):
        print(f'  Layer {li}: pred_max={mc_max[li]:.3f} pred_nz={mc_nz[li]:.3f}  gt_max={gt_max[li]:.3f} gt_nz={gt_nz[li]:.3f}')

    # MCLY predicted vs ground truth
    mcly_logits = out['mcly'][0].cpu()
    mcly_pred = mcly_logits.argmax(dim=0).numpy().astype(np.int32)  # (16, 16)

    # Need reverse vocab to map class → texture name for display
    rev_vocab = {v: k for k, v in ckpt['vocab'].items()}

    # Load metadata for texture names
    sidecar_path = Path(tile_path).with_name(Path(tile_path).stem + '_metadata.json')
    texture_names = []
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            texture_names = json.load(f).get('mcly_texture_names', [])

    if 'mcly_texture_ids' in data:
        mcly_gt = torch.from_numpy(data['mcly_texture_ids'].astype(np.int64))
        if mcly_gt.dim() == 3 and mcly_gt.shape[-1] == 4: mcly_gt = mcly_gt[..., 0]
        mcly_gt = mcly_gt.numpy()

        # Visualize MCLY as color-coded 16×16
        # Use a color map
        colors = np.array([
            [255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
            [128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],
            [192,192,192],[128,128,128],[64,0,0],[0,64,0],[0,0,64],[64,64,0],
            [64,0,64],[0,64,64],[255,128,0],[128,255,0],[0,255,128],[128,0,255],
            [0,128,255],[255,0,128],[255,128,128],[128,255,128],[128,128,255],
            [255,255,128]], dtype=np.uint8)

        pred_vis = np.zeros((16, 16, 3), dtype=np.uint8)
        gt_vis = np.zeros((16, 16, 3), dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                pid = mcly_pred[y, x]
                gid = mcly_gt[y, x]
                if gid >= 0 and gid < len(texture_names):
                    cidx = hash(texture_names[gid]) % len(colors)
                    gt_vis[y, x] = colors[cidx]
                if pid < len(ckpt['vocab']):
                    tid = rev_vocab.get(int(pid), -1)
                    if tid >= 0 and tid < len(texture_names):
                        cidx = hash(texture_names[tid]) % len(colors)
                        pred_vis[y, x] = colors[cidx]
        # Scale up 16×16 to 256×256 for visibility
        pred_vis = np.repeat(np.repeat(pred_vis, 16, axis=0), 16, axis=1)
        gt_vis = np.repeat(np.repeat(gt_vis, 16, axis=0), 16, axis=1)
        Image.fromarray(pred_vis).save(f'{base}_mcly_pred.png')
        Image.fromarray(gt_vis).save(f'{base}_mcly_gt.png')

        # Accuracy
        valid = mcly_gt >= 0
        correct = (mcly_pred == mcly_gt) & valid
        acc = correct.sum() / max(valid.sum(), 1)
        print(f'  MCLY accuracy: {acc:.3f} ({correct.sum()}/{valid.sum()})')

    # Residual predicted vs ground truth
    rs = out['residual'][0].cpu()
    # Load from composited if available
    cp_paths = [Path(tile_path).with_name(f'{stem}_composited.npz')]
    if args.residual_dir:
        cp_paths.append(Path(args.residual_dir) / f'{stem}_composited.npz')
    cp = next((p for p in cp_paths if p.exists()), None)
    if cp:
        cz = np.load(cp)
        gt_rs = torch.from_numpy(cz['texture_residual_256'].astype(np.float32)) / 255.0
        cz.close()
        if gt_rs.dim() == 3 and gt_rs.shape[-1] == 3: gt_rs = gt_rs.permute(2, 0, 1)
    else:
        gt_rs = torch.zeros(3, 256, 256)

    # Save residual as image (shift from [-1,1] to [0,1] for display)
    def save_residual(t, path):
        t = t.clamp(-1, 1)
        t = (t + 1) / 2
        Image.fromarray((t.permute(1,2,0).numpy() * 255).astype(np.uint8)).save(path)

    save_residual(rs, f'{base}_residual_pred.png')
    save_residual(gt_rs, f'{base}_residual_gt.png')

    print(f'Done. Visualizations in {out_dir}/')


if __name__ == '__main__':
    main()
