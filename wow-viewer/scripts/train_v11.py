import argparse, json, gc, sys, math, random, shutil, traceback, time
import os
os.environ.setdefault('TORCHINDUCTOR_CPP_WRAPPER', '0')
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
# LR is manual — no scheduler objects needed
import numpy as np

try:
    import timm
except ImportError:
    timm = None

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

N_CHANNELS = 26
CHANNEL_NAMES = [
    "minimap_rgb_0", "minimap_rgb_1", "minimap_rgb_2",
    "mcal_alpha_0", "mcal_alpha_1", "mcal_alpha_2", "mcal_alpha_3",
    "mcnr_normal_xyz_0", "mcnr_normal_xyz_1", "mcnr_normal_xyz_2",
    "mccv_rgb_0", "mccv_rgb_1", "mccv_rgb_2",
    "coarse_height_17_prior",
    "liquid_mask", "liquid_height",
    "object_mask", "object_precise_mask",
    "pm4_path_mask", "pm4_building_mask", "pm4_mprl_mask",
    "hole_mask_upsampled",
    "minimap_luma", "minimap_detail_gradient",
    "height_range_context", "detail_energy_context",
]

# ═══════════════════════════════════════════════════════════════════════════
# V11 Model — ConvNeXt V2 Tiny Encoder + ConvNeXt Decoder + Multi-Task Heads
# ═══════════════════════════════════════════════════════════════════════════

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

    def forward(self, x):
        r = self.conv_dw(x)
        r = self.norm(r)
        r = self.mlp(r)
        return x + r

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        fuse_ch = out_ch + skip_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_ch, out_ch, 3, padding=1),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            LayerNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class V11TerrainModel(nn.Module):
    def __init__(self, in_channels=5, decoder_dim=256, num_texture_classes=128):
        super().__init__()

        if timm is None:
            raise RuntimeError("timm is required for ConvNeXt backbone. Install with: pip install timm")

        backbone = timm.create_model('convnextv2_tiny', pretrained=False)
        self.encoder_stages = nn.ModuleList(backbone.stages)
        self.encoder_channels = [96, 192, 384, 768]

        # Overlapping conv stem for full 26-channel input
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3),
            LayerNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(96),
            nn.GELU(),
        )

        skips = self.encoder_channels
        dec_dim = decoder_dim

        self.dec3 = DecoderBlock(skips[3], skips[2], dec_dim)
        self.dec2 = DecoderBlock(dec_dim, skips[1], dec_dim)
        self.dec1 = DecoderBlock(dec_dim, skips[0], dec_dim)

        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(dec_dim, dec_dim // 2, 4, stride=4),
            LayerNorm2d(dec_dim // 2),
            nn.GELU(),
            ConvNeXtBlock(dec_dim // 2),
            ConvNeXtBlock(dec_dim // 2),
            ConvNeXtBlock(dec_dim // 2),
            nn.Conv2d(dec_dim // 2, dec_dim // 4, 3, padding=1),
            LayerNorm2d(dec_dim // 4),
            nn.GELU(),
        )
        self.refine_dim = dec_dim // 4

        self.head_height_17 = nn.Sequential(
            nn.Conv2d(self.refine_dim, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.head_height_65 = nn.Sequential(
            nn.Conv2d(self.refine_dim, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.head_height_257 = nn.Sequential(
            nn.Conv2d(self.refine_dim, 64, 3, padding=1), nn.GELU(),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1))
        self.head_mcal = nn.Sequential(
            nn.Conv2d(self.refine_dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 4, 3, padding=1))
        self.head_mcly = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(self.refine_dim, 128, 1), nn.GELU(),
            nn.Conv2d(128, num_texture_classes, 3, padding=1))
        self.head_hole = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(self.refine_dim, 32, 1), nn.GELU(),
            nn.Conv2d(32, 1, 1))

        self.register_parameter('log_height_sigma', nn.Parameter(torch.tensor(2.0)))
        self.register_parameter('log_mcal_sigma', nn.Parameter(torch.tensor(2.0)))
        self.register_parameter('log_mcly_sigma', nn.Parameter(torch.tensor(2.0)))
        self.register_parameter('log_hole_sigma', nn.Parameter(torch.tensor(2.0)))

    def forward(self, x):
        """Single path — the stem handles all inputs. Training masks aux channels
        before calling forward (set by training loop via _input_mask)."""
        x = self.stem(x)

        feats = []
        for stage in self.encoder_stages:
            x = stage(x)
            feats.append(x)

        assert len(feats) == 4, f"Expected 4 encoder features, got {len(feats)}"
        f0, f1, f2, f3 = feats

        d = self.dec3(f3, f2)
        d = self.dec2(d, f1)
        d = self.dec1(d, f0)
        d = self.dec0(d)

        out = {}
        out['height_17'] = F.interpolate(self.head_height_17(d), size=17, mode='bilinear', align_corners=False)
        out['height_65'] = F.interpolate(self.head_height_65(d), size=65, mode='bilinear', align_corners=False)
        out['height_257'] = F.interpolate(self.head_height_257(d), size=257, mode='bilinear', align_corners=False)
        out['mcal_alpha'] = torch.sigmoid(self.head_mcal(d))
        out['mcly_logits'] = self.head_mcly(d)
        out['hole_logits'] = self.head_hole(d)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def discover_npz_paths(input_paths):
    paths = []
    for p in input_paths:
        p = str(p)
        if p.endswith('.npz'):
            paths.append(p)
        elif p.endswith('.json'):
            with open(p) as f:
                manifest = json.load(f)
            entries = manifest.get('entries', manifest.get('samples', manifest.get('data', [])))
            if not entries and isinstance(manifest, list):
                entries = manifest
            for e in entries:
                if isinstance(e, str):
                    paths.append(e)
                elif isinstance(e, dict):
                    for key in ('shard_path', 'npz_path', 'path', 'file'):
                        if key in e:
                            paths.append(str(e[key]))
                            break
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith('.npz'):
                        paths.append(os.path.join(root, f))
    return list(set(paths))


def compute_signal_stats(shard_paths, verbose=True):
    signal_counts = defaultdict(int)
    n_signal = 0
    sample_paths = []
    for path in shard_paths:
        try:
            with np.load(path) as npz:
                signals = set(npz.files)
                signal_counts['total'] += 1
                required = {'height_257', 'height_17'}
                if required.issubset(signals):
                    n_signal += 1
                    sample_paths.append(path)
                    for s in signals:
                        signal_counts[s] += 1
        except Exception:
            continue

    if verbose:
        print(f"Discovered {len(shard_paths)} total npz files, {n_signal} with height signals")
        print("Signal coverage (top 30):")
        sorted_signals = sorted(signal_counts.items(), key=lambda x: -x[1])
        for name, count in sorted_signals[:30]:
            print(f"  {name}: {count}/{n_signal} ({count * 100 / max(n_signal, 1):.0f}%)")

    return sample_paths, dict(signal_counts)


def build_mcly_vocab(shard_paths, min_occurrences=3):
    from collections import Counter
    counter = Counter()
    for path in shard_paths:
        try:
            with np.load(path) as npz:
                if 'mcly_texture_ids' in npz:
                    ids = npz['mcly_texture_ids']
                    for i in ids.flat:
                        if i >= 0:
                            counter[int(i)] += 1
                if 'mcly_layer_mask' in npz and 'mcly_texture_ids' not in npz:
                    pass
        except Exception:
            continue
    vocab = {tid: idx for idx, (tid, count) in enumerate(counter.items()) if count >= min_occurrences}
    unk_idx = len(vocab)
    return vocab, unk_idx


def _ablated_minimap(mcal, mcly_ids, tileset_cache, texture_names):
    """Composite a minimap with 1-3 random texture layers ablated."""
    if tileset_cache is None or not texture_names:
        return None
    mcal = mcal.astype(np.float32)
    if mcal.ndim == 3 and mcal.shape[0] in (1, 4):
        mcal = mcal.transpose(1, 2, 0)
    if mcly_ids.ndim == 3 and mcly_ids.shape[0] == 4:
        mcly_ids = mcly_ids.transpose(1, 2, 0)

    keep_count = random.randint(1, 3)
    keep_layers = sorted(random.sample([0, 1, 2, 3], keep_count))

    tile_chunks, chunk_alpha = 16, 64
    out_size = chunk_alpha * tile_chunks
    synthetic = np.zeros((out_size, out_size, 3), dtype=np.float32)
    any_tex = False

    for cy in range(tile_chunks):
        for cx in range(tile_chunks):
            for layer in keep_layers:
                if layer >= mcly_ids.shape[-1]:
                    continue
                tid = mcly_ids[cy, cx, layer]
                if tid < 0:
                    continue
                # MCLY id maps to MTEX index — texture_names[tid] is the BLP path
                tex_name = texture_names[int(tid)] if int(tid) < len(texture_names) else ""
                if not tex_name:
                    continue
                tex = tileset_cache.get(tex_name)
                if tex is None:
                    continue
                any_tex = True
                tex_h, tex_w = tex.shape[:2]
                for ly in range(chunk_alpha):
                    ty = int(ly * tex_h / chunk_alpha) % tex_h
                    for lx in range(chunk_alpha):
                        tx = int(lx * tex_w / chunk_alpha) % tex_w
                        a = mcal[cy * chunk_alpha + ly, cx * chunk_alpha + lx, layer]
                        if a <= 0.005: continue
                        py, px = cy * chunk_alpha + ly, cx * chunk_alpha + lx
                        synthetic[py, px] += tex[ty, tx].astype(np.float32) * a

    if not any_tex:
        return None
    return synthetic.clip(0, 255).astype(np.uint8)


class V11Dataset(Dataset):
    def __init__(self, shard_paths, mcly_vocab=None, mcly_unk_idx=0,
                 signal_dropout=0.15, augment=False,
                 height_mean=0.0, height_std=1.0,
                 max_cache_mb=2048, tileset_cache=None, layer_ablate_prob=0.4):
        self.paths = shard_paths
        self.mcly_vocab = mcly_vocab or {}
        self.mcly_unk_idx = mcly_unk_idx
        self.signal_dropout = signal_dropout
        self.augment = augment
        self.height_mean = height_mean
        self.height_std = height_std
        self._cache = {}
        self._cache_order = []
        self._max_cache_bytes = max_cache_mb * 1024 * 1024
        self._cache_bytes = 0
        self._tileset_cache = tileset_cache
        self._layer_ablate_prob = layer_ablate_prob
        # Per-channel dropout multipliers — higher = more likely to drop.
        # MCCV vertex colors (ch 10-12) are artist-painted, no geometric link
        # to height. At 70% base dropout, these drop ~95% of the time.
        self._dropout_mult = torch.ones(N_CHANNELS)
        if N_CHANNELS > 12:
            self._dropout_mult[10:13] = 2.0  # MCCV: 2x base dropout
        self._cache_order = []
        self._max_cache_bytes = max_cache_mb * 1024 * 1024
        self._cache_bytes = 0

    def __len__(self):
        return len(self.paths)

    def _load(self, idx):
        path = self.paths[idx]
        if path in self._cache:
            return self._cache[path]

        with np.load(path) as npz:
            raw = {k: v.copy() for k, v in npz.items() if isinstance(v, np.ndarray)}

        # Load texture names from sidecar for layer ablation
        sidecar = Path(path).with_name(Path(path).stem + '_metadata.json')
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    meta = json.loads(f.read())
                raw['_texture_names'] = meta.get('mcly_texture_names', [])
            except:
                raw['_texture_names'] = []
        else:
            raw['_texture_names'] = []

        if 'mcal_alpha_pack_256' in raw:
            a = raw['mcal_alpha_pack_256']
            if a.shape[-1] == 4 and a.shape[-2] > 256:
                ds = a.shape[-2] // 256
                a = a.reshape(a.shape[0] // ds, ds, a.shape[1] // ds, ds, 4).mean(axis=(1, 3))
                raw['mcal_alpha_pack_256'] = a.astype(np.float32)
            elif a.ndim == 3 and a.shape[0] == 4 and a.shape[1] > 256:
                ds = a.shape[1] // 256
                a = a.reshape(4, a.shape[1] // ds, ds, a.shape[2] // ds, ds).mean(axis=(2, 4))
                raw['mcal_alpha_pack_256'] = a.astype(np.float32)

        while self._cache_bytes > self._max_cache_bytes and self._cache_order:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                self._cache_bytes -= sum(arr.nbytes for arr in self._cache[oldest].values() if hasattr(arr, 'nbytes'))
                del self._cache[oldest]

        self._cache[path] = raw
        self._cache_order.append(path)
        self._cache_bytes += sum(arr.nbytes for arr in raw.values() if hasattr(arr, 'nbytes'))
        return raw

    def __getitem__(self, idx):
        data = self._load(idx)

        def get(k):
            return data.get(k)

        # Per-tile z-score normalization for height
        # Each tile normalized to its own mean/std, not global dataset stats.
        # This removes absolute elevation bias — model learns shape only.
        h257_in = get('height_257')
        if h257_in is not None:
            h_t = torch.from_numpy(h257_in.astype(np.float32)).squeeze()
            if h_t.dim() == 2:
                tile_mean = h_t.mean()
                tile_std = max(h_t.std(), 0.01)
                height_17_raw = get('height_17')
                if height_17_raw is not None:
                    h17_t = torch.from_numpy(height_17_raw.astype(np.float32)).squeeze()
                    h17_norm = (h17_t - tile_mean) / tile_std
                else:
                    h17_norm = torch.zeros(17, 17)
            else:
                tile_mean = torch.tensor(0.0)
                tile_std = torch.tensor(1.0)
                h17_norm = torch.zeros(17, 17)
        else:
            tile_mean = torch.tensor(0.0)
            tile_std = torch.tensor(1.0)
            h17_norm = torch.zeros(17, 17)

        def safe_float(arr, shape, default=0.0):
            if arr is None or arr.size == 0:
                return torch.full(shape, default, dtype=torch.float32)
            return torch.from_numpy(arr.astype(np.float32)).squeeze()

        inp = torch.zeros(N_CHANNELS, 256, 256, dtype=torch.float32)
        ci = 0

        def place(ch, name, fallback=None):
            nonlocal ci
            arr = get(name)
            if arr is None and fallback is not None:
                arr = get(fallback)
            if arr is not None:
                a = torch.from_numpy(arr.astype(np.float32)).squeeze()
                if a.dim() == 2:
                    a = a.unsqueeze(0)
                # Detect HWC layout: last dim is small (≤ch), first two are large spatial
                if a.dim() == 3 and a.shape[-1] <= ch and a.shape[-2] > 64 and a.shape[-3] > 64:
                    a = a.permute(2, 0, 1)
                if a.shape[-1] != 256 or a.shape[-2] != 256:
                    a = F.interpolate(a.unsqueeze(0), size=256, mode='bilinear', align_corners=False).squeeze(0)
                if a.shape[0] != ch:
                    a = a[:ch] if a.shape[0] > ch else a.repeat(ch // max(a.shape[0], 1) + 1, 1, 1)[:ch]
                inp[ci:ci + ch] = a
            ci += ch
        place(3, 'minimap_rgb_256')
        place(3, 'mcnr_normal_xyz', 'normal_rgb_256')
        place(3, 'mccv_rgb')
        # Skip global height_17 — replaced with per-tile normalized version below
        ci += 1
        place(1, 'unified_liquid_mask', 'liquid_mask_257')
        place(1, 'unified_liquid_height', 'liquid_height_257')
        place(1, 'object_mask_257')
        place(1, 'object_precise_mask_257', 'object_mask_precise_257')
        place(1, 'pm4_path_mask', 'pm4_mask_257')
        place(1, 'pm4_building_footprint_mask')
        place(1, 'pm4_mprl_mask')

        hole = get('hole_mask_16')
        if hole is not None:
            hole_t = torch.from_numpy(hole.astype(np.float32)).unsqueeze(0)
            hole_t = F.interpolate(hole_t.unsqueeze(0), size=256, mode='nearest').squeeze(0)
        else:
            hole_t = get('hole_mask_16x16')
            if hole_t is not None:
                hole_t = torch.from_numpy(hole_t.astype(np.float32)).unsqueeze(0)
                hole_t = F.interpolate(hole_t.unsqueeze(0), size=256, mode='nearest').squeeze(0)
            else:
                hole_t = torch.zeros(1, 256, 256)
        inp[ci] = hole_t.squeeze(0)
        ci += 1

        # Per-tile normalized height prior replaces global height_17
        inp[13] = F.interpolate(h17_norm.unsqueeze(0).unsqueeze(0), size=256,
                                mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        luma = inp[0:3].mean(dim=0, keepdim=True)
        inp[22] = luma.squeeze(0)

        grad_k = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        grad_x = F.conv2d(luma.unsqueeze(0).repeat(1, 3, 1, 1), grad_k.repeat(3, 1, 1, 1), padding=1, groups=3).abs().mean(dim=1, keepdim=True)
        grad_y = F.conv2d(luma.unsqueeze(0).repeat(1, 3, 1, 1), grad_k.transpose(2, 3).repeat(3, 1, 1, 1), padding=1, groups=3).abs().mean(dim=1, keepdim=True)
        inp[23] = (grad_x + grad_y).squeeze(0).squeeze(0)[:256, :256]

        h17 = inp[12:13]
        height_range = h17.max() - h17.min() + 1e-8
        inp[24] = torch.full((256, 256), height_range.clamp(0, 512) / 512.0)

        inp[25] = torch.full((256, 256), 0.0)

        if self.signal_dropout > 0 and getattr(self, '_is_training', False):
            p = torch.clamp(self.signal_dropout * self._dropout_mult, 0, 0.95)
            mask = torch.rand(N_CHANNELS) >= p
            mask[0:3] = True  # always keep minimap
            inp *= mask.unsqueeze(1).unsqueeze(2).float()

        targets = {}
        for key, shape in [('height_17', (1, 17, 17)), ('height_65', (1, 65, 65)), ('height_257', (1, 257, 257))]:
            arr = get(key)
            if arr is not None:
                t = torch.from_numpy(arr.astype(np.float32)).squeeze()
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                # Per-tile normalization: each tile zero-mean, unit-variance
                t_mean = tile_mean
                t_std = tile_std
                if isinstance(t_mean, torch.Tensor):
                    t_mean = t_mean.item()
                if isinstance(t_std, torch.Tensor):
                    t_std = t_std.item()
                t_std = max(t_std, 0.01)
                targets[key] = (t - t_mean) / t_std
            else:
                targets[key] = torch.zeros(shape)

        mcal_arr = get('mcal_alpha_pack_256')
        if mcal_arr is not None:
            a = torch.from_numpy(mcal_arr.astype(np.float32))
            if a.shape[-1] == 4:
                a = a.permute(2, 0, 1)
            if a.shape[-1] > 256 or a.shape[-2] > 256:
                a = F.avg_pool2d(a.unsqueeze(0), a.shape[-1] // 256).squeeze(0)
            targets['mcal_alpha'] = a
            targets['has_mcal'] = torch.tensor(1.0)
        else:
            targets['mcal_alpha'] = torch.zeros(4, 256, 256)
            targets['has_mcal'] = torch.tensor(0.0)

        mcly_ids = get('mcly_texture_ids')
        if mcly_ids is not None and self.mcly_vocab:
            ids = torch.from_numpy(mcly_ids.astype(np.int64))
            if ids.dim() == 3 and ids.shape[-1] == 4:
                ids = ids[..., 0]
            if ids.dim() == 3 and ids.shape[0] == 4:
                ids = ids[0]
            mapped = ids.clone()
            vocab_keys = set(self.mcly_vocab.keys())
            for old_id, new_id in self.mcly_vocab.items():
                mapped[ids == old_id] = new_id
            unk_mask = (ids < 0) | ~torch.tensor([id.item() in vocab_keys for id in ids.flatten()]).reshape(ids.shape)
            mapped[unk_mask] = self.mcly_unk_idx
            targets['mcly_labels'] = mapped.unsqueeze(0)
            targets['has_mcly'] = torch.tensor(1.0)
        else:
            targets['mcly_labels'] = torch.full((1, 16, 16), self.mcly_unk_idx, dtype=torch.long)
            targets['has_mcly'] = torch.tensor(0.0)

        hole_arr = get('hole_mask_16')
        if hole_arr is not None:
            targets['hole_mask'] = torch.from_numpy(hole_arr.astype(np.float32)).unsqueeze(0)
            targets['has_hole'] = torch.tensor(1.0)
        else:
            targets['hole_mask'] = torch.zeros(1, 16, 16)
            targets['has_hole'] = torch.tensor(0.0)

        stem = Path(self.paths[idx]).stem.replace('_v10', '').replace('_v11', '')
        targets['tile_name'] = stem

        # Layer ablation: replace minimap with a stripped version (fewer texture layers visible)
        if self._tileset_cache is not None and random.random() < self._layer_ablate_prob \
                and 'mcal_alpha_pack_256' in data and 'mcly_texture_ids' in data:
            try:
                ablated = _ablated_minimap(
                    data['mcal_alpha_pack_256'], data['mcly_texture_ids'],
                    self._tileset_cache, data.get('_texture_names', []))
                if ablated is not None:
                    ablated_t = torch.from_numpy(ablated.astype(np.float32)).permute(2, 0, 1)
                    inp[0:3] = ablated_t / 255.0
            except Exception:
                pass

        # Save per-tile normalization stats for validation denormalization
        targets['tile_mean'] = torch.tensor(float(tile_mean.item() if isinstance(tile_mean, torch.Tensor) else tile_mean))
        targets['tile_std'] = torch.tensor(float(tile_std.item() if isinstance(tile_std, torch.Tensor) else tile_std))

        if self.augment:
            if random.random() > 0.5:
                inp = torch.flip(inp, dims=[-1])
                for k in targets:
                    if isinstance(targets[k], torch.Tensor) and targets[k].dim() >= 2:
                        targets[k] = torch.flip(targets[k], dims=[-1])
            if random.random() > 0.5:
                inp = torch.flip(inp, dims=[-2])
                for k in targets:
                    if isinstance(targets[k], torch.Tensor) and targets[k].dim() >= 2:
                        targets[k] = torch.flip(targets[k], dims=[-2])

        return inp, targets


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_hf_weight', torch.tensor(1.0))
        self.register_buffer('_lf_weight', torch.tensor(1.0))
        self._detail_pulse_epochs = 0

    def set_frequency_weights(self, epoch):
        """Two-phase schedule with detail pulses every 20 epochs after phase 1.
        
        Phase 1 (0-60): detail-first ramp (hf=1→0.1, lf=0.1→1)
        Phase 2 (60+): detail pulse every 20 epochs, 8 epochs each:
          hf spikes to ~0.55 on pulse start, decays to 0.1
        """
        if epoch < 60:
            t = min(epoch / 60, 1.0)
            hf = max(1.0 - t * 0.9, 0.01)
            lf = min(0.1 + t * 0.9, 1.0)
        else:
            pulse_cycle = (epoch - 60) % 20
            if pulse_cycle < 8:  # detail pulse active
                t = 1.0 - pulse_cycle / 8  # 1→0 over 8 epochs
                hf = 0.1 + 0.45 * t
                lf = 1.0 - hf
            else:  # between pulses — shape only
                hf = 0.1
                lf = 1.0

        self._hf_weight = torch.tensor(max(hf, 0.01))
        self._lf_weight = torch.tensor(min(lf, 1.0))

    def forward(self, pred, target, model, prefix=''):
        losses = {}
        total = 0.0
        B = pred['height_17'].shape[0]

        # Frequency-banded height loss with ramp weights
        t17 = target['height_17']
        p17 = F.interpolate(pred['height_17'], size=17, mode='bilinear', align_corners=False)
        t65 = target['height_65']
        p65 = F.interpolate(pred['height_65'], size=65, mode='bilinear', align_corners=False)
        t257 = target['height_257']
        p257 = F.interpolate(pred['height_257'], size=257, mode='bilinear', align_corners=False)

        # Low frequency: coarse 17x17
        lf_l1 = F.l1_loss(p17, t17)
        # Mid frequency: detail in 65 - up(17)
        mid_t = t65 - F.interpolate(t17, size=65, mode='bilinear', align_corners=False)
        mid_p = p65 - F.interpolate(p17, size=65, mode='bilinear', align_corners=False)
        mid_l1 = F.l1_loss(mid_p, mid_t)
        # High frequency: detail in 257 - up(65)
        hf_t = t257 - F.interpolate(t65, size=257, mode='bilinear', align_corners=False)
        hf_p = p257 - F.interpolate(p65, size=257, mode='bilinear', align_corners=False)
        hf_l1 = F.l1_loss(hf_p, hf_t)

        # Sobel edge loss on full-resolution height — forces sharp edges
        sx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=p257.device, dtype=p257.dtype)
        sy = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=p257.device, dtype=p257.dtype)
        edge_p = (F.conv2d(p257, sx.repeat(p257.shape[1],1,1,1), padding=1).abs() +
                  F.conv2d(p257, sy.repeat(p257.shape[1],1,1,1), padding=1).abs())
        edge_t = (F.conv2d(t257, sx.repeat(t257.shape[1],1,1,1), padding=1).abs() +
                  F.conv2d(t257, sy.repeat(t257.shape[1],1,1,1), padding=1).abs())
        edge_l1 = F.l1_loss(edge_p, edge_t)

        hf_w = self._hf_weight.item()
        lf_w = self._lf_weight.item()
        total = total + lf_w * lf_l1 + 0.5 * mid_l1 + hf_w * hf_l1 + 3.0 * edge_l1

        losses[f'{prefix}lf_l1'] = lf_l1.item()
        losses[f'{prefix}mid_l1'] = mid_l1.item()
        losses[f'{prefix}hf_l1'] = hf_l1.item()
        losses[f'{prefix}edge'] = edge_l1.item()

        def apply_uncertainty(loss_val, log_sigma, weight=1.0):
            ls = log_sigma.clamp(-3, 3)
            s = torch.exp(ls)
            return loss_val * weight / (s * s + 0.01) + ls * weight * 0.1

        mcal_mask = target.get('has_mcal', torch.zeros(B, device=pred['height_17'].device))
        if mcal_mask.sum() > 0:
            mcal = F.l1_loss(pred['mcal_alpha'], target['mcal_alpha'], reduction='none')
            mcal = (mcal.mean(dim=(1, 2, 3)) * mcal_mask).sum() / mcal_mask.sum()
            losses[f'{prefix}mcal_l1'] = mcal.item()
            total = total + apply_uncertainty(mcal, model.log_mcal_sigma, mcal_mask.mean())

        mcly_mask = target.get('has_mcly', torch.zeros(B, device=pred['height_17'].device))
        if mcly_mask.sum() > 0:
            mcly_target = target['mcly_labels'].squeeze(1)
            mcly = F.cross_entropy(pred['mcly_logits'], mcly_target,
                                   ignore_index=model.mcly_unk_idx, reduction='none')
            mcly = (mcly.view(B, -1).mean(dim=1) * mcly_mask).sum() / mcly_mask.sum()
            losses[f'{prefix}mcly_ce'] = mcly.item()
            total = total + apply_uncertainty(mcly, model.log_mcly_sigma, mcly_mask.mean())

        hole_mask_flag = target.get('has_hole', torch.zeros(B, device=pred['height_17'].device))
        if hole_mask_flag.sum() > 0:
            hole = F.binary_cross_entropy_with_logits(pred['hole_logits'], target['hole_mask'],
                                                       reduction='none')
            hole = (hole.view(B, -1).mean(dim=1) * hole_mask_flag).sum() / hole_mask_flag.sum()
            losses[f'{prefix}hole_bce'] = hole.item()
            total = total + apply_uncertainty(hole, model.log_hole_sigma, hole_mask_flag.mean())

        losses[f'{prefix}loss'] = total.item()
        return total, losses


def compute_seam_loss(pred, targets, batch_size):
    """Penalize height mismatch at tile boundaries for adjacent tiles in a batch."""
    if batch_size < 2:
        return 0.0
    
    # Parse tile coordinates from tile_name
    coords = []
    for i in range(batch_size):
        tn = targets.get('tile_name', [f'tile_{j}' for j in range(batch_size)])
        name = tn[i] if isinstance(tn, (list, tuple)) else str(tn)
        parts = name.rsplit('_', 2)
        if len(parts) == 3:
            coords.append((int(parts[-1]), int(parts[-2])))  # (X, Y) from Map_X_Y
        else:
            coords.append((i, i))
    
    seam = 0.0
    count = 0
    h257 = pred['height_257']  # [B, 1, 257, 257]
    
    for i in range(batch_size):
        xi, yi = coords[i]
        for j in range(i + 1, batch_size):
            xj, yj = coords[j]
            if xi == xj and abs(yi - yj) == 1:
                # Vertical neighbors: tile i's right edge vs tile j's left edge
                edge_i = h257[i, :, :, -1]  # rightmost column
                edge_j = h257[j, :, :, 0]   # leftmost column
                seam = seam + F.l1_loss(edge_i, edge_j)
                count += 1
            elif yi == yj and abs(xi - xj) == 1:
                # Horizontal neighbors: tile i's bottom edge vs tile j's top edge
                edge_i = h257[i, :, -1, :]  # bottom row
                edge_j = h257[j, :, 0, :]   # top row
                seam = seam + F.l1_loss(edge_i, edge_j)
                count += 1
    
    return seam / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadows = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self):
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                self.shadows[k] = self.decay * self.shadows[k] + (1 - self.decay) * v

    def apply(self):
        self.model.load_state_dict(self.shadows)

    def store(self):
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def restore(self):
        self.model.load_state_dict(self.backup)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def create_optimizer(model, opt_name='adamw', lr=2e-4, weight_decay=0.05):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'norm' in name or 'bias' in name:
            no_decay.append(p)
        else:
            decay.append(p)

    if opt_name.lower() == 'lion':
        from lion_pytorch import Lion
        return Lion([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0},
        ], lr=lr)
    else:
        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0},
        ], lr=lr, betas=(0.9, 0.95))




def save_preview_from_batch(model, inp, targets, device, height_mean, height_std, save_path, num_rows=4):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    ROW_TITLE_H = 18
    LABEL_H = 18
    PAD = 4
    FONT = ImageFont.load_default()

    model.eval()
    B = min(num_rows, inp.shape[0])
    inp_b = inp[:B].to(device)
    with torch.no_grad():
        out = model(inp_b)
    inp_cpu = inp.cpu()

    rows_list = []
    for b in range(B):
        # Minimap: first 3 channels, uint8 [0,255]
        mm = inp_cpu[b, 0:3].float().numpy().transpose(1, 2, 0)
        mm = mm.clip(0, 255).astype(np.uint8)
        h, w = mm.shape[:2]

        # Height panels: denormalize per-tile
        t_mean = targets.get('tile_mean', torch.tensor(height_mean))[b].item()
        t_std = max(targets.get('tile_std', torch.tensor(height_std))[b].item(), 0.01)
        pred_h = out['height_257'][b, 0].float().cpu().numpy() * t_std + t_mean
        targ_h = targets['height_257'][b, 0].float().cpu().numpy() * t_std + t_mean
        diff = pred_h - targ_h

        # Resize all to minimap dimensions via PIL
        def to_pil(arr):
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode='L').resize((w, h), Image.NEAREST)

        def diff_pil(arr):
            mx = max(abs(arr.min()), abs(arr.max()), 1e-5)
            norm = (arr / mx).clip(-1, 1)
            r = np.where(norm > 0, norm * 255, 0).astype(np.uint8)
            b = np.where(norm < 0, -norm * 255, 0).astype(np.uint8)
            g = (255 - abs(norm) * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([r, g, b], axis=-1)).resize((w, h), Image.NEAREST)

        panels = [
            (Image.fromarray(mm), "Minimap"),
            (to_pil(targ_h), "Target"),
            (to_pil(pred_h), "Prediction"),
            (diff_pil(diff), "Error (R+/B-)"),
        ]

        pane_w, pane_h = w, h
        row_w = pane_w * len(panels)
        row_h = ROW_TITLE_H + LABEL_H + pane_h
        row_img = Image.new("RGB", (row_w, row_h), (24, 24, 24))
        draw = ImageDraw.Draw(row_img)
        draw.rectangle((0, 0, row_w, ROW_TITLE_H), fill=(32, 32, 32))
        tname = targets.get('tile_name', [f'tile_{b}'] * B)
        draw.text((PAD, 2), str(tname[b] if isinstance(tname, list) else tname), fill=(200, 200, 200), font=FONT)

        x = 0
        for pil_img, label in panels:
            draw.rectangle((x, ROW_TITLE_H, x + pane_w, ROW_TITLE_H + LABEL_H), fill=(48, 48, 48))
            tw = draw.textbbox((0, 0), label, font=FONT)
            draw.text((x + max(PAD, (pane_w - (tw[2] - tw[0])) // 2), ROW_TITLE_H + 2), label, fill=(255, 255, 255), font=FONT)
            row_img.paste(pil_img, (x, ROW_TITLE_H + LABEL_H))
            x += pane_w

        rows_list.append(np.asarray(row_img))

    if rows_list:
        comp = rows_list[0] if len(rows_list) == 1 else np.concatenate(rows_list, axis=0)
        Image.fromarray(comp).save(save_path)


def _checkpoint_mcly_classes(resume_path, output_dir):
    """Read MCLY head output channels from checkpoint, or None."""
    candidates = []
    if resume_path:
        candidates.append(resume_path)
    auto = output_dir / 'last.pt'
    if auto.exists():
        candidates.append(str(auto))
    for path in candidates:
        if not path:
            continue
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model', {})
            w = sd.get('head_mcly.3.weight')
            if w is None:
                w = sd.get('_orig_mod.head_mcly.3.weight')
            if w is not None:
                return w.shape[0]
        except Exception as e:
            print(f"  (checkpoint MCLY read failed: {e})")
            continue
    return None

def train_v11(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    preview_dir = output_dir / 'previews'
    preview_dir.mkdir(exist_ok=True)

    print("Discovering NPZ shards...")
    shard_paths = discover_npz_paths(args.input)
    print(f"  Found {len(shard_paths)} npz files")

    sample_paths, signal_stats = compute_signal_stats(shard_paths)

    if len(sample_paths) == 0:
        print("ERROR: No usable shards found (none have height_257 + height_17)")
        sys.exit(1)

    if args.max_samples and len(sample_paths) > args.max_samples:
        random.seed(args.seed)
        sample_paths = random.sample(sample_paths, args.max_samples)
        print(f"  Subsampled to {len(sample_paths)}")

    random.seed(args.seed)
    random.shuffle(sample_paths)
    # Sort by map + coordinates so adjacent tiles cluster together
    def _sort_key(p):
        stem = Path(p).stem.replace('_v10','').replace('_v11','')
        parts = stem.rsplit('_', 2)
        if len(parts) == 3:
            return (parts[0], int(parts[2]), int(parts[1]))  # map, X, Y
        return (stem, 0, 0)
    sample_paths.sort(key=_sort_key)

    val_count = max(1, min(int(len(sample_paths) * args.val_fraction), len(sample_paths) // 2))
    train_paths = sample_paths[val_count:]
    val_paths = sample_paths[:val_count]
    if len(train_paths) == 0 or len(val_paths) == 0:
        print("ERROR: Not enough samples for train/val split")
        sys.exit(1)
    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}")

    mcly_vocab, mcly_unk_idx = build_mcly_vocab(train_paths, min_occurrences=args.mcly_min_occurrences)
    print(f"MCLY vocab: {len(mcly_vocab)} classes (+1 unknown = {len(mcly_vocab) + 1})")

    height_vals = []
    for p in train_paths[:500]:
        try:
            with np.load(p) as npz:
                if 'height_257' in npz:
                    h = npz['height_257'].astype(np.float32)
                    height_vals.extend(h.flat[:1000])
        except Exception:
            continue
    height_mean = float(np.mean(height_vals)) if height_vals else 0.0
    height_std = max(float(np.std(height_vals)) if height_vals else 1.0, 0.01)
    print(f"Height stats: mean={height_mean:.1f}, std={height_std:.1f}")

    train_ds = V11Dataset(
        train_paths, mcly_vocab, mcly_unk_idx,
        signal_dropout=args.signal_dropout, augment=True,
        height_mean=height_mean, height_std=height_std)
    val_ds = V11Dataset(
        val_paths, mcly_vocab, mcly_unk_idx,
        signal_dropout=0, augment=False,
        height_mean=height_mean, height_std=height_std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)

    model = V11TerrainModel(
        in_channels=N_CHANNELS,
        decoder_dim=args.decoder_dim,
        num_texture_classes=len(mcly_vocab) + 1)
    model.mcly_unk_idx = mcly_unk_idx
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} ({total_params * 4 / 1e6:.1f}MB @ fp32)")

    model = model.to(device)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # Monitoring trackers
    loss_ema = None
    grad_norm_ema = None
    stall_epochs = 0
    best_val_l1 = float('inf')
    last_val_l1 = -1

    optimizer = create_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)

    history = []
    start_epoch = 0

    resume_path = args.resume
    if not resume_path:
        auto = output_dir / 'last.pt'
        if auto.exists():
            resume_path = str(auto)
            print(f"Auto-resuming from {resume_path}")

    # Compile AFTER loading
    if args.use_compile and device.type == 'cuda':
        try:
            model = torch.compile(model, dynamic=False)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed: {e}")

    if args.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("  channels_last memory format enabled")

    ema = EMA(model, decay=args.ema_decay)

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_sd = ckpt['model']
        raw_sd = {k.removeprefix('_orig_mod.'): v for k, v in raw_sd.items()}
        for key in list(raw_sd.keys()):
            if 'head_mcly' in key:
                cur = model.state_dict().get(key)
                if cur is not None and cur.shape != raw_sd[key].shape:
                    print(f"  Skipping {key}: ckpt={raw_sd[key].shape} model={cur.shape}")
                    del raw_sd[key]
        model.load_state_dict(raw_sd, strict=False)
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', [])
        if 'ema' in ckpt:
            try:
                ema_raw = ckpt['ema']
                for k in list(ema_raw.keys()):
                    if 'head_mcly' in k:
                        cur = ema.shadows.get(k)
                        if cur is not None and cur.shape != ema_raw[k].shape:
                            del ema_raw[k]
                ema.shadows = ema_raw
            except Exception:
                pass
        if 'log_sigmas' in ckpt:
            model.log_height_sigma.data = torch.tensor(ckpt['log_sigmas'][0])
            model.log_mcal_sigma.data = torch.tensor(ckpt['log_sigmas'][1])
            model.log_hole_sigma.data = torch.tensor(ckpt['log_sigmas'][2])
        print(f"Resumed from epoch {start_epoch} (model+EMA weights, optimizer fresh)")

    loss_fn = UncertaintyWeightedLoss()
    _t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        loss_fn.set_frequency_weights(epoch)
        model.train()
        train_ds._is_training = True
        train_ds.signal_dropout = args.signal_dropout
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch_idx, (inp, targets) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}

            # Minimap-only pass: 60% of batches zero aux channels
            if random.random() < 0.6:
                inp[:, 3:22] = 0
                inp[:, 24:26] = 0

            pred = model(inp)
            loss, losses = loss_fn(pred, targets, model)

            # NaN guard: skip batch if loss or any gradient is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARN: NaN/Inf loss at E{epoch} B{batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            # Loss spike guard: skip if >5x running EMA
            lv = loss.item()
            if loss_ema is None:
                loss_ema = lv
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * lv
            if lv > 5 * loss_ema + 0.5 and n_batches > 10:
                print(f"  WARN: loss spike {lv:.3f} vs EMA {loss_ema:.3f}, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if gn > 0:
                    if grad_norm_ema is None:
                        grad_norm_ema = gn
                    else:
                        grad_norm_ema = 0.99 * grad_norm_ema + 0.01 * gn
                optimizer.step()
                optimizer.zero_grad()
            ema.update()

            for k, v in losses.items():
                epoch_losses[k] += v
            n_batches += 1

            if batch_idx % max(1, len(train_loader) // 5) == 0:
                ls = losses.get('loss', 0)
                if not (math.isnan(ls) or math.isinf(ls)):
                    print(f"  E{epoch:3d} B{batch_idx:4d}/{len(train_loader)} loss={ls:.4f}")

        # Manual LR schedule with floor
        if epoch < args.warmup_epochs:
            lr_val = args.learning_rate * (0.01 + 0.99 * epoch / max(args.warmup_epochs, 1))
        else:
            frac = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            lr_val = args.learning_rate * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
            lr_val = max(lr_val, args.learning_rate * 0.01)  # floor at 1% of peak
        for pg in optimizer.param_groups:
            pg['lr'] = lr_val

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        lr = lr_val

        _t0 = time.time()  # noqa: F841

        # Lightweight validation every 5 epochs
        if epoch % 5 == 0 and len(val_loader) > 0:
            model.eval()
            val_l1 = 0.0
            v_count = 0
            with torch.no_grad():
                for inp_v, targets_v in val_loader:
                    inp_v = inp_v[:4].to(device)
                    tv = {k: (v[:4].to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in targets_v.items()}
                    pv = model(inp_v)
                    val_l1 = val_l1 + F.l1_loss(pv['height_257'], tv['height_257']).item()
                    v_count += 1
                    if v_count >= 2:
                        break
            val_l1 /= v_count
            last_val_l1 = val_l1
            stall_epochs = stall_epochs + 1 if epoch > 60 and val_l1 > best_val_l1 else 0
            if epoch == 0 or val_l1 < best_val_l1:
                best_val_l1 = val_l1
                ema.store()
                ema.apply()
                raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save({
                    'model': {k.removeprefix('_orig_mod.'): v for k, v in raw.state_dict().items()},
                    'epoch': epoch, 'val_l1': val_l1,
                    'height_mean': height_mean, 'height_std': height_std,
                    'mcly_vocab': mcly_vocab, 'mcly_unk_idx': mcly_unk_idx,
                    'log_sigmas': [model.log_height_sigma.item(), model.log_mcal_sigma.item(), model.log_hole_sigma.item()],
                }, output_dir / 'best.pt')
                ema.restore()
                print(f"  -> NEW BEST (val={val_l1:.4f})")
            # Preview on val epochs
            try:
                save_preview_from_batch(model, inp_v, targets_v, device,
                    0, 1, preview_dir / f'epoch_{epoch:04d}.png', num_rows=4)
                for old_p in sorted(preview_dir.glob('epoch_*.png'))[:-5]:
                    old_p.unlink(missing_ok=True)
            except Exception:
                pass
            model.train()
            train_ds._is_training = True
        gn_str = f" g={grad_norm_ema:.1f}" if grad_norm_ema is not None else ""
        vl_str = f" val={last_val_l1:.4f}" if epoch % 5 == 0 and last_val_l1 >= 0 else ""
        print(f"E{epoch:3d} loss={avg.get('loss', 0):.4f} lr={lr:.2e}"
              f" lf={avg.get('lf_l1', 0):.4f} mid={avg.get('mid_l1', 0):.4f} hf={avg.get('hf_l1', 0):.4f}"
              f" eg={avg.get('edge', 0):.4f} mc={avg.get('mcal_l1', 0):.3f} ml={avg.get('mcly_ce', 0):.2f}{gn_str}{vl_str}")

        history.append({
            'epoch': epoch,
            'lr': lr,
            **avg,
        })

        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        ckpt = {
            'epoch': epoch,
            'model': {k.removeprefix('_orig_mod.'): v for k, v in raw_model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'scheduler': {},  # removed — manual LR schedule
            'history': history,
            'height_mean': height_mean,
            'height_std': height_std,
            'mcly_vocab': mcly_vocab,
            'mcly_unk_idx': mcly_unk_idx,
            'log_sigmas': [model.log_height_sigma.item(), model.log_mcal_sigma.item(), model.log_hole_sigma.item()],
            'args': vars(args),
        }
        # Save checkpoint
        ckpt_path = str(output_dir / 'last.pt')
        try:
            torch.save(ckpt, ckpt_path)
        except PermissionError:
            import time as _time
            _time.sleep(1)
            torch.save(ckpt, ckpt_path)

        if epoch % args.save_every == 0:
            shutil.copy(output_dir / 'last.pt', checkpoint_dir / f'epoch_{epoch:04d}.pt')

    print(f"\nTraining complete — {epoch} epochs, output: {output_dir}")

    metrics = {
        'total_params': total_params,
        'train_samples': len(train_paths),
        'val_samples': len(val_paths),
        'mcly_vocab_size': len(mcly_vocab) + 1,
        'height_mean': height_mean,
        'height_std': height_std,
        'history': history,
        'args': vars(args),
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Wrote metrics.json")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='V11 Terrain Model Trainer')

    g = p.add_argument_group('Data')
    g.add_argument('input', nargs='+', help='NPZ shards, manifest JSONs, or directories')
    g.add_argument('--output-dir', '-o', default='runs/v11_default')
    g.add_argument('--max-samples', type=int, default=1200, help='Target curated pool size')
    g.add_argument('--val-fraction', type=float, default=0.12)
    g.add_argument('--mcly-min-occurrences', type=int, default=3)

    g = p.add_argument_group('Model')
    g.add_argument('--decoder-dim', type=int, default=256)
    g.add_argument('--signal-dropout', type=float, default=0.70, help='Dropout rate for non-essential input channels — forces model to learn from minimap alone')

    g = p.add_argument_group('Training')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--batch-size', type=int, default=16)
    g.add_argument('--learning-rate', type=float, default=2e-4)
    g.add_argument('--weight-decay', type=float, default=0.05)
    g.add_argument('--optimizer', choices=['adamw', 'lion'], default='adamw')
    g.add_argument('--warmup-epochs', type=int, default=5)
    g.add_argument('--freq-ramp-epochs', type=int, default=60, help='Epochs over which to ramp from detail-first to shape-first loss')
    g.add_argument('--gradient-clip', type=float, default=1.0)
    g.add_argument('--gradient-accumulation', type=int, default=2)
    g.add_argument('--ema-decay', type=float, default=0.999)
    g.add_argument('--seed', type=int, default=1337)
    g.add_argument('--use-compile', action='store_true', help='Enable torch.compile')
    g.add_argument('--channels-last', action='store_true', help='Use channels_last memory format')

    g = p.add_argument_group('I/O')
    g.add_argument('--num-workers', type=int, default=4)
    g.add_argument('--device', default='cuda')
    g.add_argument('--save-every', type=int, default=10)
    g.add_argument('--preview-every', type=int, default=5)
    g.add_argument('--resume', help='Path to last.pt checkpoint')

    args = p.parse_args()
    print("=" * 60)
    print("V11 Terrain Model Trainer")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    train_v11(args)


if __name__ == '__main__':
    main()
