# V11.1 Terrain Model Specification

## Overview

Multi-task terrain reconstruction model that predicts height, texturing (MCAL/MCLY), and hole masks from 26 input signals. Uses a ConvNeXt V2 Tiny backbone with overlapping stem and frequency-banded loss schedule.

**Design philosophy:** Learn terrain as a composition of independent frequency bands. Detail (high-frequency) is learned first, then mid structure, then coarse shape — the reverse of traditional coarse-to-fine training. This prevents high-frequency detail from being lost during refinement.

---

## 1. Architecture

### Encoder: ConvNeXt V2 Tiny (modified)

| Layer | Output Size | Channels | Description |
|-------|-------------|----------|-------------|
| Stem Conv1 | 128×128 | 96 | 7×7 conv, stride 2, padding 3 |
| Stem LN+GELU | 128×128 | 96 | LayerNorm + GELU |
| Stem Conv2 | 64×64 | 96 | 3×3 conv, stride 2, padding 1 |
| Stem LN+GELU | 64×64 | 96 | LayerNorm + GELU |
| Stem Conv3 | 64×64 | 96 | 3×3 conv, stride 1, padding 1 |
| Stem LN+GELU | 64×64 | 96 | LayerNorm + GELU |
| Stage 0 (3 blocks) | 64×64 | 96 | ConvNeXt blocks |
| ↓ Downsample | 32×32 | 192 | LayerNorm + 2×2 conv stride 2 |
| Stage 1 (3 blocks) | 32×32 | 192 | ConvNeXt blocks |
| ↓ Downsample | 16×16 | 384 | LayerNorm + 2×2 conv stride 2 |
| Stage 2 (9 blocks) | 16×16 | 384 | ConvNeXt blocks |
| ↓ Downsample | 8×8 | 768 | LayerNorm + 2×2 conv stride 2 |
| Stage 3 (3 blocks) | 8×8 | 768 | ConvNeXt blocks |

**Stem difference from stock ConvNeXt:** The original uses a single 4×4 stride-4 non-overlapping conv, which creates a 64×64 grid artifact that propagates through the decoder. The overlapping stem (two stride-2 convs) preserves high-frequency spatial information and eliminates the grid pattern.

### Decoder: Progressive U-Net

| Stage | Input | Skip Connection | Output |
|-------|-------|----------------|--------|
| Dec3 | 8×8 (768ch) ↑2× | Stage 2 (384ch @ 16×16) | 256ch @ 16×16 |
| Dec2 | 16×16 ↑2× | Stage 1 (192ch @ 32×32) | 256ch @ 32×32 |
| Dec1 | 32×32 ↑2× | Stage 0 (96ch @ 64×64) | 256ch @ 64×64 |
| Dec0 | 64×64 ↑4× | — | 64ch @ 256×256 |

Each DecoderBlock: ConvTranspose2d (2× up) → concat skip → Conv→LN→GELU→Conv→LN→GELU.

Dec0 uses 3× ConvNeXt refinement blocks at 256×256 resolution for detail recovery.

### Task Heads

| Head | Input Dim | Architecture | Output Shape | Loss |
|------|-----------|-------------|-------------|------|
| height_17 | 64ch | Conv1×1→GELU→Conv1×1 | 1×17×17 | LF L1 (frequency-banded) |
| height_65 | 64ch | Conv1×1→GELU→Conv1×1 | 1×65×65 | Mid L1 (frequency-banded) |
| height_257 | 64ch | 3×(Conv3×3→GELU)→Conv3×3 | 1×257×257 | HF L1 (frequency-banded) |
| mcal_alpha | 64ch | 2×(Conv3×3→GELU)→Conv3×3→sigmoid | 4×256×256 | L1 on alpha weights |
| mcly_logits | 64ch | Pool→16→Conv1×1→GELU→Conv3×3 | N×16×16 | Cross-entropy |
| hole_logits | 64ch | Pool→16→Conv1×1→GELU→Conv1×1 | 1×16×16 | BCE |

### Parameter Count

| Configuration | Params | VRAM (fp32) | VRAM (compiled) |
|-------------|--------|-------------|-----------------|
| decoder_dim=256 (default) | 35.5M | 142MB | ~70MB bf16 |
| decoder_dim=96 | 29.6M | 118MB | ~59MB bf16 |

Both fit in 8GB VRAM at batch_size=8 with torch.compile. Batch_size=16 uses ~11GB.

---

## 2. Input Signals (26 channels)

| # | Name | Source NPZ Array | Range | Dropout | Rationale |
|---|------|-----------------|-------|---------|-----------|
| 0-2 | minimap_rgb | `minimap_rgb_256` | uint8 [0,255] | 1× | Direct terrain appearance, slope shading |
| 3-6 | mcal_alpha | `mcal_alpha_pack_256` | float [0,1] | 1× | Texture blend weights = artist-painted slope proxy |
| 7-9 | mcnr_normal | `mcnr_normal_xyz` / `normal_rgb_256` | float [-1,1] | 1× | Computed from height → geometrically exact |
| 10-12 | mccv_rgb | `mccv_rgb` | float [0,1] | **3×** | Vertex colors are artist-painted, no causal link to height |
| 13 | coarse_height | `height_17` | float (z-scored) | 1× | WDL or downsampled coarse terrain |
| 14 | liquid_mask | `unified_liquid_mask` / `liquid_mask_257` | float [0,1] | 1× | Water presence = flat surface |
| 15 | liquid_height | `unified_liquid_height` / `liquid_height_257` | float | 1× | Water surface Z value |
| 16 | object_mask | `object_mask_257` | float [0,1] | 1× | Buildings on flat ground |
| 17 | object_precise | `object_precise_mask_257` | float [0,1] | 1× | Precise object footprints |
| 18 | pm4_path | `pm4_path_mask` / `pm4_mask_257` | float [0,1] | 1× | Pathfinding lines follow terrain |
| 19 | pm4_building | `pm4_building_footprint_mask` | float [0,1] | 1× | Building footprints on flat |
| 20 | pm4_mprl | `pm4_mprl_mask` | float [0,1] | 1× | Portal placement on terrain |
| 21 | hole_mask | `hole_mask_16` / `hole_mask_16x16` | float [0,1] | 1× | Mesh holes to ignore |
| 22 | minimap_luma | derived from minimap | float [0,1] | 1× | Brightness channel |
| 23 | minimap_gradient | derived from minimap | float | 1× | Edge content / texture detail |
| 24 | height_range | derived from coarse height | float [0,1] | 1× | Per-tile height variance context |
| 25 | detail_energy | derived from height | float | 1× | Placeholder for future use |

**Excluded signals:** Shadow masks (`mcsh_shadow_mask_256`, `shadow_residual_mask_256`) — never present on minimap tiles, would be dead channels.

---

## 3. Loss Function

### Frequency-Banded Height Loss

Height is decomposed into three independent frequency bands:

```
LF (coarse):  height_17 → 17×17 grid
Mid:          height_65 - upsample(height_17) → residual at 65×65
HF (detail):  height_257 - upsample(height_65) → residual at 257×257
```

Each band gets an independent L1 loss, weighted by a time-varying schedule:

```python
t = epoch / freq_ramp_epochs  # 0 → 1 over ramp_epochs (default 60)
hf_weight = 1.0 - t * 0.9     # decays from 1.0 to 0.1
lf_weight = 0.1 + t * 0.9     # rises from 0.1 to 1.0
mid_weight = 0.5               # constant
```

**Why detail-first:** In terrain reconstruction, high-frequency detail (ridges, cliffs, roads) is what makes the output look realistic. Coarse shape (hills, valleys) is easier to learn from the WDL prior and other signals. If coarse shape is learned first, detail gets washed out in refinement. By forcing detail first, the model locks in fine structure early and shape learning doesn't destroy it.

### Auxiliary Task Losses

| Task | Loss | Weighting | Notes |
|------|------|-----------|-------|
| MCAL alpha | L1 | Uncertainty-weighted (learned σ) | Only if MCAL present in sample |
| MCLY class | Cross-entropy | Uncertainty-weighted (learned σ) | ignore_index for unknown textures |
| Hole mask | BCE | Uncertainty-weighted (learned σ) | Only if hole mask present |

Uncertainty weighting: each auxiliary task has a learned log-sigma parameter. Loss contribution = `task_loss / (2·σ²) + log(σ)`. This automatically balances task magnitudes.

### Total Loss

```
total = hf_weight * HF_L1 + 0.5 * mid_L1 + lf_weight * LF_L1
      + height_uncertainty
      + mcal_loss / (2·σ_mcal²) + log(σ_mcal)   [if MCAL present]
      + mcly_loss / (2·σ_mcly²) + log(σ_mcly)   [if MCLY present]
      + hole_loss / (2·σ_hole²) + log(σ_hole)    [if hole present]
```

---

## 4. Training Schedule

### Learning Rate

- **Warmup:** Linear 1% → 100% over first 5 epochs
- **Decay:** Cosine annealing from epoch 5 to epoch 300
- **Peak LR:** 2e-4 (AdamW) or 1e-4 (Lion)

### Frequency Ramp

| Epochs | hf_weight | lf_weight | Behavior |
|--------|-----------|-----------|----------|
| 0-15 | 1.0→0.78 | 0.1→0.33 | Detail dominates. Model learns texture/edges. |
| 15-30 | 0.78→0.55 | 0.33→0.56 | Balance. Mid structure fills in. |
| 30-45 | 0.55→0.33 | 0.56→0.78 | Shape catching up. Coarse structure refines. |
| 45-60 | 0.33→0.10 | 0.78→1.0 | Shape dominates. Final coarse refinement. |
| 60+ | 0.10 | 1.0 | Plateau. Detail preserved from early learning. |

### Data Augmentation

- Random horizontal flip (50%)
- Random vertical flip (50%)
- Signal dropout (15% per channel) — forces robustness to missing signals

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW or Lion |
| Peak LR | 2e-4 (AdamW) |
| Weight decay | 0.05 |
| Beta1/Beta2 | 0.9 / 0.95 (AdamW) |
| Gradient clip | 1.0 |
| Gradient accumulation | 1 |
| EMA decay | 0.999 |
| Batch size | 8 (16GB VRAM: 16) |
| Precision | bf16 AMP with GradScaler |
| torch.compile | Optional, dynamic=False |

---

## 5. Dataset

### Sources (6 clients, 14+ map variants)

| Client | Maps | Sample Tiles | Notes |
|--------|------|-------------|-------|
| 4.0.0.11927 | Azeroth, Kalimdor, LostIsles, EmeraldDream, MountHyjal | ~1,530 | Cata beta, split ADTs |
| 3.3.5.12340 | Azeroth, Kalimdor, Northrend, PVPZone01 | ~1,478 | Wrath, split ADTs |
| 3.0.1.8303 | Northrend | ~346 | Pre-Wrath, split ADTs |
| 0.7.0.3694 | Azeroth, Kalimdor | ~1,331 | Pre-BC, monolithic WDT |
| 0.5.5.3494 | Azeroth, Kalimdor, EmeraldDream | ~1,290 | Alpha, per-map MPQs |
| 0.5.3.3368 | Azeroth, Kalimdor | ~1,137 | Earliest alpha, per-map MPQs |
| **Total curated** | | **~7,000** | After curation gates |

### Curation Gates

Tiles rejected if:
- `height_range < 4.0` (flat/degenerate terrain)
- Missing `height_257` or `height_17` arrays
- Non-finite height values
- Mean WDL delta > 512 (extreme deviation from prior)

### NPZ Shard Contents

Per tile, each NPZ contains the arrays listed in §2 (Input Signals) plus the targets.
Cache at `output/tmp/v11_cache/`:

```
shards/<dataset_key>/<tile>.npz
v9_tensor_cache_manifest.json
```

---

## 6. Training Output

### Checkpoints

| File | Contents |
|------|----------|
| `last.pt` | Full training state (model, optimizer, EMA, loss sigmas, history) |
| `epoch_NNNN.pt` | Snapshot every `--save-every` epochs |
| Previews | Not saved during training (run `validate_v11.py` separately) |

### Metrics Tracked

Per epoch: `loss`, `lf_l1`, `mid_l1`, `hf_l1`, `mcal_l1`, `mcly_ce`, `hole_bce`, `lr`.

Post-training: `metrics.json` with full history.

---

## 7. Inference

```
python scripts/infer_v11.py <checkpoint> <shard_dir> --export-obj
```

Output per tile:
- `height_257.npy` — full heightmap (denormalized to world units)
- `mcal_alpha_pack_256` — predicted alpha weights
- `mcly_labels_16` — predicted texture class per chunk
- `hole_mask_16` — predicted hole locations
- `*.obj + *.mtl + texture.png` — 3D mesh with texture

---

## 8. Signals Not Yet Used (Future)

| Signal | Source | Status |
|--------|--------|--------|
| `synthetic_minimap_256` | MCAL×tileset compositing | Script exists at `scripts/synthesize_minimap.py`, needs harvested tileset BLPs |
| `texture_residual_256` | Real - synthetic minimap | Computed by synthesize script |
| Tileset pattern features | `PatternMiner` / `MinimapTilesetPatternMatcher` | Code on roadmap branch, needs MathNet dependency resolved |
