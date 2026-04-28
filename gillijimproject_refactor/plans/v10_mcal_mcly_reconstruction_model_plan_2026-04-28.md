# V10 MCAL/MCLY Reconstruction Model Plan

**Date:** 2026-04-28  
**Status:** Planning  
**Target:** `wow-viewer` library + training scripts

## Problem Statement

Given a minimap image (or low-resolution terrain image), reconstruct the terrain texturing data:
- **MCLY**: 16x16 grid of 4-layer texture combinations per chunk
- **MCAL**: 4 alpha masks at 1024x1024 (64x64 per chunk, 16 chunks per tile)

This is fundamentally underconstrained — you cannot uniquely decompose a composite minimap into 4 alpha layers and their textures. But we can make it tractable by:
1. Using the mined MCLY dictionary (35 retained texture combinations) as the output vocabulary
2. Using the mined MCAL brush dictionary (32 retained brush patterns) as the alpha vocabulary
3. Generating synthetic training data with known ground truth
4. Fine-tuning on real MCAL/MCLY data from the 11 development tiles that have it

## Architecture Insight: Pattern Composition Hierarchy

The terrain system is **not** simple texture blending. It's a layered pattern composition system — a digital canvas where Blizzard pre-baked optimizations into the textures themselves because this predates vertex shading by nearly a decade.

### The Hierarchy

```
Base textures (solid colors, simple gradients)
  ↓
Pre-baked patterns overlaid on base textures (stored in tileset BLPs)
  ↓
Brush patterns (MCAL alpha masks) control which pattern appears where
  ↓
MCLY combinations select which tileset textures to blend per chunk
  ↓
Final minimap = composite of all layers
```

### Key Hypothesis

**The pattern library embedded in tileset BLPs is likely the same library used for MCAL brushes.** Many tileset textures are base images with different repeating patterns laid on top. If we can:
1. Extract the hidden pattern library from tileset textures (via FFT, autocorrelation, or tile decomposition)
2. Map those patterns to the mined MCAL brush patterns
3. Use pattern matching to decompose minimaps into their constituent layers

...then the reconstruction problem becomes significantly more tractable because we're matching known patterns rather than doing blind decomposition.

### Mixture of Experts Architecture

Instead of one monolithic 35-class classifier, use a **Mixture of Experts (MoE)** approach:

```
Input: 256x256 minimap RGB
  ↓
Router: Coarse design-kit classifier (which zone family does this look like?)
  ↓
Expert: Small per-design-kit classifier (4-8 classes within that kit)
  ↓
Output: MCLY grid + MCAL alpha masks
```

Benefits:
- Different zones use completely different texture families — a Westfall expert doesn't need to know about Durotar textures
- Reduces classification from 35-class global to 4-8 class local per expert
- Experts can be trained independently with zone-specific synthetic data
- Router can use user-provided hints to narrow the search space

### Interactive Hinting System

Users can point at areas of a minimap image and label which design kit/tileset belongs there. This serves two purposes:

1. **Training data annotation**: Mark up existing minimaps with ground-truth design-kit labels to train the router
2. **Inference-time hints**: When reconstructing an unknown minimap, user hints constrain which experts are active, dramatically narrowing the search space

This is critical for historical reconstruction (like the pre-June 2002 Azeroth map) where the model has no ground truth but a human knows "this area is Westfall, this area is Redridge."

## Data Organization Reality

### No Biomes, Just Tilesets
WoW terrain texturing uses **tilesets** organized in folder hierarchies by zone/art style. Blizzard calls these **Design Kits** — families of art assets set aside for specific zones. The folder structure and texture names are the only organization:

```
World/Art/Tileset/
  Azeroth/
    Westfall/
      WF_Grass.blp
      WF_Dirt.blp
      WF_Rock.blp
    Redridge/
      RR_Grass.blp
      RR_Stone.blp
  Kalimdor/
    Durotar/
      DT_Sand.blp
      DT_Rock.blp
  Expansion01/
    BoneWastes/
      BoneWastesDirtShadow.blp
  Expansion02/
    SholazarBasin/
      SB_SoilA.blp
```

### Texture Naming Conventions
- `_s` suffix: specular highlight variant
- `*alpha` or alpha-era naming: semi-translucent textures from early game versions
- Prefix patterns: `WF_` (Westfall), `DT_` (Durotar), `SB_` (Sholazar Basin)
- Legacy names persist: Westfall was "Westwood" in 1999, so `WW_` prefixes exist in Westfall folders
- Spelling mistakes and inconsistent naming are common

### The Pre-Processing Problem
Raw tileset data is messy. Before training, we need to:
1. **Harvest** all tileset BLPs from client builds
2. **Index** by folder path, texture name, era, and naming patterns
3. **Normalize** legacy names (Westwood → Westfall, etc.)
4. **Group** into Design Kits (tileset families)
5. **Build dictionaries** mapping MCLY texture IDs/paths to organized tileset entries

This pre-processing step is critical. Without it, synthetic data generation will produce garbage because we won't know which textures belong together or what era they're from.

### MCLY Dictionary Reality
The mined MCLY dictionary stores texture combinations as raw paths:
```json
{
  "texture_names": [
    "Tileset/Expansion02/SholazarBasin/SB_SoilA.blp",
    "Tileset/Expansion01/BoneWastes/BoneWastesDirtShadow.blp",
    "",
    ""
  ],
  "texture_ids": [0, 1, -1, -1]
}
```

These paths ARE the organization. We need to:
- Parse folder paths to extract zone/design-kit membership
- Parse texture names to extract type hints (grass, dirt, rock, sand, etc.)
- Map texture IDs to paths per tile (MCLY stores local IDs, not global paths)
- Build a tileset database that can be queried by zone, era, or texture type

### Era-Specific Tilesets
Different client eras have different tilesets:
- **0.5.3**: Pre-release alpha textures, Westwood naming, rough art style
- **0.5.5/0.7.0**: Late alpha, some textures renamed, art improved
- **3.0.1**: WotLK pre-release, development map base textures
- **3.3.5**: WotLK retail, polished textures
- **4.0.0**: Cata beta, new zones (Lost Isles), updated tilesets

Textures can change appearance, be renamed, or disappear between eras. The synthetic data generator needs to respect era boundaries — don't mix 0.5.3 textures with 4.0.0 alpha patterns unless explicitly testing cross-era reconstruction.

## Current Data Inventory

### Real MCAL/MCLY Ground Truth
- **11 out of 64** v10 development tiles have MCAL + MCLY data
- **35 retained MCLY texture combinations** (mined dictionary)
- **32 retained MCAL brush patterns** (mined brush dictionary)
- **2,816 labeled 16x16 chunks** for MCLY classification
- **11,264 layer patches** for MCAL brush mining (9,681 rejected as uniform, 1,583 candidates)

### Tileset Textures (Available in Clients)
| Client | Path | Notes |
|--------|------|-------|
| 0.5.3.3368 | `H:\CLIENTS\...` | Alpha terrain textures, pre-release look |
| 0.5.5.3494 | WoWArchive | Early alpha textures |
| 0.7.0.3694 | WoWArchive | Late alpha textures |
| 3.0.1.8303 | `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303` | WotLK pre-release, development map base |
| 3.3.5.12340 | `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340` | WotLK retail |
| 4.0.0.11927 | `H:\CLIENTS\World of Warcraft Cata beta 11927` | Cata beta, development map overlay |

### Mined Dictionaries
- `output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json` — 35 texture combinations with raw texture paths, ID tuples, and example chunk coordinates
- `output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json` — 32 brush patterns with 64x64 stamps
- `output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json` — 32 chunk-level alpha compositions
- `output/build-validation/v10-wave2-mcly-labels/v10_mcly_label_manifest.json` — 11 tiles with 16x16 label grids

## Architecture

### Pattern Mining (New Phase 0.5)

Before synthetic data generation, extract the hidden pattern library from tileset BLPs:

- **Autocorrelation analysis**: Detect repeating patterns within each tileset texture
- **FFT frequency analysis**: Identify dominant spatial frequencies (pattern sizes)
- **Tile decomposition**: Split textures into repeating units and cluster by similarity
- **Pattern-to-brush mapping**: Compare mined patterns against MCAL brush dictionary to find overlaps
- **Output**: `pattern_library.json` with pattern stamps, frequencies, and brush-pattern relationships

This is critical because the patterns embedded in tileset textures may be the same library used for MCAL brushes. If confirmed, we can use pattern matching as a strong prior for decomposition.

### Mixture of Experts Model

#### Router: Design-Kit Classifier
- **Input:** 256x256 minimap RGB + optional user hints (masked regions with design-kit labels)
- **Output:** Per-chunk design-kit probabilities (N kits, where N = number of unique design kits in tileset database)
- **Architecture:** Lightweight CNN (MobileNetV2-style) → 16x16 spatial probability map
- **Loss:** Cross-entropy with hint-masked regions weighted higher
- **Training data:** Synthetic data with known design-kit labels + user-annotated real minimaps

#### Experts: Per-Design-Kit MCLY + MCAL Predictors
- **Input:** 256x256 minimap RGB + router probabilities (as attention weights)
- **Output:** MCLY grid (16x16xK_kit) + MCAL alpha (4x256x256) where K_kit = classes within that kit (typically 4-8)
- **Architecture:** Shared encoder → kit-specific heads (classification + regression)
- **Loss:** Cross-entropy (MCLY) + BCE (MCAL) + pattern consistency loss
- **Training data:** Synthetic data generated from that kit's textures + real tiles from that zone

#### Composition Layer
- Combines expert outputs weighted by router probabilities
- Resolves conflicts where multiple experts predict overlapping regions
- Outputs final MCLY grid (16x16x35 global classes) + MCAL alpha (4x256x256)

### Two-Stage Alternative (Fallback)

If MoE proves too complex for initial implementation:

#### Stage 1: MCLY Grid Classifier
- **Input:** 256x256 minimap RGB
- **Output:** 16x16 grid of class indices (35 classes = retained MCLY combinations)
- **Architecture:** ResNet-style encoder → 16x16 conv head with per-chunk classification
- **Loss:** Cross-entropy with ignore_index=-100 for unknown combinations
- **Training data:** 2,816 real chunks from 11 tiles + synthetic data

#### Stage 2: MCAL Alpha Predictor
- **Input:** 256x256 minimap RGB + predicted MCLY grid (as embedding)
- **Output:** 4 alpha masks at 256x256 (downsampled from 1024x1024)
- **Architecture:** U-Net encoder-decoder with MCLY conditioning via cross-attention or channel concatenation
- **Loss:** BCE + brush-pattern consistency loss
- **Training data:** Synthetic + 11 real tiles

## Synthetic Data Generation

### Core Concept
Composite tileset textures using mined MCAL brush patterns to create infinite training examples with perfect ground truth.

### Generation Pipeline
1. **Load tileset BLPs** from the pre-processed texture database (organized by Design Kit / zone)
2. **Select MCLY combination** from mined dictionary (e.g., `[SB_SoilA.blp, BoneWastesDirtShadow.blp, "", ""]`)
3. **Sample MCAL brush patterns** from mined dictionary for each active layer
4. **Composite:** `minimap = alpha_0 * tex_0 + alpha_1 * tex_1 + ... + (1 - sum_alphas) * base_color`
5. **Output:** synthetic minimap + exact MCAL alpha masks + exact MCLY combination label

### Controlled Variations
- **Texture swaps within Design Kit:** Same alpha pattern, different textures from the same zone family
- **Cross-zone mixing:** Same alpha pattern, textures from different zones (tests generalization)
- **Alpha variations:** Same textures, different brush patterns from mined dictionary
- **Noise injection:** Add minimap-style noise, compression artifacts
- **Resolution degradation:** Downsample + upsample to simulate low-res inputs
- **Era-specific:** Use 3.0.1 textures with 3.0.1 alpha patterns, 4.0.0 with 4.0.0, etc.
- **Cross-era:** Mix textures from different eras to test era-agnostic reconstruction
- **Partial coverage:** Some chunks have 1 layer, some have 2-4

### Scale
With 35 MCLY combinations × 32 brush patterns × 6 clients × 10 variations = **67,200 base combinations**, each with infinite positional variations.

## Implementation Plan

### Phase 0: Tileset Pre-Processing & Dictionary Building (`wow-viewer` tool)
- [x] Add `wowviewer-converter index-tilesets --client-root <path> --output-dir <dir> --era <era-tag>` command
- [x] Scan `World/Art/Tileset/` and `World/Textures/Terrain/` for BLP files
- [x] Extract folder paths, texture names, dimensions, format info
- [x] Parse naming patterns: zone prefixes (`WF_`, `DT_`, `SB_`), type suffixes (`_s`, `alpha`)
- [x] Normalize legacy names (Westwood → Westfall, etc.) using a mapping table
- [x] Build `tileset_database.json` with:
  - Texture path, folder, zone/design-kit name, era tag
  - Parsed name components (prefix, base, suffix)
  - Dimensions, format, alpha channel presence
  - Cross-era texture matches (same name across different client builds)
- [ ] Run against all 6 client builds (0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0)
- [ ] Merge per-era databases into a unified cross-era tileset index
- [ ] Output: `output/ml-training/v10_tileset_database/tileset_database.json`

### Phase 0.5: Pattern Library Mining (NEW)
- [ ] `wow-viewer/src/core/WowViewer.Core/Datasets/PatternMiner.cs` — autocorrelation + FFT analysis
- [ ] `wowviewer-converter mine-tileset-patterns --tileset-db <file> --output-dir <dir>`
- [ ] For each tileset BLP:
  - Compute 2D autocorrelation to detect repeating patterns
  - FFT to identify dominant spatial frequencies
  - Extract repeating tile units
  - Cluster similar patterns across textures
- [ ] Compare mined patterns against MCAL brush dictionary (`mcal_brush_dictionary.json`)
- [ ] Build `pattern_library.json` with:
  - Pattern stamps (64x64, 32x32, 16x16 at various scales)
  - Dominant frequencies and orientations
  - Source texture references
  - Brush-pattern similarity scores
- [ ] Output: `output/ml-training/v10_pattern_library/pattern_library.json`

### Phase 0.7: Interactive Hinting Tool (NEW)
- [ ] `wow-viewer/src/viewer/WowViewer.App/Tools/MinimapSegmentationTool.cs` — UI for marking regions
- [ ] `wowviewer-converter export-hints --minimap <file> --hints <json> --output <annotated.npz>`
- [ ] Users load a minimap image and paint regions with design-kit labels
- [ ] Export as NPZ with hint mask: `hint_mask_256` (256, 256) int64 with design-kit class indices
- [ ] Training uses hints as additional input channel with higher loss weight on hinted regions
- [ ] Output: `output/ml-training/v10_hint_annotations/` with NPZ files + `hint_manifest.json`

### Phase 1: Tileset Texture Harvest (`wow-viewer` tool)
- [ ] Add `wowviewer-converter harvest-tileset-blps --tileset-db <file> --output-dir <dir>` command
- [ ] Convert BLPs to RGBA numpy arrays using existing BLP decoder
- [ ] Store as `.npy` files with matching index
- [ ] Output: `output/ml-training/v10_texture_database/` with `.npy` files + `texture_index.json`

### Phase 2: Synthetic Data Generator (`wow-viewer` library + script)
- [ ] `wow-viewer/src/core/WowViewer.Core/Datasets/McalMclySynthesizer.cs` — core compositing logic
- [ ] `wowviewer-converter generate-mcal-mcly-synth --texture-db <dir> --mclay-dict <file> --brush-dict <file> --pattern-db <file> --output-dir <dir> --count <n>`
- [ ] Output format: NPZ shards matching v10 training contract:
  - `minimap_rgb_256` (256, 256, 3) uint8
  - `mcal_alpha_pack_256` (256, 256, 4) float32
  - `mcly_texture_ids` (16, 16, 4) int32
  - `mcly_label_grid` (16, 16) int64 (class indices)
  - `design_kit_label` (16, 16) int64 (design-kit class per chunk)
  - `metadata.json` with generation parameters
- [ ] Generate 50,000+ synthetic shards
- [ ] Include pattern-layer metadata: which patterns were used, at what scale, with what alpha

### Phase 3: Router Training (Design-Kit Classifier)
- [ ] New script: `wow-viewer/scripts/train_v10_router_classifier.py`
- [ ] Input: NPZ shards with `minimap_rgb_256` + `design_kit_label`
- [ ] Architecture: MobileNetV2-style lightweight CNN → 16x16 spatial probability map
- [ ] Training: synthetic first, then fine-tune on user-annotated real tiles
- [ ] Output: `router_classifier.pt` + `design_kit_index.json`

### Phase 4: Expert Training (Per-Kit MCLY + MCAL)
- [ ] New script: `wow-viewer/scripts/train_v10_expert_models.py`
- [ ] Input: NPZ shards filtered by design kit + `mcly_label_grid` + `mcal_alpha_pack_256`
- [ ] Architecture: Shared encoder → kit-specific heads (classification + regression)
- [ ] Training: synthetic data from that kit's textures only
- [ ] One model per design kit (or merged for small kits)
- [ ] Output: `expert_<kit>.pt` for each design kit

### Phase 5: Composition & Inference
- [ ] `wow-viewer/scripts/train_v10_composition_layer.py` — train the router→expert composition
- [ ] Joint inference pipeline: router → expert selection → weighted composition → output
- [ ] Support user hints as additional router input (masked regions with higher weight)
- [ ] Output: unified inference wrapper

### Phase 6: wow-viewer Library Integration
- [ ] `wow-viewer/src/core/WowViewer.Core/Datasets/McalMclyReconstructionModel.cs` — inference wrapper
- [ ] `wowviewer-converter reconstruct-mcal-mcly --minimap <file> --router-model <file> --expert-dir <dir> --hints <json> --output <file>`
- [ ] Viewer integration: minimap → terrain texturing preview with hint overlay

## Training Configuration

### Router (Design-Kit Classifier)
```
--epochs 80 --batch-size 32 --learning-rate 1e-3 --weight-decay 0.01
--warmup-epochs 10 --gradient-clip 1.0
--num-workers 0 --device cuda
--hint-weight 5.0  (higher loss weight on user-hinted regions)
```

### Per-Kit Experts
```
--epochs 100 --batch-size 8 --learning-rate 1e-3 --weight-decay 0.01
--warmup-epochs 10 --gradient-clip 1.0
--num-workers 0 --device cuda
--pattern-loss-weight 0.3  (consistency loss with mined pattern library)
```

### Composition Layer
```
--epochs 50 --batch-size 4 --learning-rate 5e-4 --weight-decay 0.01
--warmup-epochs 5 --gradient-clip 1.0
--num-workers 0 --device cuda
--freeze-experts true  (experts frozen, only composition layer trained)
```

## Validation Strategy

### Synthetic Validation
- Hold out 10% of synthetic data for validation
- Verify model can reconstruct exact ground truth on synthetic data

### Real Validation
- Hold out 2 of 11 real tiles for validation
- Evaluate MCLY accuracy (per-chunk classification accuracy)
- Evaluate MCAL quality (BCE loss, visual inspection of alpha masks)

### End-to-End Validation
- Feed real minimap → MCLY classifier → MCAL predictor → composite terrain
- Compare composite against original ADT terrain (where available)

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Only 11 real tiles for fine-tuning | Heavy synthetic pre-training, aggressive augmentation, user hint annotations |
| Texture appearance changes across eras | Era-tagged synthetic data, era-specific fine-tuning, cross-era training |
| Low-res input degradation | Resolution degradation in synthetic training |
| Underconstrained decomposition | Constrain output to mined vocabulary (35 MCLY + 32 MCAL patterns) + pattern library priors |
| Brush patterns don't generalize | Mine brushes from broader corpus, not just development |
| Messy tileset naming/organization | Pre-processing phase with legacy name normalization, Design Kit grouping |
| Missing textures in some eras | Cross-era texture matching, fallback to nearest-era equivalent |
| Spelling mistakes in texture paths | Fuzzy matching in tileset database, alias table for known misspellings |
| MoE router misclassifies zones | User hints override router; fallback to full 35-class scan if router confidence < threshold |
| Pattern library doesn't match brushes | Fall back to direct brush-pattern matching without texture-layer decomposition |
| Too many design kits for practical MoE | Merge small kits into "misc" expert; use hierarchical routing (continent → zone → kit) |
| User hinting too tedious for large maps | Semi-automatic: router proposes labels, user corrects errors only |

## File Locations

### Library Code
- `wow-viewer/src/core/WowViewer.Core/Datasets/PatternMiner.cs`
- `wow-viewer/src/core/WowViewer.Core/Datasets/McalMclySynthesizer.cs`
- `wow-viewer/src/core/WowViewer.Core/Datasets/TextureDatabase.cs`
- `wow-viewer/src/core/WowViewer.Core/Datasets/McalMclyReconstructionModel.cs`
- `wow-viewer/src/core/WowViewer.Core/Datasets/RouterClassifier.cs`
- `wow-viewer/src/core/WowViewer.Core/Datasets/ExpertModel.cs`

### Training Scripts
- `wow-viewer/scripts/train_v10_router_classifier.py`
- `wow-viewer/scripts/train_v10_expert_models.py`
- `wow-viewer/scripts/train_v10_composition_layer.py`

### Tool Commands
- `wowviewer-converter index-tilesets`
- `wowviewer-converter mine-tileset-patterns`
- `wowviewer-converter harvest-tileset-blps`
- `wowviewer-converter generate-mcal-mcly-synth`
- `wowviewer-converter export-hints`
- `wowviewer-converter reconstruct-mcal-mcly`

### Output Directories
- `output/ml-training/v10_tileset_database/` — pre-processed tileset index
- `output/ml-training/v10_pattern_library/` — mined pattern library
- `output/ml-training/v10_texture_database/` — harvested tileset BLPs as RGBA numpy
- `output/ml-training/v10_hint_annotations/` — user-annotated minimap hints
- `output/ml-training/v10_mcal_mcly_synth/` — synthetic training data
- `output/ml-training/v10_router/` — router model + checkpoints
- `output/ml-training/v10_experts/` — per-kit expert models + checkpoints
- `output/ml-training/v10_composition/` — composition layer + checkpoints
