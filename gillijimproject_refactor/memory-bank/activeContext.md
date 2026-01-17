# Active Context

## Current Focus: VLM Dataset Generation for V6 Training (Jan 16, 2026)

### Status Summary
**CRITICAL FIX COMPLETED**: Per-tile normalization was causing visible seams when tiles were stitched. Created `regenerate_heightmaps_global.py` to generate BOTH per-tile (shows hidden details) AND global (proper training data) heightmaps.

### V6 Training Plan
**Model Inputs**: minimap RGB + normalmap RGB + WDL 17x17 hint (required)

**Targets**:
- Dual heightmaps: global + local
- Alpha mask auxiliary targets (4 layers)
- Height bounds: tile min/max + global min/max

**Loss**:
- Height loss weighted toward local detail (0.3 global / 0.7 local)
- Alpha aux loss weighted (0.3)
- Bounds MSE weighted (x10)

**Dataset handling**:
- WDL required (skip tiles missing `wdl_heights.outer_17`)
- Alpha optional unless required
- Sanity report prints missing WDL/heightmaps/inputs counts

**Generalization**:
- Map-level train/val split (holdout maps or fraction)
- Minimap color jitter augmentation

**Inference output (planned)**:
- Timestamped folder with dataset-style JSON + assets
- OBJ + MTL mesh output

**Usage**:
```bash
python regenerate_heightmaps_global.py <dataset_dir> [--no-stitch] [--max-size 16384]
```

**Outputs**:
- `images/*_heightmap.png` - Per-tile global normalized
- `images/*_heightmap_local.png` - Per-tile local normalized  
- `stitched/*_full_heightmap_global.png` - Stitched global
- `stitched/*_full_heightmap_local.png` - Stitched local
- `global_height_bounds.json` - Height range metadata

### Dataset Exports In Progress
| Map | Status | Tiles | Output |
|-----|--------|-------|--------|
| Azeroth | ✅ Done | 685 | `053_azeroth_v10/` |
| Kalimdor | 🔧 Pending | ~951 | `053_kalimdor_v6/` |
| Kalidar | ✅ Done | 56 | `053_kalidar_v1/` |
| DeadminesInstance | 🔧 Pending | ? | `053_deadmines_v1/` |
| RazorfenKraulInstance | ✅ Done | 6 | `053_razorfen_v1/` |

**Client Path**: `H:\053-client\`

### Immediate Next Steps
1. Complete Kalimdor v6 export (951 tiles)
2. Complete DeadminesInstance v1 export
3. Begin V6 model training with dual heightmaps
4. Implement absolute height encoding strategy

### Key Files Updated
- `VlmDatasetExporter.cs` - Fixed GenerateHeightmap + added StitchHeightmapsToWebP
- `HeightmapBakeService.cs` - Updated to use Alpha MCVT format
- `render_heightmaps.py` - Updated to use Alpha MCVT format
- `render_normalmaps.py` - Updated to use Alpha MCNR format
- `scripts/regenerate_heightmaps_global.py` - New dual-mode heightmap generator

### Technical Notes
- Alpha MCVT format: 81 outer vertices (9×9) FIRST, then 64 inner vertices (8×8)
- Resolution upgraded from 129×129 to 256×256 (uses ALL 145 vertices per chunk)
- MCNR padding handled: Alpha exports 448 bytes, LK exports 435 bytes (truncate to 435)
