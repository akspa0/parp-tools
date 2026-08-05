# Implementation Plan: Terrain Brush Signature Classification

**Branch**: `132-terrain-brush-signature-classification` | **Date**: 2026-08-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/132-terrain-brush-signature-classification/spec.md`

## Summary

The WoW terrain was built using fractal brushes in WoWEdit that simultaneously affected heightmap, alpha layers, and texture tilesets. When textures were re-done, the alpha layers were replaced but the heightmap brush scars remained — creating a broken relationship between 3D shape and 2D texture. This plan implements the tooling to detect, classify, and correlate these brush signatures across all maps and builds.

Six user stories, implemented in priority order:

1. **Three-tier classification** (P1) — strong/normal/weak signal classes
2. **Nested weak signal detection** (P2) — multiple tiers of buried brush data
3. **Brush-texture correlation** (P2) — heightmap scars vs alpha mask patterns
4. **Cross-map fragment alignment** (P2) — rotated/mirrored copy-paste detection
5. **Pre-rescale boundary detection** (P2) — 33.33% horizontal roll = Nov 2001 rescale
6. **Predictive model** (P3) — texture from heightmap via learned brush relationships

## Technical Context

**Language/Version**: C# / .NET 10 (core tooling), Python 3.14 (analysis scripts)

**Primary Dependencies**: `WowViewer.Tool.Harvest` (NPZ shard extraction), `WowViewer.Core.IO` (ADT/MCAL/MCLY readers), NumPy/SciPy (signal processing), Zarr (store format)

**Storage**: V50-format Zarr stores under `wow-viewer/output/datasets/v50/` per build. Per-client Zarr stores under `wow-viewer/output/archaeology/<build_id>/store/`.

**Testing**: xUnit for C# tooling, pytest for Python analysis scripts. Corpus-wide validation on all 15 1.x clients.

**Target Platform**: Windows 11 / PowerShell 7. CLI: `dotnet WowViewer.Tool.Harvest.dll harvest-map-mpq` for extraction, `uv run python scripts/` for analysis.

**Scale**: 15 1.x Windows clients, ~10-30 terrain maps each, ~100-1000 tiles per map. Total ~50,000+ tiles across all builds.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All new code under `wow-viewer/`. No paths outside. |
| II. Library-First | PASS | Analysis scripts in `data-harvester/scripts/`. Harvest tool is existing. |
| III. Real-Data Validation | PASS | All 15 1.x clients are real client data on H:\CLIENTS. |
| IV. Per-Signal Evidence | PASS | Each classification tier reported independently. |
| V. Streaming-First | PASS | Zarr stores are the output format. |
| VI. No Hardcoded Paths | PASS | Client root is CLI argument. |
| Read-Only Reference | PASS | No writes to `gillijimproject_refactor`. |
| One Phase at a Time | PASS | Six phases, each ending in validation. |

## Project Structure

```
wow-viewer/
├── data-harvester/scripts/
│   ├── v50_tile_classify.py              # NEW — three-tier classifier
│   ├── v50_nested_signal.py              # NEW — nested weak signal detector
│   ├── v50_brush_correlate.py            # NEW — brush-texture correlator
│   ├── v50_fragment_align.py             # NEW — cross-map fragment alignment
│   ├── v50_rescale_boundary.py           # NEW — 33.33% rescale boundary detector
│   └── v50_brush_model.py                # NEW — predictive model (texture from heightmap)
├── data-harvester/src/harvester/v50/
│   ├── classify.py                       # NEW — classification logic
│   ├── nested_signal.py                  # NEW — nested signal detection
│   ├── brush_correlate.py                # NEW — brush-texture correlation
│   ├── fragment_align.py                 # NEW — fragment alignment
│   ├── rescale_boundary.py               # NEW — rescale boundary detection
│   └── brush_model.py                    # NEW — model training/inference
├── output/archaeology/
│   └── <build_id>/
│       ├── npz/<map>/                    # NPZ shards from harvest
│       ├── store/<map>.zarr/             # V50 Zarr store
│       └── archaeo/<map>/                # Archaeology results
└── specs/132-terrain-brush-signature-classification/
    ├── spec.md                           # This feature's spec
    ├── plan.md                           # This file
    ├── research.md                       # Phase 0 findings
    ├── data-model.md                     # Entities and relationships
    ├── quickstart.md                     # Exact commands
    └── contracts/                        # API schemas
```

## Phases

### Phase 1: Three-tier classification (US1)

**Goal**: Classify every tile as strong, normal, or weak signal with published criteria.

**Implementation**:
1. Add `normal_signal` class to `v50_tile_inventory.py` with criteria: height range 5-50 units, surviving height levels 8-64, or alpha-texture correlation <0.3
2. Update `v50_archaeology.py` to emit three-tier classification in summary
3. Update `v50_tile_composite.py` to render normal-signal tiles with a distinct outline color (green)
4. Validate on all 15 1.x clients

**Gate**: Every tile in the development corpus is classified with published criteria.

### Phase 2: Nested weak signal detection (US2)

**Goal**: Detect multiple tiers of progressively weaker brush data within a single tile.

**Implementation**:
1. Build `v50_nested_signal.py` that quantizes height data at progressively coarser precision levels
2. Count surviving height levels at each precision tier
3. Report tier boundaries and compression ratios
4. Validate on known weak-signal tiles from Expansion01

**Gate**: At least one weak-signal tile shown to contain multiple tiers.

### Phase 3: Brush-texture correlation (US3)

**Goal**: Correlate heightmap brush scars with alpha-layer patterns.

**Implementation**:
1. Extract brush scar features from heightmap (edge detection, ridge/valley finding)
2. Extract brush scar features from alpha layers (texture boundary detection)
3. Compute correlation score between the two feature sets
4. Report broken relationships (low correlation = re-textured)
5. Validate on DeadminesInstance vs Westfall

**Gate**: DeadminesInstance alpha masks shown to not match Westfall's current heightmap.

### Phase 4: Cross-map fragment alignment (US4)

**Goal**: Find copy-pasted terrain fragments across maps with rotation/mirror detection.

**Implementation**:
1. Build `v50_fragment_align.py` that takes a source tile and searches target maps
2. Use phase correlation + template matching for rotation/mirror/scale detection
3. Report sub-tile offsets for non-aligned pastes
4. Validate on DeadminesInstance alpha masks vs Westfall originals

**Gate**: DeadminesInstance fragment found in Westfall with correct rotation/mirror reported.

### Phase 5: Pre-rescale boundary detection (US5)

**Goal**: Detect the 33.33% horizontal weak-signal roll marking the Nov 2001 rescale.

**Implementation**:
1. Build `v50_rescale_boundary.py` that scans each tile for horizontal signal discontinuities
2. Report boundary position, confidence, and pre/post-rescale classification
3. Build a library of all pre-rescale tiles across all maps and builds
4. Validate on DeadminesInstance tiles known to carry the pattern

**Gate**: At least one DeadminesInstance tile confirmed to carry the 33.33% boundary.

### Phase 6: Predictive model (US6)

**Goal**: Train a model that predicts alpha-layer patterns from heightmap shape.

**Implementation**:
1. Build training dataset from tiles with intact brush-texture relationships
2. Train a CNN that maps heightmap patches to alpha-layer pattern distributions
3. Evaluate on tiles with broken relationships (re-textured zones)
4. Report confidence scores alongside predictions

**Gate**: Model predicts alpha patterns with >60% accuracy vs random baseline.
