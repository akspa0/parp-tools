# Feature Specification: V25 SegFormer Decompiler and Terrain-Texture Convergence Network (Spec 102)

**Feature Branch**: `102-v25-terrain-convergence`

**Created**: 2026-07-11

**Status**: Draft

**Owner**: wow-viewer

**Parent**: Spec 098 (V24 Lattice Reconstruction Vision)

**Research**: 
- *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (NVIDIA, NeurIPS)
- CVPR 2026: *Dual Graph Regularized Deep Unfolding Network for Guided Depth Map Super-resolution* (**LapNet**)
- `data-harvester/src/harvester/pm4_asset_matching/` (PM4 segment bounds and scorer math)
- `docs/architecture/wowfileformatdefs-vision-2026-06-16.md` (MTEX, MCLY, MCAL structure)
- `docs/architecture/v22-dataset-signals-2026-06-30.md` (Zarr signals)

**Input**: User description — "pm4 files cannot be processed solely one by one, we already handle reading the data properly in our c# tooling, stop reinventing things that we have already"

---

## Problem Statement

The terrain decompiler and texture convergence model must be **completely universal** to process any map from a single raw RGB minimap image.

To prevent duplicating PM4 file parsers, all PM4 binary parsing and cataloging are handled by our existing C# tooling. The resulting PM4 segment data is stored in the database / Zarr stores. The post-processing alignment handler (`V25Pm4GuideHandler`) does not parse raw PM4 files; instead, it consumes pre-parsed lists of segment boundary signals (`Pm4SegmentSignalRecord`) loaded from our Zarr databases.

The system is decoupled into two separate parts:
1.  **Universal Decompiler Model (`V25SegformerDecompiler`)**: A neural network that runs from a single raw RGB minimap tile and predicts detailed heights, object instance bounds/rotations, MTEX selectors, MCLY assignments, and MCAL alpha maps (using a **Differentiable Fractal Noise Generator** to represent repeating artist brushes). It has **no** PM4 dependencies.
2.  **PM4 Guided Post-Processing Handler (`V25Pm4GuideHandler`)**: A separate utility class. It takes predicted placements and aligns them to pre-parsed PM4 segment boundary signals loaded from the database.

The pipeline runs entirely offline within an **8 GB / 10 GB VRAM GPU** limit using gradient checkpointing, 8-bit AdamW, and Zarr slice preloading.

---

## User Scenarios & Testing

### User Story 1 — Universal Single-Image Reconstruction (Priority: P1)

The user drops a custom minimap PNG (e.g. from an expansion or custom edit) with no PM4 files available. The model performs forward inference and generates a structured Zarr dataset directory.

**Why this priority**: Core universal pipeline.

**Independent Test**: Run inference on an arbitrary screenshot tile without specifying any `--pm4` arguments.

**Acceptance Scenarios**:
1. **Given** a raw minimap, **When** processed by the CLI, **Then** it generates a valid textured Zarr store utilizing only visual feature maps, showing that the network behaves universally.

---

### User Story 2 — Separate PM4-Guided Post-Processing (Priority: P1)

The user runs inference on a development map. The CLI invokes the post-processing handler to snap predicted placements to PM4 centroids and matches WMO/M2 models, producing the final aligned development arrays in the Zarr dataset.

**Why this priority**: Bounded development map reconstruction.

**Independent Test**: Verify that the PM4 handler runs as a separate step on top of the universal network's output predictions.

**Acceptance Scenarios**:
1. **Given** predicted object placements from the universal model, **When** passed through the separate `V25Pm4GuideHandler` alongside pre-parsed segment bounds, **Then** coordinates snap to centroids and names are resolved.

---

## Requirements

### Functional Requirements

#### Slice 0: Lean V25 Training Dataset (Zarr)
- **FR-102-001**: A dedicated V25 Zarr datastore MUST be built fresh from the existing V18 substrate, the V22 enrichment (tileset vocabulary + placements), and — when available — the V24 store's pre-computed cleaned minimaps. It MUST NOT be a copy of the V22 schema.
- **FR-102-002**: Only signals that serve the V25 model as inputs, targets, or loss masks are carried over: `minimap_rgb`, `clean_minimap_256`, `object_mask_256`, `height_257`, `wdl_height_33`, `alpha_256`, `mcly_layer_mask`, `mcly_vocab_ids`, plus the liquid/flag loss signals `liquid_mask_256`, `liquid_type_256`, `liquid_height_256`, `mcnk_flags_16` (user-directed 2026-07-11 — liquid areas must be maskable out of height supervision, and era restoration needs MH2O/MCLQ facts), plus `index.parquet`, `placements.parquet`, and `tileset_vocab.parquet` sidecars. Normals, `holes_16` (inverted at the C# source per the V24 audit), roof/visibility/instance masks, ground-intent heights, and asset payload groups remain excluded.
- **FR-102-003**: `wdl_height_33` MUST be derived from `height_257` with the exact `WdlDownsampler` stride-8 math at build time, so high-res heights and WDL priors are aligned by construction (SC-102-004).
- **FR-102-004**: Cleaned minimaps are a dataset-build step (not per-load): prefer the V24 `cleaned_minimap_256` array via the `v18_row` join, else compute `harvester.v24.clean_minimap` once at build.
- **FR-102-005**: The builder MUST support the V18 curation manifest (`--curation-manifest`, `--difficulty-bucket`), map filters, and tile limits, and MUST read source arrays through contiguous Zarr slices (no per-row random access).
- **FR-102-006**: Pre-parsed PM4 segment records (C# export JSON) are attachable to the store as `pm4_segments.parquet` and loadable back as `Pm4SegmentSignalRecord` lists — Python never opens raw `.pm4` files.
- **FR-102-007**: All arrays use Blosc LZ4 level-1 compression with per-tile chunks.
- **FR-102-008** (amended 2026-07-11, second pass): The builder MUST accept multiple index-paired (V18, V22, V24) source triples and emit one combined multi-era corpus (user-directed: start with 0.5.3 + 3.3.5, later re-target the image side to any era). The tileset vocabulary is **era-scoped** — keyed by (build, normalized tileset path) — because tileset content changed across eras even under identical names ("the images are literally different between them"). Grass in 0.5.3 and grass in 3.3.5 are distinct vocab entries. Builds without a path table fall back to build-scoped id keys. Vocab sizing must cover every era's tilesets without frequency truncation (0.5.3 + 3.3.5 = 1,070 entries → 2048 slots).
- **FR-102-008b**: Tileset texture **images** ride in the store: `WowViewer.Tool.Harvest extract-tilesets` decodes each era's BLPs from that era's own MPQs to PNGs + manifest, and `attach_tileset_images` writes a vocab-aligned `tilesets` group (`tileset_rgb_256` (V,256,256,3) uint8, `tileset_present` (V,)). Unresolvable textures stay flagged absent, never silently substituted.
- **FR-102-009**: Curation provenance MUST be baked into the store (user-directed 2026-07-11 — months of curation work must not be lost at build time): every curation-manifest column (difficulty buckets, quality/usefulness/difficulty scores, coverage stats, profiles) is joined per tile into `index.parquet`, the mismatch-audit severity/reason columns ride along, and the trainer exposes `--difficulty-buckets` filtering against the baked metadata. When a mismatch-repair store carries a build (`height_corrected_257`), the builder MUST use the corrected heights instead of raw — never process bad data when a repaired version exists.
- **FR-102-010**: Full-signal completeness (user-directed 2026-07-11 — "every signal we will ever need"): the store also carries `normal_xyz_257` (int8, MCNR-native), `shadow_mask_256` (MCSH), `object_visibility_256` (renderer-truth visibility), `ground_intent_height_257` (object-inpainted intended ground), and `object_instance_mask`. Only derivable or deprecated signals remain excluded, each documented in `dataset.py`.
- **FR-102-011**: True hole bitmasks (user-directed 2026-07-11 — "WoWViewer flips them perfectly for every build"): the corrupt V18 `holes_16` is replaced end-to-end. The C# extractor defect is fixed at the source (`AdtTensorPackBuilder.ReadMcrfAndHoles` now reads the MCNK `holes` uint16 at header offset 0x3C — the same field `WorldTerrainHoleMask` renders — instead of flags bits 8-15; `AlphaWdtReader` preserves the full per-chunk ushort masks it already parsed at offset 0x40). A new `WowViewer.Tool.Harvest extract-holes` command dumps era-aware raw hole bitmasks per tile to JSON, and `attach_holes_bits` joins them into the store as `holes_bits_16` (int32, -1 = unknown). Mismatch-repair stores are **sparse overlays** (NaN except repaired tiles): the builder merges per cell and hard-counts non-finite heights (`nonfinite_height_tiles` attr must be 0 for a trainable store).

#### Slice 1: Universal SegFormer Decompiler (Stage 1)
- **FR-102-101**: Model MUST use a `SegformerForSemanticSegmentation` backbone (`nvidia/mit-b0`). No Depth Anything models allowed.
- **FR-102-102**: Implement `TerrainInpaintHead` outputting the clean terrain-shadow map ($3\times256\times256$ RGB) and `ObjectMaskDecoder` outputting the object footprint mask ($1\times256\times256$).
- **FR-102-103**: Implement `ObjectPlacementHead` predicting classifications, translations, and rotations.

#### Slice 2: High-Res Height & WDL Prior Generator (Stage 2)
- **FR-102-201**: Implement progressive `V25StageBPredictor` scaling heights progressively ($33 \rightarrow 65 \rightarrow 129 \rightarrow 257$) using the Sylvester solver.
- **FR-102-202**: Implement `WdlDownsampler` which mathematically averages the $257\times257$ heightmap to yield the $33\times33$ quincunx WDL prior heights.

#### Slice 3: Differentiable Fractal Generator and Parameter Head (Stage 3)
- **FR-102-301**: Implement `DifferentiableFractalGenerator` in PyTorch generating multi-octave Simplex or Perlin noise on a $256\times256$ grid.
- **FR-102-302**: Implement `FractalParameterHead` predicting boundary masks, translations, frequency, amplitude, and persistence.

#### Slice 4: Decoupled PM4 Reconstruction Utility (Stage 4)
- **FR-102-401**: PM4 matching code MUST live entirely separate from the main neural network forward and backward pipelines.
- **FR-102-402**: `V25Pm4GuideHandler` is a post-processing step running on top of predicted placements to snap coordinates and rank WMO/M2 names via `pm4_asset_matching.scorer`. It consumes pre-parsed database segment structures (`Pm4SegmentSignalRecord`) instead of reading raw `.pm4` files.

#### Slice 5: Trainer Memory & Zarr Optimizations
- **FR-102-501**: Training code MUST implement `--gradient-checkpointing` and `--8bit-optimizer` flags.
- **FR-102-502**: Use the sequential preloading cache `TileSource.preload()` to batch Zarr disk reads contiguously on start.

---

## Success Criteria

### Measurable Outcomes

- **SC-102-001**: Main model training fits on an 8 GB VRAM GPU (peak CUDA VRAM $< 7.0$ GB).
- **SC-102-002**: PM4 guided snapping and name matching operates as a post-processing CLI step using pre-parsed database records.
- **SC-102-003**: Reconstructed MCAL `alpha_256` maps achieve an SSIM $\geq 0.85$ on validation sets using predicted fractal parameters.
- **SC-102-004**: High-res heights and WDL priors are mathematically aligned.
- **SC-102-005**: SegFormer semantic object footprint segmentation achieves an IoU $\geq 0.85$.

---

## What This Spec Does NOT Do

- **No PM4 binary file parser**: Python code does not open or parse binary PM4 files; it reads pre-extracted segment record schemas loaded by the dataset builder.
- **No Python ADT Format Writer**: The model outputs are written directly to a structured Zarr dataset store to be compiled or consumed by other C# tooling.

---

## Implementation Order

0. **Slice 0**: Lean V25 Zarr dataset builder (`build_v25_dataset.py` + `harvester/v25/dataset.py`) sourcing V18/V22/V24.
1. **Slice 1**: SegFormer frontend and decompiler decoders (inpainting and object placements).
2. **Slice 2**: GPU Sylvester Solver and progressive `V25StageBPredictor` for $257\times257$ heights.
3. **Slice 3**: `WdlDownsampler` to generate WDL priors from high-res heightmaps.
4. **Slice 4**: Decoupled `V25Pm4GuideHandler` post-processor for alignment and asset matching.
5. **Slice 5**: `DifferentiableFractalGenerator` in PyTorch and `FractalParameterHead` predicting seeds and parameters.
6. **Slice 6**: Texture decoders (`MtexPredictor`, `MclyDecoder`).
7. **Slice 7**: Unified `V25SegformerDecompiler` model architecture.
8. **Slice 8**: Multi-task `V25UnifiedLoss` training function.
9. **Slice 9**: Trainer script `train_v25_decompiler.py` with 8 GB VRAM optimizations.
10. **Slice 10**: Inference CLI `infer_v25_decompiler.py` outputting a structured Zarr dataset (Blosc LZ4, level 1 compression).
