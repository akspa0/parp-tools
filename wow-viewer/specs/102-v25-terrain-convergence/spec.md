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

**Input**: User description — "hold off on ADT writing, we already have code to handle that from heightmaps, built in to our viewer... We also added the ability to generate new ADT's, entirely from code in our c# tooling... make sure no new regressions or bugs exist in the changeset, and then update memory bank files, speckit plans, and commit to git. We will continue implementation and testing in a fresh chat"

---

## Problem Statement

The terrain decompiler and texture convergence model must be **completely universal** to process any map from a single raw RGB minimap image.

To keep codebases decoupled and avoid duplicating compilation tools, python-side ADT writing is out of scope. Instead, the inference CLI writes the predicted terrain geometry, objects, and textures to a unified **Numpy `.npz` and JSON payload**. The existing C# terrain compiler reads these payloads and compiles them into game-ready binary ADT files.

The system is decoupled into two separate parts:
1.  **Universal Decompiler Model (`V25SegformerDecompiler`)**: A neural network that runs from a single raw RGB minimap tile and predicts detailed heights, object instance bounds/rotations, MTEX selectors, MCLY assignments, and MCAL alpha maps (using a **Differentiable Fractal Noise Generator** to represent repeating artist brushes). It has **no** PM4 dependencies.
2.  **PM4 Guided Post-Processing Handler (`V25Pm4GuideHandler`)**: A separate CLI/utility class. When a PM4 file is passed during inference (`--pm4`), it snaps the universal model's predicted object coordinates to PM4 segment bounds and matches WMO/M2 models from the asset library.

The pipeline runs entirely offline within an **8 GB / 10 GB VRAM GPU** limit using gradient checkpointing, 8-bit AdamW, and Zarr slice preloading.

---

## User Scenarios & Testing

### User Story 1 — Universal Single-Image Reconstruction (Priority: P1)

The user drops a custom minimap PNG (e.g. from an expansion or custom edit) with no PM4 files available. The model performs forward inference and generates a textured high-res terrain ADT mesh.

**Why this priority**: Core universal pipeline.

**Independent Test**: Run inference on an arbitrary screenshot tile without specifying any `--pm4` arguments.

**Acceptance Scenarios**:
1. **Given** a raw minimap, **When** processed by the CLI, **Then** it generates a valid textured ADT file utilizing only visual feature maps, showing that the network behaves universally.

---

### User Story 2 — Separate PM4-Guided Post-Processing (Priority: P1)

The user runs inference on a development map and supplies the corresponding PM4 file. The CLI invokes the post-processing handler to snap predicted placements to PM4 centroids and matches WMO/M2 models, producing the final aligned development ADT.

**Why this priority**: Bounded development map reconstruction.

**Independent Test**: Verify that the PM4 handler runs as a separate step on top of the universal network's output predictions.

**Acceptance Scenarios**:
1. **Given** predicted object placements from the universal model, **When** passed through the separate `V25Pm4GuideHandler` alongside a PM4 file, **Then** coordinates snap to PM4 centroids and names are resolved.

---

## Requirements

### Functional Requirements

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
- **FR-102-402**: `V25Pm4GuideHandler` is a post-processing step running on top of predicted placements to snap coordinates and rank WMO/M2 names via `pm4_asset_matching.scorer`.

#### Slice 5: Trainer Memory Optimizations
- **FR-102-501**: Training code MUST implement `--gradient-checkpointing` and `--8bit-optimizer` flags.
- **FR-102-502**: Use the sequential preloading cache `TileSource.preload()` to batch Zarr disk reads contiguously on start.

---

## Success Criteria

### Measurable Outcomes

- **SC-102-001**: Main model training fits on an 8 GB VRAM GPU (peak CUDA VRAM $< 7.0$ GB).
- **SC-102-002**: PM4 guided snapping and name matching operates as a post-processing CLI step (`--pm4`).
- **SC-102-003**: Reconstructed MCAL `alpha_256` maps achieve an SSIM $\geq 0.85$ on validation sets using predicted fractal parameters.
- **SC-102-004**: High-res heights and WDL priors are mathematically aligned.
- **SC-102-005**: SegFormer semantic object footprint segmentation achieves an IoU $\geq 0.85$.

---

## What This Spec Does NOT Do

- **No PM4 neural layers**: The neural network weights have no knowledge of PM4 files or coordinate schemas.
- **No Python ADT Format Writer**: The model outputs are written to raw numpy `.npz` and JSON placement catalogs to be compiled directly by C# tooling.

---

## Implementation Order

1. **Slice 1**: SegFormer frontend and decompiler decoders (inpainting and object placements).
2. **Slice 2**: GPU Sylvester Solver and progressive `V25StageBPredictor` for $257\times257$ heights.
3. **Slice 3**: `WdlDownsampler` to generate WDL priors from high-res heightmaps.
4. **Slice 4**: Decoupled `V25Pm4GuideHandler` post-processor for alignment and asset matching.
5. **Slice 5**: `DifferentiableFractalGenerator` in PyTorch and `FractalParameterHead` predicting seeds and parameters.
6. **Slice 6**: Texture decoders (`MtexPredictor`, `MclyDecoder`).
7. **Slice 7**: Unified `V25SegformerDecompiler` model architecture.
8. **Slice 8**: Multi-task `V25UnifiedLoss` training function.
9. **Slice 9**: Trainer script `train_v25_decompiler.py` with 8 GB VRAM optimizations.
10. **Slice 10**: Inference CLI `infer_v25_decompiler.py` outputting `.npz` and `.json` data packages.
