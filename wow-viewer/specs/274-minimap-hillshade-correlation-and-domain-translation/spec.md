# Spec 274: Minimap Hillshade Correlation & Domain Translation

**Feature Branch**: `v0.6.1-dev`  
**Created**: 2026-10-16  
**Status**: Specified; in progress  
**Owning Epics**: Epic 254 (Datasets, Client Datastore & Terrain ML) / Epic 250 (Map Reconstruction & Editor Platform)  
**Input**: Operator prompt: *"use speckit, this pertains to correlation in a model to train on the synthetic minimaps with no MCAL/image data, just raw hillshade data against existing real minimap tiles, to form a single correlation model that can reconstruct from minimap tile, the terrain implication data. Immediate Next Steps: Batch-Export Clean Pairs... Train Domain Translation..."*

---

## 1. Overview & Problem Statement

Reconstructing authentic 3D terrain elevation from 2D minimaps has previously been confounded by surface texturing and placed objects:
1. **Texture Albedo & MCAL Interference**: Authentic minimaps contain multi-layer texture splats (`MCLY`/`MCAL`), cobblestone roads, snow caps, grass fields, and dirt paths. Direct neural or photometric estimators (such as Poisson Shape-from-Shading) conflate dark diffuse ground albedo with steep shadow indentations, leading to high height errors (land MAE $> 30$ yards, max error $> 170$ yards).
2. **Object & Roof Occlusion**: Doodads (trees, rocks, fences) and WMO buildings (roofs, towers, castle battlements) obscure the ground plane, injecting false vertical high-frequency features into terrain regressors.
3. **The Clean Canonical Shading Truth**: In contrast, our synthetic renderer (`TerrainMinimapCompositor` / `ComposeShadowArray`) produces pure **object-free, texture-free synthetic hillshade** ($N \cdot L$ Lambertian illumination, analytic ray-traced cast shadows, and neutral white albedo). When encoded through the client's BC1/DXT1 4x4 block cycle (`Dxt1TileCodec`), it matches the exact compression artifact domain of real minimap tiles without any confusing albedo or object clutter.

### The Solution: Decoupled Domain Translation & Terrain Implication Correlation
By decoupling the reconstruction into:
1. **Domain Translation (Style-Transfer & Object Stripping)**: Learning a lightweight translation network mapping real textured minimaps $\to$ object-free synthetic DXT1 hillshade minimaps.
2. **Terrain Implication Correlation**: Directly correlating the clean, uncorrupted hillshade representation to low-resolution macro WDL lattices ($33 \times 33$) and full-resolution ADT elevation grids ($257 \times 257$).

---

## 2. User Stories

### User Story 1 - Batch-Export Clean Pairs Across Zarr v3 Map Revisions (Priority: P1)
As an ML dataset engineer, I want a headless batch-export command in `parp-tools` that iterates across all 37 Zarr v3 map revisions (e.g. Wrath 3.0.1.8303 and 3.3.5 stores in `output/datasets/v60/`, spanning 4,400+ tiles), dumping a synchronized, clean paired dataset pairing:
1. Real textured minimap tiles (`minimap_rgb`, $256 \times 256 \times 3$ uint8).
2. Object-free synthetic DXT1 minimap tiles (`synthetic_hillshade_dxt1`, $256 \times 256 \times 3$ uint8, raw hillshade passed through the DXT1 parity cycle).
3. Low-resolution WDL height grids (`wdl_outer_17` + `wdl_inner_16`, combined $33 \times 33$ float32).
4. Full ADT heightmaps (`height_257` / `ground_intent_height_257`, $257 \times 257$ float32).
5. Parquet index with map name, tile coordinates $(X, Y)$, build ID, elevation relief range ($\Delta Z$), and deterministic train/validation split assignment.

### User Story 2 - Lightweight Domain Translation Network (Priority: P1)
As an ML researcher, I want a lightweight style-transfer network ($\le 10\text{M}$ parameters) trained on the paired dataset that accepts real textured minimaps as input and predicts object-free synthetic hillshade minimaps, automatically learning:
1. Doodad and WMO roof stripping (replacing structures with continuous terrain shading).
2. Ground texture normalization and albedo de-mixing (removing roads, grass, dirt, and snow color variations).
3. Preservation of genuine terrain relief shading, ridge crest highlights, and self-shadowed valleys.

### User Story 3 - Terrain Implication Correlation & Reconstructive Inference (Priority: P1)
As a map reconstruction architect, I want a correlation pipeline that ingests clean predicted hillshades and reconstructs the underlying terrain implication data (macro WDL lattice + micro ADT elevation), achieving $> 0.85$ Pearson correlation against ground-truth heights and $\le 8.0$ yards land MAE.

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Headless Batch-Export Command | Tool scans all Zarr v3 map stores and exports paired arrays with deterministic schema | $\ge 2,500$ valid paired tiles exported with 0 NaNs and complete signals |
| **AC-002** | DXT1 Parity in Synthetic Targets | Synthetic hillshade targets undergo DXT1 4x4 block quantization to eliminate codec confound | $256 \times 256 \times 3$ uint8 RGB matching real minimap compression format |
| **AC-003** | Macro & Micro Elevation Alignment | Every exported pair contains synchronized WDL ($33 \times 33$) and ADT ($257 \times 257$) elevation grids | 100% spatial coordinate alignment between tile $(X, Y)$ and elevation targets |
| **AC-004** | Lightweight Domain Translation Model | Parameter budget strictly $\le 10\text{M}$ parameters, runnable on standard consumer GPUs (RTX 4070 Ti) | Model size $\le 40\text{ MB}$, forward latency $\le 15\text{ ms/tile}$ |
| **AC-005** | Object & Texture Normalization Fidelity | Domain translation model achieves high structural fidelity to synthetic hillshade | Validation SSIM $\ge 0.82$, $L_1$ photometric error $\le 0.06$ |
| **AC-006** | Terrain Elevation Correlation Benchmark | Chained pipeline (`Minimap -> Hillshade -> Elevation`) reconstructs authentic relief | Pearson $r \ge 0.85$, Land MAE $\le 8.0\text{ yds}$, 0 inverted normals |

---

## 4. Architectural Boundaries & Governance

- **Governance (AGENTS.md §4, §9)**:
  - C# format readers (`MpqArchiveCatalog`, `StandardTerrainAdapter`, `LkAdtReader`) remain frozen and intact.
  - Python scripts and ML pipelines reside exclusively under `wow-viewer/data-harvester/` using `uv`.
  - Storage conforms to the unified Zarr v3 / Parquet schema.
  - God-class freeze (AGENTS.md §10): Zero members added to `WorldScene.cs` or `ViewerApp.cs`.
- **Operator-Owned Operations (AGENTS.md §3, §5)**:
  - Batch export across the complete 4,400+ tile corpus and neural network training runs are operator-owned.
  - CLI scripts must provide exact PowerShell 7 command invocations and verify all dry-runs/unit tests before operator execution.
