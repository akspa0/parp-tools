# V14 Modular Residual Model System

**Date:** 2026-05-06
**Branch:** `v0.4.9-strict-guards`
**Base commit:** `971fff2`

---

## 0. Core Philosophy

**No monolithic models. No multi-task training. No shared weights between models.**

The V14 system decomposes a minimap into its constituent parts, then uses the residuals to reconstruct terrain geometry. The pipeline is:

```
Minimap image → Decompose into tilesets + alpha masks → Residuals → Terrain mesh reconstruction
```

### How Minimaps Are Built

A minimap tile is composed of:
1. **Tileset textures** (0-2 layers). Some tiles have no tilesets at all — they appear as solid white plates.
2. **Alpha mask layers** (1-2 layers) that blend the tilesets together.
3. **Everything else** — objects, MCCV vertex color painting, atmospheric effects, and compositing errors. This is the **residual**.

The residual is the only signal fed into the terrain reconstruction model. It encodes what the minimap shows beyond the basic tileset composition.

### Lessons from V12

V12's core idea was correct: decompose the minimap into tileset textures (MCAL/MCLY) plus a residual. The execution failed because the compositing path was broken, producing incorrect residuals.

### What Ground Truth Means

Ground truth is data that comes directly from the game client files:

| Signal | Ground Truth Source |
|--------|-------------------|
| Height values | MCVT chunk vertices |
| Alpha weights | MCAL chunk bytes |
| Texture IDs | MCLY chunk entries |
| Hole masks | MCNH chunk bits |
| Liquid data | MH2O/MCLQ chunks |
| Minimap pixels | Minimap PNG/BLP files |
| Texture pixels | BLP texture files |

What is **NOT** ground truth:
- MdxViewer's rendered terrain (different lighting/renderer than the original game)
- Any synthesized/composited minimap
- Derived residuals
- Any signal produced by a model

**Rule:** If a signal passes through a renderer, compositor, or model before reaching the training pipeline, it is not ground truth.

### Code Audit: What Exists vs What Is Needed

An audit of `wow-viewer/src/` was performed to determine readiness for the harvester tool.

#### Exists (Ready to Use)

| Component | Location | Status |
|-----------|----------|--------|
| MCAL alpha reading | `WowViewer.Core.IO/Maps/AdtMcalDecoder.cs` | Complete — handles compressed, big-alpha, packed 4-bit |
| MCLY texture layer reading | `WowViewer.Core.IO/Maps/AdtMcalDecoder.cs` + data models | Complete |
| ADT reading (root, _tex0, _obj0) | `WowViewer.Core.IO/Maps/AdtTextureReader.cs` + others | Complete |
| NPZ shard writing | `WowViewer.Core.IO/Maps/NpzTileSerializer.cs` | Complete |
| Terrain tensor pack assembly | `WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs` | Complete |
| Training supervision export | `WowViewer.Core.IO/Maps/AdtTextureTrainingSupervisionExporter.cs` | Produces tileset index + mask PNGs |

#### Missing (Needs to Be Created)

| Component | What It Does | Priority |
|-----------|-------------|----------|
| Synthetic minimap compositor | No code composites tileset textures + MCAL alpha into a 256×256 RGB image. A new `TerrainMinimapCompositor.cs` is needed in `WowViewer.Core.IO`. | High |
| Harvester CLI tool | No `WowViewer.Tool.Harvest` project exists. Needs to be created. | High |
| `data-harvester/` directory | Python training environment does not exist yet. | High |

**Note:** BLP pixel decoding is now available via SereniaBLPLib (added to `WowViewer.Core.IO.csproj`). Tests pass confirming `GetPixels()` and `GetImage()` work correctly.

#### External Dependency Violations (RESOLVED)

| Violation | Location | Fix Applied |
|-----------|----------|-------------|
| `WowViewer.Core.IO.csproj` referenced `gillijimproject_refactor/lib/wow.tools.local/DBCD/DBCD/DBCD.csproj` | Project file | Updated to reference `libs/wowdev/DBCD/DBCD/DBCD.csproj` |
| `WowViewer.Core.IO.csproj` referenced `gillijimproject_refactor/lib/WoWDBDefs/definitions/` | Project file | Updated to reference `libs/wowdev/WoWDBDefs/definitions/` |
| Runtime fallback paths to `gillijimproject_refactor/` in source files | Multiple files | Removed from 5 source files + 3 test files |
| CPM conflict with DBCD | DBCD.csproj | Added `Directory.Packages.props` to `libs/wowdev/DBCD/` to disable CPM |

**wow-viewer now has zero references to `gillijimproject_refactor` in source code.** Build succeeds with 0 errors. Tests pass (AreaIdMapper: 7/7, BLP pixel decode: 2/2).

**All alpha map reading code must reside in wow-viewer.** The MCAL decoder is already ported and complete. The BLP pixel decoder is available via SereniaBLPLib (added to WowViewer.Core.IO). The remaining work is the synthetic minimap compositor. Any code ported from MdxViewer must be self-contained within wow-viewer with zero references to `gillijimproject_refactor` or other external projects.

---

## 1. The Pipeline

### Stage 1: Decomposition

```
Input: Minimap image (256×256×3 RGB)
Output: Tileset layers (0-2) + Alpha masks (1-2) + Residual (256×256×3)
```

The decomposition extracts what can be explained by known tileset textures and alpha blending. What remains is the residual.

### Stage 2: Terrain Reconstruction

```
Input: Residual (256×256×3)
Output: Terrain mesh (height_257, hole_mask, liquid_mask, etc.)
```

The reconstruction model learns to predict terrain geometry from the residual alone. The residual encodes terrain appearance that isn't explained by tilesets — which correlates with height, holes, liquids, and other terrain features.

### Stage 3: Output

The system can produce:
- **OBJ mesh + materials** — for external use
- **Synthesized high-resolution minimap** — rendered from the reconstructed terrain
- **Patched ADT** — texturing and mesh data written back to the game file

### End-User Workflow (Future)

1. User loads a minimap image (single tile or large map image constrained to 64×64 tile bounds)
2. UI allows alignment of the image to tile boundaries
3. User presses "Process"
4. System decomposes → reconstructs → outputs OBJ/materials/minimap/patched ADT

---

## 2. The Models

### Model D1: Tileset Decomposition

| Property | Value |
|----------|-------|
| **Input** | minimap_rgb_256 (3ch) |
| **Output** | tileset_layer_1 (256×256×3), tileset_layer_2 (256×256×3), alpha_mask_1 (256×256), alpha_mask_2 (256×256) |
| **Architecture** | Small U-Net: 4-layer encoder, 4-layer decoder, 4 output heads |
| **Params** | ~3M |
| **Loss** | L1 (tilesets) + BCE (alpha masks) |

Decomposes the minimap into 0-2 tileset layers and 1-2 alpha masks. Some tiles have no tilesets — the model should output blank layers in those cases.

### Model D2: Residual Computation

This is not a model — it's a subtraction:

```
residual = minimap - composite(tileset_layers, alpha_masks)
```

The residual is computed deterministically from D1's output and the original minimap.

### Model R1: Terrain Reconstruction

| Property | Value |
|----------|-------|
| **Input** | residual (256×256×3) |
| **Output** | height_257 (257×257), hole_mask_16 (16×16), liquid_mask_256 (256×256) |
| **Architecture** | U-Net: 5-layer encoder, 5-layer decoder with skips, 3 output heads |
| **Params** | ~5M |
| **Loss** | L1 (height) + BCE (holes, liquids) + Sobel edge (height) |

Reconstructs terrain geometry from the residual alone. The residual encodes terrain appearance that correlates with height, holes, and liquids.

---

## 3. Why This Works

The minimap contains terrain appearance information. By decomposing it into known tileset textures and alpha masks, we isolate the **unexplained** portion — the residual. This residual correlates with terrain geometry because:

- Height variations affect how tileset textures appear (perspective, lighting)
- Holes create visual gaps in the minimap
- Liquid areas have distinct appearance
- MCCV vertex color painting encodes terrain detail

The reconstruction model learns these correlations: given a residual, predict the terrain geometry that produced it.

---

## 4. Data Requirements

### Per-Model Array Requirements

| Model | Required arrays | Optional arrays |
|-------|----------------|-----------------|
| D1 | minimap_rgb_256, mcal_alpha_pack_256, mcly_texture_ids | — |
| R1 | residual (computed from D1 output), height_257, hole_mask_16, liquid_mask_257 | — |

### Shard Generation

The harvester tool (`WowViewer.Tool.Harvest`) produces NPZ shards with all available signals. Models only read what they need.

---

## 5. Implementation Order

### Phase 0: Dependency Cleanup (Before Any Model Work)
- [x] Clone DBCD into `wow-viewer/libs/wowdev/DBCD/` (already existed)
- [x] Clone WoWDBDefs into `wow-viewer/libs/wowdev/WoWDBDefs/` (already existed)
- [x] Update `WowViewer.Core.IO.csproj` to reference local DBCD (`libs/wowdev/DBCD/DBCD/DBCD.csproj`)
- [x] Update `WowViewer.Core.IO.csproj` to reference local WoWDBDefs definitions (`libs/wowdev/WoWDBDefs/definitions/`)
- [x] Add `Directory.Packages.props` in `libs/wowdev/DBCD/` to disable CPM for DBCD projects
- [x] Add missing PackageVersion entries to wow-viewer's `Directory.Packages.props` (DBDefsLib, Microsoft.CSharp, Microsoft.SourceLink.GitHub, System.Runtime.CompilerServices.Unsafe)
- [x] Remove runtime fallback paths from `Pm4CoordinateService.cs`
- [x] Remove runtime fallback paths from `AreaIdMapper.cs`
- [x] Remove runtime fallback paths from `AlphaAreaAudioCatalogReader.cs`
- [x] Remove runtime fallback paths from `StormLibPatchArchiveReader.cs`
- [x] Remove runtime fallback paths from `WowViewerArchiveBootstrap.cs`
- [x] Update test path references in `MapFileSummaryReaderTests.cs`
- [x] Update test path references in `Pm4ResearchIntegrationTests.cs`
- [x] Update test path references in `AreaIdMapperTests.cs`
- [x] Verify `wow-viewer` builds with zero references to sibling folders (build succeeds)
- [x] Verify zero `gillijimproject_refactor` references in source files (grep confirms)
- [x] Add SereniaBLPLib reference to `WowViewer.Core.IO.csproj` (BLP pixel decoding via existing library)
- [x] Add BLP pixel decoding tests (`BlpPixelDecoderTests.cs` — 2 tests pass)
- [ ] Copy remaining WoW test data from `gillijimproject_refactor/test_data/` to `wow-viewer/test_data/` if needed
- [ ] Verify all alpha map reading code (MCAL, MCLY, alpha blending) resides in wow-viewer

### Phase 1: D1 (Tileset Decomposition)
- [ ] Create `data-harvester/scripts/train_d1.py`
- [ ] Small U-Net: 4-layer encoder, 4-layer decoder, 4 output heads
- [ ] Train on development tiles
- [ ] Validate: tileset L1 < 0.1, alpha mask accuracy > 80%

### Phase 2: Residual Computation
- [ ] Implement residual = minimap - composite(D1 output)
- [ ] Validate: residual is not uniform noise, not identical to minimap
- [ ] Visual inspection: residual captures objects, vertex colors, detail

### Phase 3: R1 (Terrain Reconstruction)
- [ ] Create `train_r1.py`
- [ ] U-Net: 5-layer encoder, 5-layer decoder, 3 output heads
- [ ] Train on development tiles
- [ ] Validate: height MAE < 5m, hole accuracy > 90%, liquid accuracy > 85%

### Phase 4: Output Generation
- [ ] Implement OBJ mesh export from height_257
- [ ] Implement material generation from tileset decomposition
- [ ] Implement synthesized minimap rendering
- [ ] Implement ADT patching (write mesh + texture data back to game file)

### Phase 5: UI Integration
- [ ] Create wow-viewer UI for loading minimap images
- [ ] Implement image alignment to tile boundaries
- [ ] Implement "Process" button that runs D1 → D2 → R1 → output
- [ ] Support single tiles and large map images (constrained to 64×64 tile bounds)

---

## 6. Model Size Summary

| Model | Params | Train Time | GPU Memory |
|-------|--------|------------|------------|
| D1 | ~3M | < 3h | < 4GB |
| R1 | ~5M | < 4h | < 5GB |
| **Total** | **~8M** | **< 7h** | **< 5GB** |

---

## 7. What This Solves

| Problem | Previous Approach | V14 |
|---------|-------------------|-----|
| Training complexity | Monolithic model predicting everything | Two models: decompose, then reconstruct |
| Input signals | 25+ channels, most sparse | 3 channels (residual only) |
| Debugging | Unclear failure source | Test D1 and R1 independently |
| Output flexibility | Fixed output format | OBJ, materials, minimap, or patched ADT |
| End-user workflow | CLI only | UI with image alignment and one-click processing |

---

## 8. Execution Guardrails

### Rule 1: One Phase at a Time

Work on Phase N+1 only after Phase N is complete. Complete means:
- Code is written and committed
- Tests pass
- Validation against ground truth succeeds
- Results are documented

### Rule 2: No Scope Creep Within a Phase

If a phase specifies a set of tasks, complete only those tasks. Do not:
- Add features not in the phase checklist
- Change architecture mid-phase
- Add validation metrics not in the checklist
- Begin the next phase early

### Rule 3: Validation Before Completion

A phase is not complete when code compiles. It is complete when:
- Output matches ground truth (raw game file data)
- Results are reproducible
- Metrics meet the phase's validation criteria

### Rule 4: Trust Raw Data Over Derived Data

When raw game file data conflicts with derived signals (residuals, composites, synthesized minimaps), trust the raw data. Verify derived data problems against ground truth before investing effort.

---

## 9. What NOT to Do

- Do NOT combine models into a single training script
- Do NOT add new heads to existing models
- Do NOT train multiple models in one pass
- Do NOT make models depend on each other's weights (only on each other's outputs)
- Do NOT skip the residual computation — always predict from the residual, not the full minimap
- Do NOT modify `gillijimproject_refactor`
- Do NOT train R1 before D1 produces valid decomposition
- Do NOT trust a residual computed from broken decomposition
- Do NOT proceed to model training if visual inspection of residuals shows compositing artifacts instead of actual terrain features

## 10. LkToAlpha Conversion Validation

The `convert-lk-to-alpha` pipeline converts Cataclysm 4.0.0 split ADTs into Alpha-compatible monolithic WDT/WDL files. Validation is through legacy MdxViewer rendering.

### 10.1 Pipeline stages

1. **Read source**: `LkAdtReader` parses Cataclysm split ADTs (`_tex0.adt` + `_obj0.adt`) into domain models
2. **Convert**: `LkToAlphaConverter` builds `AlphaTileData` with heightmap, normals, texture layers, liquid, placements
3. **Write**: `AlphaWdtWriter` emits monolithic Alpha WDT binary matching legacy reader expectations
4. **Bundle** (optional): `--bundle-tilesets` extracts textures, `--target-client-root` filters placements

### 10.2 Critical structural constraints (2026-05-09 fixes)

- **MAIN grid**: The 0.5.3 client uses row-major (`tileY * 64 + tileX`) after reading raw `MAIN` entries into `areaInfo`.
- **MCNK emission**: All 256 MCNKs must be emitted with valid MCVT/MCNR/MCLY/MCRF data, even for empty tiles.
- **MCRF**: Always emit the MCRF chunk (even with 0 entries). Legacy `McnkAlpha` reads it unconditionally.
- **Chunk IDs**: Use `FourCC.FromString().ToFileBytes()` for writing; readers expect reversed FourCC on disk.

### 10.3 MdxViewer validation workflow

See `wow-viewer/README.md` "Manual Validation with MdxViewer" section for the complete step-by-step.

### 10.4 Known limitations

- MCRF per-chunk reference indices are not written to Alpha MCNK (only placement names are preserved at WDT level)
- Ame to D current converter rebuilds a reduced terrain-domain model; non-decoded chunk families are dropped
- WMO v14↔v17 converters are not yet ported
- M2/MDX converters are not yet ported
- AreaID crosswalk is not wired

## 11. Future: MdxViewer Port to wow-viewer

The legacy `gillijimproject_refactor/src/MdxViewer` contains the runtime rendering and world-session logic used for validation. A long-range goal is to port this into `wow-viewer/src/viewer/WowViewer.App`.

### 11.1 Why port

- Eliminate dependency on legacy reference codebase for runtime validation
- Allow wow-viewer's shared I/O libraries to be consumed directly without adapter layers
- Enable standalone viewer builds

### 11.2 What needs porting

- WorldScene (terrain + WMO + doodad rendering pipeline)
- TerrainManager (AOI streaming, tile loading)
- AlphaTerrainAdapter and StandardTerrainAdapter
- WMO v14/v17 renderer (MomoWmoRenderer etc.)
- M2/skin runtime rendering (WowViewerM2RuntimeBridge)
- UI surfaces (ImGui panels, file browser, settings, inspector)
- Minimap renderer
- Capture automation (startup flags, queue, framebuffer save)

### 11.3 Current state

`WowViewer.App` exists in `wow-viewer/src/viewer/` with:
- Shell application framework
- World session bootstrapper
- Some M2 preview functionality
- WMO preview via `WmoPreviewLoader`/`WmoGpuPreviewRenderer`

The viewer is not yet a replacement for MdxViewer.
