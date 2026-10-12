# Spec 270 — PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Owner**: Epic 253 (PM4 Navmesh Research) · **Branch**: `v0.6.0-dev` · **Status**: Active (2026-10-10)

---

## 1. Background & Problem Statement

Across the historical `original_development` map dataset (`wow-viewer/test_data/original_development/World/Maps/development/`):
- There are **616 authentic `.pm4` navmesh/collision files**.
- There are **426 companion `_obj0.adt` placement files**.
- Exactly **399 tiles** contain both `.pm4` collision and `_obj0.adt` object placements, representing a massive, ground-truth pairing corpus.
- The remaining **217 `.pm4` tiles** are unauthored in terms of object placements (no `_obj0.adt` exists), representing missing developer settlements, dungeons, camps, and outposts.

Previous attempts to populate these tiles relied on heuristic guesses or generic shape bounding-box approximations (`ShapeOf(min, max)` in `Pm4PlacementGeneratorSupport`), which lacked persistent indexing, ignored fine geometric features, dropped M2 doodad linkages, and failed to recover authentic model identities and rotations.

Over two years of deep reverse-engineering of the PM4 binary format by the operator yielded fundamental, settled architectural facts:
1. **Coordinate Frame**: `MSVT` is stored in ADT placement space:
   $$\text{placement} = (17066.666 - \text{MSVT}.X, 17066.666 - \text{MSVT}.Y, \text{MSVT}.Z)$$
   with $92.4\%$ MDDF/MODF positions falling precisely inside their paired PM4 footprint.
2. **Asset Partitioning via CK24**:
   - A non-zero keyed `CK24` represents exactly one placed WMO ($99.3\%$ inside MODF bounding boxes; $0$ false positives across WMO-free tiles).
   - `CK24 == 0` is the per-tile bucket containing all M2 doodads and terrain collision remainder.
3. **M2 Doodad Identity**: Spatially connected components within `CK24 == 0` (welded at $\epsilon = 0.25$) land on MDDF doodads at $95.1\%$ rate; `MSLK.GroupObjectId` provides **$99.9\%$ distinctness** as the authentic per-doodad grouping key.
4. **Placement Z Key**: `MSUR._0x1C` (`PackedParams` read as single-precision float) matches the producing placement's Z coordinate bit-exactly on $93.58\%$ of objects.

This specification unifies these proven discoveries into a **persistent, high-precision Rosetta Ledger** and automated placement pipeline to restore ground-truth WMO and M2 placements across all unauthored PM4 tiles.

---

## 2. User Stories & Acceptance Criteria

### US-1: Ground-Truth Association Harvesting (The PM4 Ledger)
As a world builder and researcher, I want a tool that audits all 399 paired development tiles and serializes every PM4 collision object $\leftrightarrow$ ADT placement match into a persistent, queryable ledger, so that we have an exhaustive dictionary of how PM4 collision meshes map to authentic Blizzard assets.

- **AC-1.1**: The harvester scans all 399 paired tiles in `test_data/original_development/`.
- **AC-1.2**: For every WMO (`MODF`) and M2 (`MDDF`) placement, extracts:
  - Canonical asset path (`World\wmo\...` or `World\...mdx`).
  - Spatial bounding box, centroid, surface count, index count, vertex count.
  - Distinct `CK24` key, `MSLK.GroupObjectId`, and `PackedParams` Z-float.
  - 3D vertex fingerprint / covariance eigenvalues (scale & shape invariant).
- **AC-1.3**: Serializes the ledger to `output/pm4_object_association_ledger.json` (and `.parquet` for high-speed ML/query pipelines) with zero dropped records.
- **AC-1.4**: Emits an audit ledger summary verifying $>95\%$ match rate on paired tiles.

### US-2: Multi-Signal Geometric Fingerprinting
As an ML engineer, I want the ledger to store rich geometric invariants (surface triangle counts, bounding extents, aspect ratios, surface density, height spans, and planar normals) so that unauthored PM4 objects can be matched to assets with high confidence even when positions are displaced or orientations differ.

- **AC-2.1**: Each ledger asset entry indexes the distribution of shapes observed across multiple instances (min/max/mean surfaces, extents $[L, W, H]$).
- **AC-2.2**: Unique assets placed in multiple locations provide cross-instance invariance validation.

### US-3: Automated Unauthored Tile Placement Generator & Monolithic LK ADT Assembly
As a developer, I want a pipeline that takes unauthored PM4 tiles, queries the PM4 ledger and neighbor context with strict spillover deduplication, recovers authentic 2D PCA Yaw orientation, and outputs standard monolithic LK ADTs (`development_XX_YY.adt`) merged with surviving terrain meshes.

- **AC-3.1**: Ground-truth preservation: The 255 hand-authored tiles in `test_data/WoWMuseum/335-dev/` are strictly protected and never overwritten or altered.
- **AC-3.2**: Spillover deduplication: Large neighbor WMOs crossing tile boundaries are capped to 1 instance per parent placement with bit-exact Z validation, eliminating duplicate stacking.
- **AC-3.3**: Tight geometric distance threshold ($\le 0.18$) with vertex/surface plausibility to prevent false building assignment.
- **AC-3.4**: Orientation recovery: Derives authentic Yaw rotation in degrees from 2D PCA on the object's $(X, Y)$ vertices, aligned to the model's canonical frame.
- **AC-3.5**: Monolithic LK ADT format: Injects recovered placements into the base terrain from `335-dev` using `BlankAdtFactory.WithPlacements` and writes standard monolithic `.adt` files via `LkAdtWriter.Build(..., MapConversionTargetFormat.LkAdtV18)`.
- **AC-3.6**: Round-trip verification: Generated monolithic `.adt` files parse losslessly via `LkAdtReader.Read()` with 0 missing/failed tiles in `WoWViewer`.

### US-4: Viewer Outliner & Visual Verification
As an operator, I want to inspect restored placements directly in `WoWViewer` alongside the PM4 collision overlay to visually confirm alignment with zero clipping or drift.

- **AC-4.1**: Restored monolithic `.adt` files load seamlessly in `WoWViewer` with 0 missing/failed tile errors.
- **AC-4.2**: Visual parity between PM4 collision surfaces and rendered 3D geometry with authentic building yaw orientation.

### US-5: Multi-Source Placement Convergence & Deduplication
As a world builder, I want the restoration pipeline to converge across all available sources (`WoWMuseum/335-dev/`, `original_development/`, and `.pm4` navmeshes), keeping all existing authored placements intact, magnetically aligning transforms where PM4 collision geometry is superior, and injecting only net-new unauthored objects with strict tile-boundary containment so that large spanned models (e.g. Stormwind Harbor) exist only once.

- **AC-5.1**: Existing authored placements (`MDDF`/`MODF`) from `335-dev` and `original_development` `_obj0.adt` are strictly preserved.
- **AC-5.2**: Matches between existing placements and PM4 collision components are updated with authentic 2D PCA Yaw rotation and magnetic centroid offsets without creating duplicate records.
- **AC-5.3**: Net-new PM4 collision components ($CK24 \neq 0$ WMOs and welded M2 doodads) are synthesized and appended only when no existing placement covers the collision component.
- **AC-5.4**: Spanned multi-tile models are anchored strictly within their placement tile bounds (`IsWithinPlacementTileBounds`), preventing multi-tile cloning.

### US-6: Minimap PNG $\to$ MCCV Baking for Untextured & Missing Tiles
As an operator, I want all untextured, missing, or synthetic terrain tiles to be shaded with authentic colors from the 2,252 development minimap PNGs (`test_data/development/World/Textures/Minimap/development_{tileX}_{tileY}.png`) baked directly into 145-vertex `MCCV` chunks, eliminating pitch-black voids and giving every tile instant geographic recognition in `WoWViewer`.

- **AC-6.1**: Reads authentic $256 \times 256$ RGBA minimap PNGs using `SixLabors.ImageSharp`.
- **AC-6.2**: Samples minimap pixel colors at each of the 145 vertex locations across all 256 chunks ($37,120$ vertices per tile) into 580-byte BGRA payloads.
- **AC-6.3**: Injects synthesized `MCCV` payloads into `MCNK` chunks and enables the `McnkHasMccvFlag` (`0x40`).
- **AC-6.4**: Assigns clean neutral base diffuse layers in `MCLY`/`MTEX` so the terrain shader modulates diffuse lighting with the baked minimap colors.
- **AC-6.5**: Zero black tiles: all generated monolithic LK ADTs display full-color terrain in `WoWViewer`.

---

## 3. Architecture & File Locations

- **Core Ledger Harvest**:
  - `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ObjectLedgerHarvester.cs`
  - `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ObjectLedger.cs`
- **Core Placement & Rotation Resolver**:
  - `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4UnauthoredPlacementResolver.cs`
  - `wow-viewer/src/core/WowViewer.Core.IO/Maps/BlankAdtFactory.cs`
  - `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkAdtWriter.cs`
- **MCCV Image Sampler & Minimap Bridge**:
  - `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMccvImageSampler.cs`
  - `wow-viewer/src/core/WowViewer.Core.IO/Maps/MccvTerrainShadowService.cs`
- **Tooling & CLI**:
  - `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4RestoreUnauthoredSupport.cs`
- **Ground Truth & Minimap Base**:
  - `wow-viewer/test_data/WoWMuseum/335-dev/World/Maps/development/`
  - `wow-viewer/test_data/development/World/Textures/Minimap/`
- **Output Artifacts**:
  - `wow-viewer/output/datasets/pm4_object_association_ledger.json`
  - `wow-viewer/output/restored_development/World/Maps/development/` (monolithic `.adt` files)

---

## 4. Scope Amendments

### 2026-10-10: Monolithic LK ADT Delivery & 2D PCA Yaw
- Replaced split `_obj0.adt` output with monolithic 3.3.5 `.adt` files to resolve the 66 failed/unreadable tile defect.
- Enforced guard on all 255 hand-authored tiles from `WoWMuseum/335-dev` byte-for-byte.
- Added principal component covariance analysis on $(X, Y)$ vertices to compute authentic building yaw rotation matching PM4 collision bounds.
- Capped border-crossing WMOs to single instances per parent placement to prevent duplicate stacking.

### 2026-10-11: Multi-Source Placement Convergence & Minimap MCCV Baking (Phase 5)
- **Multi-Source Convergence**: Pull from `335-dev`, `original_development` `_obj0.adt`, and `.pm4` navmeshes. Retain all existing authored placements, magnetically align matches, and append only net-new PM4 collision objects.
- **Minimap PNG $\to$ MCCV Baking**: Sample all 2,252 authentic $256 \times 256$ development minimap PNGs into 145-vertex `MCCV` BGRA chunks for every untextured or missing tile, eliminating black tiles across the map.
- **Monolithic Preservation**: Strip all companion `_obj0.adt` and `_tex0.adt` files to prevent split-file chunk desync in `WoWViewer`.
