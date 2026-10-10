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

### US-3: Automated Unauthored Tile Placement Generator
As a developer, I want a pipeline that takes any unauthored PM4 tile (e.g. `development_17_13.pm4`, `17_14.pm4`), queries the PM4 ledger, and generates a valid, authentic `_obj0.adt` file containing recovered WMO and M2 placements.

- **AC-3.1**: Generates standard ADT placement chunks (`MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`).
- **AC-3.2**: Accurately places WMOs at the exact PM4 centroids with correct base elevations.
- **AC-3.3**: Places recovered M2 doodads matching the connected component clusters in `CK24 == 0`.
- **AC-3.4**: Round-trip verified: generated `_obj0.adt` files must parse cleanly with `AdtPlacementReader.Read()`.

### US-4: Viewer Outliner & Visual Verification
As an operator, I want to inspect restored placements directly in `WoWViewer` alongside the PM4 collision overlay to visually confirm alignment with zero clipping or drift.

- **AC-4.1**: Restored `_obj0.adt` files load seamlessly in `WoWViewer` with the PM4 Scene Graph panel.
- **AC-4.2**: Visual parity between PM4 collision surfaces and rendered 3D geometry.

---

## 3. Architecture & File Locations

- **Core Ledger Harvest**:
  - `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ObjectLedgerHarvester.cs`
  - `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ObjectLedger.cs`
- **Tooling & CLI**:
  - `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4ObjectLedgerCommand.cs`
  - `wow-viewer/data-harvester/scripts/build_pm4_object_ledger.py`
- **Output Artifacts**:
  - `wow-viewer/output/datasets/pm4_object_association_ledger.json`
- **Restored Placements**:
  - `wow-viewer/output/restored_development/World/Maps/development/`
