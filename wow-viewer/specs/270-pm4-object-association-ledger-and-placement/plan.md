# Spec 270 — Technical Plan: PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Owner**: Epic 253 · **Date**: 2026-10-10 · **Branch**: `v0.6.0-dev`

---

## 1. Architectural Strategy

The PM4 format reverse-engineering conducted over 2 years provides mathematically grounded relationships between PM4 chunks (`MSVT`, `MSVI`, `MSUR`, `MSLK`, `MSPV`, `MSCN`) and WoW world placements (`MODF`, `MDDF`).

```
Authentic Development Assets (399 paired tiles)
[ .pm4 collision files ]  +  [ _obj0.adt placement files ]
          │                             │
          └──────────────┬──────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │   Pm4ObjectLedgerHarvester       │
        │   - ADT placement space coord    │
        │   - Keyed CK24 -> WMO matching   │
        │   - CK24=0 cluster -> M2 doodad  │
        │   - MSUR._0x1C bit-exact Z       │
        │   - Geometric signature hash     │
        └────────────────┬─────────────────┘
                         ▼
        [ pm4_object_association_ledger.json ]
        (Persistent Canonical Asset Ledger)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   Pm4UnauthoredPlacementResolver │
        │   - Scans unauthored PM4 tiles   │
        │   - Resolves asset by signature  │
        │   - Derives translation & yaw    │
        │   - Synthesizes _obj0.adt chunks │
        └────────────────┬─────────────────┘
                         ▼
     [ Restored development_*_obj0.adt ]
```

---

## 2. Technical Details

### 2.1 Coordinate Conversion
In accordance with `workstream-pm4-decode.md`:
$$\begin{aligned}
\text{Placement}_X &= 17066.666 - \text{MSVT}_X \\
\text{Placement}_Y &= 17066.666 - \text{MSVT}_Y \\
\text{Placement}_Z &= \text{MSVT}_Z
\end{aligned}$$
No axis swap. Horizontal components mapped directly.

### 2.2 WMO Matching (Keyed CK24)
For each object with $CK24 \neq 0$:
1. Collect surface vertices via `MSVI` and `MSVT`.
2. Compute bounding box $[X_{\min}, Y_{\min}, Z_{\min}] \dots [X_{\max}, Y_{\max}, Z_{\max}]$.
3. Find candidates in `MODF` containing or overlapping centroid.
4. Match with `MSUR._0x1C` as float against `MODF.Position.Z`. Delta tolerance: $\le 1.0$ yds.
5. Compute geometric features:
   - Extents $E = (X_{\max} - X_{\min}, Y_{\max} - Y_{\min}, Z_{\max} - Z_{\min})$
   - Surface count, index count, vertex count.
   - PCA / Oriented extents for rotation invariance.

### 2.3 M2 Doodad Matching (CK24 == 0)
For $CK24 == 0$:
1. Run vertex-welded connected component analysis ($\epsilon = 0.25$).
2. Identify distinct `MSLK.GroupObjectId` for each component ($99.9\%$ distinctness).
3. Correlate with nearest `MDDF` placement within $r = 24.0$ yds.
4. Extract M2 asset model path, scale, and orientation.

### 2.4 Persistent Ledger Schema (`Pm4ObjectLedgerEntry`)
```json
{
  "asset_path": "World\\wmo\\Dungeon\\AZ_Subway\\Subway.wmo",
  "asset_kind": "WMO",
  "ck24": 128472,
  "surface_count": 842,
  "vertex_count": 1264,
  "extents": [45.2, 32.1, 14.8],
  "oriented_extents": [45.2, 32.1, 14.8],
  "surface_signature_hash": "a1f8...",
  "instances": [
    {
      "pm4_tile": "development_16_32.pm4",
      "placement_pos": [1245.2, 5821.1, 142.0],
      "placement_rot": [0.0, 45.0, 0.0],
      "unique_id": 14002,
      "z_delta": 0.0001
    }
  ]
}
```

### 2.5 Placement Synthesis & Monolithic LK ADT Assembly (`LkAdtWriter` / `BlankAdtFactory`)
For unauthored tiles (e.g. `development_13_50`, `14_36`, `16_36`):
1. Extract all objects ($CK24 \neq 0$) and doodad components ($CK24 == 0$).
2. Query ledger and neighbor context using strict spillover deduplication and multi-feature shape matching (extents + surface count + vertex density).
3. Compute 2D principal axis yaw rotation ($\theta_{\text{pca}} = \frac{1}{2}\text{atan2}(2C_{xy}, C_{xx} - C_{yy})$) and apply canonical model orientation offset.
4. Construct `MODF` and `MDDF` placement records.
5. Ingest surviving base terrain (`MCNK`, `MTEX`, `MCVT`, `MCNR`) from `test_data/WoWMuseum/335-dev/` via `LkAdtReader.Read()`.
6. Inject resolved placements via `BlankAdtFactory.WithPlacements()`.
7. Serialize monolithic LK 3.3.5 ADT (`.adt`) using `LkAdtWriter.Build(..., MapConversionTargetFormat.LkAdtV18)`.

### 2.6 Multi-Source Placement Convergence & Minimap MCCV Ingestion (`AdtMccvImageSampler`)
To resolve black tiles and unify disparate assets into a single cohesive development map:
1. **Multi-Source Catalog Aggregation**:
   - Ingest existing placements from `335-dev` and `original_development` `_obj0.adt`.
   - Maintain spatial hash map of existing placements.
   - For each PM4 collision component:
     - If an existing placement is within proximity ($\le 25\text{ yd}$), update its Yaw rotation to the PM4 2D PCA orientation and magnetically adjust its centroid without adding a second instance.
     - If no existing placement matches the collision component, synthesize a new placement using shape-matching and append it.
   - Enforce `IsWithinPlacementTileBounds` so that models spanning multiple tiles (e.g. Stormwind Harbor) exist only in their designated host tile.
2. **Minimap PNG $\to$ MCCV Baking**:
   - Ingest $256 \times 256$ RGBA PNG from `test_data/development/World/Textures/Minimap/development_{tileX}_{tileY}.png`.
   - Sample each chunk vertex at $(cx + u, cy + v) / 16 \times (W-1, H-1)$ into BGRA color bytes.
   - If an ADT tile lacks texture layers (`Chunks[0].Layers.Length == 0`), populate its 256 chunks with the synthesized `MCCV` payload, set `McnkHasMccvFlag` (`0x40`), and assign a clean whiteplate base layer (`tileset/ocean/westfallseafloor.blp` or neutral ground).
3. **Monolithic LK ADT Delivery**:
   - Write standard monolithic `.adt` files.
   - Purge stray `_obj0.adt` and `_tex0.adt` files to prevent split-file chunk loader desynchronization.

---

## 3. Phased Implementation Roadmap

- **Phase 1: Ledger Engine & Extraction (Complete)**
  - Implement `Pm4ObjectLedgerHarvester` scanning 399 paired tiles.
  - Produce `output/datasets/pm4_object_association_ledger.json`.
  - Validate $>95\%$ match accuracy across the 399 tiles.
- **Phase 2: High-Precision Placement Resolver & Rotation Engine (Complete)**
  - Implement 2D PCA Yaw rotation calculation on $(X, Y)$ vertices.
  - Implement strict spillover deduplication (1 instance per parent WMO across border).
  - Tighten geometric distance threshold to $\le 0.18$.
- **Phase 3: Ground Truth Protection & Monolithic LK ADT Assembly (Complete)**
  - Enforce pristine preservation for all 255 hand-authored tiles in `WoWMuseum/335-dev/`.
  - Read base terrain from `335-dev`, strip placeholder `OILPLATFORM_LOW`, and inject resolved placements.
  - Serialize monolithic 3.3.5 `.adt` files with 100% round-trip validation.
- **Phase 4: Viewer Verification & Receipts (Complete)**
  - Verify 0 missing/failed tiles in `WoWViewer`.
  - Confirm visual rotation alignment with PM4 collision wireframes.
  - Update completion receipts.
- **Phase 5: Multi-Source Placement Convergence & Minimap MCCV Ingestion**
  - Implement `AdtMccvImageSampler` in `WowViewer.Core.IO` for PNG $\to$ 145-vertex MCCV chunk sampling.
  - Update `Pm4RestoreUnauthoredSupport.cs` to converge Museum, Original `_obj0`, and PM4 placements with deduplication.
  - Bake minimap PNGs onto all untextured/missing tiles, eliminating black tiles.
  - Purge stray `_obj0`/`_tex0` files from output root and verify in `WoWViewer`.
