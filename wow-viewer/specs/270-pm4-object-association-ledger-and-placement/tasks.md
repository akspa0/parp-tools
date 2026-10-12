# Spec 270 — Tasks: PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Owner**: Epic 253 · **Status**: Completed (2026-10-10) · **Branch**: `v0.6.0-dev` · **Receipt**: [receipt-spec270.md](file:///I:/parp/parp-tools/wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/receipt-spec270.md)

---

## Phase 1 — Ledger Engine & 399-Tile Ground-Truth Extraction

- [x] **T01**: Implement `Pm4ObjectLedgerHarvester` (C# / CLI tool) harvesting paired tiles in `original_development/`.
- [x] **T02**: Implement multi-feature signature extractor (extents, vertex count, surface count, Z-key `MSUR._0x1C`, `CK24`, and `MSLK.GroupObjectId`).
- [x] **T03**: Run harvest across all paired tiles and serialize `output/datasets/pm4_object_association_ledger.json` (1.89 MB, 1,929 objects).
- [x] **T04**: Verify ledger statistics: record count, asset catalog coverage, and match fraction $\ge 90\%$ (904 matched WMOs, bit-exact Z, 95.1% M2 match).

---

## Phase 2 — Placement Resolver & Unauthored Tile Synthesis

- [x] **T05**: Implement `Pm4UnauthoredPlacementResolver` using ledger geometric distance scoring and adjacent spillover.
- [x] **T06**: Implement ADT placement builder `SplitAdtObjWriter` writing authentic `MVER`, `MMDX`, `MWMO`, `MDDF`, `MODF`, and `MCNK` chunks.
- [x] **T07**: Run placement reconstruction on unauthored development tiles (e.g. `development_16_36`, `14_36`, `13_50`, and full map batch of 130 tiles).
- [x] **T08**: Round-trip check: ensure generated `_obj0.adt` reads back cleanly with `AdtPlacementReader.Read()` (130 / 130 PASS).

---

## Phase 4 — High-Precision Monolithic LK Pipeline & Ground Truth Guard (2026-10-10)

- [x] **T12**: Implement 2D PCA Yaw rotation calculation (`ComputeFootprintYawDegrees`) in `Pm4UnauthoredPlacementResolver.cs` using $(X, Y)$ covariance principal axes.
- [x] **T13**: Implement spillover deduplication (max 1 instance per parent WMO) and tighten geometric distance threshold to $\le 0.18$ in `Pm4UnauthoredPlacementResolver.cs`.
- [x] **T14**: Implement ground truth barrier in `Pm4RestoreUnauthoredSupport.cs` using `WoWMuseum/335-dev/` (strictly preserve all 255 hand-authored tiles).
- [x] **T15**: Implement monolithic LK 3.3.5 ADT generator in `Pm4RestoreUnauthoredSupport.cs` using `LkAdtReader.Read`, `BlankAdtFactory.WithPlacements`, and `LkAdtWriter.Build(..., MapConversionTargetFormat.LkAdtV18)`.
- [x] **T16**: Run reconstruction on unauthored PM4 tiles (e.g. `13_50`, `14_36`, `16_36`, and full corpus) producing valid monolithic `.adt` files.
- [x] **T17**: Validate with C# test suite (`dotnet test`) and update completion receipt (`evidence/receipt-spec270.md`).

---

## Phase 5 — Multi-Source Placement Convergence & Minimap MCCV Ingestion (2026-10-11)

- [x] **T18**: Implement `AdtMccvImageSampler` in `WowViewer.Core.IO/Maps/` to sample $256 \times 256$ RGBA minimap PNGs into 256 145-vertex `MCCV` BGRA chunks (580 bytes each) and set `McnkHasMccvFlag` (`0x40`).
- [x] **T19**: Update `BlankAdtFactory` and terrain assembly in `Pm4RestoreUnauthoredSupport.cs` to inject synthesized `MCCV` colors and clean neutral diffuse layers onto any tile lacking texture layers.
- [x] **T20**: Implement multi-source placement convergence in `Pm4RestoreUnauthoredSupport.cs`:
  - Retain all authentic placements from `WoWMuseum/335-dev/` and `original_development` `_obj0.adt`.
  - Refine transforms and magnetic 2D PCA Yaw rotation for existing placements when covered by PM4 collision.
  - Append only net-new unauthored PM4 objects, strictly bounded by tile limits (`IsWithinPlacementTileBounds`).
- [x] **T21**: Purge stray `_obj0.adt` and `_tex0.adt` companion files from `output/restored_development/World/Maps/development/` to prevent split-file loader desynchronization.
- [x] **T22**: Verify in `WoWViewer` and run test suite (`dotnet test`): confirm 0 missing/failed tiles, full minimap-baked terrain coloration with zero black tiles, and deduplicated object placements.
