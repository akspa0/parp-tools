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

## Investigation continuation — 2026-10-20

Operator request and limits: see the dated note in `spec.md` and investigation steps in `plan.md`.
Historical implementation checkmarks above are not new visual or surface-correlation proof.

- [x] **I01**: Audit current height-byte routing and actual surface-correlation test coverage.
  Receipt: [investigation-2026-10-20.md](evidence/investigation-2026-10-20.md).
- [x] **I02**: Probe at most five paired real tiles with input hashes, exact-Z/footprint evidence,
  current resolver routing, unkeyed-component sparsity/ambiguity and proximity controls, and explicitly
  synthetic matcher controls; do not modify source data or production WMO behavior.
  Receipt: [investigation-2026-10-20.md](evidence/investigation-2026-10-20.md), raw five-pair report linked there.
- [x] **I03**: Run focused tests and record commands, exit statuses, findings and proof gaps in a receipt.
  Receipt: [investigation-2026-10-20.md](evidence/investigation-2026-10-20.md), four tests pass; no real asset-match proof.
- [x] **I04**: Synchronize the investigation result and next bounded step in the memory bank.
  Receipt: [investigation-2026-10-20.md](evidence/investigation-2026-10-20.md), dated continuity updates
  and exact-content archive check; production behavior and old completion claims are not re-certified.

**Operator correction after this diagnostic:** I01–I04 do not certify map-wide interpretation or
matching accuracy. All PM4 files must be considered as one map object with retained chunk/link
context. The working WMO baseline (~70% accuracy, operator-reported) is preserved. Next continuation
must establish that map-wide context before isolated M2 fragment comparisons; vital semantics may
remain undecoded in chunks whose bytes are already parsed.

## Whole-map zero-bucket instrument — 2026-10-20

Operator scope and validation limits: dated `spec.md` amendment and `plan.md` §5. One instrument
phase; broad corpus execution is operator-owned, not a prerequisite fabricated as agent proof.

- [x] **J01**: Record code-level context loss and competing MSUR/MSLK interpretations.
- [x] **J02**: Add provenance-preserving map-level surface/link indexing and partition hypotheses.
- [x] **J03**: Add scale-normalized candidate-family diagnostics retaining absolute scale/location.
- [x] **J04**: Provide a whole-map runner retaining all decoded chunks and source fingerprints.
- [x] **J05**: Validate synthetic multi-file/scale/link/invalid-record controls and build the runner.
- [ ] **J06**: Operator runs the complete map audit; assess outputs before naming object identities.
- [x] **J07**: Record instrument receipt, exact operator command and current proof gap in continuity.

Receipt for J01–J05/J07: [map-context-instrument-2026-10-20.md](evidence/map-context-instrument-2026-10-20.md).
These are instrument/code/control deliveries, not proof of whole-map runtime or correct partitions;
J06 remains the real-data gate.

## Operator-authored object pools — after J05

- [x] **K01**: Define portable named candidate pools and explicit map/tile-range assignments with notes.
- [x] **K02**: Add inspectable opt-in candidate filtering; preserve unannotated matching and raw scores.
- [x] **K03**: Provide file-backed assignment utility and filtered surface-database output for tooling.
- [x] **K04**: Validate persistence, eligibility/isolation/failure controls and record receipt/handoff.
- [ ] **K05**: Viewer authoring UI follow-on: owned service, shared widgets, authoritative home/inventory;
  not implemented by the file-backed tooling phase.
- [x] **K06**: Import WL-family filename hints with configurable abbreviation aliases, retained unknowns
  and source provenance; no automatic unproved tile assignment or hard filter.

Receipt for K01–K04/K06: [map-context-instrument-2026-10-20.md](evidence/map-context-instrument-2026-10-20.md).
CLI persistence/filter smoke succeeded; no actual development tile allocation was fabricated.

## Structural foundations — 2026-10-20

- [x] **L01**: Inventory every PM4 input and review detector controls before semantic conclusions.
- [x] **L02**: Re-measure MSUR/MSLK window/reference structure with controls over the entire map.
- [x] **L03**: Cross-check candidate relationships against geometry/independent authored context.
- [x] **L04**: Publish measured/refuted/unknown ledger and next experiment with exact run receipts.

Receipt L01–L04: [structural-foundations-2026-10-20.md](evidence/structural-foundations-2026-10-20.md).
All 616 files measured; addressed geometry/back-link evidence is real data. No new authored-asset
identity proof or complete format decode is claimed; editor route-node/source-object layers remain open.
