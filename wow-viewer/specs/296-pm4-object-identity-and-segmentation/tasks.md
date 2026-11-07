# Tasks 296 — PM4 Object Identity, Topological Segmentation & Conflict Decoupling

## Phase 1 — Data Models & Unique Object Identity (FR-001, FR-002)

- [ ] T001 [Core] Define `Pm4ObjectIdentity` in `Pm4ShapeMatchModels.cs` with `TileX`, `TileY`, `ComponentId`, `LinkGroupObjectId`, `PlacementZ`, `Centroid`, `BoundingSize`, and `SurfaceClass`.
- [ ] T002 [Core] Add `UniqueHullId` calculation helper `Pm4HullShapeInput.MakeUniqueHullId(int tileX, int tileY, int componentId)` and accessor on `Pm4HullShapeInput`.
- [ ] T003 [Core] Update `Pm4ShapeMatchProposal` constructor/record to carry `ulong UniqueHullId` without breaking existing `uint HullKey` consumers.

## Phase 2 — Analysis & Conflict Decoupling (FR-003, FR-005)

- [ ] T004 [Core] Update `Pm4ShapeMatchAnalysisService.Analyze` to derive `HullKey` uniquely from `(TileX, TileY, ObjectPartId)` so distinct components at the same $Z$ never share a key.
- [ ] T005 [Core] Update `Pm4ShapeMatchAnalysisService.FindHullConflicts` to preserve distinct component inputs and avoid self-conflict anomalies.

## Phase 3 — UI Disambiguation & Batch Auditing (FR-004, FR-005, FR-007)

- [ ] T006 [UI] Update `Pm4ShapeMatchBatchPanelService` conflict rendering to display `[<Tile>] Obj #<PartId> (Z=<z> yd)` instead of raw `CK24 0x... vs 0x...`.
- [ ] T007 [UI] Update `Pm4ShapeMatchBatchService` summary rows and JSON/Markdown export formats to include `ComponentId`, `LinkGroupObjectId`, and physical `PlacementZ (yd)`.

## Phase 4 — Unit Testing & Verification Gates (SC-001, SC-002, SC-003, SC-005)

- [ ] T008 [Tests] Create `Pm4ObjectIdentityTests.cs` verifying `Pm4ObjectIdentity` string formatting, unique hull ID generation, and conflict disambiguation for hulls at the same elevation.
- [ ] T009 [Verification] Run full solution build (`dotnet build WowViewer.slnx -c Debug`) and verify 0 errors.
- [ ] T010 [Verification] Run unit tests (`dotnet test`) and verify 100% pass.
- [ ] T011 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
