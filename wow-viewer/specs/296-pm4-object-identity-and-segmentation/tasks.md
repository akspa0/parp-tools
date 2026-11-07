# Tasks 296 — PM4 Object Identity, Topological Segmentation & Conflict Decoupling

Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md)

## Phase 1 — Data Models & Unique Object Identity (FR-001, FR-002)

- [x] T001 [Core] Define `Pm4ObjectIdentity` in `Pm4ShapeMatchModels.cs` with `TileX`, `TileY`, `ComponentId`, `LinkGroupObjectId`, `PlacementZ`, `Centroid`, `BoundingSize`, and `SurfaceClass`. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T002 [Core] Add `UniqueHullId` calculation helper `Pm4HullShapeInput.MakeUniqueHullId(int tileX, int tileY, int componentId)` and accessor on `Pm4HullShapeInput`. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T003 [Core] Update `Pm4ShapeMatchProposal` constructor/record to carry `ulong UniqueHullId` without breaking existing `uint HullKey` consumers. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).

## Phase 2 — Analysis & Conflict Decoupling (FR-003, FR-005)

- [x] T004 [Core] Update `Pm4ShapeMatchAnalysisService.Analyze` to derive `HullKey` uniquely from `(TileX, TileY, ObjectPartId)` so distinct components at the same $Z$ never share a key. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T005 [Core] Update `Pm4ShapeMatchAnalysisService.FindHullConflicts` to preserve distinct component inputs and avoid self-conflict anomalies. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).

## Phase 3 — UI Disambiguation & Batch Auditing (FR-004, FR-005, FR-007)

- [x] T006 [UI] Update `Pm4ShapeMatchBatchPanelService` conflict rendering to display `[<Tile>] Obj #<PartId> (Z=<z> yd)` instead of raw `CK24 0x... vs 0x...`. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T007 [UI] Update `Pm4ShapeMatchBatchService` summary rows and JSON/Markdown export formats to include `ComponentId`, `LinkGroupObjectId`, and physical `PlacementZ (yd)`. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).

## Phase 4 — Unit Testing & Verification Gates (SC-001, SC-002, SC-003, SC-005)

- [x] T008 [Tests] Create `Pm4ObjectIdentityTests.cs` verifying `Pm4ObjectIdentity` string formatting, unique hull ID generation, and conflict disambiguation for hulls at the same elevation. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T009 [Verification] Run full solution build (`dotnet build WowViewer.slnx -c Debug`) and verify 0 errors. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T010 [Verification] Run unit tests (`dotnet test`) and verify 100% pass. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
- [x] T011 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings. Receipt: [`evidence/receipt-spec296.md`](evidence/receipt-spec296.md).
