# Tasks 298 — PM4 Authentic Pairing Telemetry, Markdown Elimination of CK24 & Multi-Object GLB Export

Receipt: `evidence/receipt-spec298.md`

## Phase 1 — Authentic Pairing Telemetry & Serialization (FR-001)

- [ ] T001 [Core] Extend `TaughtPairing` record in `Pm4DoodadDuplicationService.cs` with authentic placed object telemetry (`PlacementPosition`, `PlacementRotation`, `PlacementScale`, `UniqueId`) and PM4 hull telemetry (`TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId`, `HullCentroid`, `HullBoundsMin`, `HullBoundsMax`).
- [ ] T002 [Core] Update `TeachPairing` to populate all telemetry fields from the live `ObjectInstance` and `Pm4OverlayObject`.
- [ ] T003 [Core] Verify `Pm4PairingProjectStore` round-trip serialization and manifest generation with the enriched telemetry.

## Phase 2 — Purge CK24 from All Markdown Outputs (FR-002)

- [ ] T004 [Batch] Update `FormatMarkdownSummary` in `Pm4ShapeMatchBatchService.cs` to remove the `CK24` column and display authentic object ID, world coordinates, dimensions, model path, and unique ID.
- [ ] T005 [LLM Bundle] Update `BuildPm4LlmBundleMarkdown` in `Pm4WorkbenchService.Collections.cs` to replace raw `CK24` lines with genuine object coordinates and part IDs.
- [ ] T006 [Overlay Report] Update `ExportPm4OverlayReport` in `Pm4WorkbenchService.SceneGraph.cs` to replace `CK24` column with `Placement Z (yd)`.

## Phase 3 — PM4 GLB Exporter Engine & UI Integration (FR-003, FR-004)

- [ ] T007 [Export Engine] Implement `Pm4GlbExporter.cs` in `WoWViewer.Export` supporting single-tile and multi-tile combined GLB export with individual named object nodes and color-coded PBR collision materials.
- [ ] T008 [Export Engine] Integrate terrain geometry with layer textures, minimap imagery, and vertex MCCV into the GLB export when terrain is present.
- [ ] T009 [UI] Add `[ Export Active Tile PM4 to GLB... ]` and `[ Export All Loaded PM4 to GLB... ]` buttons in `ExportWorkbenchService.cs` and `Pm4WorkbenchService.Exports.cs`.
- [ ] T010 [UI Registry] Register new export controls in `ui-controls.json` and regenerate `UI-REFERENCE.md`.

## Phase 4 — Verification & Receipts (FR-005, SC-001, SC-002, SC-003, SC-004)

- [ ] T011 [Tests] Add unit tests for `TaughtPairing` telemetry serialization and clean markdown output in `Pm4DoodadDuplicationTests.cs` and `Pm4ShapeMatchAnalysisTests.cs`.
- [ ] T012 [Verification] Run full solution build (`dotnet build WowViewer.slnx -c Debug`) and verify 0 errors.
- [ ] T013 [Verification] Run unit tests (`dotnet test`) and verify 100% pass.
- [ ] T014 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [ ] T015 [Receipt] Record verification commands, exit status, and criterion-to-evidence matrix in `evidence/receipt-spec298.md`.
