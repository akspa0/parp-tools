# Tasks: Spec 259 — Ground Effects & WMO Detail Doodad Engine

## Phase 1: DBC/DB2 Format Expansion & WMO MDDL Reader

- [x] T001 [US1] Extend `GroundEffectLookup.cs` with full record definitions (`GroundEffectDoodadRecord`, `GroundEffectTextureRecord`), parsing flags `0x1` (AlignToNormal) and `0x2` (IgnoreMCCV), density, and scale.
- [x] T002 [US2] Register `Mddl`, `Moc2`, and `Mgi2` in `WmoChunkIds.cs`.
- [x] T003 [US2] Implement `WmoMddlReader.cs` to parse root `MDDL` chunk (layers, doodad entries, group data).
- [x] T004 [US2] Update `WmoRenderDocument.cs` and `WmoRenderDocumentReader.cs` to expose `IReadOnlyList<WmoDetailDoodadLayer>` and placements.
- [x] T005 [P] Unit tests for `GroundEffectLookup` flags and `WmoMddlReader` synthetic payload decoding under `tests/WowViewer.Core.Tests/DetailDoodads/`.
- [x] Gate A: Verify `dotnet build` and unit tests pass with exit code 0.

---

## Phase 2: Placement Generator & Client Rules Conformance

- [x] T006 [US1] Implement `GroundEffectPlacementModels.cs` in `WowViewer.Core.Runtime/DetailDoodads/`.
- [x] T007 [US1] Implement `GroundEffectPlacementGenerator.cs` enforcing:
  - Slope limit test ($Z_{\text{normal}} \ge 0.4$, $\approx 66^\circ$).
  - Normal-aligned rotation (Flag `0x1`) vs upright world-$Z$ rotation.
  - MCCV color interpolation across terrain triangle vertices (Flag `0x2` unset) vs plain white (Flag `0x2` set).
  - MCSH shadow alpha attenuation.
- [x] T008 [US2] Implement `WmoDetailDoodadDecoder.cs` decoding group RLE data onto WMO batch triangles.
- [x] T009 [P] Unit tests for slope culling, orientation matrices, and color blending under `tests/WowViewer.Core.Tests/DetailDoodads/`.
- [x] Gate B: Verify all placement logic unit tests pass.

---

## Phase 3: Viewer Integration & UI Controls

- [x] T010 [US4] Implement `GroundEffectSceneService.cs` under `Workbench/Services/GroundEffects/` managing active detail doodad instances per loaded tile/WMO.
- [x] T011 [US4] Add Ground Effects toggle, density slider (`groundEffectDensity`), and distance slider (`groundEffectDist`) to `RenderQualityService.cs` and UI settings.
- [x] T012 [US3] Integrate CASC / DB2 model loading for modern 1.60.1 data and legacy MPQ data.
- [x] T013 [P] Build solution and run full test suite.
- [x] Gate C: Final receipt authored in `specs/259-ground-effects-and-wmo-detail-doodads/evidence/receipt-spec259.md`.
