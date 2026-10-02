# Receipt: Spec 259 Phase 1 (Gate A)

**Date**: 2026-10-02  
**Scope**: DBC/DB2 Format Expansion & WMO MDDL Reader  
**Status**: Gate A Complete (All unit tests passing, solution builds cleanly)  

---

## 1. Files Changed

### Added
- [`wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/spec.md`](file:///I:/parp/parp-tools/wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/spec.md)
- [`wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/plan.md`](file:///I:/parp/parp-tools/wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/plan.md)
- [`wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/tasks.md`](file:///I:/parp/parp-tools/wow-viewer/specs/259-ground-effects-and-wmo-detail-doodads/tasks.md)
- [`wow-viewer/src/core/WowViewer.Core/Wmo/WmoDetailDoodadModels.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Wmo/WmoDetailDoodadModels.cs)
- [`wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoMddlReader.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoMddlReader.cs)
- [`wow-viewer/tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectDetailDoodadTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectDetailDoodadTests.cs)

### Modified
- [`wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs): Added `Mddl`, `Moc2`, `Mgi2`.
- [`wow-viewer/src/core/WowViewer.Core/Wmo/WmoRenderDocument.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Wmo/WmoRenderDocument.cs): Added optional `DetailDoodads` property.
- [`wow-viewer/src/core/WowViewer.Core.IO/Dbc/GroundEffectLookup.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Dbc/GroundEffectLookup.cs): Full record definitions with flags (`0x1` AlignToNormal, `0x2` IgnoreMCCV), density, FileDataID, and helper methods.
- [`wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRenderDocumentReader.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRenderDocumentReader.cs): Integrated `WmoMddlReader.ReadDetailDoodads`.
- [`wow-viewer/specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md): Registered Spec 259.
- [`wow-viewer/memory-bank/activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md): Updated active lane for Spec 259.

---

## 2. Verification Commands & Exit Status

1. `dotnet build wow-viewer/src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj -c Debug`  
   **Exit Code**: 0 (Build succeeded)

2. `dotnet build wow-viewer/WowViewer.slnx -c Debug`  
   **Exit Code**: 0 (Build succeeded)

3. `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DetailDoodad"`  
   **Exit Code**: 0 (2 passed, 0 failed)

4. `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DbcLookupTests"`  
   **Exit Code**: 0 (3 passed, 0 failed)

---

## 3. Criterion $\rightarrow$ Evidence Mapping

| Spec Acceptance Criterion | Evidence / Real Output |
|---|---|
| **AC-001**: `GroundEffectLookup` parses both legacy DBC and modern DB2 for flags (`0x1` AlignToNormal, `0x2` IgnoreMCCV), density, models/FDIDs | Verified by `GroundEffectLookup_ParsesFlagsDensityAndModelsCorrectly` test: `d101.AlignToNormal == true`, `d102.IgnoreMCCV == true`, `d103.FileDataId == 123456`, `tex.Density == 16`. |
| **AC-006**: `WmoMddlReader` parses WMO root `MDDL` chunk (layers, doodad entries, and per-group RLE data) | Verified by `WmoMddlReader_ParsesLayersAndGroupDataRleCorrectly` test: parsed density, doodad counts, weights, rollAll (`0x8000`), loc ranges, and single-location bits with exact decoded structures. |
| **Backward Compatibility**: Existing callers of `GroundEffectLookup` and `WmoRenderDocument` continue working | `DbcLookupTests` (3/3 passing), `VlmDatasetExporter` compilation unaffected, default `DetailDoodads = null` preserves all existing constructors. |
