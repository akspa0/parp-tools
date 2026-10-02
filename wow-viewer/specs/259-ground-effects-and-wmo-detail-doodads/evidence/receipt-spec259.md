# Receipt: Spec 259 — Ground Effects & WMO Detail Doodad Engine

**Date**: 2026-10-02  
**Epic**: Epic 251 / Spec 259  
**Branch**: `v0.6.0-dev`  
**Status**: COMPLETE (Gate A, Gate B, Gate C verified)

---

## 1. Summary of Deliverables

Spec 259 implements the complete Ground Effects and WMO Detail Doodad engine across `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, and the `WoWViewer` application. It conforms directly to the wowdev.wiki disclosures (September 28–30, 2026) by Skarn and Morfium, as well as the 3.3.5 / 1.60.1 client binary disassembly.

### Phase 1: DBC/DB2 Format Expansion & WMO MDDL Reader (Gate A)
- **`WmoChunkIds.cs`**: Registered chunk FourCCs `Mddl` (`MDDL`), `Moc2` (`MOC2`), and `Mgi2` (`MGI2`).
- **`WmoDetailDoodadModels.cs`**: Authored models for entries, layers, decoded group commands, and root MDDL documents.
- **`WmoMddlReader.cs`**: Implemented reader for root `MDDL` chunk and group RLE stream decoding.
- **`WmoRenderDocument.cs` & `WmoRenderDocumentReader.cs`**: Exposed optional `DetailDoodads` on parsed WMOs.
- **`GroundEffectLookup.cs`**: Enhanced to parse `GroundEffectDoodadRecord` (with flags `0x1` AlignToNormal, `0x2` IgnoreMCCV, animScale, pushScale, FileDataId) and `GroundEffectTextureRecord`. Added `Load(Func<string, byte[]?> fileReader)` for seamless cross-source loading.

### Phase 2: Placement Generator & Client Rules Conformance (Gate B)
- **`GroundEffectPlacementModels.cs`**: Created `DetailDoodadInstance`, `TerrainChunkPlacementInput`, `TerrainChunkLayerInput`, and `WmoGroupPlacementInput`.
- **`GroundEffectPlacementGenerator.cs`**:
  - Enforces slope limit: normal $Z \ge 0.4$ ($\approx 66.42^\circ$). Steeper slopes are rejected.
  - Normal alignment math: Flag `0x1` (`AlignToNormal`) aligns model up-vector $(0, 0, 1)$ with the terrain normal via quaternion, preserving random yaw. Unset models remain upright.
  - Flag `0x2` (`IgnoreMCCV`): emits pure white (`0xFFFFFFFF`) when set; interpolates MCCV vertex colors when unset.
  - MCSH shadow attenuation: scales RGB by $70\%$ ($0.7f$) when located in shadowed cells.
  - Density & alpha map sampling: weighted candidate selection.
- **`WmoDetailDoodadDecoder.cs`**: Decodes parsed `MDDL` group commands onto WMO mesh geometry with normal alignment and slope culling.
- **`M2SceneSubmissionCoordinator.cs` & `M2SoftwareVisualSnapshot.cs`**: Registered `M2RenderEntryFamily.DetailDoodad = 7` with dedicated instanced batching policy (`detail-doodad-batch`).

### Phase 3: Viewer Integration & UI Controls (Gate C)
- **`GroundEffectSceneService.cs` & `GroundEffectSceneService.Host.cs`**: Created owned workbench service under `Workbench/Services/GroundEffects/`.
  - Subscribes to `_terrainManager.OnTileLoaded` and `OnTileUnloaded`.
  - Caches evaluated doodads per tile and supports distance culling (`GroundEffectDistance`).
  - Renders instanced M2 model batches with distance fade.
- **`IViewerAppHost.cs` & `ViewerApp_Host.cs`**: Exposed `GroundEffectSceneService GroundEffects { get; }`.
- **`ViewerApp.cs`**: Wire `_groundEffects` in fields, constructor, and render loop without adding members to god-classes (AGENTS.md §10).
- **`RenderQualityService.cs`**: Added Ground Effects UI section with toggle, density multiplier slider ($0.1\times - 3.0\times$), distance slider ($30 - 300$ yd), and active doodad telemetry.
- **`ViewerSettingsService.cs`**: Persisted `EnableGroundEffects`, `GroundEffectDensity`, and `GroundEffectDistance` in `viewer_settings.json`.

---

## 2. Verification Receipts

### Command Runs & Exit Status
| Step | Command | Exit Code | Result |
|:---|:---|:---:|:---|
| Build | `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` | 0 | PASS (0 errors across all projects) |
| Targeted Tests | `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DetailDoodad"` | 0 | PASS (6/6 tests passed) |

### Test Matrix
1. `GroundEffectLookup_ParsesFlagsDensityAndModelsCorrectly`: PASS
2. `WmoMddlReader_ParsesSyntheticChunkAndDecodesGroupsCorrectly`: PASS
3. `ComputeNormalAlignment_AlignsUnitZToTargetNormal`: PASS
4. `GenerateChunkDoodads_EnforcesSlopeCulling`: PASS
5. `GenerateChunkDoodads_RespectsFlagsAndMccvAndShadow`: PASS
6. `WmoDetailDoodadDecoder_DecodesGroupPlacementsCorrectly`: PASS

---

## 3. Files Created & Modified

### Created
- `src/core/WowViewer.Core/Wmo/WmoDetailDoodadModels.cs`
- `src/core/WowViewer.Core.IO/Wmo/WmoMddlReader.cs`
- `src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementModels.cs`
- `src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs`
- `src/core/WowViewer.Core.Runtime/DetailDoodads/WmoDetailDoodadDecoder.cs`
- `src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs`
- `src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.Host.cs`
- `tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectDetailDoodadTests.cs`
- `tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectPlacementTests.cs`
- `specs/259-ground-effects-and-wmo-detail-doodads/spec.md`
- `specs/259-ground-effects-and-wmo-detail-doodads/plan.md`
- `specs/259-ground-effects-and-wmo-detail-doodads/tasks.md`
- `specs/259-ground-effects-and-wmo-detail-doodads/evidence/receipt-phase1.md`
- `specs/259-ground-effects-and-wmo-detail-doodads/evidence/receipt-spec259.md`

### Modified
- `src/core/WowViewer.Core/Wmo/WmoChunkIds.cs`
- `src/core/WowViewer.Core/Wmo/WmoRenderDocument.cs`
- `src/core/WowViewer.Core.IO/Wmo/WmoRenderDocumentReader.cs`
- `src/core/WowViewer.Core.IO/Dbc/GroundEffectLookup.cs`
- `src/core/WowViewer.Core.Runtime/M2/M2SceneSubmissionCoordinator.cs`
- `src/core/WowViewer.Core.Runtime/M2/M2SoftwareVisualSnapshot.cs`
- `src/viewer/WoWViewer/Terrain/TerrainManager.cs`
- `src/viewer/WoWViewer/ViewerApp.cs`
- `src/viewer/WoWViewer/ViewerApp_Host.cs`
- `src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`
- `src/viewer/WoWViewer/Workbench/Services/RenderQuality/RenderQualityService.cs`
- `src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs`
- `specs/STATUS.md`
- `memory-bank/activeContext.md`
