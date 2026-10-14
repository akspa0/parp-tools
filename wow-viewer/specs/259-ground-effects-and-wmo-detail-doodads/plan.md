# Plan: Spec 259 — Ground Effects & WMO Detail Doodad Engine

## Architectural Overview

The Ground Effect & Detail Doodad system operates in three decoupled layers:

```
[DBC / DB2 & WMO Parsers]
 (GroundEffectLookup, WmoMddlReader, WmoMoc2Reader)
         │
         ▼
[Runtime Placement Generator]
 (GroundEffectPlacementGenerator, WmoDetailDoodadDecoder)
  - Slope limit test (Z_norm >= 0.4)
  - Normal alignment vs upright rotation (Flag 0x1)
  - MCCV color interpolation & MCSH shadow test (Flag 0x2)
  - Density and distance culling
         │
         ▼
[GroundEffectSceneService & GPU Instancer]
  - Batched/Instanced M2 rendering
  - RenderQuality controls (density, dist, toggle)
```

---

## Detailed File Changes

### 1. `src/core/WowViewer.Core/`
- `Wmo/WmoChunkIds.cs`: Register `Mddl` (`MDDL`), `Moc2` (`MOC2`), and `Mgi2` (`MGI2`).
- `Wmo/WmoSummary.cs`: Add properties for MDDL layer count and detail doodad counts.

### 2. `src/core/WowViewer.Core.IO/`
- `Dbc/GroundEffectLookup.cs`:
  - Enhance with `GroundEffectDoodadRecord` (`Id`, `ModelPath`, `FileDataId`, `Flags`, `AnimScale`, `PushScale`).
  - Flag constants: `AlignToNormal = 0x1`, `IgnoreMCCV = 0x2`.
  - Enhance `GroundEffectTextureRecord` (`Id`, `DoodadIds`, `Density`).
  - Expose helper lookup methods for placement generators.
- `Wmo/WmoMddlReader.cs`:
  - Implements reader for root chunk `MDDL`.
  - Decodes `Layer detailDoodadLayers[]` and per-group RLE placement stream.
- `Wmo/WmoRenderDocument.cs` & `WmoRenderDocumentReader.cs`:
  - Include parsed detail doodad layers and placements.

### 3. `src/core/WowViewer.Core.Runtime/`
- `DetailDoodads/GroundEffectPlacementModels.cs`:
  - `DetailDoodadInstance` struct (`Vector3 Position`, `Quaternion Orientation`, `float Scale`, `uint ColorBgra`, `uint ModelIdOrFdid`, `string ModelPath`).
  - `GroundEffectChunkDefinition`: Data per terrain chunk or WMO group.
- `DetailDoodads/GroundEffectPlacementGenerator.cs`:
  - Evaluates terrain chunk triangles against `GroundEffectTexture` rules.
  - Implements slope culling ($Z_{\text{normal}} \ge 0.4$), normal orientation (`0x1`), MCCV interpolation (`0x2`), and shadow attenuation.
- `M2/M2SceneSubmissionCoordinator.cs`:
  - Register `M2RenderEntryFamily.DetailDoodad` with dedicated fast-path instancing.

### 4. `src/viewer/WoWViewer/`
- `Workbench/Services/GroundEffects/GroundEffectSceneService.cs`:
  - Manages active detail doodads across loaded ADT chunks and WMOs.
  - Updates when camera moves, culling doodads beyond `groundEffectDist`.
- `Workbench/Services/RenderQuality/`:
  - Add UI toggles and sliders for Ground Effects.
- `IViewerAppHost.cs`:
  - Expose `IGroundEffectSceneService GroundEffects { get; }`.

---

## Verification & Receipts Plan
- **Unit Tests**: Add tests under `WowViewer.Core.Tests/DetailDoodads/`:
  - Flag parsing tests (`0x1`, `0x2`).
  - Slope culling test ($Z < 0.4$ rejected, $Z \ge 0.4$ accepted).
  - Normal alignment quaternion generation test.
  - MDDL chunk round-trip and decoding tests.
- **Solution Build**: `dotnet build wow-viewer/WowViewer.slnx -c Debug` (0 errors).
- **Execution**: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug`.

## Operator amendment — lighting propagation (2026-10-14; deferred after regression)

Requested behavior: carry a per-instance RGB tint with each terrain detail doodad. The first attempt
computed the terrain-equivalent MCCV multiplier (`raw RGB * 2`, neutral missing-MCCV input 1.0),
preserved `IgnoreMCCV` as white, applied MCSH attenuation, and forwarded that tint through
`IModelRenderer` into MDX material color. The operator then reported that the viewer could not load
content. That implementation is reverted. Do not reapply the renderer change without an isolated
reproduction, root-cause diagnosis, and a safer opt-in implementation. AC-009 remains open.
