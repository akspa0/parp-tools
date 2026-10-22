# Plan: Spec 259 — Ground Effects & WMO Detail Doodad Engine

**2026-10-22 precedence:** original architecture below is historical intent, not proof of current
GPU instancing or a directive to replay completed work. The active bounded G roadmap below
governs distance/density/opacity refresh; AC-009 remains reverted/open.

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

## Active refresh — 2026-10-22

**Branch/context:** 286-view-controls-userguide; this spec remains flora owner.
**Operator intent:** turn bounded renderer research into a phased plan, distinguish opacity,
density and culling, and establish evidence before claiming a dense-distant-effects fix.
No reader change, global material adjustment, instancing overhaul or tint reapplication is implied.

### Technical Context / adopted design artifacts

Existing C#/.NET 10 placement/runtime services and viewer OpenGL model routes; xUnit tests;
existing quality settings persist density/distance. No new dependency or storage. Adopt the shared
[research](../249-epic-renderer-performance-and-correctness/research.md),
[evidence model](../249-epic-renderer-performance-and-correctness/data-model.md),
[contracts](../249-epic-renderer-performance-and-correctness/contracts/render-fidelity.md) and
[quickstart](../249-epic-renderer-performance-and-correctness/quickstart.md) by reference rather
than duplicate diagnostic/receipt schemas. UI owns disclosure, not placement/telemetry collection.

### Observed seams (not automatic write permission)

| Seam | Current behavior / bounded candidate |
|---|---|
| [Placement generator](../../src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs:112) | Density creates candidates at generation time; shared pure policy/test seam |
| [Scene-service generation](../../src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs:352) | WMO material path has a separate density/triangle-step rule; keep separate from MDDL placements |
| [Render admission](../../src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs:467) | 3D radius rejection; fade over last 25 units; no distance-density thinning |
| [Model submission](../../src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs:539) | Per-instance opaque submission with fade; grouping is not proof of GPU instancing |
| [Compatibility material](../../src/viewer/WoWViewer/Rendering/ModelRenderer.cs:1446) | Existing 0.50 cutout/0.05 faded branch; Spec 276 work already present |
| [Quality controls](../../src/viewer/WoWViewer/Workbench/Services/RenderQuality/RenderQualityService.cs:242) | Actual 0.1–3× density / 30–300 yd differs from original spec ranges; decide explicitly before changing |
| [Existing caller](../../src/viewer/WoWViewer/ViewerApp.cs:1118) | Fixed light/fog inputs; observe mismatch, do not grow app or silently bundle lighting correction |

### Constitution Check — before and after design

PASS in design: explicit operator planning scope, shared policy/library ownership, no reader or
legacy reference writes, no new app/scene members, no oversized renderer growth, <=10 steps per
phase, operator-owned real proof and repository-contained receipts. No exceptions. Native versus
compatibility appearance and true complaint cause are evidence gates, not assumed findings.

### G0 — reproducible distance/density/fade baseline
1. Consume Epic 249 Gate E identity/reference card. Pick a real terrain sample and WMO detail sample
   where available, keeping explicit camera/settings, build and model identity.
2. Trace placement counts, submitted counts, 3D distance, radius, fade value, material route and
   caller lighting. Compare near, fade-band and beyond-radius stations; one setting changes at a time.
3. **Gate G:** separate source-confirmed facts from overlap/route/density/lighting hypotheses;
   operator reproduction proves what remains wrong. Do not lower global density/opacity by guess.

### G1 — one evidence-backed admission/fade correction
1. Only after Gate G diagnosis, extract/test a small pure admission/fade policy and implement the
   approved correction in the scene service/adapter, preserving deterministic placements and defaults.
2. If the defect is inside shared model materials, isolate an opt-in detail-doodad path and scope
   it explicitly; obey existing oversized-file split follow-up before renderer growth. If evidence
   needs a different feature (distance thinning, instancing, lighting), stop for a bounded amendment.
3. **Gate GP:** focused placement/material-route tests, build, then operator same-input near/fade/far
   captures with unaffected models checked. Code/test receipt does not prove visual density parity.

### G2 — existing tint regression stays independent
1. Existing T014 first requires isolated reproduction/root-cause for the reverted viewer-load
   regression. No restored renderer API based solely on the fade investigation.
2. A separately approved safe implementation addresses neutral/IgnoreMCCV/MCCV/MCSH behavior
   while preserving ambient/directional lighting and loading in both affected model routes.
3. **Gate GT:** existing T015 operator real-terrain tint/shadow smoke; AC-009 remains unchecked
   until actual output. Existing T016 oversized-renderer follow-up is not silently waived.

### Phase 2 handoff

Generate new unchecked G0/G1 tasks after this planning pass; retain T014–T016 meanings/status.
Do not replay original reader expansion or Spec 276 cutout/fade work. Address observed range/unit
drift only after an explicit decision. Every implementation phase stops at its own evidence gate.
