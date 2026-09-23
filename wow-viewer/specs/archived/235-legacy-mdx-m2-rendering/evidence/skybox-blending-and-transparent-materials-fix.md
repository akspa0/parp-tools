# Evidence Receipt: Skybox Blending & Transparent Cutout Materials Corrections

**Date**: 2026-09-12  
**Feature Branch**: `235-legacy-mdx-m2-rendering`  
**Owner Spec**: [spec.md](../spec.md)  
**Governance Standard**: `AGENTS.md` §9.2 (Receipts required)

---

## 1. Problem Statements & Root Causes

### 1.1 Skybox Rendered as Low-Res "N64 Polygonal Blobs"
- **Symptom**: Sky domes, cloud layers, star fields, and horizon glows rendered as solid, hard-edged polygonal discs and bands across the sky rather than smooth celestial gradients.
- **Root Cause (`M2Renderer.cs`)**:
  ```csharp
  // Former line 731:
  if (!backdrop && transparent)
  {
      _gl.Enable(EnableCap.Blend);
      ConfigureBlendMode(section.Material.BlendMode);
      _gl.DepthMask(false);
  }
  else
  {
      _gl.Disable(EnableCap.Blend);
      _gl.DepthMask(!backdrop);
  }
  ```
  When rendering backdrops (`backdrop: true`), `!backdrop && transparent` was ALWAYS `false`. The `else` branch executed for every section of the skybox, unconditionally calling `_gl.Disable(EnableCap.Blend)`. Every blended atmospheric gradient, cloud layer, star field, and glow in the skybox was forced into opaque rendering without alpha or additive blending.
- **Remediation**: Changed condition to `if (transparent)` so blending is enabled for all transparent sections regardless of `backdrop`.

### 1.2 Foliage / Plant Leaves & Spiderwebs Rendered as Solid Opaque Polygons
- **Symptom**: Plant leaves (e.g. `zangarplantgroup05.m2`) and spiderwebs rendered as solid opaque geometric planes instead of alpha-tested cutouts.
- **Root Cause 1 (`M2Era100ModelReader.cs`)**:
  In 1.0.0 (build 3980), embedded division section records were 32 bytes (0x20). In 0x104/0x106 models, section records include `sortCenterPosition` (12 bytes) and `sortRadius` (4 bytes), making the stride 48 bytes (0x30). The hardcoded 32-byte stride caused Section[1] to read from the middle of Section[0]'s bounding sphere fields, corrupting `vertexStart`, `vertexCount`, and `indexCount`.
- **Root Cause 2 (`WarcraftNetM2Adapter.cs`)**:
  `DiscoverProfiledRenderFlags` scanned header offsets 0x48 to 0xA0. At offset 0x48 in `zangarplantgroup05.m2`, it read a table with 1,408 entries where the high word was 0. Because `EvaluateRenderFlagQuality` added `+ renderFlags.Count`, this garbage array scored 5,552 points, while the genuine 3-element `RenderFlags` array scored 27 points. `ShouldPreferProfiledRenderFlags` overwrote genuine render flags with 0s (`blendMode = 0 (Opaque)`), disabling `section.AlphaCutout`.
- **Root Cause 3 (`M2StaticRenderModelBuilder.cs`)**:
  `bool isProjected = batch is not null && (((batch.Flags & 0x4) != 0) || ((batch.GeosetIndex & 0x2) != 0));`
  Testing bit `0x2` of `GeosetIndex` erroneously flagged any batch with geoset index 2, 3, 6, 7... as projected diffuse.
- **Root Cause 4 (`WowViewerM2RuntimeBridge.cs`)**:
  Collapsed all non-opaque blend modes onto `Mod` instead of mapping `M2BlendMode.AlphaKey` to `M2CombinerEffectFamily.AlphaKey`.

---

## 2. Files Changed

| Component | File Path | Description of Change |
|---|---|---|
| Native M2 Renderer | [M2Renderer.cs](../../../../src/viewer/WoWViewer/Rendering/M2Renderer.cs) | Changed `if (!backdrop && transparent)` to `if (transparent)` so blending is enabled for transparent sections during backdrop / skybox rendering. |
| Core M2 Era 100 Constants | [M2Era100Constants.cs](../../../../src/core/WowViewer.Core.IO/M2Era100/M2Era100Constants.cs) | Added `SectionStrideEra104 = 0x30` (48 bytes) for 2.x/0x104+ section records containing sort center and radius. |
| Core M2 Era 100 Reader | [M2Era100ModelReader.cs](../../../../src/core/WowViewer.Core.IO/M2Era100/M2Era100ModelReader.cs) | Added dynamic calculation of `sectionStride` in `ReadGeometry` using `(batchesOfs - sectionsOfs) / sectionsCount` (supporting 32 B and 48 B strides). |
| Static Model Builder | [M2StaticRenderModelBuilder.cs](../../../../src/core/WowViewer.Core.Runtime/M2/M2StaticRenderModelBuilder.cs) | Removed erroneous `((batch.GeosetIndex & 0x2) != 0)` check from `isProjected`. |
| M2 Runtime Bridge | [WowViewerM2RuntimeBridge.cs](../../../../src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs) | Mapped `M2BlendMode` to `M2CombinerEffectFamily` using full switch expression preserving `AlphaKey`, `Decal`, `Add`, and `Fade`. |
| M2 Adapter | [WarcraftNetM2Adapter.cs](../../../../src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs) | Prevented speculative header scans in `ShouldPreferProfiledRenderFlags` from overwriting valid render flags when `current` already has valid blend modes (<= 6); capped candidate count to 128 and score contribution to 32. |
| Unit Tests | [M2Era100ModelReaderTests.cs](../../../../tests/WowViewer.Core.Tests/M2Era100ModelReaderTests.cs) | Added permanent regression test `ReadDetailed_ZangarPlantGroup05_ParsesMaterialsAndEmbeddedSectionsCorrectly` asserting blend modes, 48-byte section stride, vertex counts (650), and batch-material associations. |

---

## 3. Verification Commands & Outputs

### 3.1 Regression Unit Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Era100ModelReaderTests"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:    14, Skipped:     0, Total:    14, Duration: 391 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 3.2 Runtime Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Runtime"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:    26, Skipped:     0, Total:    26, Duration: 152 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 3.3 Solution Build
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: `0`
- **Output**:
```text
    330 Warning(s)
    0 Error(s)
Time Elapsed 00:00:15.80
```

---

## 4. Acceptance Criteria & Evidence Matrix

| Criterion | Requirement | Evidence |
|---|---|---|
| AC-1 | Skybox transparent passes (clouds, sun, celestial domes, stars) must blend properly | `M2Renderer.cs`: `if (transparent)` enables `GL.Enable(Blend)` and `ConfigureBlendMode` when `backdrop == true`. |
| AC-2 | Plant foliage and spiderwebs with `BlendMode == AlphaKey` must enable alpha cutout | `zangarplantgroup05.m2`: `Material[0].BlendMode == 1 (AlphaKey)` preserved; `M2Renderer.cs` sets `section.AlphaCutout = true` and shader discards fragments with `a < 0.5`. |
| AC-3 | Multi-section models with 48-byte embedded division section records must parse without corruption | `ReadGeometry`: `sectionStride` computes 48 B; `zangarplantgroup05.m2` parses Section[0] (233 verts) and Section[1] (417 verts) totaling 650 verts with 0 index skips. |
| AC-4 | Code complies with `AGENTS.md` God-Class Freeze (§10) | Zero new members added to `WorldScene` or `ViewerApp`. |
