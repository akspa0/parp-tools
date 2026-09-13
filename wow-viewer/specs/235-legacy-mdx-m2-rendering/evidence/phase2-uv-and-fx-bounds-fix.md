# Phase 2 Evidence Receipt: FX Emitter Bounds & UV Clamping Corrections

**Date**: 2026-09-12  
**Feature Branch**: `235-legacy-mdx-m2-rendering`  
**Owner Spec**: [spec.md](../spec.md)  
**Governance Standard**: `AGENTS.md` §9.2 (Receipts required)

---

## 1. Problem Statement

1. **Giant Solid Opaque White Bounding-Box Cubes**:
   Emitter doodads (`stratholmefloatingembers.m2`, `stratholmefiresmokeemberm.m2`, `hellfire_fireparticle.m2`) appeared as massive solid opaque white monoliths (spanning up to 63×63 yards) occluding the scene.
   - **Root Cause**: These models have `rawVertexCount == 0` on disk by design. `WowViewerM2RuntimeBridge.BuildEra100StaticRenderModel` received `InlineEra100Geometry == null` and treated all 0-vertex models as decoding failures under FR-005, invoking `BuildBoundingBoxFallbackModel` which synthesized an 8-vertex, 12-triangle cube with `M2BlendMode.Opaque` and no texture.

2. **3D Mesh Texture Streaks / Clamping Defect**:
   Complex 3D objects (`ballistaruined.m2`, broken siege engines, beams, wheels, arrows) rendered as flat untextured surfaces or single-color streaks, while only flat planar components (shields) textured correctly.
   - **Root Cause**: In `M2Renderer.cs`, `clampS` and `clampT` were computed as:
     ```csharp
     bool clampS = (candidate.TextureFlags & 0x1u) == 0;
     bool clampT = (candidate.TextureFlags & 0x2u) == 0;
     ```
     Because of `== 0`, any texture with `TextureFlags == 0` (the default for M2 textures like `WOOD02.BLP` and `BATTLEGLADEARROW.BLP`) forced `TextureWrapMode.ClampToEdge` in OpenGL. When UV coordinates outside `[0.0, 1.0]` (e.g. `UV0.X = 8.741`) were clamped to edge, the edge border pixels smeared across the mesh into flat single-color streaks. Only planar submeshes with UVs strictly in `[0.0, 1.0]` (like shields) were unaffected. Furthermore, in `ModelRenderer.cs`, `NormalizeAdaptedM2TextureSampling` forcibly clamped all non-opaque textures to edge.

---

## 2. Files Changed

| Component | File Path | Description of Change |
|---|---|---|
| Viewer Runtime | [WowViewerM2RuntimeBridge.cs](../../../src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs) | Reads `rawVertexCount` from `M2Era100Constants.VertexCountOffset` (`0x44`). If `rawVertexCount == 0`, immediately returns an empty `M2StaticRenderModel` without fallback geometry, reserving FR-005 bounding box fallback strictly for non-zero vertex models where mesh decoding fails. |
| Native Renderer | [M2Renderer.cs](../../../src/viewer/WoWViewer/Rendering/M2Renderer.cs) | Inverted clamping condition from `== 0` to `!= 0`. Textures repeat by default (`TextureWrapMode.Repeat`); clamping (`TextureWrapMode.ClampToEdge`) is only applied when flag bit `0x1` (S) or `0x2` (T) is set. |
| Adapter Renderer | [ModelRenderer.cs](../../../src/viewer/WoWViewer/Rendering/ModelRenderer.cs) | Corrected `clampS`/`clampT` evaluation for M2 adapter models to check for `WrapWidth`/`WrapHeight` flag presence; disabled aggressive `ClampToEdge` override in `NormalizeAdaptedM2TextureSampling` which broke repeating alpha textures. |
| Core Tests | [M2Era100ModelReaderTests.cs](../../../tests/WowViewer.Core.Tests/M2Era100ModelReaderTests.cs) | Added unit test `Era100Reader_ZeroVertexModel_HasNullInlineGeometry` validating that 0-vertex models correctly report null inline geometry without error. |

---

## 3. Verification Commands & Outputs

### 3.1 Unit Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Era100ModelReaderTests"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:    13, Skipped:     0, Total:    13, Duration: 67 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 3.2 Solution Build
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: `0`
- **Output**:
```text
0 Error(s)
```

---

## 4. Acceptance Criteria & Evidence Matrix

| Criterion | Requirement | Evidence |
|---|---|---|
| AC-1 | Particle/emitter doodads with 0 vertices must not render giant solid white fallback cubes | `WowViewerM2RuntimeBridge.cs`: `rawVertexCount == 0` returns empty `M2StaticRenderModel` with 0 sections. |
| AC-2 | M2 models with `TextureFlags == 0` must repeat and not clamp | `M2Renderer.cs`: `(candidate.TextureFlags & 0x1u) != 0` evaluates to `false`, mapping to `TextureWrapMode.Repeat`. |
| AC-3 | Models with clamp flags (`0x1` / `0x2` / `0x3`, e.g. `hellfire_oldorcbanner01.m2`) clamp correctly | `(candidate.TextureFlags & 0x1u) != 0` evaluates to `true`, mapping to `TextureWrapMode.ClampToEdge`. |
| AC-4 | Code complies with `AGENTS.md` God-Class Freeze (§10) | No new members added to `WorldScene` or `ViewerApp`. |
