# Evidence Receipt: Bone Quaternion Normalization & AlphaKey Cutout Material Transfer Corrections

**Date**: 2026-09-13  
**Feature Branch**: `235-legacy-mdx-m2-rendering`  
**Owner Spec**: [spec.md](../spec.md)  
**Governance Standard**: `AGENTS.md` §9.2 (Receipts required)

---

## 1. Problem Statements & Root Causes

### 1.1 Defect A: Character Models Crumple into Mangled Limbs When Animated
- **Symptom**: 0x100 character models (e.g. `TrollFemale.m2`) collapsed into a ball of mangled limbs upon playback or loading in the viewer / editor.
- **Root Cause (`M2Era100ModelReader.cs`)**:
  In classic 0x100 M2 models, bone rotation tracks are stored on disk as **16-byte uncompressed IEEE 754 float quaternions (`C4Quaternion`: 4 floats: X, Y, Z, W)**, rather than 8-byte `M2CompQuaternion` (4 int16 values). In `TrollFemale.m2`, all 61 non-empty rotation tracks contain 16-byte float quaternions.
  When read with a fixed 8-byte compressed quaternion stride, identity rotations `(0, 0, 0, 1.0f)` were read as `(0, 0, 0, 0)` in the lower 8 bytes, which decoded to `(-0.5, -0.5, -0.5, -0.5)`. This twisted the root bone by 120° and warped child bones down the hierarchy, collapsing the skeleton into a crumpled ball.
- **Remediation**:
  1. Added `IsUncompressedQuaternionTrack(byte[] data, int offset)` to detect normalized 4-float quaternions ($X^2 + Y^2 + Z^2 + W^2 \approx 1.0$).
  2. In `ReadBones`, dynamically determine if each bone rotation track is uncompressed (stride 16) or compressed (stride 8).
  3. In `M2Era100PayloadAppender.NormalizeQuaternionTrack`, convert 16-byte float quaternions into canonical 8-byte `M2CompQuaternion` structures in `_extension` so standard runtime sampling (`M2TrackSampler.SampleCompressedQuaternion`) works uniformly without special casing across the entire engine.

### 1.2 Defect B: Foliage and Canopies Rendered with Solid Black / White Margins
- **Symptom**: 0x100 foliage and canopy doodads (`terokkartreelarge.mdx`, `razorfen_canopy01_hole.mdx`) rendered with solid black borders or opaque white planes around leaf textures instead of alpha-tested transparency cutout.
- **Root Cause 1 (`WarcraftNetM2Adapter.cs`)**:
  In `ParseEra100Model`, `data.RenderFlags` had a placeholder loop:
  `for (int i = 0; i < geometry.Batches.Count; i++) data.RenderFlags.Add(new ParsedRenderFlagData { Flags = 0, BlendingMode = 0 });`
  This hardcoded all 0x100 models to `BlendingMode = 0` (`Opaque`). Models like `terokkartreelarge` (`Material[1]` and `[2]` have `BlendMode = 1` `AlphaKey`) and `razorfen_canopy01_hole` (`Material[0]` has `BlendMode = 1` `AlphaKey`) had their blend modes discarded.
- **Root Cause 2 (`M2Renderer.cs`)**:
  When initializing `SectionBuffers` in `InitBuffers()`, `buffers.AlphaCutout` was not set from the section material blend mode:
  `buffers.AlphaCutout = section.Material.BlendMode == WowViewer.Core.M2.M2BlendMode.AlphaKey;`
- **Remediation**:
  1. Populated `data.RenderFlags` from `geometry.Materials[i].Flags` and `geometry.Materials[i].BlendMode` in `ParseEra100Model`.
  2. Initialized `buffers.AlphaCutout` in `M2Renderer.cs` based on `section.Material.BlendMode == M2BlendMode.AlphaKey`.

---

## 2. Files Changed

| Component | File Path | Description of Change |
|---|---|---|
| Core M2 Era 100 Reader | [M2Era100ModelReader.cs](../../../src/core/WowViewer.Core.IO/M2Era100/M2Era100ModelReader.cs) | Added `IsUncompressedQuaternionTrack`, dynamic rotation track stride (16 B vs 8 B), and `NormalizeQuaternionTrack` float-to-`M2CompQuaternion` conversion. |
| M2 Adapter | [WarcraftNetM2Adapter.cs](../../../src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs) | Corrected `ParseEra100Model` to populate `data.RenderFlags` directly from `geometry.Materials` flags and blend modes. |
| Native M2 Renderer | [M2Renderer.cs](../../../src/viewer/WoWViewer/Rendering/M2Renderer.cs) | Set `buffers.AlphaCutout = section.Material.BlendMode == WowViewer.Core.M2.M2BlendMode.AlphaKey;` in `InitBuffers()`. |
| Unit Tests | [M2Era100ModelReaderTests.cs](../../../tests/WowViewer.Core.Tests/M2Era100ModelReaderTests.cs) | Added synthetic tests `Era100_Synthetic_UncompressedQuaternion_NormalizedAndSampled` and `Era100_Synthetic_MaterialsWithAlphaKey_ParsedCorrectly`, plus real TrollFemale sampling test. |
| Embedded Profile Tests | [M2EmbeddedProfileRealDataTests.cs](../../../tests/WowViewer.Core.Tests/M2EmbeddedProfileRealDataTests.cs) | Added unit test `BuildEmbeddedStaticRenderModel_SyntheticEra100_TransfersAlphaKeyBlendMode` verifying `AlphaKey` material transfer. |

---

## 3. Verification Commands & Outputs

### 3.1 Era 100 Reader Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Era100ModelReaderTests"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:    17, Skipped:     0, Total:    17, Duration: 447 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 3.2 Embedded Profile & AlphaKey Adapter Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2EmbeddedProfileRealDataTests"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:     3, Skipped:     0, Total:     3, Duration: 54 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 3.3 Full Solution Compilation
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: `0`
- **Output**:
```text
    42 Warning(s)
    0 Error(s)
Time Elapsed 00:00:08.89
```

---

## 4. Acceptance Criteria & Evidence Matrix

| Criterion | Requirement | Evidence |
|---|---|---|
| AC-1 | 0x100 models with uncompressed float quaternion bone rotation tracks must sample identity and valid rotations without hierarchical distortion | `Era100_Synthetic_UncompressedQuaternion_NormalizedAndSampled`: synthesized 16-byte float quaternion samples as $(0.7071, 0, 0, 0.7071)$ with lenSq $1.0000$. `Era100_TrollFemale_QuaternionNormalizationAndSampling`: Root bone 0 samples as exact identity $(0, 0, 0, 1)$ and bone 66 lenSq is $1.0000$. |
| AC-2 | 0x100 foliage/canopy models with `AlphaKey` materials must transfer blend mode to render model and enable cutout | `BuildEmbeddedStaticRenderModel_SyntheticEra100_TransfersAlphaKeyBlendMode`: returns `Section[0].Material.BlendMode == M2BlendMode.AlphaKey`. `M2Renderer.cs`: sets `buffers.AlphaCutout = true`. |
| AC-3 | Code complies with `AGENTS.md` God-Class Freeze (§10) | Zero new members added to `WorldScene` or `ViewerApp`. |
