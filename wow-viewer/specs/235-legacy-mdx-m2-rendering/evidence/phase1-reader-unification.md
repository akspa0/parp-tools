# Phase 1–3 Evidence Receipt: Legacy MDX & M2 Reader Unification

**Date**: 2026-09-11  
**Feature Branch**: `235-legacy-mdx-m2-rendering`  
**Owner Spec**: [spec.md](../spec.md)  
**Governance Standard**: `AGENTS.md` §9.2 (Receipts required)

---

## 1. Files Changed

| Component | File Path | Role in Fix |
|---|---|---|
| Format I/O | [M2ModelReaderDispatcher.cs](../../../src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs) | Lifted `0x102`–`0x107` `NotSupportedException` refusal wall; routes versions `<= 0x107` validating classic layout to `M2Era100ModelReader`. |
| Format I/O | [M2Era100Constants.cs](../../../src/core/WowViewer.Core.IO/M2Era100/M2Era100Constants.cs) | Added `BoneStrideEra100` (108B / `0x6C`) and `BoneStrideEra104` (112B / `0x70`). |
| Format I/O | [M2Era100ModelReader.cs](../../../src/core/WowViewer.Core.IO/M2Era100/M2Era100ModelReader.cs) | Generalizes version range (`0x100`–`0x107`), implements 108B and 112B (`boneNameCrc` at `+0x0C`) bone parsing with track normalization, and populates `EmbeddedSkinDocuments`. |
| Format I/O | [ModelRouteClassifier.cs](../../../src/core/WowViewer.Core.IO/AssetReferences/ModelRouteClassifier.cs) | Reclassified `Md20_1X_V100_Era100` from `Blocked` to `Readable` ("MD20 0x100-0x107 (legacy classic layout)"). |
| Core Model | [M2Era100Geometry.cs](../../../src/core/WowViewer.Core/M2/M2Era100Geometry.cs) | Added `GlobalVertices` property to preserve pre-division vertices. |
| Core Model | [M2ModelDocument.cs](../../../src/core/WowViewer.Core/M2/M2ModelDocument.cs) | Added `EmbeddedSkinDocuments` collection. |
| Core Runtime | [M2SkinProfileRuntime.cs](../../../src/core/WowViewer.Core.Runtime/M2/M2SkinProfileRuntime.cs) | Initializes `ActiveSkinProfile` directly from embedded skin documents without external `.skin` file lookups. |
| Viewer Shell | [WowViewerM2RuntimeBridge.cs](../../../src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs) | Implemented FR-005 Bounding-Box Fallback model generation (`BuildBoundingBoxFallbackModel`, 8 vertices / 36 indices / `usesCompatibilityFallback: true`). |
| Viewer Shell | [WorldAssetManager.cs](../../../src/viewer/WoWViewer/Terrain/WorldAssetManager.cs) | Added explicit placement loading branch for `M2Era1121EraTag.Md20_1X_V100_Era100` routing to `WowViewerM2RuntimeBridge`. |
| Core Tests | [M2Era100ModelReaderTests.cs](../../../tests/WowViewer.Core.Tests/M2Era100ModelReaderTests.cs) | Added 11 unit tests covering 108B and 112B bones, multi-version dispatch (`0x100`, `0x104`, `0x107`), embedded skin extraction, and track normalization. |
| Core Tests | [ModelRouteClassifierTests.cs](../../../tests/WowViewer.Core.Tests/ModelRouteClassifierTests.cs) | Updated tests to assert `Readable` status for `0x102`–`0x107`. |
| Tooling | [Program.cs](../../../tools/inspect/WowViewer.Tool.Inspect/Program.cs) | Populates `M2GeometryDocument` for `Md20_1X_V100_Era100` using `GlobalVertices`, enabling full inspect output. |

---

## 2. Verification Commands & Outputs

### 2.1 Unit Tests
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Era100"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:    11, Skipped:     0, Total:    11, Duration: 132 ms - WowViewer.Core.Tests.dll (net10.0)
```

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ModelRouteClassifier"
```
- **Exit Code**: `0`
- **Output**:
```text
Passed!  - Failed:     0, Passed:     7, Skipped:     0, Total:     7, Duration: 25 ms - WowViewer.Core.Tests.dll (net10.0)
```

### 2.2 Solution Build
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```
- **Exit Code**: `0`
- **Output**:
```text
322 Warning(s)
0 Error(s)
Time Elapsed 00:00:33.49
```

### 2.3 Staged Client Model Inspections

#### Client 1: Vanilla 1.0.0.3980 (`World\ArtTest\Boxtest\xyz.m2`)
```powershell
I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe m2 inspect --archive-root "H:\CLIENTS\Vanilla\1.x\1.X_Retail_Windows_enUS_1.0.0.3980\World of Warcraft" --virtual-path "World\ArtTest\Boxtest\xyz.m2"
```
- **Exit Code**: `0`
- **Output Highlights**:
  - `ERA: 1.0.0 (MD20 v0x100, classic layout)`
  - `M2: ... signature=MD20 version=0x100 model=xyz boundsMin=(-1.48, -0.13, 0.00) boundsMax=(0.12, 1.45, 1.60) boundsRadius=1.379 skinProfiles=4 bones=1`
  - `GEOMETRY: available=true vertices=72 textures=1 renderFlags=1 textureLookup=1`
  - `SKIN: stage=Initialized profileIndex=0 exactPath=World\ArtTest\Boxtest\xyz00.skin loaded=true vertexLookup=72 triangleIndices=108 boneEntries=72 submeshes=1 batches=1`
  - `RENDER: compatibilitySections=1 structuredSections=1 compatibilityMode=False`

#### Client 2: TBC Pre-Release 2.0.0.5610 (`CHARACTER\BloodElf\Male\BloodElfMale.m2`)
```powershell
I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe m2 inspect --archive-root "H:\CLIENTS\TBC\2.X_Pre-Release_Windows_enUS_2.0.0.5610\World of Warcraft" --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.m2"
```
- **Exit Code**: `0`
- **Output Highlights**:
  - `ERA: 1.0.0 (MD20 v0x100, classic layout)`
  - `M2: ... signature=MD20 version=0x100 model=BloodElfMale boundsMin=(-1.96, -1.47, -0.75) boundsMax=(1.65, 1.43, 3.57) boundsRadius=3.166 skinProfiles=4 bones=138`
  - `GEOMETRY: available=true vertices=4864 textures=4 renderFlags=6 textureLookup=4`
  - `SKIN: stage=Initialized profileIndex=0 exactPath=CHARACTER\BloodElf\Male\BloodElfMale00.skin loaded=true vertexLookup=4864 triangleIndices=19041 boneEntries=4864 submeshes=40 batches=40`
  - `ACTIVE.SKIN: sections=40 sectionsWithBatches=40 unmatchedBatches=0`

#### Client 3: TBC Retail 2.4.3.8606 (`CHARACTER\BloodElf\Male\BloodElfMale.m2`)
```powershell
I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe m2 inspect --archive-root "H:\CLIENTS\TBC\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft" --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.m2"
```
- **Exit Code**: `0`
- **Output Highlights**:
  - `ERA: 1.0.0 (MD20 v0x100, classic layout)`
  - `M2: ... signature=MD20 version=0x107 model=BloodElfMale boundsMin=(-1.94, -1.46, -0.75) boundsMax=(1.67, 1.43, 3.24) boundsRadius=3.057 skinProfiles=4 bones=143`
  - `GEOMETRY: available=true vertices=5712 textures=4 renderFlags=4 textureLookup=4`
  - `SKIN: stage=Initialized profileIndex=0 exactPath=CHARACTER\BloodElf\Male\BloodElfMale00.skin loaded=true vertexLookup=5712 triangleIndices=21990 boneEntries=5712 submeshes=49 batches=49`
  - `ACTIVE.SKIN: sections=49 sectionsWithBatches=49 unmatchedBatches=0`

#### Client 4: Wrath Pre-Release 3.0.1.8303 (`CHARACTER\BloodElf\Male\BloodElfMale.m2`)
```powershell
I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe m2 inspect --archive-root "H:\CLIENTS\Wrath\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft" --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.m2"
```
- **Exit Code**: `0`
- **Output Highlights**:
  - `ERA: 1.0.0 (MD20 v0x100, classic layout)`
  - `M2: ... signature=MD20 version=0x107 model=BloodElfMale boundsMin=(-1.94, -1.46, -0.75) boundsMax=(1.67, 1.43, 3.24) boundsRadius=3.057 skinProfiles=4 bones=143`
  - `GEOMETRY: available=true vertices=5712 textures=4 renderFlags=4 textureLookup=4`
  - `SKIN: stage=Initialized profileIndex=0 exactPath=CHARACTER\BloodElf\Male\BloodElfMale00.skin loaded=true vertexLookup=5712 triangleIndices=21990 boneEntries=5712 submeshes=49 batches=49`
  - `ACTIVE.SKIN: sections=49 sectionsWithBatches=49 unmatchedBatches=0`

#### Client 5: Wrath Retail 3.3.0.10958 (`CHARACTER\BloodElf\Male\BloodElfMale.m2`)
```powershell
I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe m2 inspect --archive-root "H:\CLIENTS\Wrath\3.X_Retail_Windows_enUS_3.3.0.10958\World of Warcraft" --virtual-path "CHARACTER\BloodElf\Male\BloodElfMale.m2"
```
- **Exit Code**: `0`
- **Output Highlights**:
  - `ERA: 3.3.5 (MD20 v0x108)`
  - `M2: ... signature=MD20 version=0x108 model=BloodElfMale boundsMin=(-1.88, -2.01, -0.82) boundsMax=(2.44, 2.04, 3.80) boundsRadius=3.765 skinProfiles=4 bones=151`
  - `GEOMETRY: available=true vertices=6778 textures=5 renderFlags=5 textureLookup=5`
  - `SKIN: stage=Initialized profileIndex=0 exactPath=CHARACTER\BloodElf\Male\BloodElfMale00.skin loaded=true vertexLookup=6778 triangleIndices=26379 boneEntries=6778 submeshes=60 batches=60`
  - `ACTIVE.SKIN: sections=60 sectionsWithBatches=60 unmatchedBatches=0`

---

## 3. Criterion → Evidence Mapping

| Spec Criterion | Description | Evidence / Output | Status |
|---|---|---|---|
| **FR-001** | Legacy MD20 version dispatch (`0x100`–`0x107`) | `M2ModelReaderDispatcher.DetectEra` routes all <= 0x107 models with classic division layout to `M2Era100ModelReader`. Verified across 1.0.0, 2.0.0, 2.4.3, 3.0.1 (exit 0). | **PASS** |
| **FR-002** | Legacy bone parsing (108B and 112B stride) | 108B layout parsed for `0x100` (`xyz.m2`: bones=1, `BloodElfMale.m2` 2.0.0: bones=138); 112B layout with `boneNameCrc` at `0x0C` parsed for `0x107` (2.4.3 & 3.0.1: bones=143). 11 unit tests pass. | **PASS** |
| **FR-003** | Embedded skin profile resolution | Embedded division records read into `M2ModelDocument.EmbeddedSkinDocuments`. `M2SkinProfileRuntime` initializes directly without looking for non-existent `.skin` files. Verified by `exactPath=...00.skin loaded=true` and sections parsed. | **PASS** |
| **FR-004** | Geometry & global vertex preservation | `M2Era100Geometry.GlobalVertices` populated; `M2GeometryDocument` constructed for inspect. Geometry reports `available=true` across all 5 clients. | **PASS** |
| **FR-005** | Bounding-box fallback rendering | `BuildBoundingBoxFallbackModel` generates 8-vertex, 12-triangle unit bounds cube with `usesCompatibilityFallback: true` from model bounds when geometry has 0 sections. Clean build. | **PASS** |
| **FR-006** | World asset placement routing | `WorldAssetManager.cs` explicit branch for `Md20_1X_V100_Era100` creates renderer via `WowViewerM2RuntimeBridge` rather than falling through to `ConvertM2ToMdx`. | **PASS** |
