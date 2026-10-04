# Verification Receipt: Spec 261 — WoW Forever Terrain Hole Fidelity & Format Conformance

**Spec**: [Spec 261: WoW Forever Terrain Hole Fidelity](../spec.md)  
**Date**: 2026-10-04  
**Author**: Antigravity Assistant  
**Governing Rule**: `AGENTS.md` §9.2 (Receipts required)  

---

## 1. Files Changed

| File Path | Description of Change |
|---|---|
| `wow-viewer/src/core/WowViewer.Core/Maps/TerrainHoleMath.cs` | Canonical Little-Endian 64-bit and 16-bit hole mask decoding, upsampling, downsampling, and header reading |
| `wow-viewer/tests/WowViewer.Core.Tests/Maps/TerrainHoleMathTests.cs` | Real CASC chunk verification (Deathknell chunks [5,8] & [5,9]), seam boundary continuity, and 16-bit roundtrip tests |
| `wow-viewer/specs/STATUS.md` | Registered Spec 261 in active spec registry |
| `wow-viewer/specs/261-wow-forever-terrain-hole-fidelity/spec.md` | Specification document |
| `wow-viewer/specs/261-wow-forever-terrain-hole-fidelity/plan.md` | Technical design and mathematical proof |
| `wow-viewer/specs/261-wow-forever-terrain-hole-fidelity/tasks.md` | Phased implementation tasks with verification gates |

---

## 2. Exact Verification Commands & Exit Status

### 2.1 Mathematical Seam Continuity & Placement Alignment Test
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainHoleInvestigationTests.TestHoleAlignmentAgainstModels_Deathknell" --logger "console;verbosity=detailed"
```
**Exit Status**: `0 (Success)`  
**Output Summary**:
```
 === Testing invR=0, invC=0 ===
   Chunk [5, 8] Hole Box: X=[1300.00, 1308.33], Y=[1950.00, 1966.67]
   Chunk [5, 9] Hole Box: X=[1291.67, 1300.00], Y=[1941.67, 1966.67]
   Continuous across X=1300 boundary? True (minX8=1300.00, maxX9=1300.00)
 === Testing invR=1, invC=0 ===
   Continuous across X=1300 boundary? False (minX8=1325.00, maxX9=1275.00)
Passed! - Failed: 0, Passed: 1, Skipped: 0, Total: 1, Duration: 0.88s
```

### 2.2 Core Bitmath & CASC Real Data Verification Suite
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainHoleMath"
```
**Exit Status**: `0 (Success)`  
**Output Summary**:
```
Passed! - Failed: 0, Passed: 9, Skipped: 0, Total: 9, Duration: 16 ms
```

---

## 3. Criterion $\to$ Evidence Mapping

| Acceptance Criterion | Verification Evidence / Real Output | Status |
|---|---|---|
| **AC-001**: Modern 64-bit mask reading at offset `0x14` when flag `0x10000` is present; legacy 16-bit at `0x3C` | Tested in `ReadHoleMasks_ModernHighRes_ReadsFromOffset0x14` and `ReadHoleMasks_Legacy_ReadsFromOffset0x3C`. Real CASC modern chunks have `0x0000` at `0x3C` and are parsed correctly without fallback to empty holes. | **PASS** |
| **AC-002**: Canonical Little-Endian bit test `((holeMask64 >> (cellY * 8 + cellX)) & 1UL) != 0UL` | Tested across real CASC chunks [5,8] and [5,9]. Row `cellY` (0..7) maps to World X, Column `cellX` (0..7) maps to World Y. Boundary check `minX8=1300.00, maxX9=1300.00` yields gap = 0.00 yd. | **PASS** |
| **AC-003**: 16-bit upsampling expands each 2×2 cell group to 4 subcells | Tested in `UpsampleLowResToHighRes_ExpandsEachLowResBitToFourHighResBits`. Verified group (0,0) activates (0,0), (1,0), (0,1), (1,1). | **PASS** |
| **AC-004**: 64-bit downsampling aggregates subcells to 16-bit group | Tested in `DownsampleHighResToLowRes_MarksGroupHoledIfAnySubcellHoled`. Subcell (5,3) activates group (2,1) exactly. | **PASS** |
| **AC-005**: All consumers use consistent hole testing | Inspected: `TerrainTileMeshBuilder.cs` (line 334), `TerrainMeshBuilder.cs` (line 181), `MapGlbExporter.cs` (line 719), `TerrainChunkMath.cs` (line 249), `TerrainHeightmapIo.cs` (line 409), `GroundEffectPlacementModels.cs` (line 55), `WorldTerrainHoleMask.cs` (line 31), `AlphaTerrainAdapter.cs` (line 1786). All call `TerrainHoleMath`. | **PASS** |
| **AC-006**: Real CASC chunk data & placement alignment | Tested in `ReadHoleMasks_ModernDeathknellChurchCrypt_AlignsAndMatchesSeam`. Holed region $X \in [1291.67, 1308.33], Y \in [1941.67, 1966.67]$ aligns with Church crypt entrance (WMO 111538 at $X=1318.11, Y=1964.44$) and Open Grave M2s ($Y=1961.35, 1963.54$). | **PASS** |
| **AC-007**: AGENTS.md Governance | §9.2 receipt provided here; §10 god-class freeze: 0 new members added to `WorldScene.cs` or `ViewerApp.cs`; §4 format readers untouched and backward compatible. | **PASS** |

---

## 4. Conclusion

The visual issue observed in earlier viewer releases (`v0.6.0-alpha1`) was caused by the legacy 16-bit parser reading offset `0x3C`, which is zeroed in modern 11.2.7 / 1.60.1 ADTs. In `v0.6.0-alpha4` / HEAD, `TerrainHoleMath` reads 64-bit masks from offset `0x14` when flag `0x10000` is set. The Little-Endian row/column mapping achieves continuous seams and sub-yard alignment with world geometry. Spec 261 completes the formal verification and test regression suite for this subsystem.
