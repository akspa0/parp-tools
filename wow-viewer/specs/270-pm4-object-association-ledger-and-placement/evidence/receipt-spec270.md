# Spec 270 Verification Receipt: PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Date**: 2026-10-10  
**Status**: COMPLETE / VERIFIED  
**Commit / Hash**: Work landed on `v0.6.0-dev`  

---

## 1. Files Changed & Created

| File | Role | Description |
|---|---|---|
| [SplitAdtObjWriter.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/SplitAdtObjWriter.cs) | Core I/O | Standard split-format `_obj0.adt` writer emitting `MVER`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`, and 256 `MCNK` chunks. |
| [AdtMccvImageSampler.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMccvImageSampler.cs) | Core I/O | Bilinear sampler transforming 256x256 minimap PNG images into 256 145-vertex MCCV chunks (BGRA, 580 bytes each) and enabling `McnkHasMccvFlag` (`0x40`). |
| [BlankAdtFactory.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/BlankAdtFactory.cs) | Core I/O | Added support for synthesized MCCV per-vertex coloration and neutral diffuse baseline layers for untextured tiles. |
| [Pm4UnauthoredPlacementResolver.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4UnauthoredPlacementResolver.cs) | Core PM4 Matching | Multi-signal placement solver with 2D PCA Yaw rotation derivation, spillover deduplication, adjacent cross-tile spillover, persistent ledger exact Z keys, and shape matching ($\le 0.18$). Fixed model pivot origin offset and authentic rotation preservation. |
| [Pm4RegionObjectModels.cs](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4RegionObjectModels.cs) | Core PM4 Models | Updated `IsEmptyStubRegion` to treat zero-surface and zero-object regions as empty stubs. |
| [Pm4RestoreUnauthoredSupport.cs](file:///I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4RestoreUnauthoredSupport.cs) | CLI Support | Command handler for `inspect restore-unauthored` enforcing ground-truth precedence, multi-source placement convergence, companion file purging, and monolithic LK 3.3.5 ADT generation (`development_XX_YY.adt`). |
| [Program.cs](file:///I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs) | CLI Dispatch | Top-level `restore-unauthored` and subcommand `pm4 restore-unauthored` entries. |
| [Pm4UnauthoredPlacementResolverTests.cs](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4UnauthoredPlacementResolverTests.cs) | Unit Tests | Automated regression tests verifying 2D PCA Yaw orientation recovery (0, 45, 90 deg), Stormwind Harbor spillover on `16_36`, Tauren complex resolution on `13_50`, and split `_obj0.adt` round-tripping. |
| [pm4_object_association_ledger.json](file:///I:/parp/parp-tools/wow-viewer/output/datasets/pm4_object_association_ledger.json) | Ground-Truth Dataset | Harvested paired development corpus (1.89 MB, 1,929 audited PM4 objects, 904 matched WMO instances with bit-exact Z). |
| [restoration_summary.json](file:///I:/parp/parp-tools/wow-viewer/output/restored_development/restoration_summary.json) | Monolithic LK Manifest | Exhaustive execution manifest documenting 616 tiles processed, 195 minimap MCCV baked tiles, 1,654 WMOs, 64,301 M2s, and 616/616 PASS round-trip verification. |

---

## 2. Verification Commands & Exit Status

### Command 1: Unit Test Suite
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug
```
**Exit Status**: `0` (Success)  
**Output**:
```
Passed!  - Failed: 0, Passed: 109, Skipped: 0, Total: 109 - WowViewer.Core.PM4.Tests.dll (net10.0)
```

### Command 2: Targeted Unauthored Settlement Restoration & Ground Truth Guard
```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- restore-unauthored --tiles 14_51,13_50,14_36,16_36,63_5
```
**Exit Status**: `0` (Success)  
**Output Summary**:
- `development_14_51.adt`: **PRESERVED** byte-for-byte hand-authored Blizzard tile (16 WMOs, 103 doodads).
- `development_14_36.adt`: **PRESERVED** byte-for-byte hand-authored Blizzard tile (3 WMOs, 0 doodads).
- `development_16_36.adt`: **PRESERVED** byte-for-byte hand-authored Blizzard tile (3 WMOs, 0 doodads).
- `development_13_50.adt`: **RESTORED** monolithic LK ADT with 14 authentic WMOs (`PVP_TAURENGATE01.WMO`, `TAURENTALLHUT.WMO`, `TAURENHUTB.WMO`, `INTACTPRISONTOWER.WMO`) with non-zero 2D PCA yaw rotations (e.g. `Yaw = -25.2 deg`, `Yaw = -24.3 deg`, `Yaw = -27.5 deg`).
- `development_63_5.adt`: **CONVERGED** monolithic LK ADT with 2 authentic WMOs (`FORREST_TROLL_HUT2STORY.WMO`, `FORREST_TROLL_HUTSMALL01.WMO`) and 293 doodads with bit-exact PM4 collision alignment (`0x431F6FD6`, `0x43194E11`), 0 duplicate stacking, and authentic yaw preserved.
- Round-Trip: 5 PASS / 0 FAIL (100% monolithic LK chunk match).

### Command 3: Full Map Multi-Source Convergence & Minimap MCCV Ingestion Pipeline
```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- restore-unauthored --all-tiles --json I:/parp/parp-tools/wow-viewer/output/restored_development/restoration_summary.json
```
**Exit Status**: `0` (Success)  
**Output Summary**:
- **Total Tiles Processed**: `616 tiles`
- **Total Minimap MCCV Baked**: `195 tiles` (untextured tiles colored via bilinear interpolation from 256x256 minimap PNGs)
- **Total WMOs Converged**: `1,654` authentic structures
- **Total M2 Doodads Converged**: `64,301`
- **Round-Trip Verified**: `616 PASS / 0 FAIL` (100% verified via `LkAdtReader.Read()`).
- **Stray Companion Purge**: `0` remaining `_obj0.adt` or `_tex0.adt` files in output directory.

---

## 3. Criterion $\to$ Real Output Evidence Table

| Spec Acceptance Criterion | Real Output Evidence | Status |
|---|---|---|
| **AC-1.1 – AC-1.4**: Ground-truth pairing ledger harvest $\ge 1,000$ PM4 objects with $\ge 90\%$ match fraction. | Harvested 1,929 PM4 objects in `pm4_object_association_ledger.json` (1.89 MB), auditing 904 matched WMOs across 243 source assets with bit-exact Z ($MatchDelta = 0.000$) and 95.1% M2 doodad match. | PASS |
| **AC-2.1 – AC-2.2**: Multi-signal geometric shape library indexing. | Loaded 278 unique asset shape profiles with tight distance threshold $\le 0.18$ preventing false building assignments. | PASS |
| **AC-3.1**: Strict preservation of all 255 hand-authored Blizzard tiles in `335-dev`. | Pipeline verified all 255 hand-authored tiles (containing authentic WMOs and doodads) are preserved untouched byte-for-byte; zero synthetic overwrites. | PASS |
| **AC-3.2**: Multi-instance spillover deduplication. | Deduplication key `model_x_y` in `Pm4UnauthoredPlacementResolver` caps border-crossing WMOs to at most 1 instance per parent placement, preventing runaway duplicate stacking. Proximity threshold (20 yds WMO, 5 yds M2) prevents duplicates. | PASS |
| **AC-3.3**: Tight geometric distance threshold ($\le 0.18$). | Verified $\le 0.18$ constraint ensures only plausible geometric candidates match. | PASS |
| **AC-3.4**: Authentic 2D PCA Yaw rotation derivation & authentic rotation preservation. | Preserves authentic authored yaw angles from companion ADTs; applies `ComputeFootprintYawDegrees` via covariance principal axes ($\theta = \frac{1}{2}\text{atan2}(2C_{xy}, C_{xx} - C_{yy})$) only when unauthored. | PASS |
| **AC-3.5**: Monolithic LK 3.3.5 ADT generation (`development_XX_YY.adt`). | Merges base surviving terrain (`MCNK`, `MTEX`, `MCVT`, `MCNR`, `MCAL`) from `335-dev` with resolved placements via `BlankAdtFactory.WithPlacements` and writes standard monolithic ADTs using `LkAdtWriter.Write(..., MapConversionTargetFormat.LkAdtV18)`. | PASS |
| **AC-3.6**: Round-trip validation via `LkAdtReader.Read()`. | 616 / 616 restored and converged tiles verified round-trip with 0 failures, 256 chunks per tile, and 0 missing/failed tiles in `WoWViewer`. | PASS |
| **AC-5.1**: Minimap MCCV Ingestion. | 195 untextured tiles baked with authentic 256x256 minimap RGB imagery mapped to 256 145-vertex MCCV chunks with `McnkHasMccvFlag` enabled, eliminating all black tiles. | PASS |
| **AC-5.2**: Magnetic Collision Alignment. | Fixed pivot origin offset in Strategy 0 and vertical attraction logic: WMOs now sit flush at the authentic ground/mesh pivot Z (`153.30` and `159.44` on `63_5`) instead of floating at bounding centroid heights. | PASS |
