# Spec 209 Evidence: Liquid Convergence & Shoreline Mechanism Measurement

**Date**: 2026-09-02  
**Command**: `inspect adt liquid-convergence --client "H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft" --map Azeroth`  
**Client**: `0.5.3.3368` (Alpha WDT + `misc.mpq` WL* files)  
**Subject**: `Azeroth` (500 liquid tiles, including the Wetlands coast `Azeroth_31_29` .. `Azeroth_31_30`)

---

## Executive Summary

1. **Union Invariant Verified (SC-002 / FR-004)**:
   - **0 cells** covered by either source were dropped from the unified liquid array (`MissingFromUnified = 0`).
   - The merge in `BuildUnifiedLiquid` is a union and cannot drop coverage. Gaps are NOT caused by cell dropping in the merge.

2. **Mechanism A (MCLQ shoreline sag)**:
   - Fixed via `LiquidSurfaceInterpolation` (presence-weighted quad interpolation across 7 unit tests).
   - On the 0.5.3 Alpha WDT path, MCNK liquid is already normalized to the $257 \times 257$ grid.

3. **Mechanism B (Waterline Culling) MEASURED & CONFIRMED**:
   - `WlLiquidRasterizer.KeepOnlyAboveTerrain` culled **585,108 WL* cells** across Azeroth where terrain elevation rose above or met the water plane.
   - Crucially, **459,374 (78.5%) of those culled WL* cells had NO MCLQ coverage** to replace them.
   - This population is concentrated exactly along coastlines and shorelines. On the Wetlands coast (`Azeroth_31_29`):
     - Overlapping water (`Both`): 27,545 cells
     - WL* cells culled by terrain: **24,857 cells**
     - Culled WL* cells without MCLQ: **23,192 cells** (93.3% of culled cells had no MCLQ).
   - **Diagnosis**: This proves Mechanism B is the root cause of the shoreline gaps. At the shoreline, WL* polygons extend slightly into rising coastal terrain; `KeepOnlyAboveTerrain` strictly culls any sample where $\text{terrain} \ge \text{water}$. Because MCLQ coverage ends abruptly or has holes along the Wetlands coast, the culled WL* water leaves empty terrain strips (gaps) along the water's edge.

---

## Map-Wide Aggregate Measurements

| Metric | Measured Value |
|---|---|
| Total liquid tiles analyzed | **500** |
| Tiles with MCLQ | **495** |
| Tiles with WL* | **111** |
| Tiles with BOTH MCLQ and WL* | **108** |
| MCLQ-only cells | **16,960,049** |
| WL*-only cells | **13,942** |
| Overlapping cells (`Both`) | **695,208** |
| Dry terrain cells (`Neither`) | **15,355,301** |
| WL* cells culled by `KeepOnlyAboveTerrain` | **585,108** |
| **Culled WL* cells WITHOUT MCLQ** | **459,374** |
| **Cells covered by source but absent from unified** | **0** (100% union preservation) |

---

## Focus: Wetlands Coastline Tiles

| Tile | MCLQ-Only | WL*-Only | Both (Overlap) | Δ Height Mean / Max | WL* Culled | Culled Without MCLQ (Gaps) |
|---|---|---|---|---|---|---|
| `Azeroth_30_29` | 3,512 | 0 | 20,018 | 0.00 / 0.00 | 5,096 | **3,815** |
| `Azeroth_31_29` | 6,469 | 0 | 27,545 | 0.00 / 0.06 | 24,857 | **23,192** |
| `Azeroth_32_29` | 2,083 | 0 | 13,264 | 0.00 / 0.00 | 4,197 | **3,625** |
| `Azeroth_35_29` | 3,007 | 0 | 14,919 | 0.00 / 0.00 | 8,121 | **6,603** |
| `Azeroth_36_29` | 3,324 | 0 | 42,179 | 0.00 / 0.05 | 6,637 | **5,657** |
| `Azeroth_37_29` | 1,502 | 0 | 36,412 | 0.00 / 0.03 | 9,944 | **8,827** |
| `Azeroth_30_30` | 8,308 | 0 | 26,995 | 0.00 / 0.03 | 8,916 | **7,301** |
| `Azeroth_31_30` | 6,831 | 0 | 38,788 | 0.00 / 0.12 | 10,555 | **7,796** |
| `Azeroth_32_30` | 496 | 0 | 9,045 | 0.00 / 0.02 | 2,556 | **2,104** |
| `Azeroth_36_30` | 2,116 | 2 | 8,765 | 1.71 / 15.42 | 8,003 | **6,250** |

Notice that where both sources overlap in open water (`Both`), their surface heights agree almost identically (mean $\Delta H = 0.00$, max $\le 0.12$ across most Wetlands tiles). The issue is not surface height disagreement in the open water; it is that `KeepOnlyAboveTerrain` cuts away WL* at the shoreline when terrain elevation rises above the water plane, and MCLQ does not supply shoreline water in those cells.
