# Spec 295 — Validation-Tile Centroid Calibration Measurement Receipt (T002)

**Date:** 2026-11-05  
**Scope:** T002 — Measured signed X/Y/Z offset between solved PM4 footprint centers and authored WMO placement / footprint centers on the validation tile (`development_00_00.pm4` + `development_0_0_obj0.adt`).

## Validation tile & repository fixture fingerprint

| Artifact | Path | Size (bytes) | SHA-256 |
|---|---|---:|---|
| PM4 validation tile | `wow-viewer/test_data/original_development/World/Maps/development/development_00_00.pm4` | 1,267,923 | `F34256CA621DBC97FFC18E9CC36D490B75B6EF968BE53900738C37DDEC488163` |
| Companion ADT object tile | `wow-viewer/test_data/original_development/World/Maps/development/development_0_0_obj0.adt` | 5,629 | `88242A89E6AE5F8518C37E7E797AAD1FABC18AE53F882D9209778DC7DE469591` |

## Per-hull measurement (15 authored WMO placements matched by `MSUR.0x1C` / `CK24`)

Each row matches an authored `MODF` placement in `development_0_0_obj0.adt` to its exact `CK24` (`MSUR.0x1C >> 8`) surface group in `development_00_00.pm4` (`|ΔZ| < 0.001` yd) and compares:
- `PrismCentroid`: extruded 2D convex-hull footprint center from `Pm4FootprintExtruder.ExtrudeFootprint`
- `AuthoredPos`: `MODF.Position`
- `AuthoredBoundsCenter`: `(MODF.BoundsMin + MODF.BoundsMax) * 0.5`
- `DeltaBoundsCenter`: `AuthoredBoundsCenter - PrismCentroid` (the signed correction `Offset` required to move the solved footprint center onto the authored footprint center)

| CK24 | Asset | RotZ (°) | PrismCentroid (X, Y, Z) | AuthoredPos (X, Y, Z) | DeltaPos (X, Y, Z) | DeltaBoundsCenter (X, Y) | Planar residual (yd) |
|---|---|---:|---|---|---|---|---:|
| `0x41D4B1` | `GUARDTOWER_DESTROYED_CONSTRUCTION.WMO` | 195.0 | `(16632.643, 16879.398, 26.586)` | `(16626.645, 16878.500, 26.586)` | `(-5.998, -0.898, 0.000)` | `(-0.344, +0.604)` | `0.695` |
| `0x421809` | `ND_IRONDWARF_LARGEBUILDING.WMO` | 315.0 | `(16812.543, 16802.918, 38.009)` | `(16812.543, 16802.918, 38.009)` | `(0.000, 0.000, 0.001)` | `(0.000, 0.000)` | `0.000` |
| `0x410C11` | `WG_WALL01C.WMO` | 0.0 | `(16744.812, 17030.219, 8.754)` | `(16744.807, 17030.188, 8.754)` | `(-0.006, -0.031, 0.000)` | `(0.000, 0.000)` | `0.000` |
| `0x412CDC` | `WG_WALL01D.WMO` | 0.0 | `(16783.381, 17028.547, 10.804)` | `(16783.375, 17028.521, 10.804)` | `(-0.006, -0.025, 0.000)` | `(0.000, +0.008)` | `0.008` |
| `0x415881` | `WG_WALL02D.WMO` | 0.0 | `(16801.807, 17013.562, 13.531)` | `(16801.828, 17013.584, 13.532)` | `(+0.021, +0.021, 0.000)` | `(0.000, +0.014)` | `0.014` |
| `0x418D9F` | `GuardTower_intact.wmo` | 0.0 | `(16830.938, 16936.062, 17.703)` | `(16837.561, 16936.191, 17.703)` | `(+6.623, +0.129, 0.000)` | `(+0.773, +0.141)` | `0.786` |
| `0x40B023` | `WG_Wall01.wmo` | 0.0 | `(16745.781, 16982.707, 5.504)` | `(16745.775, 16982.682, 5.504)` | `(-0.006, -0.025, 0.000)` | `(0.000, +0.008)` | `0.008` |
| `0x409E75` | `WG_Wall01.wmo` | 0.0 | `(16718.064, 16982.578, 4.952)` | `(16718.059, 16982.553, 4.952)` | `(-0.006, -0.025, 0.000)` | `(0.000, +0.008)` | `0.008` |
| `0x40DB09` | `WG_Wall02.wmo` | 0.0 | `(16689.566, 16983.020, 6.845)` | `(16689.588, 16983.027, 6.845)` | `(+0.021, +0.008, 0.000)` | `(0.000, 0.000)` | `0.000` |
| `0x407F21` | `WG_WALL03.WMO` | 0.0 | `(16774.361, 16983.010, 3.986)` | `(16774.383, 16983.031, 3.986)` | `(+0.021, +0.021, 0.000)` | `(+0.021, +0.021)` | `0.030` |
| `0x40D51E` | `ND_HUMAN_WALL.WMO` | 180.0 | `(16667.398, 16952.990, 6.660)` | `(16667.398, 16952.990, 6.660)` | `(0.000, 0.000, 0.000)` | `(0.000, 0.000)` | `0.000` |
| `0x417964` | `ND_HUMAN_WALL_END.WMO` | 183.5 | `(16666.512, 16927.094, 15.587)` | `(16666.512, 16927.094, 15.587)` | `(0.000, 0.000, 0.000)` | `(0.000, 0.000)` | `0.000` |
| `0x40883F` | `ND_HUMAN_WALL_END.WMO` | 179.0 | `(16667.057, 16978.264, 4.258)` | `(16667.057, 16978.264, 4.258)` | `(0.000, 0.000, 0.000)` | `(0.000, 0.000)` | `0.000` |
| `0x41C0F5` | `WG_GATE01.WMO` | 90.0 | `(16583.037, 16619.863, 24.120)` | `(16583.037, 16619.854, 24.120)` | `(0.000, -0.010, 0.000)` | `(+0.041, -0.023)` | `0.047` |
| `0x40AA0A` | `GuardTower_intact.wmo` | 0.0 | `(16716.418, 16930.574, 5.314)` | `(16723.041, 16930.703, 5.314)` | `(+6.623, +0.129, 0.000)` | `(+0.773, +0.141)` | `0.786` |

## Root cause of the operator-reported `-X` offset

1. **Asymmetric footprint center vs hull convex hull (`GuardTower_intact.wmo`, `RotZ = 0°`):** On the two unrotated guard towers (`0x418D9F` and `0x40AA0A`), the PM4 walkable floor hull centroid (`PrismCentroid.X`) sits `-0.773` yards along **-X** relative to the WMO bounding-box center (`DeltaBoundsCenter = (+0.773, +0.141)` yds), and on `WG_Wall01` / `WG_WALL01C` / `WG_WALL01D` the uncalibrated position is `-0.006` yards along **-X** (`DeltaPos.X = -0.006` yds).
2. **Legacy shape-library `AttractPositionToPm4Mesh` call (`Pm4UnauthoredPlacementResolver.cs:439`):** Passed `modelBoundsMin = Vector3.Zero` and `modelBoundsMax = bestRefShape` (all-positive span), shifting the assumed model center by `+bestRefShape * 0.5f` instead of `0`.
3. **Aggregate `Pm4ShapeMatchAnalysisService.MeasureCalibration` result on `development_00_00.pm4` (15/15 WMO hulls):**
   - `SampleCount`: `15`
   - `MeanOffset`: `(+0.084, +0.061, 0.000)` yds (bounds-center reference) / `(+0.773, +0.141, 0.000)` yds on `GuardTower_intact.wmo`
   - Post-calibration SC-001 check: **15 / 15 (100.0%)** of proposals place the model's footprint center within `1.0` yard of the hull's footprint center (target $\ge 90\%$).

