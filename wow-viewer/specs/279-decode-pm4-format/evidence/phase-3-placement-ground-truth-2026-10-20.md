# Phase 3 — Authored-placement ground truth (2026-10-20)

## Question

The operator's core concern: WMO matching works (~70%), but M2 is unresolved because M2
pathfinding geometry is sparse and merged. Is there a field that partitions the M2 surfaces from
the WMO surfaces? Work backwards from authored `_obj0.adt` placements (MDDF = M2, MODF = WMO).

## Method

New read-only probe [`Pm4PlacementGroundTruthProbe`](../../../src/core/WowViewer.Core.PM4/Research/Pm4PlacementGroundTruthProbe.cs):

1. For each PM4 tile, resolve its companion `_obj0.adt` with
   [`Pm4CoordinateService.TryGetObj0PathForPm4`](../../../src/core/WowViewer.Core.PM4/Services/Pm4CoordinateService.cs:131)
   and read placements with [`AdtPlacementReader`](../../../src/core/WowViewer.Core.IO/Maps/AdtPlacementReader.cs:8).
   The reader already reverses on-disk chunk signatures, so the earlier ASCII scan for `MDDF`/`MODF`
   was a false negative; the on-disk bytes are `FDDM`/`FDOM`.
2. Convert every MSVT vertex to ADT placement space with
   [`Pm4LocalToAdtPlacement`](../../../src/core/WowViewer.Core.PM4/Services/Pm4CoordinateService.cs:54).
3. A surface covers a placement when the placement origin is inside the surface's projected polygon
   or within 0.05 of its boundary.
4. Aggregate surface features by class: M2-only, WMO-only, both, neither.

This is a correlation instrument. Coverage is origin-in-polygon, not full model footprint, and a
merged surface can cover several objects.

## Result

Corpus: 616 PM4 files, 411 with a companion `_obj0.adt`; 308,892 surfaces.

| Measure | M2 (MDDF) | WMO (MODF) |
|---|---|---|
| Authored placements | 59,544 | 1,116 |
| Covered by ≥1 surface | 12,147 (20.4%) | 883 (79.1%) |
| Covered by exactly 1 | 7,781 | 215 |
| Covered by >1 | 4,366 | 668 |

| Surface class | Surfaces | `Ck24Type == 0` | `GroupKey == 3` | Dominant type | Dominant group |
|---|---|---|---|---|---|
| M2-only | 16,338 | 14,489 (88.7%) | 14,489 (88.7%) | 0 | 3 |
| WMO-only | 1,961 | 57 (2.9%) | 51 (2.6%) | 64–67 | 16–19 |
| Both | 218 | 110 | 110 | mixed | mixed |
| Neither | 290,375 | 98,162 (33.8%) | 96,516 (33.3%) | 66/67 | 18 |

`Ck24Type == 0` and `GroupKey == 3` select the same 14,489 M2-only surfaces: redundant encodings of
one partition. The reverse direction is weaker — only 12.8% of all `Ck24Type == 0` surfaces cover an
authored M2 origin — so the zero bucket is the **M2 class**, a superset, not a per-object identity.

Connection kinds separate too: `MSLK.TypeFlags` 17, 18, 20, 26, 28 occur on WMO-only surfaces and on
zero M2-only surfaces; kinds 1, 2, 4, 12 are M2-leaning. `AttributeMask` 0 is M2-leaning (2,369 vs
8); masks 9 and 10 are WMO-leaning.

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4PlacementGroundTruthProbe.cs` (new)
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/Program.cs` (`--placements` mode)
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj` (Core.IO reference)
- `wow-viewer/docs/research/pm4-format-reference.md` (§3.5, MSUR rows, CK24 paragraph, E-PLACE)
- `wow-viewer/specs/279-decode-pm4-format/evidence/placement-ground-truth-2026-10-20.json` (report)

## Verification

```powershell
dotnet build "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --nologo -v q
# exit 0, 0 errors

dotnet run --project "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --no-build -- --placements "wow-viewer/test_data/development/World/Maps/development" "wow-viewer/specs/279-decode-pm4-format/evidence/placement-ground-truth-2026-10-20.json"
# exit 0; 616 files, 411 with placements, M2 12147/59544 covered, WMO 883/1116 covered, 17.572s
```

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| M2 and WMO surfaces are separable by a measured field | `Ck24Type == 0` / `GroupKey == 3`: 14,489 M2-only vs 57 WMO-only |
| The partition is not an artifact of one tile | Whole-map: 616 files, 411 with placements |
| The zero bucket is not a per-object identity | Only 12.8% of zero surfaces cover an authored M2 origin |
| Connection kinds also separate | Types 17/18/20/26/28: 0 M2-only; types 1/2/4/12 M2-leaning |

## Limits

- Origin-in-polygon coverage undercounts objects whose origin is outside their pathfinding surface.
- The probe does not yet split the M2 bucket into individual placed objects; that is the next test.
- No runtime, visual or FPS claim is made.
