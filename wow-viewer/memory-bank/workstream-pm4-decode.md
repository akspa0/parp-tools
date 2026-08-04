# Workstream — PM4 decode

Owner specs: 130 (remaining decode), 131 (scene graph / doodads), epic-pm4-restoration.
Last updated: 2026-08-04. Branch `131-pm4-scene-graph-doodads`.

This file is the durable home for PM4 decode findings. `activeContext.md` links here and stays
short; put detail here, not there.

## Settled — do not re-derive

### The coordinate transform

MSVT is stored in **ADT placement space**: a distance-from-origin coordinate, exactly like a raw
MDDF position, with its two horizontal fields in the **opposite order** to MDDF's
(`MSVT.X == MDDF.rawY`, `MSVT.Y == MDDF.rawX`). Per-axis subtraction, **no axis swap**:

```text
placement = (17066.666 - MSVT.X, 17066.666 - MSVT.Y, MSVT.Z)
```

Evidence: over the 179 development tiles with both a PM4 and a correctly named `_obj0.adt`,
**55,978 of 60,560 (92.4%)** MDDF/MODF positions fall inside their paired PM4's footprint. The
unswapped reading scores **412/60,560 (0.7%)** and is eliminated.

Earlier "absolute world with swapped axes" readings came from bounds fits, which prove only which
**band** a value is in — a reflection about the map centre maps a band onto a band. The raw band
measurement still reproduces (309/309: X in the band of the filename's SECOND number, Y in the
FIRST). What was wrong was reading a distance-from-origin band as a map tile index. **The map tile
index is `31 - band`.**

`Pm4PlacementMath.ConvertPm4VertexToWorld` is **correct — do not touch it**. It emits an
intermediate space the viewer finishes with `renderer = (MapOrigin - world.Y, MapOrigin - world.X,
world.Z)`; composing the two reproduces the transform above, so its axis swap is cancelled by the
renderer's. The 7 `PlacementMath_*` tests defending it are right.

### Region-scoped coordinate frames — REFUTED

`pm4 bounds-audit --by-region` over 1,895 CK24 objects / 309 files / 207 regions: 1,877 objects on
one frame, 1,892 with zero tile displacement, 61 of 62 multi-file regions homogeneous. No frame
family table exists.

`MSHD.Field04` is a **per-file** header value, so "objects in one region fail alike" and "objects in
one file fail alike" were always the same observation — the hypothesis was unfalsifiable on the
evidence that motivated it. Regions are still real authored areas (245 is the ~2006 Sholazar
prototype zone); they are simply not a coordinate frame key.

### The placement fitter was the bug — removed from the render path

`Pm4PlacementMath`'s MPRL-scored per-object fitter was overriding the canonical frame two ways:

- `ResolveCoordinateMode` picked `TileLocal` for data that is never tile-local, then added tile
  offsets to absolute coordinates. 18 objects, including the human tents.
- `TryComputeWorldYawCorrectionRadians` rotated 974 of 1,895 objects by 15–45°. Disproven by
  `pm4 yaw-evidence` against MODF world boxes over the 127 objects whose box can see a rotation
  (each proven by a 45° control): containment **93.3% canonical / 88.2% with yaw / 79.0% control**,
  hurting 96 and helping 3. `WG_GATE01.WMO` dropped 100% → 50%.

`WorldScene.ResolveCk24CoordinateModeResolution` now returns a constant canonical resolution and
`WorldScene.ResolvePlacementSolution` uses the identity planar transform with zero yaw, keeping the
real world centroid as the pivot. **`Pm4PlacementMath` is deliberately untouched** — all 16
`PlacementMath_*` tests still pass; the render path just stopped asking it to fit anything.

**Confirmed working in the viewer 2026-08-04**: tiles aligned, tents correctly identified, rotated
walls and buildings now correct.

### CK24 is an asset-class key

`pm4 doodad-split`, over the 179 tiles with ADT ground truth:

- **A keyed (non-zero) CK24 is one placed WMO.** 47 tiles carry no WMO placements and **none** of
  them carries a keyed object — 47 chances to falsify, zero failures. Keyed count equals WMO
  placement count exactly on 136/179 tiles and within ±1 on 163. Totals 1,045 vs 1,113. 99.3% of
  keyed objects sit inside a MODF box, against 16.5% of CK24 0.
- **CK24 0 is not an object.** Every one of the 170 tiles that has one has **exactly one**: it is
  the per-tile remainder holding everything that is not a keyed WMO. Some hold 1,000+ surfaces.

### CK24 0 splits into per-doodad components

`pm4 component-identity` builds components from geometry alone — surfaces welded through shared
vertex positions at epsilon 0.25, no field involved — then checks them against MDDF:

- **20,113 components; 19,124 (95.1%) land within 24 units of an MDDF placement.** Closest matches
  are at distance **0.00** with extents matching the model (brazier 3.9×3.9×0.6, planter 4.1×3.9).
- 0.339 components per MDDF placement. That shortfall is **expected, not a failure**: most M2
  doodads generate no collision at all.

### MSLK.GroupObjectId is the per-doodad identity

Scored on purity (constant within a component) **and** distinctness (that value used by no other
component on the tile):

| field | purity | distinctness | values/tile |
|---|---|---|---|
| **MSLK.GroupObjectId** | 16.6% | **99.9%** | 11 |
| MSLK.LinkId | 60.7% | 0.3% | 1 |
| MSLK.TypeFlags | 34.0% | 1.3% | 3 |
| MSUR.GroupKey | 100.0% | 0.0% | 1 |
| MSUR.AttributeMask | 59.7% | 1.1% | 4 |

3,343 of GroupObjectId's 3,345 pure components carry a value no other component on the tile uses.

**Distinctness is the load-bearing half.** `MSUR.GroupKey` is 100% pure and would have read as a
perfect separator, but takes 9 values across the whole corpus — a class enum. `MSLK.LinkId` is
60.7% pure and effectively constant per tile. Purity alone cannot tell an identity from a constant.

### Other settled facts

- **MSPV/MSPI is a vertical planar quad mesh** — the walls between MSUR floors. 98.05% of MSLK path
  windows hold exactly 4 indices, 99.6% coplanar, zero of 598,790 faces Z-dominant against MSUR's
  91.7%. Polyline and triangle-list eliminated.
- **MSPV, MSVT and MSCN share one chunk frame. MPRL is the only permuted chunk** — its third axis is
  MSVT's first.
- **MSUR surfaces are triangle fans.** The April 2025 export is fully accounted for: 15,096 verts =
  MSPV + MSVT, 7,382 faces = Σ(IndexCount − 2). MSPI contributed no faces then, so the connective
  geometry was never decoded in any era.
- **MPRR structural hypothesis eliminated** — no chunk's entry count matches the sentinel-delimited
  run count (best MPRL, 5/502 files). 94% of 3,171,410 runs are exactly length 3 (75.5%) or 7
  (18.5%): small fixed-shape records, not a bulk index stream.

## Open

- **Component coverage.** 34.4% of components have no MSLK link at all and only 16.6% are pure on
  GroupObjectId, so it names a minority. The anchor-only MSLK entries (`MspiFirstIndex < 0`, 53% of
  1.27M links) are the next place to look — prior art reads them as doodad placements carrying
  group ids and anchors.
- **MPRR.** The length-3 and length-7 record shapes are undecoded. The editor screenshots (below)
  show dashed inter-polygon links, which is the right shape for a navmesh adjacency graph.
- **MSCN** as a co-equal connective-geometry candidate is still untested. `MSUR._0x18 → MSCN`
  resolves 98.8% but reaches only 34.5% of MSCN points.
- **Does `MSHD.Field04` index something outside the PM4** — a master region table, DBC, or
  server-side list?
- **Viewer performance.** The overlay builds and draws all 9,207 objects regardless of camera
  position. It must cull per tile using the existing ADT Detail Tiles budget.

## Ground truth: Blizzard's WoW Editor 1.9.0 (circa 2005)

User-supplied screenshots of the editor rendering this data, with the Karazhan Crypts WMO loaded in
the same scene for comparison:

- Grey filled polygons are walkable MSUR surfaces; the finely striped band is a **staircase, one
  thin polygon per tread** — which is why MSUR runs to 518K surfaces.
- **Red edges** appear at platform lips and drop-offs — almost certainly the MSPV/MSPI vertical
  quads.
- Cyan vertical tick marks, one per surface region; identity unknown.
- White dashed lines link polygons — candidate MPRR adjacency.
- **Candelabras, cobwebs and banners have no nav polygons beneath them.** Decorative M2s contribute
  nothing. This is the constraint that makes 0.339 components per placement the expected result.

## Traps

- **Post-transform caches.** Two disk caches store geometry *after* the placement transform, so a
  cached tile replays whatever placement was in effect when written. Both must be bumped together
  for any change to placement semantics even when the byte layout is unchanged: tile overlay
  (`PM4C`, `Pm4OverlayCacheService.CacheVersion`) and per-file (`PM4F`,
  `Pm4PerFileCacheService.EntryVersion`), both now **9**. Root:
  `<app bin>/output/cache/pm4-overlay/<id>/`. This cost a full round of "still broken" on a fix that
  was already correct.
- **ADT and PM4 name their tiles differently.** PM4 zero-pads (`development_01_00.pm4`), ADT does
  not (`development_1_0_obj0.adt`). The old `GetObj0PathForPm4` built a padded name that exists for
  **none** of the 616 corpus files though 411 have a companion. Use
  `Pm4CoordinateService.TryGetObj0PathForPm4`, which returns null rather than letting callers fall
  back to an arbitrary tile. **`Pm4CollisionDumper` still falls back to the first `*_obj0.adt` in
  the folder** and should be fixed.
- **Bounds fits prove bands, not axes.** They cannot see a reflection. Three sessions were lost to
  this.
- **When one chunk renders correctly and another does not in the same file, the difference is the
  code path, not the data.** Comparing MSCN's render path against MSUR's found in minutes what
  bounds analysis could not.
- **Purity without distinctness is a false positive generator.** See `MSUR.GroupKey` above.
- **Detector power must be shown before a null result is believed.** The legacy
  `trianglesOnly` counter is zero by construction. `pm4 connective-geometry --verify-detector` and
  `pm4 yaw-evidence`'s 45° control exist for this reason.

## Commands

All take `--input`/`-i` (PM4 dir) and `--output`/`-o` (JSON; otherwise stdout).

| command | what it answers |
|---|---|
| `pm4 bounds-audit` | per-tile MSVT bounds and spill |
| `pm4 bounds-audit --by-region` | is the coordinate frame region-scoped? (no) |
| `pm4 yaw-evidence` | does the fitted yaw help or hurt? (hurts) |
| `pm4 doodad-split` | does CK24 partition by asset class? (yes — keyed = WMO) |
| `pm4 component-identity` | what splits the CK24 0 remainder? (geometry; id is GroupObjectId) |
| `pm4 connective-geometry` | what is MSPV/MSPI? (vertical quad mesh) |
| `pm4 mprr` | MPRR domain sweep and run structure |

`doodad-split` and `component-identity` also take `--placements <dir>` for the companion
`_obj0.adt` files, defaulting to `--input`. Reports land in `output/pm4-decode/`.

## Test state

`WowViewer.Core.PM4.Tests`: 102 passed, 1 failed —
`Pm4RegionObjectGrouperTests.AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects`,
**pre-existing**, confirmed failing at baseline.
