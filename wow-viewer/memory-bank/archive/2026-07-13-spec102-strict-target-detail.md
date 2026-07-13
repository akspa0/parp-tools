# Archive — Spec 102 strict object-target detail (to 2026-07-13)

Verbose detail moved out of the live memory bank. The strict fragment-trace
pipeline is committed (`4f44c7f7`) but parked; the active path is the simple
precise-mask M0 (see live `activeContext.md`).

## Strict target contract (committed, parked)

- M0 = 3,043,041-param RGB-minimap → one object mask. No DepthAnything, extra
  heads, or numeric side inputs.
- Strict target `object_geometry_visible_mask_257`: rasterize transformed M2/WMO
  triangles, compare each fragment to raw MCVT terrain Z, drop only individually
  below-terrain fragments, keep every above-ground fragment/overlap. Never erase a
  whole placement from a centroid/bounds/fallback/missing-asset decision.
- `strict-geometry-terrain-liquid-fragment-trace-v3` sidecar = variable-length
  numeric audit (raster xy, world XYZE, placement/asset/triangle ids, raw-MCVT
  three-node coords/Z/presence/weights, terrain/liquid elevations, classification,
  asset table, unresolved placements, content hash). Audit sidecar only — never a
  model input. The C# harvester emits it; the M0 model never consumes it.
- Liquid evidence WL → MCLQ → MH2O per pixel; unknown/unreadable = reject.
  Initial M0 was dry-only (any liquid coverage rejects the tile).

## Source facts

- Staged `3_3_5_12340`: 125 map records, 52 terrain-ready maps, 5,471 WDT
  locations, 5,134 raw-V18 rows. 367 locations lack required source signals.
- Eight readable maps lack canonical minimap RGB (ArgentTournamentDungeon,
  ArgentTournamentRaid, DalaranArena, development_nonweighted, ExteriorTest,
  OrgrimmarArena, QA_DVD, WintergraspRaid); six also lack MCLY/MCAL. Frozen
  staged-client source gap — not a parser bug and not reharvestable.

## Status when parked

- C# strict labeling + v3 sidecar implemented; 29 focused C# tests + 42 spec102
  Python tests green. Harvester verified to emit the strict target on a real tile.
- Reharvest of the full strict corpus was never run to completion; the simple
  precise-mask route was chosen instead to move to training.

## Historical

- `ef99e715` = trainer control-flow reference only.
- Legacy `object_precise_mask_257` was called "contaminated" (mixed legacy
  bounds/circle/coverage fallbacks + no under-terrain clipping). The simple route
  knowingly accepts that flaw for now.
- H0/H1/V23/V24/V25 results remain historical and non-comparable.
