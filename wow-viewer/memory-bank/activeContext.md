# Active Context — wow-viewer

**Last updated**: 2026-06-17 | **Focus**: PM4 surface correlation to WMO assets — recover object placements from PM4 data

## THE POINT OF THIS WORK (read this first)

**ADT placements are INCOMPLETE.** 222 of 616 dev PM4 tiles have NO ADT at all. Many ADTs that DO exist are missing placements. This is the PROBLEM, not a limitation of our matcher.

**PM4 data IS the source of truth for what objects are actually placed on a tile.** The PM4 contains the collision surfaces, scene graph placements, and pathing data for every object the game client placed there — including objects the ADT doesn't list.

**We are matching PM4 CK24 objects to WMO assets so we can REGENERATE placement data (MODF/MDDF entries) for tiles where ADT is missing or incomplete.** The GOAL is producing placement data for tiles where ADT doesn't exist.

**Do NOT frame ADT incompleteness as a "limitation of the matcher" or a "validation gap."** It is the entire reason this work exists. Validate against ADT where ADT exists (to confirm the algorithm works), but the GOAL is producing placement data for tiles where ADT doesn't exist.

## CRITICAL: The matcher is STILL producing false positives

Both approaches tried so far produce false positives:

1. **Hull footprint matching (ABANDONED)**: Ironforge/Darnassis at 0.999 overlap despite NOT being on dev map. Convex hull throws away surface structure.

2. **Surface edge-length histogram (current, STILL WRONG)**: GoldshireInn matched to tile 0_2 at 0.86 PM4 coverage. User confirmed: **NO GoldshireInn exists in PM4 data on tile 0_2. It does not exist there.** The 0.86 coverage is a histogram collision — edge-length bins match across completely different geometry with similarly-sized triangles. Different wall/floor triangles from different WMOs have similar edge lengths and bin identically.

**Edge-length-only histograms are TOO COARSE.** They are a necessary but insufficient signal. Do NOT claim matches are correct without ground truth verification. The matcher has produced false positives in BOTH approaches. Verify before claiming.

## Current approach: Surface triangle correlation (needs improvement)

PM4 MSUR surfaces → triangulated fans → per-triangle sorted edge lengths binned to integers (transform-invariant geometric hash) → histogram intersection against WMO MOVI/MOVT collision triangle histograms.

**Why hull footprints failed:** Convex hull throws away internal surface structure. A 12×12×48 box matches Ironforge, Darnassis, and dozens of other WMOs equally. Surface triangle correlation eliminated the WORST hull false positives but still produces its own false positives via histogram collisions.

## What needs work (FRESH CHAT — start here)

Surface histogram matching has hit a ceiling. Area-only gives the best result so far (P@3=25.3%, P@1=0%). Placement-invariant normal/offset descriptors made it worse. The next workstreams are:

1. **Fix WMO enumeration** (Phase 8): the DB only has 503/1985 WMO roots. Many correct WMOs are simply not in the candidate pool. This is the highest-leverage next step.
2. **Revisit `Pm4Generator.cs`**: `pm4 validate-generator-geometry` shows generated PM4 surfaces do not match real PM4 surfaces (score ~0.004). The generator likely uses the wrong source mesh, wrong simplification, or wrong transform. This is the user's stated concern.
3. **Spatial/placement filtering**: once a WMO candidate is identified, use ADT-style placement bounds or CK24 group location to disambiguate, rather than relying solely on histogram scores.

**The full pipeline (end goal)**: surface match → identify WMO → extract placement transform from PM4 → write MODF entry → regenerate ADT for tiles without one.

## Tooling

- `pm4 build-wmo-surface-db` — builds WMO surface triangle histogram DB from staged client
- `pm4 extract-pm4-surfaces` — extracts PM4 CK24 surface triangle histograms
- `pm4 match-surfaces` — matches PM4 surfaces to WMO DB (histogram intersection, F1 score)
- `pm4 validate-matches` — checks matches against ADT ground truth (for tiles that HAVE ADT)
- Legacy (hull-based, superseded): `build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`
- Legacy (ADT-based, kept for validation only): `correlate-models`, `sweep-correlate`

## Key files

- `src/core/WowViewer.Core.PM4/Models/Pm4SurfaceCorrelationContracts.cs` — surface correlation data models
- `src/core/WowViewer.Core.PM4/Services/Pm4SurfaceCorrelationExtractor.cs` — triangulate + histogram
- `src/core/WowViewer.Core.PM4/Services/Pm4SurfaceCorrelationMatcher.cs` — histogram intersection + F1
- `tools/inspect/WowViewer.Tool.Inspect/Pm4SurfaceBuildSupport.cs` — WMO surface DB builder + PM4 extraction
- `src/core/WowViewer.Core.PM4/Services/Pm4Generator.cs` — PM4 generator from WMO collision (plane clustering)
- `specs/065-pm4-correlation-to-world-assets/` — spec updated 2026-06-17: surface triangle correlation is primary, hull/footprint approach abandoned, legacy commands kept for reference.
- Phase 6 (add triangle area to histogram key) complete 2026-06-18: area-bin-size=1.0 eliminates GoldshireInn tile 0_2 false positive and boosts P@3 10.3%→25.3%, but P@1 drops to 0%.
- Phase 7 (placement-invariant descriptors + generator validation) complete 2026-06-18:
  - Added optional normal-alignment and planar-offset bins; defaults are 0 to preserve the area-only baseline.
  - Validation with bins enabled: 0 matched, 0 ambiguous, P@3=10.1% (worse than area-only 25.3%).
  - Added `pm4 validate-generator-geometry` to directly test `Pm4Generator.cs` against real PM4 tiles.
  - Generator validation on `development_16_37`: mean symmetric score 0.004, 0/4 matched placements. Generated PM4 does not reproduce real PM4 surfaces.
- `pm4 validate-matches` fixed to deserialize `Pm4SurfaceMatchOutput` (surface match format), not the old fingerprint-match format.
