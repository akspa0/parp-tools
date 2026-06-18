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

The matcher needs a stronger geometric signal. Options to explore:
- Add triangle area to histogram key (not just edge lengths)
- Use surface normal + plane distance (PM4 MSUR has Normal + Height fields) as additional histogram dimensions
- Use exact vertex positions after transform alignment (not just edge lengths)
- Match at the surface level (PM4 MSUR surface = WMO MOPY face group) not just individual triangles
- Use the full PM4 surface structure: normal + height + edge lengths + vertex count per surface
- Fix WMO enumeration (503/1985 — archive catalog probe bug or need listfile)
- 956 ambiguous still high — need stronger disambiguation

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
- `specs/065-pm4-correlation-to-world-assets/` — spec (needs updating: surface correlation is primary, hull abandoned, matcher still produces false positives)
