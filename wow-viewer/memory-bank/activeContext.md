# Active Context — wow-viewer

**Last updated**: 2026-06-17 | **Focus**: PM4 surface correlation to WMO assets — recover object placements from PM4 data

## THE POINT OF THIS WORK (read this first)

**ADT placements are INCOMPLETE.** 222 of 616 dev PM4 tiles have NO ADT at all. Many ADTs that DO exist are missing placements. This is the PROBLEM, not a limitation of our matcher.

**PM4 data IS the source of truth for what objects are actually placed on a tile.** The PM4 contains the collision surfaces, scene graph placements, and pathing data for every object the game client placed there — including objects the ADT doesn't list.

**We are matching PM4 CK24 objects to WMO assets so we can REGENERATE placement data (MODF/MDDF entries) for tiles where ADT is missing or incomplete.** When surface correlation says GoldshireInn is on tile 0_2 at 0.86 triangle coverage but ADT doesn't list it, the MATCHER IS RIGHT and the ADT is what's missing.

**Do NOT frame ADT incompleteness as a "limitation of the matcher" or a "validation gap."** It is the entire reason this work exists. Validate against ADT where ADT exists (to confirm the algorithm works), but the GOAL is producing placement data for tiles where ADT doesn't exist.

## Current approach: Surface triangle correlation

PM4 MSUR surfaces → triangulated fans → per-triangle sorted edge lengths binned to integers (transform-invariant geometric hash) → histogram intersection against WMO MOVI/MOVT collision triangle histograms.

**Why hull footprints failed:** Convex hull throws away internal surface structure. A 12×12×48 box matches Ironforge, Darnassis, and dozens of other WMOs equally — all have similarly-sized structural groups. Hull matching produced 0.999 overlap false positives for WMOs that are NOT on the map. Surface triangle correlation eliminated these false positives.

## What works (surface correlation, commit 21aa0064)

- 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible (1604 PM4 vs 2790 WMO fingerprints)
- P@3 = 10.3% (2.3x improvement over hull P@3=4.5%)
- NO false positives — Ironforge/Darnassis eliminated
- 12 correct top-1: GoldshireInn (0.86 PM4 triangle coverage on tiles 0_2/1_1), classicalelfruins, arathistonebridge, orchut
- GoldshireInn matches tiles 0_2/1_1 at 0.86 coverage — PM4 says it's there, ADT doesn't list it → PM4 is right, ADT is incomplete (this is the whole point)

## What needs work

- WMO DB coverage: archive catalog finds 503/1985 WMOs. Need to fix enumeration (listfile or archive catalog probe bug).
- Edge bin size (1.0 unit) may need tuning
- Histogram key only has edge lengths — triangle area could add discrimination
- 956 ambiguous still high — need stronger disambiguation signals
- The full pipeline: surface match → identify WMO → extract placement transform from PM4 → write MODF entry → regenerate ADT

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
- `specs/065-pm4-correlation-to-world-assets/` — spec (needs updating: surface correlation is now primary, hull is abandoned)
