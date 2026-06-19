# Active Context — wow-viewer

**Last updated**: 2026-06-18 | **Focus**: PM4 surface correlation to WMO assets — fix `Pm4Generator.cs` so generated PM4 surfaces reproduce real PM4 surfaces

## THE POINT OF THIS WORK (read this first)

**ADT placements are INCOMPLETE.** 222 of 616 dev PM4 tiles have NO ADT at all. Many ADTs that DO exist are missing placements. This is the PROBLEM, not a limitation of our matcher.

**PM4 data IS the source of truth for what objects are actually placed on a tile.** The PM4 contains the collision surfaces, scene graph placements, and pathing data for every object the game client placed there — including objects the ADT doesn't list.

**We are matching PM4 CK24 objects to WMO assets so we can REGENERATE placement data (MODF/MDDF entries) for tiles where ADT is missing or incomplete.** The GOAL is producing placement data for tiles where ADT doesn't exist.

**Do NOT frame ADT incompleteness as a "limitation of the matcher" or a "validation gap."** It is the entire reason this work exists. Validate against ADT where ADT exists (to confirm the algorithm works), but the GOAL is producing placement data for tiles where ADT doesn't exist.

## What we just did (generator fix pass)

Focused on `Pm4Generator.cs` and `pm4 validate-generator-geometry`:

1. **Switched source geometry to WMO collision faces** (`MOPY` flags `0x08` collision + `0x20` render collidable, excluding `0x04` no-collide). This cut generated triangle counts from ~350k to ~145k for stormwindharbor and is the correct input mesh.
2. **Fixed `MSVI` first-index bug**: generator wrote byte offsets (`indexCursor * 4`) instead of raw uint indices. Corrected to `(uint)indexCursor`.
3. **Fixed `WorldToPm4Raw` coordinate transform**: was swapping X/Y (`MapOrigin - Y`, `MapOrigin - X`); corrected to `MapOrigin - X`, `MapOrigin - Y`, `Z`.
4. **Added vertex welding + coplanar boundary-polygon merging** so generated surfaces match real PM4's merged polygons (index counts 3–7) instead of raw triangles.
5. **Updated `Pm4GeneratorValidationSupport`** to type-filter real fingerprints to CK24 `0x43` and to use collision geometry.
6. **Identified the correct validation tile**: `development_29_18` has 48 WMO CK24 groups; `development_16_37` has only M2 groups and is unsuitable for WMO generator validation.
7. **Found the matching real group for a generated farm**: generated farm bounds overlap real group `0x43C689` exactly (min/max within 1 unit).

## Current results

### Generator validation (per-placement WMO → PM4 correlation)

Tile `development_29_18` (48 real WMO groups, 48 ADT WMO placements), using **area-only collision fingerprints**:
- Mean symmetric score: **0.462**
- Matched groups (score >= 0.50): **36/48**
- All farm placements match real group `0x43C510`.

### Full-corpus PM4 → WMO matching (collision-only DB)

Built `wmo_collision_surface_db_335.json` from the staged 3.3.5 client (502 WMO roots, 2749 fingerprints, 5.5M collision triangles) and matched all 616 dev PM4s:
- `pm4 match-surfaces`: **30 matched**, **195 ambiguous**, 1106 unresolved, 273 ineligible.
- `pm4 validate-matches` against ADT ground truth: **P@1 = 1.2%**, **P@3 = 28.5%**.

The collision fingerprints are strong enough to produce candidate sets, but **top-1 is unreliable** because many unrelated WMOs share simple box-like collision shapes. The correct asset often appears in the top-3 or in the unresolved candidate list.

## What needs work next

1. **Disambiguation via spatial/placement constraints.** Once a WMO candidate set is identified, use tile bounds, CK24 group location, or adjacent groups to pick the right one instead of relying solely on histogram score.
2. **WMO enumeration gap.** The DB still only has ~502 roots because the archive catalog only exposes named listfile entries. A real full-client listfile would widen coverage.
3. **Cleanup.** Remove the unused boundary-polygon merge helpers in `Pm4Generator.cs`; raw collision triangles are the right output for fingerprints.

## Tooling

- `pm4 build-wmo-surface-db` — builds WMO surface triangle histogram DB from staged client
- `pm4 extract-pm4-surfaces` — extracts PM4 CK24 surface triangle histograms
- `pm4 match-surfaces` — matches PM4 surfaces to WMO DB (histogram intersection, F1 score)
- `pm4 validate-matches` — checks matches against ADT ground truth (for tiles that HAVE ADT)
- `pm4 validate-generator-geometry` — directly tests `Pm4Generator.cs` against real PM4 tiles
- Legacy (hull-based, superseded): `build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`
- Legacy (ADT-based, kept for validation only): `correlate-models`, `sweep-correlate`

## Key files

- `src/core/WowViewer.Core.PM4/Services/Pm4Generator.cs` — PM4 generator from WMO collision (now with collision-face extraction, welded boundary-polygon merging)
- `tools/inspect/WowViewer.Tool.Inspect/Pm4GeneratorValidationSupport.cs` — generator validation harness
- `tools/inspect/WowViewer.Tool.Inspect/Program.cs` — `pm4 validate-generator-geometry` and `pm4 generate-from-wmo` commands
- `src/core/WowViewer.Core.IO/Wmo/WmoGroupMeshDetailReader.cs` — MOVT/MOVI reader
- `src/core/WowViewer.Core/Wmo/WmoGroupFaceMaterialDetail.cs` — MOPY face flags
- `specs/065-pm4-correlation-to-world-assets/` — spec, plan, tasks
