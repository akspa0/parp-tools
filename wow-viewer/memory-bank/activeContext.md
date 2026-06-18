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

## Current generator validation results

Tile `development_29_18` (48 real WMO groups, 48 ADT WMO placements):
- Mean symmetric score: **0.051**
- Matched groups (score >= 0.50): **0**
- Best farm placement vs real group `0x43C689`: **14/210** real triangles matched after normal-Z orientation correction.

The generated geometry is now in the right coordinate frame and the right group location, but surface normals/tessellation still do not line up with real PM4 polygons. The remaining gap is almost certainly **WMO local-axis / rotation convention** or **surface winding orientation**.

## What needs work next

1. **Resolve WMO local-axis convention** so that placed roof/wall normals in generated PM4 match real PM4 normals. Candidate checks:
   - Swap local X/Y before placement transform.
   - Invert one local axis.
   - Remove or flip the extra `CreateRotationZ(PI)` in `BuildPlacementTransform`.
2. **Standardize surface winding** so all generated MSUR normals point in a consistent direction (e.g., positive Z for upward-facing surfaces), matching real PM4.
3. **Re-run `pm4 validate-generator-geometry` on `development_29_18`** and aim for at least one group crossing the 0.50 symmetric-score threshold.
4. **Commit the current fixes** before iterating on rotation/orientation.

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
