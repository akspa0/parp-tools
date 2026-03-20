# PM4 Current Decoding Logic (Mar 20, 2026)

## Why This Exists

PM4 behavior has historically been opaque. This document captures the active decode and viewer reconstruction logic in one place so progress is not lost and future passes do not reintroduce old mistakes.

This is a status snapshot, not a final spec.

## Current Confidence Level

- Strong: CK24 object-part cohesion now holds (parts stay together as one object).
- Strong: PM4 tile assignment is deterministic from filename (`map_x_y.pm4` -> `tileX=x`, `tileY=y`).
- Strong: duplicate PM4 tile payloads merge rather than overwrite.
- Open issue: reconstructed PM4 objects currently appear rotated about 90 degrees counter-clockwise versus expected world orientation in runtime checks.

## Active Data Contract In Use

The active overlay path reads from `WoWMapConverter.Core.Formats.PM4.Pm4File` and uses these chunks/fields:

- `MSVT`: mesh vertices (`Pm4File.MeshVertices`)
- `MSVI`: mesh index stream (`Pm4File.MeshIndices`)
- `MSUR`: surface loops (`Pm4File.Surfaces`)
  - uses `surface.Ck24` (decoded from packed params)
  - uses `surface.MsviFirstIndex` + `surface.IndexCount`
  - uses `surface.MdosIndex`, `surface.GroupKey`, `surface.AttributeMask`, `surface.Height`
- `MSLK`: link graph (`Pm4File.LinkEntries`)
  - uses `GroupObjectId`, `MsurIndex`, `RefIndex`
- `MPRL`: placement refs (`Pm4File.PositionRefs`)
  - used as world-space anchors (ADT placement order semantics)
- `MSCN`: parsed but not yet authoritative in the current orientation contract

## Tile Mapping Contract

In `WorldScene` PM4 overlay loading:

1. Parse PM4 filename tile from `*_x_y.pm4`
2. Map directly to viewer tile `(tileX=x, tileY=y)`
3. Do not reassign tiles from MPRL centroid heuristics
4. If multiple PM4 files map to one tile, merge payloads and rebase `objectPart`

This prevents sparse-dataset drift and silent overwrite loss.

## Object Assembly Pipeline

Implemented in `WorldScene.BuildPm4TileObjects(...)`:

1. Keep only `MSUR` surfaces with `CK24 != 0` for object reconstruction
2. Group surfaces by CK24
3. Detect axis convention for each CK24 candidate set
4. Build one shared planar transform per CK24 using linked MPRL anchors
  - coordinate mode (tile-local vs world-space) is selected per CK24 by MPRL fit score, not fixed once per file
5. Split CK24 by MSLK-linked components
6. Optional split by MDOS index
7. Optional split by vertex connectivity
8. Emit per-component overlay objects keyed by `(tile, ck24, objectPart)`

Important guardrail: transform resolution is now shared per CK24, not per linked subgroup, to keep all parts of a CK24 object on one coordinate plane.

## MSLK Linkage Rules Used

When associating `MSLK` links to surfaces:

- prefer `MsurIndex` as the surface reference
- only fall back to `RefIndex` as a surface id when `MsurIndex` does not resolve

For MPRL association:

- `RefIndex` is treated as position-ref index when it is in MPRL range
- refs are deduplicated by index

## Orientation Solver (Current)

`ResolvePlanarTransform(...)` evaluates candidate planar transforms (swap/invert combinations), then scores by:

1. centroid-to-nearest-MPRL distance
2. footprint score between sampled object points and MPRL planar points
3. yaw tie-break between candidate principal axis and averaged MPRL yaw
  - MPRL packed low-16 rotation is decoded as clockwise and rebased by +90° for world yaw comparison
  - basis fallback now evaluates direct/sign-flipped plus quarter-turn variants (`expected`, `-expected`, `expected +/- 90°`, `-expected +/- 90°`)
  - when linked-footprint scoring is active, strong yaw agreement can override modest footprint-distance differences
4. small penalty for winding inversion

For tile-local data it evaluates swap+invert combinations.
For world-space data it evaluates swap parity without inversion.

## Coordinate Conversion Contract

### PM4 local -> world (conceptual)

- choose planar/up axes from detected convention (`XY+Zup`, `XZ+Yup`, `YZ+Xup`)
- apply candidate planar swap/invert
- if tile-local:
  - `worldX = tileX * TileSize + mappedU`
  - `worldY = tileY * TileSize + mappedV`
  - `worldZ = localUp`
- else use world-like planar values directly

### World -> renderer space

- `rendererX = MapOrigin - worldY`
- `rendererY = MapOrigin - worldX`
- `rendererZ = worldZ + 0.5`

This matches the active terrain/object renderer basis.

## Known Residual Problem

Runtime status after restoring shared CK24 transforms:

- object parts stay together (major improvement)
- orientation is still globally offset by about 90 degrees counter-clockwise in tested scenes

Interpretation:

- assembly and grouping are closer to correct
- residual is likely a consistent yaw-basis offset, not random per-part divergence

## Next Fix Slice (Smallest Safe)

1. Keep grouping/splitting as-is
2. Add temporary debug readout for selected object:
   - chosen planar transform
   - averaged MPRL yaw
   - solved principal-axis yaw
   - delta in degrees
3. Validate the current stronger MPRL-oriented tie-break on the same known failing scene before broader changes
4. If residual offset remains, constrain any further correction to the yaw tie-break/basis layer only

## What Not To Reintroduce

- no PM4 tile reassignment from MPRL centroid heuristics
- no per-linked-subgroup transform solving that lets one CK24 split spin differently per part
- no fallback to `(tile, ck24)` identity keys without `objectPart`
