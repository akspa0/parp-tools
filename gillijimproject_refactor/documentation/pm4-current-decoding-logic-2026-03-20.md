# PM4 Viewer Reconstruction Contract (Updated Mar 21, 2026)

## Why This Exists

PM4 work in the viewer has accumulated several useful fixes, several misleading hypotheses, and at least one confirmed regression experiment. This document is the current viewer-side contract for PM4 reconstruction and debugging in the active `MdxViewer` path.

This is not a final PM4 format specification. It is the authoritative description of what the active viewer does today, what assumptions are still open, and what should not be reintroduced.

## Validation Discipline

- Build validation is useful for guarding refactors and keeping the branch coherent.
- Build success is not PM4 correctness.
- Runtime real-data signoff is still required before claiming PM4 placement/orientation closure.
- When this document and runtime behavior disagree, runtime evidence wins and the document should be updated.

## Current Confidence Level

- Strong: PM4 tile assignment is deterministic from filename: `map_x_y.pm4 -> (tileX = x, tileY = y)`.
- Strong: duplicate PM4 payloads merge rather than overwrite.
- Strong: PM4 overlay object identity is keyed by `(tile, ck24, objectPart)`, not only `(tile, ck24)`.
- Strong: coordinate mode and planar solve are resolved per CK24, not once per file.
- Strong: tile-local PM4 and world-space PM4 no longer share the same unrestricted planar candidate set.
- Strong: the linked-`MPRL` bounds-center translation experiment is no longer active.
- Strong negative result: current runtime evidence does not support an `MPRL` bounding-box or `MPRL` container-frame paradigm for viewer reconstruction.
- Open: final PM4 frame ownership is still not fully closed, but `MPRL` should currently be treated as anchor/scoring data rather than a bounding-box ownership model.
- Open: `MSCN` is parsed and important to the broader PM4 pipeline, but it is not yet the authoritative source for active viewer object extents/orientation.
- Open: runtime visual signoff is still pending for remaining alignment and visibility edge cases.

## PM4 Must Be Read At Three Layers

This is the cleanest mental model for the active codebase.

1. Raw per-file PM4 data
   - chunk contents and decoded fields from `Pm4File`
   - this is where `MSVT`, `MSVI`, `MSUR`, `MSLK`, `MPRL`, and `MSCN` exist as storage

2. Linked object-assembly view
   - viewer reconstruction groups surfaces into CK24-scoped objects and sub-objects
   - this is where coordinate mode, axis convention, planar transform, and linked-group splitting are decided

3. Final viewer render derivation
   - world-space lines/triangles, renderer-basis conversion, object-local debug transforms, bounds, and pickability
   - this is where many regressions show up even when raw PM4 decode is technically present

Confusing these layers caused several previous dead ends. A raw chunk fact is not automatically a viewer placement rule.

## Active Data Contract In Use

The active overlay path reads from `WoWMapConverter.Core.Formats.PM4.Pm4File` and currently uses:

- `MSVT`: mesh vertices via `Pm4File.MeshVertices`
- `MSVI`: mesh index stream via `Pm4File.MeshIndices`
- `MSUR`: surface loops via `Pm4File.Surfaces`
  - consumes `Ck24`, `MsviFirstIndex`, `IndexCount`, `MdosIndex`, `GroupKey`, `AttributeMask`, and `Height`
- `MSLK`: link graph via `Pm4File.LinkEntries`
  - consumes `GroupObjectId`, `MsurIndex`, and `RefIndex`
- `MPRL`: placement refs via `Pm4File.PositionRefs`
  - currently used as anchor and scoring input, especially for coordinate-mode selection, planar solve, and yaw comparison
- `MSCN`: parsed but not authoritative for the current viewer object reconstruction contract

## Tile Mapping Contract

In `WorldScene` PM4 overlay loading:

1. Parse PM4 filename tile from `*_x_y.pm4`
2. Map directly to viewer tile `(tileX = x, tileY = y)`
3. Do not reassign tiles from `MPRL` centroid heuristics
4. If multiple PM4 files map to one tile, merge payloads and rebase `objectPart`

This prevents sparse-dataset drift and silent overwrite loss.

Rendering follow-up:

- PM4 overlay rendering and picking should not require ADT tile residency for PM4-only sparse tiles.
- AOI-loaded-tile gating remains for ADT-backed tiles.

## Object Assembly Pipeline

Implemented in `WorldScene.BuildPm4TileObjects(...)`:

1. Keep only `MSUR` surfaces with `CK24 != 0` for object reconstruction
2. Group surfaces by CK24
3. Detect axis convention per CK24 candidate set
4. Resolve coordinate mode per CK24
   - tile-local vs world-space is chosen by `MPRL` fit, not once per file
5. Resolve one shared planar transform per CK24
6. Compute one CK24 world pivot from reconstructed geometry
7. Optionally apply one coarse CK24 yaw correction when the residual error is meaningfully large
8. Split CK24 by `MSLK`-linked groups
9. Optionally split by dominant `MDOS` index
10. Optionally split by connectivity
11. Emit per-component overlay objects keyed by `(tile, ck24, objectPart)`

Important guardrail: transform resolution is shared per CK24, not solved independently per linked subgroup or per tiny fragment. That keeps one real object from exploding into internally inconsistent orientations.

## MSLK And MPRL Linkage Rules

When associating `MSLK` links to surfaces:

- prefer `MsurIndex` as the surface reference
- only fall back to `RefIndex` as a surface id when `MsurIndex` does not resolve

For `MPRL` association:

- treat `RefIndex` as a position-ref index when it is in `MPRL` range
- deduplicate refs by index

## MPRL Contract In The Active Viewer

`MPRL` is important, but the current viewer contract is narrower than the earlier failed experiment.

Negative result from runtime debugging:

- PM4 geometry, PM4 bounds, and visible object extents are not currently conforming to an `MPRL` bounding-box paradigm.
- Treat that as observed falsification of the current bounding-box theory, not as something that only needs more tuning.
- In practice: if a future fix starts from "the CK24 should sit inside the `MPRL` bounds/container," that is reintroducing a disproven assumption and should require new evidence first.

What `MPRL` does today:

- helps decide tile-local vs world-space interpretation
- helps score planar transform candidates
- provides expected yaw for comparison against geometry-derived principal yaw
- provides anchor references for debugging and inspection

What `MPRL` does not do today:

- it does not translate a whole CK24 group into a linked `MPRL` world-bounds center
- it does not override every geometry-derived decision unconditionally

Reason: the linked-center translation experiment regressed runtime placement and was removed.

Working interpretation for now:

- `MPRL` still looks useful as a placement/reference signal.
- `MPRL` does not currently behave like a reliable per-object bounding box, enclosing footprint, or authoritative container for reconstructed PM4 viewer objects.

## Orientation Solver (Current)

`ResolvePlanarTransform(...)` evaluates candidate planar transforms and scores them by:

1. centroid-to-nearest-`MPRL` distance
2. footprint score between sampled object points and `MPRL` planar points when enough linked refs exist
3. yaw tie-break between candidate principal axis and averaged `MPRL` yaw
   - `MPRL` packed low-16 rotation is decoded as clockwise and rebased by `+90°` for world yaw comparison
   - basis fallback evaluates direct, sign-flipped, and quarter-turn variants
4. an explicit penalty for mirrored or winding-inverting candidates

### Candidate Sets

Tile-local PM4:

- tests only the non-swapped mirror set inside the established south-west tile basis
- does not allow the quarter-turn swap set that caused coherent non-origin tile rotations

World-space PM4:

- evaluates the rigid set first: identity, `180°`, `+90°`, `-90°`
- only falls back to mirrored candidates afterward

This separation matters. Applying the world-space quarter-turn search to tile-local PM4 caused the reported `90°` non-origin tile regression.

### Continuous Yaw Correction

After planar transform selection, the viewer may apply one CK24-scoped continuous yaw correction:

- derive candidate principal yaw from reconstructed object footprint
- derive expected yaw from linked `MPRL` refs
- compute best signed delta with basis/parity fallback
- rotate geometry around the CK24 world centroid before world-to-renderer conversion

Guardrail:

- residual yaw deltas below `12°` are ignored
- this is intentional because the principal-axis solve is useful for coarse basis recovery, but too noisy for small final alignment tweaks

## Coordinate Conversion Contract

### PM4 Local To World

- choose planar/up axes from detected convention: `XY+Zup`, `XZ+Yup`, or `YZ+Xup`
- apply the chosen planar transform
- if tile-local:
  - `worldX = tileX * TileSize + mappedU`
  - `worldY = tileY * TileSize + mappedV`
  - `worldZ = localUp`
- else use world-like planar values directly

### World To Renderer Space

- `rendererX = MapOrigin - worldY`
- `rendererY = MapOrigin - worldX`
- `rendererZ = worldZ + 0.5`

This matches the active terrain/object renderer basis.

## Diagnostics And Viewer Tooling

The active viewer already includes several PM4 debugging surfaces:

- `BuildPm4OverlayInterchangeJson(...)` exports overlay summary, tile/object metadata, alignment state, and optional geometry
- PM4 Alignment UI exposes `Dump PM4 Objects JSON`
- PM4 Alignment UI edits selected-object 9DoF only: translation, rotation, and scale
- `Flip Obj X`, `Flip Obj Y`, `Flip Obj Z` are object-local mirror tests
- object-local rotation and scale are applied around object center so the object stays in place during parity testing
- PM4 debug overlay includes `PM4 X-Ray`
- PM4 debug overlay includes `PM4 Bounds`
- PM4 debug UI can split CK24 by `MdosIndex`

Important scope note:

- current PM4 bounds come from rendered PM4 object geometry, not from `MSCN` directly
- they are a debugging aid, not final proof of authoritative PM4 extents

## Confirmed Rejected Experiments

Do not silently reintroduce these without fresh runtime evidence:

- no PM4 tile reassignment from `MPRL` centroid heuristics
- no linked-`MPRL` bounds-center translation for whole CK24 groups
- no per-linked-subgroup transform solving that lets one CK24 split spin differently per part
- no fallback to `(tile, ck24)` identity keys without `objectPart`
- no requirement that PM4-only sparse tiles must also have loaded ADT terrain to render or pick

## Open Questions

- whether `MSCN` should become authoritative for some parts of viewer-side object extents or grouping
- whether `MPRL` has a narrower semantic role than the earlier "container/bounds frame" hypothesis suggested
- whether any remaining misalignment is now in reconstruction, visibility/culling, or asset/render parity rather than PM4 basis solving itself

## Practical Next Steps

1. Use `Dump PM4 Objects JSON` on a known failing live scene.
2. Compare tile/object counts, bounds, and transform metadata against what is visible in the viewer.
3. Validate the current CK24 yaw correction on the same scene before changing ownership rules again.
4. If residual issues remain, prefer narrow reconstruction or visibility fixes over new global PM4 frame theories.
