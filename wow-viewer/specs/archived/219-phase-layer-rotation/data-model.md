# Data Model: Phase Layer Rotation Tools

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
**Created**: 2026-09-03

All entities live in `WowViewer.Core/Maps/` (library-first; no test project references the viewer).

## Entity: `PhaseLayerSettings` (extended)

Existing Spec 203 type; gains rotation and per-tile placement state.

| Field | Type | Default | Validation | Notes |
|---|---|---|---|---|
| `RotationDegrees` | `float` | `0f` | Any; quadrant-exact when a multiple of 90 | Zero short-circuits all rotation logic (FR-011) |
| `RotationOriginTileX` | `float` | computed | Tile space | Default set by adapter = centre of occupied bounds |
| `RotationOriginTileY` | `float` | computed | Tile space | As above |
| `MirrorHorizontal` | `bool` | `false` | — | Exact-grid mirror along the horizontal axis (FR-018) |
| `MirrorVertical` | `bool` | `false` | — | Exact-grid mirror along the vertical axis (FR-018) |
| `TilePlacements` | `IList<PhaseTilePlacement>` | empty | See below | Per-tile donor→target mappings (US5) |
| `CellOffsetX` | `int` | `0` | Signed terrain cells | Canonical translation axis; 8 cells/chunk, 128 cells/ADT (FR-020) |
| `CellOffsetY` | `int` | `0` | Signed terrain cells | Same; tile/chunk UI values are exact views over this field |
| `RotationApproximation` | `PhaseRotationApproximation` | derived | — | Reported, not set: `ExactGrid` (90° multiples or mirrors), `FreeRotate` (otherwise) |

Relationships: one layer → many `PhaseTilePlacement`; the layer's existing `TileOffsetX/Y`,
`Channels`, `OnlyTakeWhatThePhaseCarries` semantics are unchanged and compose per R3's order.

## Entity: `PhaseTilePlacement` (new record)

One donor→target tile mapping on a layer.

| Field | Type | Validation |
|---|---|---|
| `DonorTileX` | `int` | 0..63 (donor grid slot; may be empty in the donor WDT — resolves to "no contribution") |
| `DonorTileY` | `int` | 0..63 |
| `TargetTileX` | `int` | 0..63 (base-map tile) |
| `TargetTileY` | `int` | 0..63 |

Rules:
- Duplicate `TargetTileX/Y` across mappings: **last in list wins**, conflict reported (FR-015).
- A mapping claims its target over the whole-layer offset+rotation for that tile.
- Removal restores the prior composition exactly (FR-016).

## Entity: cell-granular translation (Phase 2B)

- Canonical unit is an integer terrain cell (one eighth of a chunk edge, one 128th of an ADT edge).
- Existing whole-tile offsets convert exactly as `cellOffset = tileOffset * 128`; chunk offsets use
  `cellOffset = chunkOffset * 8`. No parallel offset states are allowed to drift.
- A non-boundary cell offset requires a source neighborhood and re-slices transformed data into
  target chunks/ADTs. Height samples are moved exactly; no bilinear interpolation is implicit.
- Optional WDL assistance consumes Spec 196's sampler to return a candidate integer `(dx,dy)` plus
  fit score. The proposal does not mutate `CellOffsetX/Y` until accepted by the operator.

## Entity: `PhaseTileSource` (new result record)

What `PhaseCompositionPolicy.ResolveTileSource(layer, targetX, targetY)` returns — the single
lookup both adapters consume.

| Field | Type | Notes |
|---|---|---|
| `HasSource` | `bool` | False when nothing fills this target |
| `SourceTileX/Y` | `int` | Donor tile to read |
| `Via` | `PhaseTileSourceKind` | `None`, `Base`, `Offset`, `Rotation`, `PerTile` — for logs and conflict reports |
| `RotationDegrees` | `float` | Rotation to apply to the sourced content (0 for offset/base paths) |
| `Approximation` | `PhaseRotationApproximation` | Stated mode (FR-009) |

## Enum: `PhaseTileSourceKind`

`None`, `Base`, `Offset`, `Rotation`, `PerTile` — provenance of a target tile's content.

## Enum: `PhaseRotationApproximation`

`ExactGrid` (multiple of 90°), `FreeRotate` (otherwise) — reported per layer in UI and logs.

## State transitions

- Layer rotation: `0° (unrotated)` ⇄ `90° CW/CCW` ⇄ `45°/free` — removing rotation returns to the
  unrotated composition exactly (FR-008).
- Per-tile placement: `unmapped` → `mapped` → `removed`; a mapped target may be re-mapped (last
  wins, conflict reported).

## Validation rules (from FRs)

- Angles: any float accepted; quadrant detection via `Degrees / 90` integrality.
- Tile coordinates clamped to 0..63 at UI entry; policy rejects out-of-range with a diagnostic.
- Rotation applies only to channels the layer is permitted to contribute (FR-010).
- Placement transform: position rotated about origin; orientation (Z degrees) incremented by the
  same angle; exact for 90° multiples, float-exact for free angles.

## Entity: `TileContentTransform` (new static seam, FR-019)

Pure transform functions over tile content, in `WowViewer.Core/Maps/TileContentTransform.cs`.
Knows nothing about phases — the phase layer is just its first consumer (R9).

| Operation | Signature (shape) | Exactness |
|---|---|---|
| Rotate 90° | `RotateContent90(chunk, placements, origin, clockwise)` | Exact grid |
| Mirror horizontal | `MirrorContentH(chunk, placements, origin)` | Exact grid |
| Mirror vertical | `MirrorContentV(chunk, placements, origin)` | Exact grid |
| Translate | `TranslateContent(chunk, placements, dx, dy)` | Exact |
| Free rotate | `RotateContent(chunk, placements, angle, origin)` | Free (stated) |

Content covered per call: heights, normals (mirrored axis negates), texture layers + alpha maps,
hole mask, shadow map, vertex colours, liquid, and every placement (position transformed,
orientation adjusted — including the handedness flip on mirrors).

## Invariants (tested in Core)

1. `ResolveTileSource` with `RotationDegrees == 0`, no mirrors, and no placements returns exactly
   the pre-feature offset behaviour (byte-identical routing).
2. 90° CW followed by 90° CCW (same origin) is the identity mapping.
3. Mirror H twice, or mirror V twice, is the identity (SC-009 involution).
4. Per-tile mapping beats offset+rotation for its target; two mappings on one target resolve
   last-wins with a report.

## Entity: `MapContentSelection` (Phase 3)

| Field | Meaning |
|---|---|
| `Granularity` | Active magnetic grid: Tile, Chunk, or TerrainCell |
| `Units` | Canonical stable set of selected global cells/ranges; tile/chunk are lossless aggregate views |
| `EditMode` | Replace, Add, Subtract, or Toggle |
| `SourceLayerId` | Base or one phase layer supplying selected data |
| `TargetAnchor` | Explicit target grid coordinate; never inferred solely from camera position |
| `PreviewTransform` | Rotation/mirror/cell offset applied non-destructively before commit |

The orthographic canvas, 3D terrain selector, per-tile phase mapping, and editor operations all
reference this one selection. They may cache render geometry, but not own divergent selected-unit sets.

## Entity: `BaseLayerSettings`

| Field | Meaning |
|---|---|
| `Channels` | Same channel vocabulary used by phase layers |
| `Enabled` | Convenience whole-layer gate; base identity remains non-removable |

Base channel gates run before phase composition. They are data-composition controls, not renderer-only
visibility flags.

## Entity: `MapCompositionSavePreflight`

| Field | Meaning |
|---|---|
| `TargetFormat` | Explicit supported or refused map output format |
| `OutputRoot` | User-selected output-copy destination |
| `AffectedTiles` | Complete tile inventory materialized by the preview |
| `OmittedOrLossyChannels` | Every channel the target cannot represent faithfully |
| `Conflicts` | Unresolved source/target claims or off-map content |
| `RefusalReasons` | Conditions blocking any write |
| `PlannedOutputPaths` | Exact files to be produced after confirmation |

No output is written until the preflight has no blocking refusal. Native-format identity is never
inferred from a filename or relabeled compatibility output.
5. The composed result is a pure function of (layer settings, donor WDT contents) — identical
   across reloads (SC-005).
6. `TileContentTransform` is callable with no phase-system type in scope (SC-010).
