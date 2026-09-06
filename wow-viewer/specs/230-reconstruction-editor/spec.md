# Feature Specification: Reconstruction Editor — Rosetta Placement, Map Generator & Multi-Era Save (Spec 230)

**Feature Branch**: `230-reconstruction-editor`
**Created**: 2026-09-06
**Status**: Draft — authored verbatim from operator directive; not planned
**Input**: Operator directive 2026-09-06 (verbatim intent): "My form of editing is more for
reconstruction from existing data in the game, than it is about sculpting new terrain meshes or
hand-painting chunks. Maybe one day, but not today. We should, though, start planning for object
placement tooling from the Rosetta object maps, to use as an index and a means of simply
copy/pasting the asset path into new maps and new positions on maps. We would like a way to have a
'new map generator' in the UI, that allows real editing to start taking place in the editor tab. We
have the tooling already. We need to allow saving to alphaWDT or LK ADT or Cataclysm/MoP ADT's
eventually, too."

## Context

The operator's editing model is **reconstruction from existing game data**, not mesh sculpting or
hand-painting. The building blocks already exist and must be composed, not reinvented:

- **Rosetta 3D Object Library** (landed 2026-09-04): 5,832 calibrated assets with bounds and
  manifest provenance — the index for placement.
- **Terrain Template Brush & generator** (Spec 192): `terrain-generate-templated` CLI synthesizes
  walkable multi-tileset maps — the seed of a UI map generator.
- **Writers** (Spec 197): Alpha 0.5.3 WDT and LK v18 monolithic ADT output are supported; native
  Cataclysm/MoP split ADT is explicitly unavailable until a slot-aware writer exists.
- **Placement editing** (existing): staged placement edits, MODD editing planned by Spec 220.
- **Spec 222 Cartography**: multi-map tile composition is the transport for "existing data into new
  maps".

## User Stories

### US1 — Rosetta-indexed object placement (P1)
In the Editor workspace, the operator picks an asset from the Rosetta library (already searchable
with bounds/volume metadata), then copy/pastes it into the loaded (or generated) map at a chosen
position — placement records authored against the target map, batched and saveable.

**Acceptance criteria**:
- Pick from the Rosetta library directly into a placement queue; paste targets follow the terrain
  pick (with the Spec 226-class occlusion rules) or explicit coordinates.
- Placements stage through the existing placement-edit/save pipeline (single write path, no fork).
- Placement authoring works on generated maps as well as loaded ones.

### US2 — New map generator in the Editor UI (P1)
A **New Map Generator** surface in the Editor tab drives the Spec 192 generator: parameters in,
generated map out, loaded into the viewer for immediate editing.

**Acceptance criteria**:
- UI wraps the existing generator tooling (CLI composed, not reimplemented — FR-2 discipline).
- Output lands in a project-managed output root and loads as a normal map.
- The generated map is immediately usable with US1 placement and Spec 222 tile composition.

### US3 — Multi-era save targets (P2)
Saving generated/edited maps targets Alpha 0.5.3 WDT or LK v18 ADT today; Cataclysm/MoP split ADT
becomes available only when the slot-aware writer lands (Spec 197 boundary — never a lossy
best-effort).

**Acceptance criteria**:
- Save targets are explicit and refuse unsupported eras by name (existing Spec 197 rule).
- Round-trip validation receipts (Spec 221 harness) cover every new save path.

## Constraints

- All placement/writing flows through existing services — Rosetta index, placement edit pipeline,
  Spec 192 generator, Spec 197 writers, Spec 222 composition. This spec composes them behind the
  Editor UI; any missing piece is specced separately rather than forked.
- Per AGENTS.md §10, the Editor UI surfaces are owned service classes.
- Per AGENTS.md §11, new UI registers an inventory row in the same change.

## Dependencies

- Spec 192 (generator), 197 (writers/targets), 220 (WMO doodad editing), 222 (tile composition),
  190 (Rosetta datastore), 221 (validation harness), 227/228 (UI audit + service extraction).
- Sequences after Spec 227's audit so the Editor surfaces land in their post-audit shape.

## Success Criteria

- **SC-1**: Generate a map from the UI, place Rosetta assets into it, and save as Alpha WDT and LK
  ADT with validation receipts — without touching a CLI.
- **SC-2**: Every placement/save action is keybind-addressable under the Spec 229 profiles.
