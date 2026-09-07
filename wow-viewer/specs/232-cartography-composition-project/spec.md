# Spec 232 — Cartography Composition: Cell-Level Alignment, Project Persistence & Full-Map Export

Status: **Draft** (spec authored 2026-09-07 from operator directive; implementation pending —
fresh session per the Spec 231 pattern)
Owner epic: Reconstruction & Editing
Depends on: Spec 231 Phase 7 (layer rotation/mirror wired), Spec 219 (composition policy),
Spec 222 (Cartography footprints)

## Operator-originated requirement (verbatim, 2026-09-07)

> "We need a way to fine-tune the overlay/overlap, down to the terrain cell, if possible, as we
> can see the roadway for Moonbrook hidden in the texturing for DeadminesInstance, but it's not
> aligned with the buildings and there's no way to align them. We should also have a way to save
> and lock in the alignments that we do, and save these layer stacks as some sort of project
> data that gets loaded on every launch, once things are established and locked in by the user,
> so the archeology/Cartography feature is more than just a visualizer but also a means of
> restoring data. We need to ensure that the maps it writes are complete maps and not just
> patching the differences, in whatever map format the client uses, as the output format type."

Context: the operator's screenshot (2026-09-07) shows DeadminesInstance composed onto Azeroth at
90° CW + offset (19, −3) with the rotated terrain rendering correctly — the T074 visual gate
passed. The remaining gap is alignment precision, persistence, and export.

## Requirements

### FR-1 — Cell-level alignment fine-tune
- Per-layer sub-tile offset in **terrain cells** (1/16 tile), beyond the existing whole-tile
  offset — so a roadway hidden in a phase's texturing can be nudged until it aligns with the
  buildings of another layer.
- Must compose with the existing whole-tile offset, quarter-turn rotation, and mirrors (a cell
  nudge applies in the layer's own frame, after rotation).
- Terrain channels shift by the cell vector: heightmap, normals, holes, texturing/alpha, MCCV,
  shadows, liquid, area ID. Placements shift by the same world delta.
- Cross-tile content at cell granularity must pull from the neighboring donor tile (a cell shift
  moves content across tile borders); the composition policy owns the math.

### FR-2 — Alignment persistence ("project data")
- The full layer stack per base map (layers, order, enabled/channels, offsets, cell offsets,
  rotation + origin, mirrors, per-tile mappings, presence gates) persists to a project file and
  reloads automatically on launch / on map load.
- Locked layers: an operator can mark an alignment **locked**; locked layers render a locked
  badge and reject accidental edits (unlock is explicit).
- Storage: a per-base-map JSON project file under the viewer's project output area (path per
  `wow-viewer/memory-bank/data-paths.md` conventions), human-editable, diffable.

### FR-3 — Full-map export (restoration output)
- Export composes the ENTIRE base map with its locked layer stack and writes **complete client
  map data** — every tile of the 64×64 grid the base map occupies, fully self-contained ADT set
  (not patches/diffs of the source data).
- Output format: the format the target client uses (Alpha WDT/ADT for 0.x clients; LK ADT for
  3.3.5 via the existing converters where applicable). Export states the format and build.
- Export honors the per-channel gates and presence rules exactly as the live composition does —
  the exported map must match what the viewer renders.
- Export runs off the render thread with progress + re-entrancy guard (the
  `ExportPm4ObjectsObjSet` pattern).

## Acceptance criteria

- SC-1: A cell-level nudge visibly shifts the composed terrain by exactly that many cells in the
  direction the UI states, with cross-tile content carried correctly.
- SC-2: A saved project reloads on launch: offsets, rotation, mirrors, channels, locks — bit-identical
  composition without manual re-entry.
- SC-3: A locked layer ignores accidental channel/offset edits until unlocked.
- SC-4: Exported ADT set loads in the target client (or the project's established client-load
  validation path) with the composed content present on every occupied tile.
- SC-5: Composition parity: the exported tile for any (tx, ty) matches the live-rendered
  composition for the same tile (same policy code path — the exporter must consume
  `PhaseCompositionPolicy`, not a parallel implementation).

## Out of scope

- Free-angle (non-90°) rotation content transforms (remains research R2 from Spec 219).
- Editing terrain content directly (sculpting) — this spec composes existing authored data.
