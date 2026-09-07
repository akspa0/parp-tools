# Spec 232 — Cartography Composition: Cell-Level Alignment, Project Persistence, Full-Map Export & Layer UI

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

## Operator follow-up directives (2026-09-07, after first render pass)

> "the Archeology tab should default to the map layers feature, since it's a bigger feature than
> simple uniqueID sedimentary layers (which, it would be cool if we could color-code the layers,
> so all objects from a particular range of uniqueID's ends up tinted with a set color, or set
> effect on the object textures, to designate the eras better than the text representation we use
> right now. There's a lot of refinement and effects that we can improve on in the viewer's ui,
> as well as -- MDX objects do not render their effects all the time. Water renders oddly dark,
> and fire is missing almost entirely from materials that are meant to have fire effects applied.
> Not sure why that is. Smoke seems to work fine. Most things are fine, it's just some animated
> effects seem to be missing entirely. Anything that casts light, maybe? We were working on
> improving that aspect and I think we broke the light effects that objects can give off in the
> renderer."

- FR-4 — **Archaeology default tab**: the Archaeology workbench defaults to the Map Layers page
  (the composition feature) rather than the UniqueId sedimentary view.
- FR-5 — **Era color-coding**: objects within a UniqueId range render tinted with a per-range
  color (or texture effect) so sedimentary eras read visually instead of via text labels.
- FR-6 — **MDX effect regression**: water renders oddly dark and fire effects are missing almost
  entirely from materials that should have them (smoke works); suspected regression from the
  lighting-effect work — audit the renderer's object-emissive/light-pipeline changes, find the
  break, and restore fire/water/light-casting effects.
- FR-7 — **Force reload**: tiles do not always load everything after a placement change — provide
  a "force reload" control that evicts and re-streams every tile affected by the current layer
  stack (the existing `EvictAllTiles` path, scoped to affected tiles).
- FR-8 — **Save / import settings buttons**: explicit save + import controls for the layer-stack
  project data (alongside the FR-2 auto-load), so an alignment can be exported, shared, and
  re-imported.
- FR-9 — **Placements follow transforms**: doodads (MDDF) and world objects (MODF) must follow
  the layer's tile offset, cell offset, rotation, and mirror with the SAME transforms as the
  terrain — they are currently missing/misplaced under transformed layers.
- FR-10 — **Texture restoration for stripped phase maps**: DeadminesInstance carries MCAL alpha
  masks for all its MCLY layers but the MCLY texture references were stripped from the chunk.
  Provide a texture-remapping tool that assigns base-map textures to texture-less MCLY layers
  (matched per layer, per tile, against the terrain being overlaid), restoring the 25-year-lost
  appearance of these maps. Persisted in the layer project data (FR-2).
- FR-11 — **Base-map channel gates**: the BASE map gets the same per-channel keep/discard gates
  as phase layers — if the operator does not want the base map's liquids (or shadows, or doodads,
  etc.) in the composition, they can drop them per channel rather than being forced to keep
  everything the base map carries.
- FR-12 — **Minimap synthesis from composed maps**: the synthesized-minimap pipeline must be able
  to consume the COMPOSED, loaded map (base + locked layers, post-transform) as its input — not
  only a folder or client map — so exported/synthesized minimaps show the restored composition.
- FR-3 (restated emphasis) — **Output maps**: build real output maps from the experiments — full
  tile sets written as proper LK ADTs or Alpha WDT/ADT output, in the correct map format for the
  target client, not patches.
- Also recorded: cell-shift composition must move the layer as a RIGID map object (fixed in
  Phase 1's rigid `ResolveCellShiftedChunk` rework, same day).
- Also recorded (UI): right-sidebar page dropdowns must be sticky per top tab — switching to
  Quick and back must return to the page the operator was on, not snap to the first page.

## Out of scope

- Free-angle (non-90°) rotation content transforms (remains research R2 from Spec 219).
- Editing terrain content directly (sculpting) — this spec composes existing authored data.
