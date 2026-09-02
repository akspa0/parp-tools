# Tasks: Cross-Map Tile Transplant

**Spec**: [spec.md](./spec.md)

Status key: `[ ]` not started · `[x]` done · `[~]` in progress · `[O]` operator-owned

---

## Phase 0 — Reconcile what already exists (blocking, mostly reading)

- [x] **T001** Audit spec 195's `ChunkTranspositionService.ExtractPayload` / `TransformPayload` /
      apply path for **map awareness**. Determine exactly where a map identity would have to enter for
      the source and target to differ. This is the whole delta; get it right before writing code.
      *(Completed: Map identity enters at source chunkReader boundary and target MCLY texture remapping boundary. See [evidence/phase0-reconciliation.md](./evidence/phase0-reconciliation.md))*
- [x] **T002** Verify 195's rotate/mirror actually transforms **normals and placement rotations**, not
      only vertex positions. FR-005 depends on it. If it does not, that is a defect in 195 to fix
      there, not to work around here.
      *(Completed: Source inspection confirms 195 fails to rotate normals, placement rotations, and placement relative coords. Defect identified in 195 to fix directly. See [evidence/phase0-reconciliation.md](./evidence/phase0-reconciliation.md))*
- [x] **T003** Map `PhaseDataChannel` onto `ChunkTranspositionOptions` (FR-003). One mechanism, one
      direction of derivation. **Do not add a third channel model.**
      *(Completed: Mapped all channels to `PhaseDataChannel`. See [evidence/phase0-reconciliation.md](./evidence/phase0-reconciliation.md))*
- [x] **T004** Confirm 195's chunk-granular offset satisfies the operator's "partial tile offsets"
      (1/16 tile). If sub-chunk precision is meant, stop and re-scope — that needs interpolation and
      is a materially larger piece of work.
      *(Completed: Confirmed chunk granularity (1/16 tile = 33.33 yd) preserves raw vertex/header fidelity without lossy resampling. See [evidence/phase0-reconciliation.md](./evidence/phase0-reconciliation.md))*
- [x] **T005** Establish whether a transplant applies to the live loaded map, to a staged artifact, or
      to the datastore. FR-011 says a transplant is a proposal until applied; this decides where the
      proposal lives.
      *(Completed: Proposal lives in EditorSession as staged payload preview, commits to live map with undo/redo, exports via ARRY/ENDS. See [evidence/phase0-reconciliation.md](./evidence/phase0-reconciliation.md))*

## Phase 1 — Cross-map sourcing (US1) — FR-001..FR-004, FR-010, FR-012

- [ ] **T101** Load a source map's tile independently of the currently loaded target map.
- [ ] **T102** Extract a payload from the source map at a given tile.
- [ ] **T103** Re-map MCLY texture indices from the source's texture table into the target's (FR-010).
      Copying raw indices across maps paints the wrong textures and is the single most likely silent
      corruption in this feature.
- [ ] **T104** Apply the payload to the target under `PhaseDataChannel` selection, with the spec 203
      presence gate so an empty source channel cannot blank the target.
- [ ] **T105** Report every selected tile that could not be transplanted, with the reason (FR-012).
- [ ] **T106** Unit-test extract → transform → apply across two synthetic maps with different texture
      tables, asserting unticked channels are byte-identical before and after (SC-002).

## Phase 2 — Tile picker (US2) — FR-002, FR-009

- [ ] **T201** Pop-out window hosting a 64×64 grid for the source map.
- [ ] **T202** Draw each cell with that tile's minimap, loaded lazily so opening the window does not
      stall (SC-006).
- [ ] **T203** Three visually distinct cell states: no tile, tile with no preview, tile present.
      Measured: 0.5.3 has no loose minimap directory, so "no preview" is common and must not read as
      "no tile".
- [ ] **T204** Drag-select across cells as a group.
- [ ] **T205** Clear or re-key the selection when the source map changes; never carry coordinates
      across maps silently.
- [ ] **T206** Show the target map's grid alongside, so source and destination placement are visible
      together.

## Phase 3 — Transform (US3) — FR-005, FR-006

- [ ] **T301** Surface rotation (0/90/180/270) and mirror X/Y in the transplant UI, backed by 195's
      existing options.
- [ ] **T302** Surface a chunk-granular offset (ΔGx, ΔGy) on the global lattice.
- [ ] **T303** Round-trip test: rotate by R, then by −R, and assert heights return within tolerance
      (SC-003).
- [ ] **T304** Assert normals and placement rotations are transformed, not just positions.

## Phase 4 — Provenance (US4) — FR-007, FR-008, FR-011

- [ ] **T401** Record source map, source tile, channels and transform per target tile.
- [ ] **T402** Preserve ordering when a tile is transplanted more than once.
- [ ] **T403** Export provenance alongside the tile data, through the ARRY handoff so it reaches the
      datastore rather than living only in the UI.
- [ ] **T404** Wire undo/redo through `EditorSession` (FR-008) and assert byte-identical restore
      (SC-005).
- [ ] **T405** Review-before-apply: a transplant is a proposal until committed (FR-011).

## Operator gates

- [O] **T501** Transplant a known instance-map region onto the corresponding main-map location and
      confirm by eye that it is the older terrain.
- [O] **T502** Confirm a rotated/mirrored region can be realigned with the correct inverse transform.
- [O] **T503** Confirm the picker is usable at 64×64 on a real map.

## Deferred

- **Automatic detection of rotated/mirrored copies.** `project_weak_tile_jigsaw`'s edge-agreement
  method is the candidate approach; the operator supplies the transform for now.
- **Seam blending** — spec 196 owns neighbour mesh fitting.
- **WLW/MCLQ convergence** — spec 209.
