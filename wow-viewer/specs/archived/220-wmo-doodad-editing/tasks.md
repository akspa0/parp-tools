# Spec 220: Tasks — WMO Doodad Placement Editing, Custom Doodad Sets & WMO Writing

## Phase 0: Round-trip gate (blocking, no UI)

- [ ] 220-T001: Build the round-trip test harness in [`tests/WowViewer.Core.Tests/Editor/WmoDoodadRoundTripTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/Editor/WmoDoodadRoundTripTests.cs): locate a real doodad-bearing V14 WMO under the staged client fixtures; read → `WmoV17ToV14Converter` write → re-read.
- [ ] 220-T002: Assert field equality on MODD/MODN/MODS/MOHD and byte equality on all untouched chunks (groups included). Record which chunks the current writer regenerates.
- [ ] 220-T003: Fix [`WmoV17ToV14Converter`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Converters/WmoV17ToV14Converter.cs) only where the gate fails; no behavioral changes beyond round-trip correctness.
- [ ] 220-T004: Gate: round-trip suite green on at least one real WMO; document the fixture path + SHA256 in `evidence/phase0-round-trip.md`.

## Phase 1: Core edit operations (placement transform + delete/duplicate)

- [ ] 220-T101: Add [`WmoDoodadEditOperations`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.Editor/Operations/WmoDoodadEditOperations.cs): `MovePlacement`, `RotatePlacement`, `ScalePlacement`, `DeletePlacement`, `DuplicatePlacement` — each mutating `WmoV14Data` and returning a undo command (before/after record snapshots only).
- [ ] 220-T102: Add [`WmoDoodadUndoStack`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.Editor/WmoDoodadUndoStack.cs) with apply/undo/redo over the command records.
- [ ] 220-T103: Tests: every op mutates exactly the targeted MODD fields; undo restores them; delete also removes the def index from any MODS range covering it (or fails with a named reason if that would empty a set the operator did not ask to shrink).
- [ ] 220-T104: Gate: focused Editor tests green; full Core build 0 errors.

## Phase 2: Doodad set authoring (MODS) + MODN reuse

- [ ] 220-T201: Add `CreateDoodadSet(name, flags)` appending a MODS record; `AssignPlacementsToSet(defIndices, setId)` computing the `(StartIndex, Count)` range and failing loudly on non-contiguity unless explicitly flagged.
- [ ] 220-T202: Add MODN name-table helper: resolve-or-append model names, reusing byte offsets for existing entries.
- [ ] 220-T203: Tests: set ranges stay reader-compatible (verify against [`WmoDoodadDetailReader.ReadSets`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadDetailReader.cs)); name-table append/reuse round-trips.
- [ ] 220-T204: Gate: focused tests green; round-trip suite re-run with an authored set present.

## Phase 3: Renderer hook + live editing

- [ ] 220-T301: Expose `WmoRenderer.ReloadDoodadsFromModel()` re-running the existing `LoadActiveDoodadSet` path; expose read-only accessors for the loaded `WmoV14Data`, source path, and detected version.
- [ ] 220-T302: Wire Spec 211 selection to the editor session: selecting a `WmoDoodad` focuses the edit panel; gizmo/field edits call Core ops then `ReloadDoodadsFromModel()`.
- [ ] 220-T303: Gate: build green; operator can move a doodad and see it move without reload; undo reverts both data and render.

## Phase 4: Add-from-model + versioned save

- [ ] 220-T401: `AddPlacement(modelPath, position, rotation, scale)` with data-source resolution gate (reject unresolvable models at add-time).
- [ ] 220-T402: Save flow: preflight output path, write V14 root via the proven writer, re-emit group files byte-for-byte from source, emit provenance sidecar (source path, build, change summary, SHA256s).
- [ ] 220-T403: Refuse save for eras without a proven writer, naming the missing writer.
- [ ] 220-T404: Gate: save a real edited WMO, load it back in the viewer, confirm edits persist; round-trip suite extended to cover an edited file.
- [ ] 220-T405: Register Spec 220 in [`specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md); update [`memory-bank/activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md) in the same pass.
- [ ] 220-T406: Operator interactive verification: move/rotate/add a doodad in a placed 0.5.3 WMO, author a set, save, reload, verify.
