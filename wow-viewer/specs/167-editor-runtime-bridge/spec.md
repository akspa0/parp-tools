# Feature Specification: Editor ↔ Runtime Bridge

**Feature Branch**: `167-editor-runtime-bridge`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [166](../166-editor-plugin-host/spec.md) — host and plugin lifecycle.

## Scope

The one boundary between the Editor and the running viewer. Plugins read live scene state through it
and apply changes through it; a change lands in the source file and the viewport together.

Proven here with the narrowest useful operation — **translation-only placement moves** through the
existing, already-tested, currently-uncalled `AdtPlacementWriter` — which also retires `ViewerApp`'s
112-reference parallel staging implementation.

## User Story - The Editor changes something the user can see (Priority: P1)

A plugin reads what is selected and loaded in the running scene, expresses a change as an operation,
and the change is written to the source file *and* reflected in the viewport — without the plugin
touching the renderer, and without the renderer knowing the Editor exists.

**Independent Test**: Select a placed object in the viewport, move it, apply. The ADT is written to
the output directory, the object appears at its new position without a full map reload, and undo
restores both file and viewport. No renderer or scene type gains a reference to the Editor.

**Acceptance Scenarios**:

1. **Given** an active plugin, **When** it queries the bridge, **Then** it reads loaded map/tiles,
   selection, and camera **without** referencing the renderer, `WorldScene`, or `ViewerApp`.
2. **Given** an operation is submitted, **When** applied, **Then** the file change and the in-scene
   refresh happen together; a failure in either leaves **both** unchanged.
3. **Given** an operation is applied, **When** the viewport is observed, **Then** only the affected
   region refreshes — no whole-map reload.
4. **Given** every Editor project is removed from the solution, **When** the viewer is built and run,
   **Then** it builds and runs unchanged.
5. **Given** an operation is undone, **When** inspected, **Then** file state and scene state both
   return to pre-operation values.
6. **Given** the migration is complete, **When** the codebase is searched, **Then** no
   `_stagedPlacementEdits`/`_selectedPlacement*` field remains on `ViewerApp`.

### Edge Cases

- An operation requested with nothing selected, or against an unloaded tile.
- A source file that is not writable — reported before anything is written.
- A tile unloads between operation submission and application.
- An operation that cannot be reversed — declared non-undoable rather than faking it.

## Requirements

### Functional Requirements

- **FR-001**: Expose read-only live scene context — loaded map, loaded tiles, selection, camera —
  without exposing renderer, scene, or app types.
- **FR-002**: **No runtime, scene, or renderer type may reference any Editor type.** Enforceable as a
  build-time or test-time check, not a convention.
- **FR-003**: With every Editor project removed, the viewer builds and runs with unchanged
  non-editing behavior.
- **FR-004**: An applied operation updates source file and in-scene representation atomically from the
  user's perspective; partial application is not permitted.
- **FR-005**: Scene refresh after an operation is scoped to the affected region.
- **FR-006**: Operations are expressed as data — what changed, how to reverse it — so the host can
  reverse one without knowing which plugin produced it. This is required by
  [168](../168-editor-session-undo/spec.md)'s shared undo history; it is **not** a concession to any
  external consumer.
- **FR-007**: Placement edits are applied through the existing core placement writer. `ViewerApp`'s
  parallel staging implementation is **removed, not wrapped**.
- **FR-008**: Report inapplicable operations (nothing selected, tile not loaded, file not writable)
  with the reason, before starting.
- **FR-009**: Bridge contract lives in a shared library under `src/core/`.

## Success Criteria

- **SC-001**: Moving a selected object writes the ADT and shows it at its new position without a full
  map reload; undo restores file bytes and viewport together.
- **SC-002**: `ViewerApp.cs` placement-staging references: **112 → 0**.
- **SC-003**: `AdtPlacementWriter` production callers: **0 → 1**, and it is the only placement write
  path.
- **SC-004**: Removing every Editor project leaves a viewer that builds and runs — verified by build
  and smoke run.
- **SC-005**: A dependency check fails if any runtime/scene/renderer type gains an Editor reference.

## Out of Scope

- Full placement authoring — rotation, scale, add, delete
  ([175](../175-placement-authoring/spec.md)). Translation only here, deliberately, as the narrowest
  proof of the whole path.
- Durable journaling of staged operations ([172](../172-editor-edit-journal/spec.md)).

## Assumptions

- `AdtPlacementWriter` is correct — it is unit-tested; this spec makes it reachable. A defect found
  here is fixed in core, benefiting the CLI equally.
