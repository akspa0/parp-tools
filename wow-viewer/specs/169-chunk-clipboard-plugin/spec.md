# Feature Specification: Chunk Clipboard Plugin (Migration)

**Feature Branch**: `169-chunk-clipboard-plugin`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [166](../166-editor-plugin-host/spec.md), [167](../167-editor-runtime-bridge/spec.md),
[168](../168-editor-session-undo/spec.md).

## Scope

Move the existing terrain chunk copy/paste/selection/save tooling out of `ViewerApp` and
`ViewerApp_Sidebars` into an Editor plugin, with **no behavior change**.

**This is the host contract's only real validation.** A contract validated only against code written
to fit it is not validated. The chunk clipboard predates any contract: app-object state, direct
keyboard reads, renderer-owned terrain mutation, its own dirty set. If the contract absorbs that with
byte-identical output, it will absorb editor #7. If it cannot, the contract is wrong — and finding
that out here is the point of doing it third rather than last.

## User Story - The chunk clipboard runs as a plugin (Priority: P1)

The user reaches chunk tooling through the Editor rather than Experimental > Terrain Lab > Clipboard,
and every operation behaves exactly as before.

**Independent Test**: Run the documented workflow (select chunks, copy, move, paste with each
rotation, toggle relative heights / alpha-shadow / textures, save) on a real client tile before and
after migration. Heightmap and ADT outputs must be **byte-identical**.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the user drags to select chunks, copies, and pastes at a target,
   **Then** the result matches pre-migration output byte-for-byte — across all four paste rotations
   and each of the relative-heights, alpha/shadow, and textures toggles.
2. **Given** the plugin is active, **When** the user presses the copy/paste shortcuts, **Then** they
   work as before; **and when** the plugin is not active, **Then** those shortcuts do not fire.
3. **Given** chunks have been pasted, **When** the user checks unsaved-changes state, **Then** dirty
   tiles are reported through the host's dirty-state surface, not a plugin-private counter.
4. **Given** the migration is complete, **When** the codebase is searched, **Then** no
   `_chunkClipboard*` or `_selectedChunks` field remains on `ViewerApp`.
5. **Given** the old Terrain Lab > Clipboard page, **When** the user opens it, **Then** it either
   routes to the plugin or is gone — two divergent implementations must not coexist.

### Edge Cases

- Paste target spanning two tiles, one of which is not loaded.
- Selection made, then the map is switched.
- Undo of a paste after the affected tile unloaded.
- Save while the viewer is actively streaming the affected tile.

## Requirements

### Functional Requirements

- **FR-001**: Preserve all chunk selection, copy, paste, rotation, option toggles, target locking,
  overlay, and save behavior with **byte-identical outputs**.
- **FR-002**: Chunk-clipboard state is owned by the plugin and removed from `ViewerApp`.
- **FR-003**: Chunk dirty-tile tracking reports through the host dirty-state surface (168).
- **FR-004**: The prior Terrain Lab > Clipboard entry point routes to the plugin or is removed.
- **FR-005**: Terrain mutation goes through the bridge (167), not direct renderer access.
- **FR-006**: Paste and save become undoable Editor Operations under 168.

## Success Criteria

- **SC-001**: `_chunkClipboard*`/`_selectedChunks` references in `ViewerApp.cs`: **124 → 0**; fields:
  **18 → 0**.
- **SC-002**: The full workflow produces outputs **byte-identical** to pre-migration outputs across
  all four rotations and all three option toggles, on a real client tile.
- **SC-003**: Exactly one chunk-clipboard implementation exists after this spec.
- **SC-004**: Shortcuts fire only while the plugin is active.

## Out of Scope

- Any new terrain capability. Behavior change of any kind is a **failure** of this spec, not a bonus.
- Freehand terrain sculpting (height brushes, hole editing, texture painting) — a later spec.

## Assumptions

- Pre-migration outputs are captured **before** work starts, as the regression oracle. Without that
  baseline this spec cannot be validated.
