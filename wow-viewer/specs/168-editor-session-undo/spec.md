# Feature Specification: Editor Session — Undo, Dirty State, Save Arbitration

**Feature Branch**: `168-editor-session-undo`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [166](../166-editor-plugin-host/spec.md). Assumes operations-as-data from
[167](../167-editor-runtime-bridge/spec.md) FR-006.

## Scope

Undo, redo, dirty state, and save policy become properties of **the Editor**, not of each plugin.
Land this before plugin #3 exists, or every plugin grows its own undo stack — which is exactly how
the repo ended up with three duplicated implementations already (see epic).

## User Story - One session across plugins (Priority: P1)

The user undoes a chunk paste and a DBC cell edit through the same command, sees one unsaved-changes
indicator covering both, and is warned once on exit.

**Independent Test**: Perform edits in two different plugins, then undo/redo from the host; each
reverses correctly and the global dirty indicator clears only when both plugins are clean.

**Acceptance Scenarios**:

1. **Given** edits in two plugins, **When** the user invokes undo repeatedly, **Then** operations
   reverse in reverse chronological order **across** plugins; redo replays them forward.
2. **Given** any plugin has unsaved changes, **When** the user views the Editor, **Then** a single
   indicator reports unsaved work and **names which plugins hold it**.
3. **Given** unsaved changes exist, **When** the user closes the viewer, **Then** they are warned
   once, listing each plugin with pending changes, and can cancel.
4. **Given** an operation cannot be undone, **When** it is performed, **Then** the plugin declares it
   non-undoable and the host records the point rather than offering a broken undo.
5. **Given** a save is requested, **When** it runs, **Then** all writes land in the configured output
   directory and the written files are reported.

### Edge Cases

- Undo of an operation whose plugin has since become unavailable.
- Two plugins with edits to the same underlying file.
- Undo history growth during a long session.
- A save that succeeds for one plugin and fails for another.

## Requirements

### Functional Requirements

- **FR-001**: Maintain one cross-plugin undo/redo history. Plugins contribute reversible operations
  and may declare an operation non-undoable rather than fake its reversal.
- **FR-002**: Aggregate dirty state across plugins into a single unsaved-changes surface that names
  the plugins holding changes.
- **FR-003**: Warn before any action that would discard unsaved plugin changes, and allow
  cancellation.
- **FR-004**: No plugin may write to a configured game-install path. All writes go to a configured
  output directory. Attempts to write inside a game install are refused and logged.
- **FR-005**: No plugin may write, repack, or emit an MPQ, CASC, or any other Blizzard container — in
  this spec or any later one (**Constitution VII**). Client *content* formats (ADT/WMO/M2/BLP/DBC/DB2)
  are written directly as loose files; that is the intended output.
- **FR-006**: Undo granularity is one user-initiated operation (a paste, a cell edit, a row
  add/delete), not per-keystroke.
- **FR-007**: Session logic lives in a shared library under `src/core/`.

## Success Criteria

- **SC-001**: Undo reverses the most recent operation correctly regardless of which plugin produced
  it, across a mixed sequence of at least 10 operations from two plugins.
- **SC-002**: The dirty indicator clears only when every plugin is clean.
- **SC-003**: Closing with unsaved changes warns exactly once and lists every plugin holding changes.
- **SC-004**: No file inside any configured game install is created, modified, or deleted during any
  validation run — verified by hashing the install tree before and after.

## Out of Scope

- Durable/crash-safe persistence of the history ([172](../172-editor-edit-journal/spec.md)). This spec
  holds the session in memory; 172 makes it survive a crash.

## Assumptions

- Operations are data with a reverse. If they are not, this spec cannot be built as written — which is
  why that requirement sits in 167 rather than here.
