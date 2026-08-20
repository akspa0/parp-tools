# Feature Specification: Editor Plugin Host

**Feature Branch**: `166-editor-plugin-host`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**; it carries the measured baselines, hard constraints, and the mistakes not to repeat.
**Depends on**: nothing. This is the epic's foundation.

## Scope

An **Editor** destination in the viewer that lists registered plugins and runs one at a time. Host and
registry only — the sole plugin that ships here is a reference plugin proving registration works.

## User Story - A capability is added by writing a plugin, not by editing ViewerApp (Priority: P1)

A developer adds an editing capability by writing one self-contained plugin class and registering it,
rather than editing `ViewerApp`, the workbench enums, and the navigator switch.

**Independent Test**: Register a minimal reference plugin that draws one line of text. It appears,
opens, closes, and tears down — with zero edits to `ViewerApp` or `Workbench*` beyond the one-line
registration. Removing the registration removes it completely.

**Acceptance Scenarios**:

1. **Given** the viewer runs, **When** the user opens the Editor, **Then** every registered plugin is
   listed; plugins whose requirements are unmet appear as unavailable **with a stated reason**, never
   hidden.
2. **Given** a plugin is listed, **When** activated, **Then** its surface draws and it receives an
   activation callback exactly once before its first draw.
3. **Given** a plugin is active, **When** the user switches away and back, **Then** its state is
   retained and restored.
4. **Given** a plugin throws during draw or update, **When** the frame completes, **Then** the viewer
   stays running, the failure is logged with the plugin's identity, and the plugin is marked faulted
   rather than retried every frame.
5. **Given** a new plugin class is registered, **When** the diff is inspected, **Then** no file under
   `ViewerApp*` or `Workbench*` changed except the registration site.
6. **Given** the loaded client changes, **When** availability is recomputed, **Then** each plugin's
   availability reflects its declared build eras.

### Edge Cases

- Two plugins declare the same identity — registration fails **at startup**, not at first use.
- A plugin faults during teardown — the host completes teardown of the remaining plugins.
- No data source loaded — the Editor renders with data-dependent plugins unavailable, no crash.
- A plugin declares an era range the loaded build falls outside.

## Requirements

### Functional Requirements

- **FR-001**: Expose an Editor surface listing all registered plugins, including unavailable ones with
  a stated reason.
- **FR-002**: Each plugin declares a stable unique identity, display name, description, and supported
  build eras. Duplicate identities fail registration at startup.
- **FR-003**: Drive a defined lifecycle — register, availability change, activate, update/draw,
  deactivate, dispose — each transition observable in logs with the plugin's identity.
- **FR-004**: Contain plugin faults: the viewer continues, the plugin is marked faulted with its error
  retained, and it is not re-invoked until explicitly reset.
- **FR-005**: Plugins access game data **only** through the existing `IDataSource` abstraction. That
  abstraction must suffice for a content-addressed, network-backed source (streaming reads, files
  identified by ID, no local directory listing), so a CASC reader can be added later without changing
  it.
- **FR-006**: Plugin input (keyboard shortcuts, mouse capture) is active only while that plugin is
  active, and must not conflict with existing viewer bindings.
- **FR-007**: Adding a plugin requires no edit to `ViewerApp*` or workbench routing enums beyond one
  registration site.
- **FR-008**: Host contract types live in a shared library under `src/core/`; the viewer app is host
  shell and render surface only (Constitution II).
- **FR-009**: Resolve era-scoped handlers by declared build range, deterministically, with the
  selected handler inspectable at runtime.
- **FR-010**: Recompute every plugin's availability when the loaded client changes.

## Success Criteria

- **SC-001**: A new capability appears in the Editor via one plugin class plus one registration line,
  with **zero** other file modifications.
- **SC-002**: A deliberately faulted plugin does not stop the viewer; remaining plugins keep working.
- **SC-003**: A stub data source of a new kind is consumed by plugins with **no plugin changes**.
- **SC-004**: A stub era-scoped handler for a synthetic build is selected over the default.
- **SC-005**: Duplicate plugin identities fail at startup with both identities named.

## Out of Scope

- The bridge to live scene state ([167](../167-editor-runtime-bridge/spec.md)) — plugins here draw UI
  and read data; they do not mutate the world.
- Undo, dirty state, save arbitration ([168](../168-editor-session-undo/spec.md)).
- Any real editing capability (169, 170, 175).
- Dynamic/out-of-process plugin loading. In-process only; the contract must not *preclude* separate
  assemblies later, but nothing dynamic ships.

## Assumptions

- Plugins are compiled into the solution and registered explicitly.
- "The Editor" is a destination inside the existing viewer, not a separate executable. The contract
  lives in `src/core/`, so a separate shell remains possible later.
