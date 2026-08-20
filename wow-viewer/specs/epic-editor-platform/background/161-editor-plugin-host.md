# Feature Specification: Editor Plugin Host

**Feature Branch**: `161-editor-plugin-host`

**Created**: 2026-08-19

**Status**: Draft

**Input**: User description: "Build a DBC editor into the viewer, and plan for a bunch of editors like it. Ideally we build a plugin system into the existing viewer, call all this functionality the Editor, and move the chunk clipboard/manipulation tools into it as the first part of the plugin." Scope confirmed in session: **plugin host + chunk-clipboard migration + DBC/DB2 editor**, with write support at **read + edit + save loose file**. CASC data sources and era-scoped format plugins are explicitly follow-on specs whose seams must be designed in here. **Amended 2026-08-19** to fold in the Editor↔Runtime bridge after the session finding that the map authoring layer is already library-first in `src/core/` but reachable only from CLI tools — `AdtPlacementWriter` has zero production callers while `ViewerApp` carries 112 references to a parallel staging implementation. World authoring (spec 162) is a follow-on. MCP (spec 163) is an external optional component and imposes nothing on this spec.

## Context

The viewer has accumulated editing capability without an editing architecture. Every tool that
mutates data lives as a partial-class page hung off one god object:
[ViewerApp.cs](../../src/viewer/WoWViewer/ViewerApp.cs) is **15,670 lines** across 24 partial files,
and [WorldScene.cs](../../src/viewer/WoWViewer/Terrain/WorldScene.cs) is another **15,733**. Adding
an editor today means adding a 25th partial file, another block of `_prefixed` fields on `ViewerApp`,
and another hand-wired entry in the workbench tab enums. That cost is now the binding constraint on
the stated goal — "a full-functioned, all-versions WoW editor" — not the format knowledge, which the
repo already has.

### What already exists

| Asset | State | Evidence |
|---|---|---|
| Workbench destination/page routing | Working, but every page is a hardcoded enum member + `switch` arm | [WorkbenchTab.cs](../../src/viewer/WoWViewer/Workbench/WorkbenchTab.cs), [WorkbenchNavigator.cs](../../src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs) |
| Chunk clipboard / selection / paste | Working, but **18 private fields on `ViewerApp`** and **124 `_chunkClipboard*`/`_selectedChunks` references** in `ViewerApp.cs` alone | [ViewerApp.cs:466-481](../../src/viewer/WoWViewer/ViewerApp.cs#L466-L481), [ViewerApp_Sidebars.cs:3731-3850](../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs#L3731-L3850) |
| DBCD (typed DBC/DB2 read **and write**) | Vendored and already referenced by both `WowViewer.Core.IO` and the viewer | [DBCD.csproj ref](../../src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj#L6), `DBCDStorage.Save(string)` at `libs/wowdev/DBCD/DBCD/DBCDStorage.cs:292` |
| WoWDBDefs definitions | Vendored, **1,320 definitions**, already copied to build output | [WoWViewer.csproj:77](../../src/viewer/WoWViewer/WoWViewer.csproj#L77) |
| `IDBCProvider` over the repo's archive boundary | Working; one canonical byte source for viewer and CLI tools | [ArchiveReaderDbcProvider.cs](../../src/core/WowViewer.Core.IO/Dbc/ArchiveReaderDbcProvider.cs) |
| Data-source abstraction | `IDataSource` exists with MPQ + loose implementations, and its own doc comment already names CASC as an intended third | [IDataSource.cs](../../src/viewer/WoWViewer/DataSources/IDataSource.cs) |
| Map authoring / write layer | **3,585 lines already in core** — ADT writing, alpha-WDT writing, both conversion directions | [LkAdtWriter.cs](../../src/core/WowViewer.Core.IO/Maps/LkAdtWriter.cs) (620), [AlphaWdtWriter.cs](../../src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs) (1,314), [AlphaToLkConverter.cs](../../src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs) (667), [LkToAlphaConverter.cs](../../src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs) (784) |
| Transactional placement writing | Written, unit-tested, and **called by nothing** | [AdtPlacementWriter.cs](../../src/core/WowViewer.Core.IO/Maps/AdtPlacementWriter.cs) — sole caller is `tests/WowViewer.Core.Tests/AdtPlacementWriterTests.cs` |

So the DBC editor is not blocked on parsing — DBCD plus 1,320 definitions plus a working provider is
the whole read path, and `Save` is the whole write path. It is blocked on **where a table-editing UI
would live** without adding a 25th partial file. That is what this spec builds.

### Why a host contract, not just a tab

Six named capabilities are queued behind this decision: DBC/DB2 editing, chunk manipulation, CASC
support, era-scoped format readers, and at least two more editors implied by "a bunch of editors like
it." Each one needs the same four things — a place in the UI, access to the loaded game data, a way
to say "I have unsaved changes", and a way to undo. Building those four things once as a host
contract, and proving the contract against **one existing capability** (chunk clipboard) and **one
new capability** (DBC editor), is what makes the seventh editor cheap. Building the DBC editor alone
proves nothing about the seventh.

### The migration is the proof, not a cleanup

Chunk clipboard is deliberately plugin #1. A host contract validated only against code written to
fit it is not validated. The chunk clipboard was written with no such contract, holds state on the
app object, reads keyboard input directly, mutates terrain the renderer owns, and tracks its own
dirty set (`_chunkClipboardDirtyTileChunks`). If the contract can absorb that unchanged in behavior,
it can absorb the editors that come after it. If it cannot, the contract is wrong and this spec finds
out in Phase 2 rather than in spec 168.

### The bridge is the missing half

The write layer is not missing and is not in the wrong place. It is already library-first in
`src/core/`, as the constitution requires — 3,585 lines of ADT/WDT writing and conversion, plus a
transactional placement writer. **Every one of its callers is a CLI tool.** The viewer calls none of
them.

The most direct evidence is `AdtPlacementWriter`: a complete transaction-based MDDF/MODF writer with
passing unit tests and **zero production callers**. Meanwhile `ViewerApp.cs` carries **112
references** to `_stagedPlacementEdits`/`_selectedPlacement*` — a second, parallel staging
implementation limited to translation-only saves, built on the app object because there was no path
from a live scene selection to the core writer.

That is the gap this spec closes. The pattern is the one Unreal draws between Runtime and Editor
modules: the editor depends on the runtime, the runtime never depends on the editor, and everything
the editor does to the world crosses one explicit boundary. Here that boundary — the **bridge** —
answers three questions for every plugin:

1. *What is the scene showing right now?* (loaded map/tiles, selection, camera) — read-only.
2. *Apply this change.* An operation is expressed once and lands in both the source file and the
   scene, or in neither.
3. *Undo it.* The same operation reverses both sides.

Building this once is what stops plugin #3 from growing its own `_stagedPlacementEdits`. The
direction rule is the load-bearing part: **no runtime, scene, or renderer type may reference an
Editor type.** That keeps the viewer shippable without the Editor and keeps plugins genuinely
removable.

### Out of scope (seams required, implementation deferred)

- **CASC data sources** (local install and remote/CDN). This spec must leave `IDataSource` as the
  seam a CASC implementation slots into, and must not let any plugin bypass it, but ships no CASC
  reader. Reference implementation for the follow-on: WoW.Export (TypeScript).
- **Era/version-scoped format plugins.** This spec must establish how a plugin declares which build
  eras it supports and how the host resolves per-era behavior, and must apply that to the two
  plugins it ships. It does not restructure existing format readers behind it.
- **MPQ/CASC repacking or writing into a game install.** Saves go to an output directory only.
  Note this is **not merely out of scope** — writing any Blizzard container is prohibited outright by
  Constitution VII. Reading them is unrestricted; a future CASC data source is a reader with no
  writing counterpart.
- **Out-of-process / third-party plugin loading.** Plugins are in-process and compiled into the
  solution. The contract must not *preclude* separate assemblies later, but nothing dynamic ships.
- **World authoring — ADT creation, object transfer between ADTs/alpha-WDTs, full placement editing**
  (spec 162). This spec proves the bridge with the narrowest useful case, translation-only placement
  moves through the existing `AdtPlacementWriter`, and retires the parallel staging code. It does not
  surface the ADT/WDT writers or converters in the UI.
- **MCP server and client** (spec 163). MCP is an **external, optional component** that sits outside
  the Editor and consumes whatever the Editor already exposes. It imposes no requirement on this
  spec, gets no vote on the host contract or the operation model, and nothing here is shaped to
  accommodate it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Editor host with registered plugins (Priority: P1)

A developer adds a new editing capability by writing one self-contained plugin class and registering
it, rather than by editing `ViewerApp`, the workbench enums, and the navigator switch. The user sees
an **Editor** destination in the viewer listing every registered plugin, and can open one.

**Why this priority**: Nothing else in this spec can be built without it, and it is the entire point
of the feature. A plugin that exists but has no host is just another partial file.

**Independent Test**: Register a minimal reference plugin that draws one line of text. It appears in
the Editor destination, opens, closes, and is torn down — with zero edits to `ViewerApp` or the
workbench enums beyond the one-line registration. Removing its registration removes it completely.

**Acceptance Scenarios**:

1. **Given** the viewer is running, **When** the user opens the Editor destination, **Then** every
   registered plugin is listed with its display name, and plugins whose requirements are unmet are
   listed as unavailable with a stated reason rather than hidden.
2. **Given** a plugin is listed, **When** the user activates it, **Then** its surface is drawn and it
   receives an activation callback exactly once before its first draw.
3. **Given** a plugin is active, **When** the user switches to another plugin, **Then** the first
   plugin is deactivated but retains its state, and reactivating it restores that state.
4. **Given** a plugin throws during draw or update, **When** the frame completes, **Then** the viewer
   stays running, the failure is logged with the plugin's identity, and that plugin is marked faulted
   rather than retried every frame.
5. **Given** a new plugin class is added to the solution, **When** it is registered, **Then** no file
   under `ViewerApp*` or `Workbench*` requires modification other than the registration site.

---

### User Story 2 - Chunk clipboard runs as a plugin (Priority: P1)

The existing terrain chunk copy/paste/selection/save tooling is moved out of `ViewerApp` and
`ViewerApp_Sidebars` and into an Editor plugin. The user reaches it through the Editor rather than
through Experimental > Terrain Lab > Clipboard, and every operation behaves exactly as before.

**Why this priority**: This is the host contract's only real validation. It also removes 18 fields
and 124 references from the largest file in the repo.

**Independent Test**: Run the documented chunk-clipboard workflow (select chunks, copy, move, paste
with each rotation, toggle relative heights / alpha-shadow / textures, save) on a real client tile
before and after migration; heightmap and ADT outputs must be byte-identical.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the user drags to select chunks, copies, and pastes at a target,
   **Then** the result matches pre-migration output byte-for-byte, including all four paste rotations
   and each of the relative-heights, alpha/shadow, and textures toggles.
2. **Given** the plugin is active, **When** the user presses the copy/paste keyboard shortcuts,
   **Then** they work as before; **and when** the plugin is not active, **Then** those shortcuts do
   not fire.
3. **Given** chunks have been pasted, **When** the user checks the Editor's unsaved-changes state,
   **Then** the dirty tiles are reported through the host's dirty-state surface rather than a
   plugin-private counter.
4. **Given** the migration is complete, **When** the codebase is searched, **Then** no
   `_chunkClipboard*` or `_selectedChunks` field remains on `ViewerApp`, and the old Terrain Lab >
   Clipboard page either forwards to the plugin or is removed.

---

### User Story 3 - The Editor acts on the live scene through a bridge (Priority: P1)

A plugin changes something the user can see. It reads what is currently selected and loaded in the
running scene, expresses a change as an operation, and the change is both written to the source file
and reflected in the viewport — without the plugin touching the renderer directly, and without the
renderer knowing the Editor exists.

**Why this priority**: This is the half of the host contract that makes an *editor* rather than a
form. It is also the seam that is measurably missing today (see "The bridge is the missing half"),
and both P1 plugins already need it — chunk clipboard mutates terrain the renderer owns, and the
placement staging code exists solely because there was no bridge to write through.

**Independent Test**: With the existing, already-tested `AdtPlacementWriter` as the write vehicle,
select a placed object in the viewport, move it, and apply. The ADT is written to the output
directory, the object appears at its new position without a full map reload, and undo restores both
the file state and the viewport. No renderer or scene type gains a reference to the Editor.

**Acceptance Scenarios**:

1. **Given** a plugin is active, **When** it queries the bridge, **Then** it can read current scene
   context — loaded map and tiles, current selection, camera position — without holding a reference
   to the renderer, `WorldScene`, or `ViewerApp`.
2. **Given** a plugin submits an operation through the bridge, **When** it is applied, **Then** the
   source-file change and the in-scene refresh happen together, and a failure in either leaves both
   unchanged.
3. **Given** an operation is applied, **When** the viewport is observed, **Then** only the affected
   region is refreshed; a whole-map reload is not required.
4. **Given** the Editor is removed from the build entirely, **When** the viewer is compiled and run,
   **Then** it builds and runs unchanged — no runtime, scene, or renderer type references any Editor
   type in either direction.
5. **Given** a bridge operation is undone, **When** the result is inspected, **Then** both the file
   state and the scene state return to their pre-operation values.
6. **Given** the migration is complete, **When** the codebase is searched, **Then** no
   `_stagedPlacementEdits`/`_selectedPlacement*` field remains on `ViewerApp`, and placement writes go
   through `AdtPlacementWriter` rather than a second staging implementation.

---

### User Story 4 - Browse client tables with real types (Priority: P2)

A user opens the DBC editor, picks a table from the loaded client, and sees a grid with named,
correctly typed columns — not raw uint32 fields. They can search, sort, filter, and click a foreign
key to jump to the referenced row in the referenced table.

**Why this priority**: This is the capability the user asked for, and it is the payoff that makes the
host worth having. Browsing is separable from editing and is useful on its own — the repo already
hand-maintains several narrow DBC readers ([AreaIdMapper.cs](../../src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs)
is 1,093 lines) that a general table browser makes inspectable.

**Independent Test**: Open a 0.5.3 client and a modern CASC-era client's extracted tables; confirm
column names/types match the WoWDBDefs definition for that build, and that navigating a known foreign
key (e.g. an area's parent area) lands on the correct row.

**Acceptance Scenarios**:

1. **Given** a client is loaded, **When** the user opens the DBC editor, **Then** the tables present
   in that client are listed, with tables that have no usable definition for that build shown as
   unavailable with the reason stated.
2. **Given** a table is open, **When** the grid renders, **Then** each column shows its definition
   name and its values are rendered per their declared type (integer, float, string, localized
   string, array element), not as raw words.
3. **Given** a table is open, **When** the user types a search term, **Then** rows are filtered
   across all displayed columns; **and when** the user clicks a column header, **Then** rows sort by
   that column, stably and in both directions.
4. **Given** a cell holds a foreign key with a declared relation, **When** the user activates it,
   **Then** the referenced table opens at the referenced row, and the user can return to where they
   came from.
5. **Given** a table with 100,000+ rows, **When** it is open, **Then** scrolling and searching remain
   responsive, with no full-table materialization per frame.
6. **Given** the same table name exists in two clients of different builds, **When** each is opened,
   **Then** each resolves its own build-appropriate definition and layout, and the build used is
   displayed.

---

### User Story 5 - Edit rows and save a loose table (Priority: P2)

The user changes cell values, adds rows, and deletes rows, then saves the modified table as a loose
`.dbc`/`.db2` file into an output directory. The game install is never written to.

**Why this priority**: Editing is the difference between a viewer and an editor. It is separated from
US3 because a typed browser is independently valuable and independently testable, and because the
write path carries the risk.

**Independent Test**: Load a table, change a known value, add a row, delete a row, save, then reload
the saved file in a fresh session and confirm the changes persisted and the file parses. Round-trip
an *unmodified* table and confirm the output is byte-identical to the input.

**Acceptance Scenarios**:

1. **Given** a cell is editable, **When** the user enters a value, **Then** it is validated against
   the column's declared type and range before being accepted, and rejected values leave the cell
   unchanged with an explanation.
2. **Given** a table is modified, **When** the user views it, **Then** modified cells and added/
   deleted rows are visually distinguished from unmodified ones.
3. **Given** a table is modified, **When** the user saves, **Then** a loose file is written to the
   configured output directory, the file's path is reported, and no file inside the game install is
   created, modified, or deleted.
4. **Given** a table is loaded and immediately saved with no edits, **When** the output is compared
   to the source bytes, **Then** they are identical.
5. **Given** the user adds a row, **When** no ID is supplied, **Then** the editor proposes an unused
   ID; **and when** a duplicate ID is supplied, **Then** the save is refused with the conflict named.
6. **Given** a table is modified, **When** the user closes it or the viewer without saving, **Then**
   they are warned and can cancel.

---

### User Story 6 - One Editor session across plugins (Priority: P3)

Undo, redo, dirty-state, and save are properties of the Editor as a whole, not of each plugin. The
user can undo a chunk paste and a DBC cell edit through the same command, sees one unsaved-changes
indicator covering both, and is warned once on exit.

**Why this priority**: Without this, each new plugin reinvents its own undo stack and its own dirty
flag, which is the exact duplication this spec exists to prevent. It is P3 because both P1/P2
plugins are usable before it lands — but every plugin after them is cheaper because of it.

**Independent Test**: Perform an edit in each plugin, then undo/redo from the host; confirm each
operation reverses correctly and that the global dirty indicator clears only when both plugins are
clean.

**Acceptance Scenarios**:

1. **Given** edits in two different plugins, **When** the user invokes undo repeatedly, **Then**
   operations reverse in reverse chronological order across plugins, and redo replays them forward.
2. **Given** any plugin has unsaved changes, **When** the user views the Editor, **Then** a single
   indicator reports unsaved work and names which plugins hold it.
3. **Given** unsaved changes exist, **When** the user closes the viewer, **Then** they are warned
   once, listing each plugin with pending changes, and can cancel.
4. **Given** a plugin's operation cannot be undone, **When** it is performed, **Then** the plugin
   declares it non-undoable and the host records the point rather than silently offering a broken
   undo.

---

### User Story 7 - Extension seams proven, not promised (Priority: P3)

The two deferred capabilities — CASC data sources and era-scoped format handling — are demonstrably
addable without changing the host contract.

**Why this priority**: The user's stated goal is reach across all game versions. A host that would
need re-architecting to accept CASC has failed at its one job. This story buys that guarantee cheaply
now instead of expensively later.

**Independent Test**: Register a stub data source that reports itself as a distinct kind and serves a
handful of files; confirm both shipped plugins read through it with no plugin code changes. Register
a stub era-scoped handler for a synthetic build and confirm the host selects it over the default.

**Acceptance Scenarios**:

1. **Given** a new data source is registered, **When** plugins request files, **Then** they resolve
   through it without any plugin knowing which kind of source it is.
2. **Given** a plugin declares supported build eras, **When** a client outside that range is loaded,
   **Then** the plugin is listed as unavailable with the build stated, rather than failing at use.
3. **Given** two handlers claim the same format for different eras, **When** a client is loaded,
   **Then** the host selects by declared build range deterministically and the choice is inspectable.
4. **Given** no data source is loaded, **When** the Editor is opened, **Then** it renders with all
   data-dependent plugins marked unavailable, and does not crash or block the viewer.

### Edge Cases

- A table exists in the client but WoWDBDefs has no definition for that build — listed as unavailable
  with the reason, never silently parsed as raw uint32.
- A definition exists but the file's field count or record size disagrees with it — refuse to open
  with both expected and actual layout reported; do not partially parse.
- Two plugins claim the same identity — registration fails loudly at startup, not at first use.
- A plugin faults during teardown — the host completes teardown of the remaining plugins.
- The output directory is unwritable or the disk is full mid-save — the source file is untouched, no
  partial file is left in place, and the failure is reported.
- The user edits a table, switches clients, and returns — pending edits are either preserved against
  the original client or discarded with an explicit warning; they are never silently applied to a
  different build's table.
- A localized string column in a build with a different locale-count than the definition assumes.
- Undo of a chunk paste that spans tiles unloaded since the paste.
- A DB2 with sparse/offset-map storage, or with a non-inline ID column, saved back out.
- The viewer is launched with no client configured at all.

## Requirements *(mandatory)*

### Functional Requirements

**Host contract**

- **FR-001**: The system MUST expose an Editor surface in the viewer that lists all registered
  plugins, including unavailable ones with a stated reason.
- **FR-002**: Each plugin MUST declare a stable unique identity, a display name, a description, and
  the build eras it supports; duplicate identities MUST fail registration at startup.
- **FR-003**: The host MUST drive a defined plugin lifecycle — register, become available/
  unavailable as data sources change, activate, update/draw, deactivate, dispose — and each
  transition MUST be observable in logs with the plugin's identity.
- **FR-004**: A plugin fault MUST be contained: the viewer continues, the plugin is marked faulted
  with its error retained for display, and it is not re-invoked until explicitly reset.
- **FR-005**: Plugins MUST access game data only through the existing data-source abstraction, never
  by opening archives or paths directly.
- **FR-006**: Input handled by a plugin (keyboard shortcuts, mouse capture) MUST only be active while
  that plugin is active, and MUST NOT conflict with existing viewer bindings.
- **FR-007**: Adding a plugin MUST NOT require editing `ViewerApp` partials or the workbench routing
  enums beyond a single registration site.
- **FR-008**: Host contract types and all plugin logic MUST live in shared libraries under
  `src/core/`, with the viewer app acting only as host shell and rendering surface (Constitution II).

**Editor ↔ Runtime bridge**

- **FR-B01**: The bridge MUST expose read-only live scene context to plugins — loaded map, loaded
  tiles, current selection, camera position — without exposing renderer, scene, or app types.
- **FR-B02**: No runtime, scene, or renderer type may reference any Editor type. The dependency runs
  one way only, and this MUST be enforceable as a build-time or test-time check rather than a
  convention.
- **FR-B03**: With every Editor project removed from the solution, the viewer MUST still build and
  run with unchanged non-editing behavior.
- **FR-B04**: An operation applied through the bridge MUST update the source file and the in-scene
  representation atomically from the user's perspective: if either half fails, neither is left
  applied, and the failure is reported.
- **FR-B05**: Scene refresh after an operation MUST be scoped to the affected region; a full map
  reload MUST NOT be required to observe an applied change.
- **FR-B06**: Bridge operations MUST be expressible as Editor Operations (FR-009), so they are
  undoable and so the bridge has one way to apply and reverse a change rather than one per plugin.
- **FR-B07**: Plugin-staged placement edits MUST be applied through the existing core placement
  writer. `ViewerApp`'s parallel staging implementation MUST be removed, not wrapped.
- **FR-B08**: The bridge MUST report when a requested operation is not applicable to current scene
  state (nothing selected, tile not loaded, source file not writable) with the reason, rather than
  failing partway through.

**Session, undo, and saving**

- **FR-009**: The host MUST maintain one cross-plugin undo/redo history; plugins contribute reversible
  operations and MUST be able to declare an operation non-undoable rather than fake its reversal.
- **FR-010**: The host MUST aggregate dirty state across plugins into a single unsaved-changes
  surface that names the plugins holding changes.
- **FR-011**: The system MUST warn before any action that would discard unsaved plugin changes, and
  allow cancellation.
- **FR-011a**: No plugin may write, repack, or emit an MPQ, CASC, or any other Blizzard container,
  in this spec or any later one (Constitution VII). Client *content* formats (ADT/WMO/M2/BLP/DBC/DB2)
  are written directly as loose files; that is the intended output.
- **FR-012**: No plugin may write to a configured game-install path. All writes go to a configured
  output directory. Attempts to write inside a game install MUST be refused and logged.

**Chunk clipboard plugin**

- **FR-013**: All chunk selection, copy, paste, rotation, option toggles, target locking, overlay,
  and save behavior MUST be preserved with byte-identical outputs.
- **FR-014**: Chunk-clipboard state MUST be removed from `ViewerApp` and owned by the plugin.
- **FR-015**: Chunk dirty-tile tracking MUST report through the host dirty-state surface.
- **FR-016**: The prior Terrain Lab > Clipboard entry point MUST either route to the plugin or be
  removed; two divergent implementations MUST NOT coexist.

**DBC/DB2 editor plugin**

- **FR-017**: The editor MUST list tables available in the loaded client and open any table with a
  usable definition for that client's build.
- **FR-018**: Column names and types MUST come from the vendored definitions, resolved against the
  client's build; the resolved build MUST be displayed.
- **FR-019**: A layout mismatch between file and definition MUST refuse the open and report both
  expected and actual layout. Silent fallback to untyped parsing is prohibited.
- **FR-020**: The grid MUST support search across displayed columns, sortable columns, and filtering,
  and MUST remain responsive on the largest tables present in supported clients.
- **FR-021**: Declared foreign keys MUST be navigable to the referenced row, with a way back.
- **FR-022**: Cell edits MUST be type- and range-validated before acceptance; rows MUST be addable
  and deletable; modified/added/deleted rows MUST be visually distinguished.
- **FR-023**: Saving MUST produce a loose `.dbc`/`.db2` in the output directory that reloads
  correctly, and an unmodified load-then-save MUST be byte-identical to the source.
- **FR-024**: Adding a row MUST propose an unused ID; a duplicate or out-of-range ID MUST block the
  save with the conflict named.

**Extension seams**

- **FR-025**: New data-source kinds MUST be registerable and consumable by plugins with no plugin
  changes; the abstraction MUST be sufficient for a content-addressed, network-backed source
  (streaming reads, files identified by ID rather than path, absent local directory listing).
- **FR-026**: The host MUST resolve era-scoped handlers by declared build range, deterministically,
  with the selected handler inspectable at runtime.
- **FR-027**: Every plugin's availability MUST be recomputed when the loaded client changes.

**Validation**

- **FR-028**: All validation MUST be performed against real clients from the configured library
  (`H:\CLIENTS`), covering at minimum one pre-CASC era build and one modern-era build's tables, with
  commands, build identity, and file hashes recorded (Constitution III).

### Key Entities

- **Editor Host**: Owns plugin registration, lifecycle, availability, fault containment, the shared
  undo history, aggregate dirty state, and the output-directory policy.
- **Editor Plugin**: A self-contained editing capability. Has identity, display metadata, supported
  build eras, availability state, a surface to draw, and its own dirty state and undoable operations.
- **Editor Operation**: A single user-initiated change contributed to the shared history, carrying
  what it changed, how to reverse it, or an explicit declaration that it cannot be reversed.
- **Editor Bridge**: The one boundary between the Editor and the running viewer. Serves read-only
  scene context to plugins and applies operations to file and scene together. Referenced by the
  Editor; never referenced by the runtime.
- **Scene Context**: A read-only snapshot of what the viewer is currently showing — loaded map,
  loaded tiles, current selection, camera — with no renderer or app types exposed.
- **Data Source**: An abstract origin for game files (loose, MPQ, later CASC). Plugins never see
  which kind they have.
- **Build Era**: The client-version scope a plugin or format handler declares support for; drives
  availability and handler selection.
- **Table Definition**: The named/typed column layout for one table at one build, resolved from the
  vendored definitions.
- **Table Session**: One open table — its rows, pending edits, added/deleted rows, resolved
  definition, source build, and dirty state.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new editing capability can be added and shown in the Editor by writing one plugin
  class plus one registration line, with **zero** other file modifications — demonstrated by the
  reference plugin.
- **SC-002**: `ViewerApp.cs` no longer contains any chunk-clipboard field or reference; the 18 fields
  and 124 references measured today reach **zero**.
- **SC-003**: The full chunk-clipboard workflow produces outputs **byte-identical** to pre-migration
  outputs across all four rotations and all three option toggles on a real client tile.
- **SC-004**: A user who has never opened the DBC editor can open a client, find a named table, and
  read correctly-labeled typed columns without consulting documentation.
- **SC-005**: Load-then-save of an unmodified table is byte-identical to the source, verified by hash
  on at least 20 tables spanning both a pre-CASC and a modern-era build.
- **SC-006**: An edited table saved and reopened in a fresh session shows exactly the intended
  changes and no others.
- **SC-007**: Search and scroll in the largest table present in the validation clients stay
  interactive with no perceptible stall.
- **SC-008**: A deliberately faulted plugin does not stop the viewer, and the remaining plugins keep
  working.
- **SC-009**: A stub data source and a stub era-scoped handler are each accepted by the host with no
  change to the host contract or to either shipped plugin.
- **SC-010**: No file inside any configured game install is created, modified, or deleted during any
  validation run, verified by hashing the install tree before and after.
- **SC-011**: `ViewerApp.cs` no longer contains any placement-staging field or reference; the 112
  `_stagedPlacementEdits`/`_selectedPlacement*` references measured today reach **zero**, and
  `AdtPlacementWriter` goes from zero production callers to being the single placement write path.
- **SC-012**: Removing every Editor project from the solution leaves a viewer that builds and runs
  with unchanged non-editing behavior — verified by a build and a smoke run.
- **SC-013**: A dependency check fails the build (or a test) if any runtime, scene, or renderer type
  gains a reference to an Editor type.
- **SC-014**: Moving a selected object in the viewport writes the ADT and shows the object at its new
  position without a full map reload, and undo restores file bytes and viewport together.
- **SC-015**: Undo reverses the most recent operation correctly regardless of which plugin produced
  it, across a mixed sequence of at least 10 operations from both plugins.

## Assumptions

- Plugins are in-process and compiled into the solution; no dynamic assembly loading ships here,
  though the contract is kept free of anything that would prevent it.
- "The Editor" is a destination within the existing viewer, not a separate `PARPEditor` executable.
  The user weighed a separate app and a plugin system in-session and chose the plugin system; a
  separate shell remains possible later because the contract lives in `src/core/`.
- DBCD's existing `Save` is the write path; this spec does not write a new DBC/DB2 serializer. If
  DBCD's writer proves unable to round-trip a storage variant present in the validation clients, that
  variant is documented as unsupported for save rather than worked around with a second writer.
- The vendored WoWDBDefs snapshot is the definition source. Refreshing it is a separate operation.
- Foreign-key navigation is limited to relations the definitions declare; no relation inference.
- Undo granularity is one user-initiated operation (a paste, a cell edit, a row add/delete), not
  per-keystroke.
- The bridge is proven here with the narrowest useful operation — translation-only placement moves
  through the existing `AdtPlacementWriter` — because that retires real duplicated code (112
  references) and exercises the full path without pulling spec 162's authoring surface forward.
- "Editor depends on runtime, runtime never depends on editor" is treated as a hard constraint with a
  mechanical check, not a review convention. Without the check it decays on the first deadline.
- Editor Operations are modeled as data (what changed, how to reverse it) rather than as UI
  callbacks, because that is what a shared cross-plugin undo history requires — an operation the host
  can reverse without knowing which plugin produced it. This is settled on its own merits and is not
  a concession to any external consumer.
- Existing narrow DBC readers under `src/core/WowViewer.Core.IO/Dbc/` are left in place; consolidating
  them onto the general table layer is a follow-on, not part of this spec.
- The viewer's current single-client-at-a-time model is retained; multi-client comparison is not
  introduced here.
- Validation clients come from `H:\CLIENTS`; a modern-era client's tables may be supplied as
  extracted loose files, since no CASC reader ships in this spec.
