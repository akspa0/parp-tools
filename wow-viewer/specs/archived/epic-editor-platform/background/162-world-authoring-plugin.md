# Feature Specification: World Authoring Plugin

**Feature Branch**: `162-world-authoring-plugin`

**Created**: 2026-08-19

**Status**: Draft

**Input**: User description: "Since we have tooling to create new ADT files with objects transferrable from existing ADT's or alphaWDT's, we should be able to move that functionality over to the plugin, or make that a bridge between the editor and viewer portions."

**Depends on**: [Spec 161 — Editor Plugin Host](../161-editor-plugin-host/spec.md). This spec is the
first consumer of the bridge that 161 establishes; it does not define plugin or bridge contracts.

## Context

The map authoring capability already exists, is already library-first, and is already tested. What it
has never had is a user.

### The write layer, and who calls it

| Capability | Lines | Location | Production callers |
|---|---|---|---|
| ADT writing | 620 | [LkAdtWriter.cs](../../src/core/WowViewer.Core.IO/Maps/LkAdtWriter.cs) | CLI only — converter, inspect, harvest |
| Alpha-WDT writing | 1,314 | [AlphaWdtWriter.cs](../../src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs) | CLI only — converter, inspect |
| Alpha → LK conversion | 667 | [AlphaToLkConverter.cs](../../src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs) | CLI only |
| LK → Alpha conversion | 784 | [LkToAlphaConverter.cs](../../src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs) | CLI only |
| Monolithic → split ADT | — | [SplitAdtToLkCommand.cs](../../tools/converter/WowViewer.Tool.Converter/SplitAdtToLkCommand.cs) | CLI only |
| Transactional placement writing | 200 | [AdtPlacementWriter.cs](../../src/core/WowViewer.Core.IO/Maps/AdtPlacementWriter.cs) | **None.** Sole caller is its unit test |

3,585 lines of correct, constitution-compliant authoring code, and the only way to reach any of it is
a command line. Meanwhile the viewer — the one place where a user can *see* which tile they mean and
*point at* the object they want to move — has no path to it.

Spec 161 closes the last row of that table: it wires `AdtPlacementWriter` to a live selection through
the bridge and deletes the 112-reference parallel staging implementation on `ViewerApp`. **This spec
closes the rest.**

### Why this is a plugin and not more CLI flags

The authoring operations that matter are the ones that need a *spatial* decision: which tile, which
object, where it lands, does it look right afterwards. Those are exactly the decisions a command line
is worst at and a viewport is best at. `convert-split-adt-to-lk` is a fine batch operation;
"take that building and put it in the next tile over" is not a flag.

The CLI commands stay. They are the right tool for bulk and pipeline work, they are the regression
oracle for this plugin (identical inputs must produce identical bytes), and spec 163 exposes them to
automation. This spec adds the interactive path over the same core functions — never a second
implementation of them.

### Era coupling is a first-class concern

Alpha-WDT and split-ADT are not two file formats; they are two eras of the same world. Object
transfer between them crosses that boundary — a placement in a 0.5.3 alpha WDT and a placement in a
3.3.5 split ADT differ in coordinate handling, chunk layout, and what a "tile" even is. This plugin
declares its supported eras through the host's era mechanism (spec 161 FR-026) and must refuse a
transfer it cannot perform faithfully rather than perform it approximately.

### Out of scope

- **Terrain sculpting** (height brushes, hole editing, texture painting). Chunk-level copy/paste is
  already covered by the chunk clipboard plugin from spec 161; freehand sculpting is a later spec.
- **Creating new maps** (new WDT with new map entry, DBC map registration). This spec creates and
  edits ADT tiles within an existing map.
- **Writing into a game install, MPQ, or CASC.** Container writing is prohibited outright by Constitution VII, not merely deferred. All output goes to an output directory, per the
  spec 161 host policy.
- **Model/WMO asset authoring.** Objects are placed and transferred by reference; their assets are
  not modified.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Author placements in the viewport (Priority: P1)

The user selects a placed doodad or WMO in the scene, moves/rotates/scales it, adds a new placement
by picking a model and a spot, or deletes one — and saves the affected ADT to the output directory.

**Why this priority**: It is the smallest complete authoring loop, it is the one the viewer is
uniquely good at, and it extends the narrow translation-only proof from spec 161 into a real editing
capability. Nothing else here is useful without it.

**Independent Test**: On a real map tile, move one object, rotate a second, add a third, delete a
fourth, save; reload the written ADT in a fresh session and confirm all four changes are present and
nothing else changed.

**Acceptance Scenarios**:

1. **Given** an object is selected in the viewport, **When** the user edits its position, rotation, or
   scale, **Then** the change is visible immediately and staged as an undoable operation.
2. **Given** a model or WMO is chosen, **When** the user places it in the world, **Then** a new
   placement entry is created with a unique ID that does not collide with any existing ID in the map.
3. **Given** a placement is selected, **When** the user deletes it, **Then** it disappears from the
   scene and its entry is removed on save; undo restores both.
4. **Given** staged placement edits, **When** the user saves, **Then** the affected ADTs are written
   to the output directory and unaffected chunks in those files are byte-identical to the source.
5. **Given** an edit is made to an object whose source file is not writable, **When** the user
   attempts to save, **Then** the reason is reported before any file is written.
6. **Given** placements are edited across several tiles, **When** the user saves, **Then** every
   affected tile is written and the set of written files is reported.

---

### User Story 2 - Transfer objects between tiles and between eras (Priority: P1)

The user selects one or more placed objects in a source tile — from a loaded ADT or from an alpha WDT
— and transfers them into a target tile, which may be in a different map or a different era's format.
Referenced model/WMO names are carried across so the target file resolves them.

**Why this priority**: This is the capability the user named, and it is the one with no CLI
equivalent — the object-name tables (MMDX/MMID/MWMO/MWID) must be merged into the target, and IDs
reconciled, which is exactly the fiddly part that makes it worth automating.

**Independent Test**: Transfer a known set of objects from an alpha-WDT tile into a split-ADT tile in
another map; confirm the target renders them at the correct world positions and that the written ADT
loads in an independent tool.

**Acceptance Scenarios**:

1. **Given** objects are selected in a source tile, **When** the user transfers them to a target tile,
   **Then** their placements appear in the target at the intended world positions.
2. **Given** transferred objects reference models not present in the target, **When** the transfer is
   applied, **Then** the target's model/WMO name tables gain the needed entries and all index
   references are correct.
3. **Given** transferred placement IDs collide with existing IDs in the target, **When** the transfer
   is applied, **Then** new non-colliding IDs are assigned and the remapping is reported.
4. **Given** source and target are different eras, **When** the transfer is applied, **Then**
   coordinates and rotations are converted correctly for the target era; **and if** the conversion
   cannot be performed faithfully, **Then** the transfer is refused with the reason, not approximated.
5. **Given** a transfer would place an object outside the target tile's bounds, **When** it is
   attempted, **Then** the user is warned with the offending objects named.
6. **Given** a transfer is applied, **When** the user undoes it, **Then** the target returns to its
   prior state including its name tables.

---

### User Story 3 - Create new ADT tiles (Priority: P2)

The user creates a new ADT tile in an existing map — empty, or seeded from an existing tile — and it
appears in the world, is saved to the output directory, and is registered in the map's tile index so
the client and the viewer both find it.

**Why this priority**: Object transfer is most useful when there is somewhere new to transfer *to*.
It is P2 because transfers between existing tiles (US2) deliver value first, and because tile-index
registration is the part most likely to need iteration.

**Independent Test**: Create a new tile at an unoccupied coordinate, transfer objects into it, save,
and confirm the tile loads with its objects in a fresh session and its existence is reflected in the
map's tile index.

**Acceptance Scenarios**:

1. **Given** an unoccupied tile coordinate, **When** the user creates a tile there, **Then** a valid
   ADT is produced that loads without error in the viewer and in an independent tool.
2. **Given** an existing tile is chosen as a seed, **When** a new tile is created from it, **Then**
   the new tile's terrain matches the seed and its placements are either copied or omitted per the
   user's choice.
3. **Given** a tile is created, **When** the map is reloaded, **Then** the map's tile index reflects
   the new tile and the viewer streams it like any other.
4. **Given** the target coordinate is already occupied, **When** creation is attempted, **Then** it is
   refused with the existing tile named; overwriting requires explicit confirmation.
5. **Given** a tile is created in a map whose era uses a monolithic WDT rather than split ADTs,
   **When** the tile is saved, **Then** it is written in that era's correct form.

---

### User Story 4 - Convert between eras from the viewer (Priority: P3)

The user runs the existing alpha↔LK and split-ADT conversions from the Editor, on the currently
loaded map, and inspects the result in the viewport before committing it.

**Why this priority**: The conversions already work from the CLI, so this is convenience rather than
new capability — but "see it before you keep it" is precisely what the viewer adds, and it removes
the last reason to leave the app for an authoring task.

**Independent Test**: Convert a loaded map through a round trip from the Editor and from the CLI with
the same inputs; the outputs must be byte-identical.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the user runs a conversion from the Editor, **Then** the output is
   byte-identical to the equivalent CLI command's output on the same inputs.
2. **Given** a conversion is running, **When** the user watches, **Then** progress and the current
   file are reported and the viewer stays responsive.
3. **Given** a conversion completes, **When** the user chooses to preview, **Then** the result can be
   loaded in the viewport without overwriting the source.
4. **Given** a conversion fails partway, **When** it aborts, **Then** partial output is either removed
   or clearly marked incomplete, and the source is untouched.

### Edge Cases

- Transferring an object whose model name exists in the target under a different index.
- A source alpha WDT and a target ADT that disagree on tile origin conventions.
- Placement ID space exhausted, or a source using IDs that are reserved in the target era.
- Deleting a placement referenced by something else in the tile.
- Saving a tile while the viewer is actively streaming it.
- A tile created at a coordinate the map's WDT index cannot represent.
- Objects selected across two tiles, transferred as one operation, where one target write fails.
- An alpha WDT whose per-tile data is present but whose object tables are empty.
- Undo of a transfer after the target file has been modified by something else.

## Requirements *(mandatory)*

### Functional Requirements

**Placement authoring**

- **FR-001**: Users MUST be able to select placed doodads and WMOs in the viewport and edit position,
  rotation, and scale, with changes visible immediately.
- **FR-002**: Users MUST be able to add a new placement by choosing a model/WMO and a world position,
  and to delete an existing placement.
- **FR-003**: New placements MUST receive IDs that do not collide with any existing ID in the map.
- **FR-004**: Every authoring action MUST be an undoable Editor Operation under the spec 161 session.
- **FR-005**: Saving MUST write only the affected tiles; chunks not touched by an edit MUST be
  byte-identical to the source.
- **FR-006**: All writes go through the existing core writers. This spec MUST NOT add a second ADT,
  WDT, or placement serializer.

**Object transfer**

- **FR-007**: Users MUST be able to transfer selected placements from a source tile to a target tile,
  including across maps.
- **FR-008**: Transfer MUST merge referenced model/WMO name tables into the target and correct all
  index references.
- **FR-009**: Colliding IDs MUST be remapped, and the remapping reported to the user.
- **FR-010**: Cross-era transfer MUST convert coordinates and rotations correctly for the target era,
  or be refused with the reason stated. Approximate transfer is prohibited.
- **FR-011**: Transfers spanning multiple target tiles MUST apply as one operation — all targets are
  written or none are.

**Tile creation**

- **FR-012**: Users MUST be able to create an ADT tile at an unoccupied coordinate, empty or seeded
  from an existing tile.
- **FR-013**: Created tiles MUST be valid for the map's era and MUST load in the viewer and in an
  independent tool.
- **FR-014**: Tile creation MUST update the map's tile index so the tile is discoverable.
- **FR-015**: Creation at an occupied coordinate MUST require explicit confirmation.

**Conversions**

- **FR-016**: The existing alpha↔LK and split-ADT conversions MUST be invocable from the Editor on
  the loaded map, producing output byte-identical to the equivalent CLI invocation.
- **FR-017**: Long-running conversions MUST report progress and MUST NOT block the viewer's frame
  loop.
- **FR-018**: A failed conversion MUST leave the source untouched and MUST NOT leave unmarked partial
  output.

**Boundaries**

- **FR-019**: All output MUST go to the configured output directory. No write into any game install.
- **FR-019a**: Authored output MUST be loose client-content files (ADT/WDT/WMO/M2/BLP). Packaging any
  of it into an MPQ or other Blizzard container is prohibited (Constitution VII) — including as a
  convenience for testing in a real client.
- **FR-020**: The plugin MUST declare its supported build eras and MUST be unavailable, with the
  reason stated, for maps outside them.
- **FR-021**: All scene reads and applied changes MUST go through the spec 161 bridge; the plugin MUST
  NOT reference renderer, scene, or app types directly.

**Validation**

- **FR-022**: Every authoring path MUST be validated against real clients from `H:\CLIENTS`, covering
  at minimum one alpha-era map and one split-ADT-era map, with commands, build identity, and file
  hashes recorded.
- **FR-023**: Written files MUST be verified to load in at least one tool outside this repo, not only
  in the viewer that wrote them.

### Key Entities

- **Placement**: One placed doodad or WMO — its model reference, unique ID, position, rotation,
  scale, and owning tile.
- **Placement Edit**: A staged change to a placement (move, rotate, scale, add, delete), undoable and
  not yet written.
- **Transfer**: A set of placements moving from a source tile/era to a target tile/era, carrying the
  name-table merges and ID remapping the move requires.
- **Tile Authoring Target**: The tile being created or written — its coordinate, map, era, source file
  when one exists, and output path.
- **Name Table Merge**: The reconciliation of model/WMO name lists and index references when
  placements arrive from another file.
- **Conversion Job**: A long-running era conversion — inputs, output directory, progress, and result.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can move, rotate, add, and delete placements and save a working ADT without
  leaving the viewer or typing a command.
- **SC-002**: Saving a tile with one placement edited leaves every unedited chunk byte-identical to
  the source, verified by chunk-level comparison.
- **SC-003**: Objects transferred between tiles render at the intended world positions, verified
  visually and by reading back the written placement values.
- **SC-004**: A cross-era transfer either round-trips correctly or is refused; no transfer produces
  silently wrong coordinates.
- **SC-005**: A newly created tile loads in the viewer and in at least one independent tool.
- **SC-006**: Editor-run conversions produce byte-identical output to the equivalent CLI command on
  the same inputs, verified by hash on at least 3 maps.
- **SC-007**: The count of ADT/WDT/placement serializers in the repo stays at its current value —
  this spec adds zero.
- **SC-008**: Every authoring action is undoable, verified across a mixed sequence of at least 15
  operations including a multi-tile transfer.
- **SC-009**: No file inside any configured game install is created, modified, or deleted during any
  validation run, verified by hashing the install tree before and after.
- **SC-010**: The viewer remains responsive during a full-map conversion — no frame-loop stall.

## Assumptions

- Spec 161 has landed. The plugin host, the bridge, Editor Operations, undo, and the output-directory
  policy all come from it and are not redefined here.
- The existing core writers and converters are correct. Where this spec finds a defect in them, it is
  fixed in core and the CLI benefits equally — not worked around in the plugin.
- CLI commands remain the supported path for bulk and pipeline work and are the regression oracle for
  the interactive path.
- "Era" means the build-era mechanism spec 161 establishes; this spec declares support for the eras
  its validation clients cover and refuses others rather than guessing.
- Terrain data (heights, textures, holes) is copied, not synthesized, when a tile is seeded from
  another tile. Generating terrain is out of scope.
- Placement unique IDs are allocated within the map being edited; global cross-map uniqueness is not
  attempted. Related prior finding: uniqueId is the world-layout chronology of record, so remapping
  is reported rather than silent.
- No CASC-sourced map is required for validation; if spec 161's follow-on CASC reader is not yet
  present, modern-era validation uses extracted loose files.
