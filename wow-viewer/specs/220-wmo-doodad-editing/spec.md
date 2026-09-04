# Spec 220: WMO Doodad Placement Editing, Custom Doodad Sets & WMO Writing

**Feature Branch**: `220-wmo-doodad-editing` (authored on `v0.5.3`; specs 193–219 follow the same convention)

## Overview & User Intent

Spec 211 made WMO interior doodads selectable and hoverable. The operator now wants to go further:

1. **Edit doodad placements inside WMOs** — move, rotate, scale, add, and delete MODD placements on a loaded WMO, with undo, using the same selection/picking pipeline Spec 211 established.
2. **Author custom doodad sets** — create/save new MODS doodad set definitions (name, MODD range, flags) so an edited WMO can expose alternative furniture configurations the way Blizzard-authored sets do.
3. **Write proper WMO files** — persist the edited WMO back to disk in the format version of the files currently open, so edits are real data, not a session-only overlay.

The non-negotiable ordering: **the internal editing model is V17** (the codebase's canonical WMO read model, produced by `WmoV14ToV17Converter` and consumed by `WmoRenderer`), and **writers target the era of the file that was opened**. Version 14 (Alpha 0.5.3) writing already exists via `WmoV17ToV14Converter.WriteWmoV14`; that is the first writer target because 0.5.3 is the operator's primary client root.

## User Stories

### US-1: Edit doodad placements on a loaded WMO (Priority: P1)
- **As a** world editor,
- **I want** to select a WMO doodad (existing Spec 211 picking) and move, rotate, scale, duplicate, or delete its MODD placement,
- **So that** interior layouts can be rearranged without leaving the viewer.

### US-2: Add new doodad placements from any loadable model (Priority: P1)
- **As a** world editor,
- **I want** to add a new MODD placement referencing any MDX/M2 model the data source can resolve,
- **So that** custom props can be furnished into interiors, not just rearranged.

### US-3: Author and save custom doodad sets (Priority: P1)
- **As a** world editor,
- **I want** to create a new MODS doodad set (name + MODD range + flags), assign placements to it, and switch between sets exactly as the existing set dropdown does,
- **So that** alternative configurations (e.g. "raided", "pre-war") live in the file as data.

### US-4: Save the edited WMO in the open file's version (Priority: P1)
- **As a** world editor,
- **I want** to write the edited WMO back to disk in the version it was read from (V14 first), with a preflighted output path and a byte-level diff summary of what changed,
- **So that** edits survive the session and are loadable by the real client of that era.

### US-5: Round-trip safety gate (Priority: P1)
- **As an** operator,
- **I want** an automated round-trip test that reads a real WMO, writes it unmodified, and diffs the re-read result against the original,
- **So that** the writer cannot silently corrupt chunks it does not understand — a failed round trip blocks the Save action.

## Acceptance Criteria

### AC-001: Placement edit ops
- Moving/rotating/scaling a selected doodad updates `WmoRenderer`'s live `_doodadInstances` transform and the MODD record in the underlying `WmoV14ToV17Converter.WmoV14Data`; the change is visible immediately and survives a WMO reload.
- Delete removes the MODD record and its set references; duplicate clones the record with a new position offset.

### AC-002: New placement
- Adding a placement appends to MODD, extends MODN with the model name (reusing an existing MODN entry when the name already exists), and appends the def index to the active set's MODS range.
- A model that cannot be resolved from the data source is rejected at add-time with a reason, not written as a dangling reference.

### AC-003: Doodad set authoring
- Creating a set appends a MODS record; assigning placements produces a contiguous or explicitly-flagged MODD range exactly as the reader model represents sets today.
- The existing set dropdown lists authored sets without special-casing.

### AC-004: Versioned save
- Save on a V14-opened WMO produces a V14 root file (and group files unchanged — doodad editing touches only root chunks: MODD/MODN/MODS/MOHD counts).
- Save on an era without a proven writer is refused with a named reason (no silent best-effort).

### AC-005: Round-trip gate
- `dotnet test` includes a round-trip suite over at least one real V14 WMO per doodad-bearing shape: read → write → re-read → assert MODD/MODN/MODS/MOHD field equality and chunk-byte equality for untouched chunks.

## Technical Constraints & Invariants

1. **Format readers frozen.** Editing composes on top of `WmoV14ToV17Converter.WmoV14Data`; no reader changes. `AlphaWdtWriter.cs` remains frozen (WDT, not WMO — untouched).
2. **Editor logic lives in `WowViewer.Core.Editor`** (no test project references the viewer). The viewer contributes picking + gizmo UI only; every mutation is a Core editor operation with unit tests.
3. **Undo is mandatory** for every mutating operation, following the existing Core.Editor operation pattern.
4. **No parallel WMO model.** If the V17 data model lacks a field an edit needs, extend the existing converter model — never fork a side copy.
5. **Provenance.** Saved files record source path + build + a change summary sidecar (JSON) next to the output.
6. **Group files are untouched** by this spec; save re-emits group files byte-for-byte from source.
