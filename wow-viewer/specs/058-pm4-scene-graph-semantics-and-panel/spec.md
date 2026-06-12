# Feature Specification: PM4 Scene Graph Semantics and Panel

**Feature Branch**: `058-pm4-scene-graph-semantics-and-panel`
**Created**: 2026-06-10
**Status**: Draft
**Owner**: wow-viewer (viewer + PM4 research lib)
**Related**: spec 046 (PM4 asset matching), spec 044 (viewer shell usability — provides the dockable panel infrastructure), spec 057 (client archive version selector — adjacent, not blocking)

**Input**: User description: "we have to start splitting up the scene graph by using the 0xAA value as a bucket for types of objects, 0xNNBBCC is our object ID, I believe - chatGPT seems to think that the BB and CC are two distinct values, not a single value. The original name of this field was PackedParams, and was noted to be a 24-bit datastructure, and I've believed that chatGPT's interpretation is probably spot on, but it may pertain to index values in the unknowns or Subtype data that we do not currently link or attribute to anything important - yet. This pertains to the scene graph entirely. ... we noticed that the 24-bit number is AABBCC, and there's padding after it that is always 00. ... the ObjectID at least gives us multiple of the same object, which implies that maybe the BBCC values can be some sort of connection between the two, as a bond of some sort, hidden in the ck24 data that we don't realize yet - Much like the AA bytes being the Type of object/mesh."

## Context

The PM4 scene-graph panel (currently embedded as a `CollapsingHeader` inside a sidebar) renders a `Pm4SelectedObjectGraphInfo` tree on every ImGui frame. As of the 2026-06-10 freeze fix (`42e83488`), the graph is cached by selection and no longer rebuilds per-frame, but the panel surface itself is still hidden, small, and gives no visual preview of the object. The user wants:

1. The scene-graph promoted to a **separate dockable panel**, expanded by default, with the layout and discoverability of a Blender outliner.
2. An **image preview** of the PM4 mesh inside the panel (also useful as a training-data input for visual object identification models later).
3. A formal capture of the **24-bit CK24 data model** the user has been sitting on — specifically the user's hypothesis that the `0xBB` and `0xCC` bytes of the 24-bit CK24 are not a single 16-bit ID but a **pair** that may carry a hidden bond between multi-instance objects, parallel to how `0xAA` is the type bucket.

A second, smaller, **regression** in the current build also belongs here: the 30-second click freeze on multi-instance PM4 containers, fixed today by commit `42e83488` (graph info cache). The cached-graph behavior is a prerequisite for the panel-redesign work, so the spec records it as already-shipped, not as a new task.

## Hard Constraints (read first)

1. **No `H:\CLIENTS` references** anywhere in new code, scripts, tests, or docs (RULE 9).
2. **No re-touching of frozen `AlphaWdtWriter.cs`** (RULE 10).
3. **The PM4 research surface is the source of truth** for the data model. New spec body MUST reference and align with the existing `Pm4Research*` analyzers, not duplicate or contradict them.
4. **Per-frame cost of the panel MUST stay O(cached)**. The current per-frame graph rebuild was the root cause of the 30s freeze; any redesign that re-introduces a per-frame rebuild regresses the fix.
5. **The unknowns/subtype linkage hypothesis is a research thread, not a settled fact.** Spec 058 captures it as a question to investigate, not as a behavior change in production code.

## Data Model (the formal capture)

This is the user's session-derived model of the PM4 24-bit identity. The user is the authority; the spec captures it so it doesn't get lost.

### Source

`MSUR._0x1C` is a 32-bit word in the PM4 file format. The format name is `PackedParams` per the original PD4 spec on wowdev.wiki. **The user notes this may be wrong or stale** (the wiki is sometimes updated without notice). Treat the format name as a working hypothesis.

### Bit layout

```
PackedParams (32 bits):
  [31:24] = 0xAA          type bucket (Ck24Type, 1 byte)
  [23:16] = 0xBB          high object byte (Ck24HighByte, 1 byte, the user's "BB")
  [15: 8] = 0xCC          low object byte  (Ck24LowByte, 1 byte, the user's "CC")
  [ 7: 0] = 0x00          reserved padding (always observed to be zero in our data; spec says low byte is reserved per PD4; treat as always-zero trailer, not identity)
```

### Derived fields (current and proposed)

| Field | Current | Proposed in this spec | Notes |
|---|---|---|---|
| `Ck24` (24-bit) | `(PackedParams >> 8) & 0x00FF_FFFF` = `0xAABBCC` | unchanged | The 24-bit identity. |
| `Ck24Type` (8-bit) | `(PackedParams >> 24) & 0xFF` = `0xAA` | unchanged | The type bucket. Used by `Pm4RegionObjectGrouper` to group surfaces by object type. |
| `Ck24ObjectId` (16-bit) | `(ushort)(Ck24 & 0xFFFF)` = `0xBBCC` | unchanged, but **relabeled in comments** as a derived convenience for backward compatibility, not a primary key. | The current code flattens `0xBB` and `0xCC` into a single 16-bit ID. The user believes this is **lossy**. |
| `Ck24HighByte` (8-bit) | **not exposed** | NEW: `(byte)((Ck24 >> 8) & 0xFF)` = `0xBB` | The high object byte. The user hypothesizes this may index into a separate unknowns/subtype table. |
| `Ck24LowByte` (8-bit) | **not exposed** | NEW: `(byte)(Ck24 & 0xFF)` = `0xCC` | The low object byte. The user hypothesizes this may be the **bond** between two CK24 objects — the relationship that makes multi-instance containers cohere. |

### The bond hypothesis (research thread)

The user's observation: "we have to start splitting up the scene graph by using the 0xAA value as a bucket for types of objects ... 0xBBCC is our object ID ... chatGPT seems to think that the BB and CC are two distinct values, not a single value. ... the BBCC values can be some sort of connection between the two, as a bond of some sort, hidden in the ck24 data."

What this implies, in concrete terms:

- **Current model**: `0xBBCC` is a single 16-bit ID. Multi-instance sharing is "every surface with the same `0xBBCC` belongs to the same logical object". This is what `Pm4ResearchLinkageAnalyzer` already measures (`Ck24ObjectId.ReuseCountPerFile`).
- **User's hypothesis**: `0xBB` and `0xCC` are **paired**, not concatenated. Two objects share a bond if their `0xBB` matches one byte and their `0xCC` matches another byte, *or* by some other pairing rule (rotation, sum, xor) that we have not yet discovered. The current `Ck24ObjectId` is a lossy projection of this pair.
- **What would prove or disprove the hypothesis**: a real-data pass over the development corpus, surfacing per-CK24 statistics on (Ck24HighByte, Ck24LowByte) co-occurrence with the unknowns/subtype tables. If `0xBB` and `0xCC` index into **different** tables, we should see them correlated with **different** unknowns/subtype fields. If they index into the **same** table at different offsets, we should see them correlated with the **same** field, just offset.

The research surface already has a `Ck24ObjectId.ReuseCountPerFile` metric (in `Pm4ResearchLinkageAnalyzer`). The natural next step is `Ck24HighByte.ReuseCountPerFile` and `Ck24LowByte.ReuseCountPerFile`, with a cross-correlation against the unknowns/subtype field indices.

This is research, not a behavior change. The spec records it as a question; the user decides when (and whether) to investigate.

## User Scenarios & Testing

### User Story 1 - Separate dockable PM4 Scene Graph panel, expanded by default (Priority: P1)

As a viewer user, I want the PM4 scene graph to live in its own dockable panel, expanded by default when a PM4 object is selected, so I can browse the graph the way I browse the Blender outliner — full panel, no hidden collapse.

**Why this priority**: This is the user's primary UX complaint. The current `CollapsingHeader` is collapsed by default; the graph is invisible until you open it. Making it visible is the entry point to the rest of the work.

**Independent Test**: Click a PM4 object. The PM4 Scene Graph panel is visible in the dock without any further clicking. The graph tree is shown. Camera position is preserved on every frame the panel is open.

**Acceptance Scenarios**:

1. **Given** a PM4 object is selected, **When** the viewer draws the UI, **Then** the PM4 Scene Graph panel is visible (not collapsed) in its default dock position, and the graph tree is shown.
2. **Given** no PM4 object is selected, **When** the viewer draws the UI, **Then** the PM4 Scene Graph panel exists in the dock and shows an empty / placeholder state with a clear "Select a PM4 object" message.
3. **Given** the user closes the PM4 Scene Graph panel, **When** they re-select a PM4 object, **Then** the panel reopens in its default dock position (or stays closed if the user has explicitly asked it to stay closed — see US3).
4. **Given** a multi-instance PM4 container (CK24 with hundreds of parts) is selected, **When** the panel is open, **Then** the graph renders in under 1 second from click to first frame, using the cache introduced in commit `42e83488`.

---

### User Story 2 - Image preview of the PM4 mesh in the panel (Priority: P1)

As a viewer user, I want the PM4 Scene Graph panel to show a small image preview of the selected PM4 mesh — the object the graph describes — so I have a visual anchor while reading the graph.

**Why this priority**: The user said the image is useful both for navigation (this is the object the graph is about) and for "training a model for visual identification of the objects". A model input is a research asset; the navigation value is immediate.

**Independent Test**: Click a PM4 object that has a known M2 or WMO file (resolved by the existing asset-corpus surface). The panel shows a 256x256 (or comparable) image of that mesh. If no mesh can be resolved, the panel shows a placeholder with a clear "no preview available" message.

**Acceptance Scenarios**:

1. **Given** a PM4 object is selected and its M2/WMO mesh can be resolved via the existing asset corpus, **When** the panel renders, **Then** it shows the mesh's image preview, sized to fit the panel without dominating.
2. **Given** a PM4 object is selected but its mesh cannot be resolved, **When** the panel renders, **Then** it shows a placeholder image area with a clear "no preview available" status (not an error, not a freeze).
3. **Given** a preview is shown, **When** the user changes the selection, **Then** the preview updates to the new object's mesh (with the same caching strategy as the graph).
4. **Given** a preview is shown, **When** the user clicks the preview, **Then** the user can optionally be sent to the source mesh (e.g., open the standalone MDX/M2 in the world). This is a stretch goal; the first slice ships the preview without click-through.

---

### User Story 3 - Per-user dock preferences (Priority: P2)

As a viewer user, I want the panel's position, expanded/collapsed state, and which dock node it lives in to persist across sessions, so the panel is where I left it.

**Why this priority**: This is the same persistence model that the existing `ShellPanel` system uses. Implementing it is consistent with the rest of the UI, not a new requirement.

**Independent Test**: Move the PM4 Scene Graph panel to a different dock, close the viewer, reopen. The panel is in the same dock node on the next start.

**Acceptance Scenarios**:

1. **Given** the user moves the PM4 Scene Graph panel to a different dock, **When** the viewer restarts, **Then** the panel is in the same dock node.
2. **Given** the user closes the PM4 Scene Graph panel, **When** the viewer restarts, **Then** the panel remains closed (the user did not explicitly reopen it).
3. **Given** the user toggles a "remember this layout" preference, **When** the viewer restarts, **Then** the previous panel state is restored (otherwise, the spec 044 default dock layout is used).

---

### User Story 4 - Expose the 24-bit CK24 byte pair (`0xBB`, `0xCC`) for research (Priority: P1)

As a research user, I want the PM4 Scene Graph panel (and the research export) to expose `Ck24HighByte` and `Ck24LowByte` separately, so I can investigate the bond hypothesis without re-decoding `PackedParams` from raw bytes every time.

**Why this priority**: This is the spec's main research contribution. The user's hypothesis is testable only if the byte pair is exposed in the data model. The current `Ck24ObjectId = 0xBBCC` is lossy by construction.

**Independent Test**: Run the existing `WowViewer.Tool.Inspect pm4 forensics` command on a development map. The JSON report contains `ck24HighByte` and `ck24LowByte` for every surface. Run a new `ck24 forensics bond-stats` command (or extend the existing one) that produces a per-file cross-correlation of `Ck24HighByte` reuse against `Ck24LowByte` reuse against unknowns/subtype table indices.

**Acceptance Scenarios**:

1. **Given** a PM4 file with N surfaces, **When** the forensics export is run, **Then** the report includes `ck24HighByte` and `ck24LowByte` for every surface, alongside the existing `ck24`, `ck24Type`, and `ck24ObjectId`.
2. **Given** a PM4 file with multi-instance containers, **When** the bond-stats command is run, **Then** the report includes:
   - `Ck24HighByte.ReuseCountPerFile` (analogous to existing `Ck24ObjectId.ReuseCountPerFile`)
   - `Ck24LowByte.ReuseCountPerFile`
   - A cross-tabulation: for each `Ck24HighByte` value, the distribution of `Ck24LowByte` values across the surfaces that share that high byte.
   - The same cross-tabulation reversed.
3. **Given** the bond-stats command is run on the development corpus, **When** a surface's `Ck24HighByte` and `Ck24LowByte` are independently correlated with unknowns/subtype field indices, **Then** the report's correlation table shows whether the two bytes index into the same table (high correlation) or different tables (low correlation).
4. **The research output is opt-in** — the bond-stats report runs as a separate command, not on every `inspect`. This keeps the existing 046 match-asset flow unchanged.

---

### User Story 5 - CK24 type-bucket grouping in the scene graph (Priority: P1)

As a viewer user, I want the scene graph to group entries by `Ck24Type` first, so the type-bucket `0xAA` is the primary visual anchor (matching the user's mental model: "we have to start splitting up the scene graph by using the 0xAA value as a bucket for types of objects").

**Why this priority**: The current graph groups by `LinkGroupObjectId` first, then by `MscnRefIndex`. The user wants the type bucket to be the top-level grouping. This is a real change in the graph shape, not a small tweak.

**Independent Test**: Click a PM4 object. The graph shows type-bucket headers (`0xAA=0x03 (M2 top)`, `0xAA=0x10 (interior WMO floor)`, etc.) at the top, with the existing link-group hierarchy nested under each.

**Acceptance Scenarios**:

1. **Given** a multi-CK24-type object is selected, **When** the graph renders, **Then** the top-level entries are type-bucket groups (one per distinct `Ck24Type` value within the selected object), not link-group clusters.
2. **Given** a single-type object, **When** the graph renders, **Then** the type-bucket header is shown at the top with the existing link-group tree nested under it (no semantic change for single-type).
3. **Given** the type-bucket grouping is on, **When** the user reads the graph, **Then** the `0xAA` value is visible at every level (header + per-entry), so the type identity is never lost in the link-group detail.
4. **Given** the new grouping changes the graph shape, **When** the existing JSON export runs, **Then** the export schema gains a top-level `typeBuckets` array; the existing per-part data is preserved under each bucket. The change is additive, not breaking.

---

### Edge Cases

- A PM4 surface where `Ck24` is `0x000000` (zero pad, zero high, zero low) — the current code already excludes these from the group analysis (`where Ck24ObjectId != 0`). Spec 058 MUST keep that exclusion.
- A multi-instance container where every part has the same `0xBB` and the same `0xCC` — confirms the existing object-ID-reuse model is correct for that case; the bond hypothesis predicts other cases where the bytes vary across parts.
- A PM4 object where the M2/WMO file lives in a different build's MPQ than the user's current data source — the image preview cannot be resolved; the panel shows the placeholder.
- A development corpus with no `unknowns` table entries at all — the bond-stats cross-tabulation still emits, with empty/unknown entries.
- The user moves the PM4 Scene Graph panel to a tab group with another panel (e.g., the existing Navigator or Inspector) — the panel must respect the tab group's docking contract without crashing or losing its tree state.
- The viewer is started with no client loaded — the panel is visible with the placeholder, no error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The PM4 Scene Graph MUST be reachable as a separate dockable panel, distinct from any existing sidebars.
- **FR-002**: The PM4 Scene Graph panel MUST be expanded by default when a PM4 object is selected. The user MUST be able to close it; if they do, the panel remembers the closed state across the session but is reopened on the next viewer restart (per the default-shell-layout contract in spec 044).
- **FR-003**: The PM4 Scene Graph panel MUST render the per-selection graph using the cached `Pm4SelectedObjectGraphInfo` introduced in commit `42e83488`. The panel MUST NOT trigger a per-frame rebuild.
- **FR-004**: The PM4 Scene Graph panel MUST group entries by `Ck24Type` first, with the existing link-group hierarchy nested under each type bucket.
- **FR-005**: The PM4 Scene Graph panel MUST show an image preview of the selected object's resolved M2 or WMO mesh. When the mesh cannot be resolved, it MUST show a placeholder.
- **FR-006**: The image preview MUST be cached per selection, using the same selection-key cache as the graph itself. Selecting a new object MUST update the preview without rebuilding the panel layout.
- **FR-007** (SHIPPED — `fe8ed85d`): The PM4 data model in the research lib (`WowViewer.Core.PM4.Models.Pm4MsurEntry`) gained two new derived fields: `Ck24HighByte` and `Ck24LowByte`. The existing `Ck24ObjectId` field is retained for backward compatibility but relabeled in its `///` XML doc as "the low 16 bits of the 24-bit CK24; not an independent identity". The original spec referenced `Pm4MslkEntry` by mistake; the correct type is `Pm4MsurEntry`.
- **FR-008**: The `WowViewer.Core.PM4.Research.Pm4ResearchLinkageAnalyzer` MUST gain new metrics `Ck24HighByte.ReuseCountPerFile` and `Ck24LowByte.ReuseCountPerFile`, computed analogously to the existing `Ck24ObjectId.ReuseCountPerFile`.
- **FR-009**: A new `WowViewer.Tool.Inspect pm4 bond-stats` subcommand MUST be added (or `pm4 forensics` MUST be extended — TBD on which is cleaner) that emits the cross-tabulation of high-byte vs low-byte reuse and the correlation with unknowns/subtype field indices.
- **FR-010**: The PM4 Scene Graph panel MUST be reachable through the existing shell-panel infrastructure (`ShellPanelId`, `ShellPanelLane`) so the user can dock it alongside the existing Navigator, Inspector, and PM4 Workbench panels.
- **FR-011**: No reference to `H:\CLIENTS` may appear in any new code, scripts, tests, or documentation. Staged client paths under `output/tmp/wowarchive-clients/` are the only trusted client access.
- **FR-012**: The new code MUST follow the Library-First principle: the data-model additions live in `WowViewer.Core.PM4`, the research analyzer additions live in `WowViewer.Core.PM4.Research`, and the panel lives in the viewer shell.

### Key Entities

- **CK24 byte pair** (research lib): `Ck24HighByte` and `Ck24LowByte` derived fields on `Pm4MsurEntry` (SHIPPED — `fe8ed85d`). Pure getters on `PackedParams`; zero new storage.
- **BondStatsReport** (research lib): the per-file output of the new bond-stats analyzer. Lists per-file `Ck24HighByte` and `Ck24LowByte` reuse distributions, a high×low cross-tabulation, and the unknowns/subtype correlation table.
- **Pm4SceneGraphPanel** (viewer): the new dockable panel, reachable as a `ShellPanelId` and rendered with cached data only.
- **Pm4MeshPreviewCache** (viewer): a small per-selection cache of the resolved mesh preview texture. Cleared on selection change.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user who clicks any PM4 object (single, multi-instance, or container) sees the scene-graph panel within 1 second, with the type-bucket grouping visible at the top level.
- **SC-002**: The graph panel renders 60+ FPS with no per-frame rebuild, on a development map with 600+ PM4 objects and a selected container with 1000+ parts.
- **SC-003**: An image preview is shown when the M2 or WMO can be resolved; a clear placeholder is shown when it cannot.
- **SC-004**: The new `Ck24HighByte` and `Ck24LowByte` fields appear in the existing `pm4 forensics` JSON export without breaking the existing schema (additive change only).
- **SC-005**: The new bond-stats subcommand runs against the development corpus in under 60 seconds and emits a structured JSON or Markdown report.
- **SC-006**: The bond-stats report's cross-tabulation lets a researcher see, for the first time, whether `0xBB` and `0xCC` index into the same unknowns/subtype table or different ones — which is the empirical test for the bond hypothesis.

## Assumptions

- The user's PM4 bit-layout interpretation (`0xAABBCC_00` with `0x00` always-zero padding) is correct for the development corpus. If a real client shows non-zero low bytes, the spec is updated and the byte-pair research becomes moot. (Per the user, the wowdev.wiki/PD4 reference for the original `PackedParams` name may itself be wrong or stale — treat as a working hypothesis, not a settled fact.)
- The user has the development corpus available locally for the bond-stats run. The research command is local-only; no MPQ archive required.
- The current `Pm4SelectedObjectGraphInfo` schema is stable; the type-bucket grouping is a **structural** change (top-level grouping), not a **data** change (the underlying `LinkGroup`/`MscnRefGroup`/`Part` records are preserved). The JSON export gains `typeBuckets` as a new top-level array; existing arrays remain.
- The image preview is rendered from the **resolved M2 or WMO geometry**, not a separate render pass. No new OpenGL machinery is needed beyond what the standalone MDX/M2 inspector already uses.
- The PM4 Scene Graph panel is **read-only** in this spec. Editing the graph (e.g., adding or removing instances from a multi-instance container) is out of scope. That would be a follow-up spec.

## Out of Scope (explicit)

- Editing the scene graph from the panel (add/remove instances, change bonds). Read-only.
- A new "bond editor" UI for manipulating `0xBB` and `0xCC` values. The research thread is observational, not interventional.
- Automatic detection of the bond rule (whatever it is). Spec 058 produces the data; the user interprets.
- A 2D topology view of multi-instance containers (e.g., a 2D scatter of parts by world position). The user mentioned an "image preview" but did not ask for a 2D map.
- Per-tile filters or "show only this tile" toggle. The graph is per-selection, not per-tile.
- Migrating any of the existing research analyzers (`Pm4ResearchLinkageAnalyzer`, `Pm4Ck24ForensicsAnalyzer`, etc.) to use the new `Ck24HighByte` / `Ck24LowByte` fields. They are independent; the new fields are additive.
- Auto-mounting the WoWArchive. (See spec 057.)

## Follow-Up Tasks (each is a small, independently-shippable slice)

1. **Already shipped** (`42e83488`): the 30s click-freeze fix on multi-instance PM4 containers. The graph is now cached by selection.
2. **Already shipped** (`ade06247`): the hot-swap missing-map fallback. (That one belongs to spec 057, not 058, but the spec-coverage is recorded here for cross-reference.)
3. **Data-model additions** (FR-007): add `Ck24HighByte` and `Ck24LowByte` to `Pm4MslkEntry`; relabel `Ck24ObjectId` in its XML doc. Pure additive; no behavior change. One-file diff in `WowViewer.Core.PM4`. Land this first because US4, US5, and the bond-stats work all depend on it.
4. **Bond-stats analyzer + CLI subcommand** (FR-008, FR-009): extend `Pm4ResearchLinkageAnalyzer` with the new reuse metrics; add a `pm4 bond-stats` subcommand to `WowViewer.Tool.Inspect`; emit a structured report. Test on the development corpus.
5. **Forensics JSON schema addition** (FR-004 in the data-model context): add `ck24HighByte` and `ck24LowByte` to the existing `pm4 forensics` export output. Additive schema change.
6. **Type-bucket grouping in the graph** (FR-004 in the panel context): the new top-level `typeBuckets` array. This is the biggest structural change; consider it a separate spec slice.
7. **Image preview** (FR-005, FR-006): resolve the selected object's M2/WMO, render to a small texture, cache per selection. Bounded, but requires the asset-resolution path to handle misses gracefully.
8. **Dockable panel extraction** (FR-001, FR-002, FR-010): promote the existing `DrawPm4SceneGraph` to a `ShellPanelId`-hosted panel with default-expanded state. The user can then drag it to any dock node.
9. **Per-user dock preferences** (US3): persist the panel's dock position across sessions. Builds on the existing `ShellPanel` persistence surface.
10. **Research thread**: empirical test of the bond hypothesis. The user drives this; the spec provides the data surface (the bond-stats report) needed to evaluate it.
