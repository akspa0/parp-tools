# Feature Specification: 045 — Scene Graph Workbench

**Feature Branch**: `045-scene-graph-workbench`
**Created**: 2026-06-03
**Status**: Draft
**Input**: User description — "if we could get a tree-view hierarchial scene graph in the right sidebar, to show all the data as a proper scene graph like a blender project does it, then we'd be GOLDEN for all PM4 data, and, ADT/AlphaWDT data, too. We need a way to look at the whole entire scene graph. That's what all these files really are, under the surface, after all!"

## Context

`WoWViewer` now has a viable dockable shell path again (spec 044), but the viewer still makes users reconstruct the scene mentally from separate PM4 utilities, terrain panels, object lists, and selection surfaces. The user wants one hierarchical outliner that exposes the loaded scene the way a Blender-style project tree does: not as raw files, but as the scene graph those files imply.

The workbench is a viewer-owned shell feature in `wow-viewer`, not a legacy `MdxViewer` archaeology surface. It must unify the already-decoded runtime data for:

- world terrain (`ADT` / `AlphaWDT`)
- placed world objects (`WMO`, `M2`, `MDX`)
- PM4 overlay hierarchy

The first slice is inspection-first and read-only. It is not a world editor, reparenting tool, or mesh authoring workflow.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Browse The Whole Loaded Scene As A Hierarchy (Priority: P1)

As a viewer user, I need one scene graph tree that shows the loaded world as a hierarchy so I can understand how terrain, objects, and PM4 structures relate without jumping between unrelated panels.

**Why this priority**: This is the core value. If the tree does not expose the loaded scene in one place, the feature does not solve the user's problem.

**Independent Test**: Load a staged client and world map with PM4 available. Open the scene graph panel and verify the root tree exposes terrain, world objects, and PM4 as separate top-level branches that can each be expanded into meaningful child nodes.

**Acceptance Scenarios**:

1. **Given** a world session is loaded, **When** I open the scene graph panel, **Then** I can see one root scene tree instead of disconnected per-domain lists.
2. **Given** terrain is loaded, **When** I expand the terrain branch, **Then** I can drill from map/session level into tiles and chunks.
3. **Given** PM4 data is loaded, **When** I expand the PM4 branch, **Then** I can drill from PM4 regions into objects and their sub-objects.
4. **Given** placed world objects are loaded, **When** I expand the object branch, **Then** I can inspect placed WMO/M2/MDX instances in the same tree.

---

### User Story 2 - Selection Sync Between Tree And Viewer (Priority: P1)

As a viewer user, I need tree selection to stay in sync with the existing viewport and inspector selection so the scene graph becomes the main way to navigate the loaded scene rather than a dead report.

**Why this priority**: A hierarchy that cannot drive or reflect selection is only documentation, not a usable workbench.

**Independent Test**: Select a PM4 node, terrain node, and placed-object node from the scene graph. Verify the active selection/inspector updates. Then select an item from the viewport or existing inspector and verify the tree highlights the matching node.

**Acceptance Scenarios**:

1. **Given** I click a selectable node in the tree, **When** the node represents a live scene element, **Then** the viewer selection changes to that element.
2. **Given** the viewer already has an active selected object or PM4 item, **When** I open or refresh the tree, **Then** the corresponding node is highlighted.
3. **Given** a node supports camera framing, **When** I invoke the node action, **Then** the camera can focus that element without changing unrelated scene state.
4. **Given** a node points to data that is currently unavailable or streamed out, **When** I select it, **Then** the UI explains the state instead of failing silently.

---

### User Story 3 - Large Scene Usability (Priority: P2)

As a user browsing large maps, I need the scene graph to remain usable even when the loaded scene contains many tiles, many PM4 objects, or many placed instances.

**Why this priority**: The feature fails if it becomes another giant unsearchable dump the moment a real world is loaded.

**Independent Test**: Load a heavier world session and verify the tree can be filtered, expanded lazily, and navigated without losing the current viewport interaction flow.

**Acceptance Scenarios**:

1. **Given** a large loaded scene, **When** I first open the tree, **Then** it does not eagerly dump every leaf node expanded by default.
2. **Given** I know a tile, CK24, region id, or object name, **When** I use the tree filter, **Then** I can narrow the visible hierarchy to matching nodes.
3. **Given** I expand and collapse branches while inspecting the world, **When** selection changes elsewhere, **Then** my expansion state is preserved where practical.
4. **Given** a branch has thousands of potential descendants, **When** I expand it, **Then** the workbench reveals children incrementally rather than freezing the whole UI.

---

### User Story 4 - Reuse The Same Graph Contract Across Scene Types (Priority: P3)

As a viewer user, I need the same scene-graph concept to work beyond one world-session shape so the tree can become the long-term inspection surface for standalone assets and future runtime slices too.

**Why this priority**: The user is asking for a real scene graph, not a one-off PM4 browser. The contract should survive future domains.

**Independent Test**: Open a standalone or reduced-content scene type after the first world-session slice is implemented and verify the panel can still present a meaningful root and node path using the same graph contract, even if some branches are absent.

**Acceptance Scenarios**:

1. **Given** a scene type has no PM4 data, **When** I open the scene graph, **Then** the PM4 branch is absent or clearly unavailable rather than showing stale placeholders.
2. **Given** a standalone asset scene is loaded in a later slice, **When** I open the scene graph, **Then** the same panel and node contract can host that scene type without inventing a second hierarchy UI.

### Edge Cases

- What happens when no client or world session is loaded yet?
- How does the tree represent streamed-out tiles or objects that were visible earlier but are no longer resident?
- How are cross-tile PM4 objects represented without duplicating the same object under every tile branch?
- What happens when a filter matches a leaf whose parents are currently collapsed or not yet materialized?
- How does the workbench behave when PM4 is unavailable for a map but terrain and objects are present?
- How does the tree represent version-specific terrain structure differences between `AlphaWDT` and split-ADT worlds without breaking the common hierarchy model?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `WoWViewer` MUST provide a hierarchical scene graph workbench for the active loaded scene.
- **FR-002**: The workbench MUST be available from the active viewer shell as a right-sidebar surface and as a dockable panel in the dockspace layout.
- **FR-003**: The first slice MUST be read-only; it MUST NOT require editing, reparenting, or deletion support.
- **FR-004**: The scene graph MUST unify at least these world-session domains when available: terrain, placed world objects, and PM4.
- **FR-005**: The terrain hierarchy MUST expose map/session-level terrain structure down to tile and chunk granularity.
- **FR-006**: The placed-object hierarchy MUST expose loaded world instances in a form users can browse and identify by type and placement identity.
- **FR-007**: The PM4 hierarchy MUST expose PM4 regions, objects, and sub-object structure using the current best-known PM4 ownership model.
- **FR-008**: Selecting a tree node MUST synchronize with the viewer's existing selection and inspection surfaces when the node maps to a live scene element.
- **FR-009**: Existing viewer selection changes from outside the tree MUST be reflectable back into the tree.
- **FR-010**: The workbench MUST support filtering/search across visible scene-graph nodes by identifiers and user-facing labels.
- **FR-011**: The workbench MUST load and expand large branches lazily enough that opening the panel does not require materializing every possible descendant immediately.
- **FR-012**: When a domain is unavailable, the workbench MUST surface that state clearly instead of implying stale or fake data.
- **FR-013**: The workbench MUST rely on existing decoded/runtime data and MUST NOT introduce duplicate file-format readers or parallel parser stacks.
- **FR-014**: The workbench MUST use stable node identity paths so expansion state and reverse-selection mapping can survive refreshes where the underlying scene still represents the same entity.
- **FR-015**: The first implementation slice MUST define a reusable scene-graph contract so later standalone-scene or asset-scene work can use the same panel rather than creating a second hierarchy UI.

### Key Entities *(include if feature involves data)*

- **Scene Graph Snapshot**: A read-only hierarchical representation of the currently loaded scene, composed from one or more scene domains.
- **Scene Graph Node**: A single node in the hierarchy with a stable identity path, type, label, summary metadata, and optional selectable target.
- **Scene Graph Domain Provider**: A provider that projects one scene domain, such as terrain, placed objects, or PM4, into scene-graph nodes under a shared root.
- **Scene Graph Selection Link**: The mapping between a node's stable identity and the viewer's live selection model.
- **Scene Graph Filter State**: The user-visible filter/search state that narrows the visible tree without destroying the underlying hierarchy contract.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a loaded world session with terrain, objects, and PM4 available, users can reach a terrain chunk, a placed object, and a PM4 sub-object from one scene graph panel without switching to separate tooling surfaces.
- **SC-002**: Selecting a supported node from the tree updates the active viewer selection in one interaction, and reverse selection from existing viewer surfaces highlights the matching tree node.
- **SC-003**: Opening the scene graph panel for a normal loaded world session does not force every leaf node visible by default; users can progressively reveal structure through expansion and filtering.
- **SC-004**: A filter can narrow the visible tree to a known identifier or label such as a tile coordinate, PM4 region id, CK24 key, or object name/uid without destroying the underlying hierarchy model.
- **SC-005**: The first slice lands without adding new duplicated parser ownership outside the existing `wow-viewer` shared/runtime surfaces.

## Assumptions

- The first slice targets active viewer sessions, not a full editing workflow.
- The first slice is read-only and inspection-oriented.
- The existing dockable shell restored by spec 044 is the correct host for this workbench.
- The current best PM4 hierarchy remains `Region -> Object -> Sub-object -> detail`, with newer PM4 field observations treated as metadata rather than a reason to invent a second PM4 tree model.
- Standalone asset scenes can adopt the same scene-graph contract later even if the first implementation slice only fully signs off world-session coverage.
