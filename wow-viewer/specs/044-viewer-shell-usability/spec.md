# Feature Specification: Viewer Shell Usability

**Feature Branch**: `044-viewer-shell-usability`

**Created**: 2026-06-03

**Status**: Phase 1+2 complete (P1 landed 2026-06-10), US4 (cursor-as-model) deferred P2

**Related specs**:
- `049-viewer-ui-consolidation` — live continuation lane for categorized Tools menu, floating window extraction, and sidebar consolidation. 044 is the foundation layer; build new viewer shell work on 049.

**Input**: User description: "`Open MK Dataset` is still in the viewer file menu, old tools should be exposed somewhere else, the sidebars are painful and do not allow placing panels into the sidebar containers, `World Maps` never expands after a client is loaded, and the mouse cursor should be rendered as a model so broken model rendering is immediately obvious."

## Context

The current `WoWViewer` shell exposes stale dataset and conversion entrypoints in primary menus, leaves map discovery hidden behind a closed `World Maps` header, and ships a half-wired dockable shell path that never actually renders the dockspace host and persists `UseDockspaceUi = false`. The first implementation slice fixes those shell regressions without bundling the larger cursor-as-model runtime seam into the same pass.

## User Scenarios & Testing

### User Story 1 - Dockable Shell Panels (Priority: P1)

As a viewer user, I need the shell panels to be genuinely dockable so I can drag them, redock them, and recover a usable layout without fighting fixed sidebars.

**Why this priority**: The current UI becomes hard to use because the dockspace path is dormant and fixed splitters remain the only practical layout mechanism.

**Independent Test**: Launch `WoWViewer`, verify the dockspace host is present, drag `Navigator` or `Selection` into a different dock position, then use `View > Reset Shell Layout` to recover the default layout.

**Acceptance Scenarios**:

1. **Given** the viewer is running, **When** I open shell panels, **Then** they render into the dockable shell path instead of the legacy fixed-sidebar-only path by default.
2. **Given** a shell panel is visible, **When** I drag it to a different dock position, **Then** ImGui docking accepts the panel instead of trapping it in the old fake-sidebar layout.
3. **Given** dockable shell mode is active, **When** I look at the viewport edges, **Then** legacy fixed splitters are not rendered over the dockable layout.
4. **Given** I want to recover from layout drift, **When** I choose `View > Reset Shell Layout`, **Then** the viewer returns to the default dockable shell layout and persists that reset.

### User Story 2 - Discoverable World Maps After Client Load (Priority: P1)

As a user who just loaded a client, I need the `World Maps` section to open itself when maps are discovered so I immediately see how to load a map.

**Why this priority**: Users currently think map loading is broken because the maps exist but stay hidden behind a closed header.

**Independent Test**: Load a staged client root. Verify the `World Maps` section expands automatically the first frame after discovery in the navigator surface.

**Acceptance Scenarios**:

1. **Given** no client is loaded, **When** I attach a client and map discovery succeeds, **Then** `World Maps` opens automatically.
2. **Given** a client load discovered maps, **When** I look at the navigator, **Then** I can immediately see the discovered map list without extra clicks.

### User Story 3 - Menu Declutter for Legacy Data and Conversion Surfaces (Priority: P1)

As a user focused on viewing worlds and models, I need stale dataset and conversion tools moved out of the primary file flow so the main menu stops implying they are part of the normal load path.

**Why this priority**: `Open MK Dataset` and related data-pipeline tools are legitimate utilities, but they confuse the main viewer flow when presented as primary `File` actions.

**Independent Test**: Open the main menu bar. Verify `File` no longer contains `Open MK Dataset...`, and the legacy data/conversion actions live under a dedicated `Tools` submenu.

**Acceptance Scenarios**:

1. **Given** the viewer is running, **When** I open `File`, **Then** dataset-specific entrypoints are not mixed into the main file loading flow.
2. **Given** I still need the old utilities, **When** I open `Tools > Offline Data / Conversion`, **Then** the MK/Zarr, dataset build, training, texture transfer, and map/WMO conversion entrypoints remain reachable.

### User Story 4 - Cursor-as-Model Diagnostic Surface (Priority: P2, Deferred)

As a user validating model rendering, I want the mouse cursor to be rendered as a version-appropriate model object so model-render failures are visible immediately.

**Why this priority**: If model rendering breaks, losing the cursor becomes an immediate diagnostic signal.

**Independent Test**: Launch the viewer on at least one M2-era and one MDX-era client and verify a version-appropriate cursor asset is rendered and follows the cursor.

**Acceptance Scenarios**:

1. **Given** a client build uses an MDX cursor, **When** the viewer renders UI/scene interaction, **Then** the cursor uses the MDX-backed asset path for that era.
2. **Given** a client build uses an M2 cursor, **When** the viewer renders UI/scene interaction, **Then** the cursor uses the M2-backed asset path for that era.
3. **Given** model rendering fails, **When** the viewer starts, **Then** the cursor model is missing too, providing an immediate visual diagnostic.

## Requirements

### Functional Requirements

- **FR-001**: `WoWViewer` MUST render the dockspace host whenever dockable shell mode is active.
- **FR-002**: Dockable shell mode MUST be the default layout path for this feature slice.
- **FR-003**: Viewer settings MUST persist the actual `UseDockspaceUi` value instead of hard-coding it to `false`.
- **FR-004**: `View` MUST expose a direct toggle for dockable shell panels and retain `Reset Shell Layout`.
- **FR-005**: Fixed sidebar splitters MUST NOT render while dockable shell mode is active.
- **FR-006**: `World Maps` MUST auto-open when a client load transitions discovered maps from `0` to `>0`.
- **FR-007**: `File` MUST NOT expose `Open MK Dataset...` or `Open Zarr Dataset...`.
- **FR-008**: Legacy dataset and conversion entrypoints MUST remain reachable under a dedicated `Tools` submenu.
- **FR-009**: Cursor-as-model remains a bounded follow-up and MUST NOT be faked with a non-model substitute when this deferred slice is implemented.

## Success Criteria

- **SC-001**: Users can drag shell panels into new dock positions in the active viewer shell.
- **SC-002**: Loading a client immediately exposes the map list without requiring users to discover the collapsed `World Maps` header manually.
- **SC-003**: `File` focuses on actual file/client loading while legacy data and conversion workflows live under `Tools > Offline Data / Conversion`.
- **SC-004**: The deferred cursor-model requirement is explicitly tracked in this spec instead of being lost as an undocumented follow-up.
