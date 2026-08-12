# Feature Specification: WoWViewer UI Overhaul

**Feature Branch**: `145-wow-ui-overhaul`
**Created**: 2026-08-12
**Status**: Draft
**Input**: User description: "Overhaul the WoWViewer UI shell with context-aware keybindings, visual shortcut help, a vertical right-sidebar tab rail, bounded non-overflowing sidebars, persistent closeable utility windows, a usable log surface, and synchronized release documentation while preserving existing viewer functionality."

## User Scenarios & Testing

### User Story 1 - Contextual shortcuts and help (Priority: P1)

As a viewer user, I want keyboard shortcuts to follow the page or tab I am using, so Capture keys do not unexpectedly control the camera while I am working elsewhere, and I want one visible help dialog that explains the active and global shortcuts.

**Why this priority**: Shortcut collisions are a direct usability and recording risk. The user must be able to discover the current controls without reading source or remembering which panel owns a key.

**Independent Test**: Open Help > Keyboard Shortcuts, select Model, World, Tools, Utilities, and Capture pages, and verify that the displayed context changes. Enable Capture keyboard authoring, leave the Capture page, and verify Capture-only keys no longer mutate the camera path.

**Acceptance Scenarios**:

1. **Given** the viewer is on a non-Capture page, **When** the user presses a Capture authoring key, **Then** the Capture path is not changed.
2. **Given** the user opens Help > Keyboard Shortcuts, **When** the help window is visible, **Then** global shortcuts and the active page's shortcuts are grouped with readable descriptions and key labels.
3. **Given** the user changes the active page, **When** the help window remains open, **Then** its active-context section updates without requiring a restart.

### User Story 2 - Stable WoW-like shell navigation (Priority: P1)

As a viewer user, I want the right sidebar's main pages to be visibly separated and easy to select, so I can understand which page owns the controls and avoid horizontal tab overflow.

**Why this priority**: The right sidebar currently hides the page hierarchy in scrolling tab strips, making the UI difficult to navigate and causing controls to be clipped.

**Independent Test**: Load the viewer at narrow and wide window sizes, select Model, World, and Tools from the vertical page rail, and verify that each page remains visible without horizontal scrolling in the main page selector.

**Acceptance Scenarios**:

1. **Given** the right sidebar is visible, **When** the user looks at its left edge, **Then** Model, World, and Tools are presented as a persistent vertical page selector.
2. **Given** the user selects a page, **When** the page content is shown, **Then** the selector remains visible and the content region owns the page's controls.
3. **Given** the window is narrower than the preferred layout, **When** the sidebar is displayed, **Then** the content clips or wraps vertically rather than requiring horizontal scrolling for the main page selector.

### User Story 3 - Useful bounded navigator and log (Priority: P1)

As a viewer user, I want the left navigator and log to preserve useful content at their current size, so loading a map does not hide the map list and log entries remain readable without horizontal scrolling.

**Why this priority**: The navigator and log are operational tools. Their current layout makes them unusable exactly when the viewer has the most data to show.

**Independent Test**: Load a world with a minimap and discovered maps, then open the Log page and generate long log messages. Verify that the map list remains reachable and log lines wrap in a vertically scrollable region.

**Acceptance Scenarios**:

1. **Given** a world and minimap are loaded, **When** the left navigator is visible, **Then** the minimap uses bounded space and the map/file lists remain independently reachable.
2. **Given** a log entry is longer than the available width, **When** it is displayed, **Then** it wraps within the log region and does not force horizontal scrolling.
3. **Given** the log window or log page is open, **When** the user clicks elsewhere in the viewer, **Then** the log remains open until its close control or owning visibility control is used.

### User Story 4 - Persistent utility windows and honest pages (Priority: P2)

As a viewer user, I want pop-out utility windows to stay open until I close them, and I want tabs with unfinished functionality to say so clearly instead of presenting unexplained text or dead controls.

**Why this priority**: Persistent windows make inspection and recording workflows practical, while honest empty states prevent the UI from implying capabilities that are not wired.

**Independent Test**: Open a utility pop-out, click the viewport and another page, then close it with its title-bar X. Inspect the LOD page and verify its state is either actionable or explicitly marked as unavailable with the missing data path.

**Acceptance Scenarios**:

1. **Given** a utility window is open, **When** focus moves to another viewer surface, **Then** the window remains visible and positionable.
2. **Given** a utility window is open, **When** the user clicks its title-bar close button, **Then** it closes and does not require selecting a different tab.
3. **Given** a page has no supported data for the current scene, **When** the page is opened, **Then** it shows a concise disabled/empty state rather than dead controls.

### User Story 5 - Truthful release surface (Priority: P2)

As a project user, I want the root and viewer READMEs, About/version metadata, and current feature notes to agree, so a release checkout does not describe stale support or branch state.

**Why this priority**: Documentation is part of the release surface and is currently contradicting the running application.

**Independent Test**: Compare the About box, assembly version, viewer README, root README, and release notes after a build; they must identify v0.5.2 consistently and label incomplete client/runtime proof as pending.

**Acceptance Scenarios**:

1. **Given** the viewer is built, **When** the user opens About, **Then** it reports v0.5.2.
2. **Given** a user reads either README, **When** they look for support and current status, **Then** partial/provisional paths and pending real-client proof are clearly distinguished from validated behavior.

## Edge Cases

- The user opens the help window before a world or any page is loaded; global shortcuts remain visible and page-specific sections show a clear unavailable state.
- The available sidebar width is smaller than the preferred width; the layout suppresses horizontal overflow and preserves access to the content scroll region.
- A shortcut is assigned to both a global action and a page action; the active page owns the page action only when the page has focus, while global actions remain explicitly marked.
- A utility window is closed with its title-bar X while its owning tab remains selected; the tab remains usable and may reopen the window through its explicit launcher.
- Long log messages contain no whitespace; they are clipped or safely wrapped without making the parent window horizontally scroll.

## Requirements

### Functional Requirements

- **FR-001**: The viewer MUST maintain an explicit active keybinding context derived from the selected main page and nested page.
- **FR-002**: Capture authoring shortcuts MUST be active only while the Capture context is active and keyboard authoring is enabled.
- **FR-003**: The Help menu MUST expose a visual Keyboard Shortcuts window with global, active-context, and available-page shortcut groups.
- **FR-004**: The Keyboard Shortcuts window MUST remain open until its title-bar close control or an explicit close action is used.
- **FR-005**: The right sidebar MUST present its main page selection as a vertical rail with a separate content region.
- **FR-006**: The main page selector MUST remain usable without horizontal scrolling at the supported compact sidebar width.
- **FR-007**: The left navigator MUST allocate bounded space to world overview/minimap content and preserve independent access to file and map discovery content.
- **FR-008**: The log surface MUST use the available content width, wrap long entries, and provide vertical scrolling without horizontal overflow.
- **FR-009**: Utility pop-outs promoted to persistent windows MUST use an explicit visibility state and a working title-bar close control.
- **FR-010**: Pages with unavailable or unfinished capabilities MUST present an honest disabled/empty state with the relevant missing data or proof boundary.
- **FR-011**: The viewer Windows and cross-platform project metadata MUST report version 0.5.2 consistently.
- **FR-012**: The root README and viewer README MUST describe the current v0.5.2 state, current support boundaries, active UI work, and pending runtime proof without claiming unverified success.
- **FR-013**: Existing terrain, WMO, MDX/MD2, minimap, capture, and bottom-bar routes MUST remain available while the UI shell is migrated incrementally.
- **FR-014**: The redesign MUST not move client-file parsing ownership into the UI layer or alter existing format-reader contracts.

### Key Entities

- **Keyboard binding**: A named action with a display key gesture, scope, description, and activation context.
- **Keyboard context**: The current main and nested page identity used to decide which page-specific bindings may run.
- **Sidebar page**: A top-level Model, World, or Tools surface with a stable selector and content region.
- **Persistent utility window**: A positionable viewer window with independent visibility and title-bar close behavior.
- **Release truth record**: The version, support boundary, validation status, and known limitations shared by application metadata and documentation.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A user can open the shortcut help window from the Help menu in one action and identify the active page's bindings without reading source code.
- **SC-002**: Capture-only shortcuts cause zero Capture mutations when the active context is not Capture.
- **SC-003**: The main right-sidebar page selector remains fully readable and selectable at the compact supported sidebar width without horizontal scrolling.
- **SC-004**: Long log entries remain readable within the log content width, and the log surface exposes vertical scrolling for all retained entries.
- **SC-005**: Loading a map with a minimap leaves the map discovery surface reachable without closing the world overview.
- **SC-006**: Every promoted pop-out can be closed directly with its title-bar X and stays visible after a viewport click until closed.
- **SC-007**: Application metadata and both READMEs identify v0.5.2 consistently, with no stale claim that 2.x support is wholly absent or that pending real-client proof is complete.

## Assumptions

- The current ImGui viewer shell and bottom bars remain the base interaction model; this feature is a structural UI overhaul, not a renderer rewrite.
- The existing tabbed workbench remains available during migration, and the left navigator remains in place until its replacement has independent proof.
- Default key gestures remain project-defined for this slice; user-editable rebinding storage is a later extension unless required by an existing route.
- Real-client runtime and visual proof remain user-run gates; source/build validation does not claim FPS or rendering signoff.
- The existing v0.5.2 release line is the intended application version, because the running title/About surface already identifies it.

## Out of Scope

- Replacing ImGui with another UI framework.
- Rewriting terrain, WMO, MDX/M2, DBC, or map readers.
- Changing capture file formats or camera-path interpolation semantics.
- Removing the left or right sidebar wholesale before replacement routes have passed independent proof.
- Claiming that unfinished LOD, client-era rendering, or performance paths are complete solely because their controls are visible.
