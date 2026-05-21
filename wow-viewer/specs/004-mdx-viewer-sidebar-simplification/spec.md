# Feature Specification: MdxViewer Right Sidebar Simplification

**Feature Branch**: `004-mdx-viewer-sidebar-simplification`

**Created**: 2026-05-21

**Status**: Draft

**Input**: User description: "The right sidebar in MdxViewer needs to be simplified — consolidate all functionality/panels/windows/tools, clean out old UI stuff for training, upgrade version to v0.5.0. The right panel is not resizeable, not customizable, and content is cut off."

## User Scenarios & Testing

### User Story 1 — Resizable Right Sidebar (Priority: P1)

As a developer using MdxViewer, I need the right sidebar to be resizable so I can read all content without text being cut off.

**Why this priority**: The immediate pain point — text is cut off and unreadable at the current max width. This blocks all other work.

**Independent Test**: Launch MdxViewer, drag the sidebar edge or use the width slider to resize the right panel to 800px+. Verify all text is readable and content reflows properly.

**Acceptance Scenarios**:

1. **Given** MdxViewer is running, **When** I drag the sidebar splitter edge left/right, **Then** the right sidebar width changes smoothly and content reflows.
2. **Given** MdxViewer is running, **When** I use the "Inspector Width" slider, **Then** the sidebar width adjusts to the slider value (range 260px–1080px).
3. **Given** the sidebar is wider than 600px, **When** I look at PM4 object details, **Then** all fields (CK24, type, surfaces, bounds, MPRL refs) are fully visible without truncation.
4. **Given** I resize the sidebar, **When** I close and reopen MdxViewer, **Then** the sidebar width is restored from saved settings.

---

### User Story 2 — Customizable Panel Visibility (Priority: P2)

As a developer, I need to show/hide individual sections in the right sidebar so I can focus on the panels I need.

**Why this priority**: The sidebar has 4 sections (Viewer Settings, Selection, World Tools, Utilities) but not all are needed at once. Collapsing helps but full visibility toggles would be better.

**Independent Test**: Toggle each section on/off. Verify hidden sections don't render and visible sections expand to fill available space.

**Acceptance Scenarios**:

1. **Given** the right sidebar is visible, **When** I click a section header, **Then** that section collapses/expands.
2. **Given** multiple sections are collapsed, **When** I look at the sidebar, **Then** visible sections use the full available height.
3. **Given** I collapse all sections, **When** I look at the sidebar, **Then** only the width control and section headers are visible.

---

### User Story 3 — Consolidated PM4 Workbench (Priority: P2)

As a developer working with PM4 data, I need the PM4 workbench controls to be accessible from the main right sidebar instead of requiring a separate panel.

**Why this priority**: The PM4 workbench is currently a separate panel that requires switching. Consolidating it into the right sidebar reduces context switching.

**Independent Test**: Open PM4 overlay controls from the right sidebar. Verify all PM4 toggles (overlay, solid fill, x-ray, bounds, CK24 bounds, MPRL refs, centroids) are accessible.

**Acceptance Scenarios**:

1. **Given** the right sidebar is visible and PM4 overlay is active, **When** I look at the "World Tools" section, **Then** all PM4 overlay controls are present.
2. **Given** PM4 overlay is active, **When** I toggle "PM4 CK24 Bounds", **Then** one merged bounding box per CK24 object appears in the viewport.
3. **Given** PM4 overlay is active, **When** I toggle "PM4 Bounds", **Then** per-sub-object bounding boxes appear (existing behavior preserved).

---

### User Story 4 — Cleaned Training UI (Priority: P3)

As a developer, I need training-related UI elements removed from the right sidebar since training is now handled by the separate data-harvester pipeline.

**Why this priority**: Training UI in MdxViewer is stale and confusing — training happens in `wow-viewer/data-harvester/`, not in the viewer.

**Independent Test**: Verify no training-related controls (dataset scan, model training, inference) appear in the right sidebar.

**Acceptance Scenarios**:

1. **Given** MdxViewer is running, **When** I look at the right sidebar, **Then** there are no training-specific controls (dataset-scan, train, infer buttons).
2. **Given** MdxViewer is running, **When** I look at the right sidebar, **Then** the "Utilities" section contains only viewer-relevant diagnostics.

---

### Edge Cases

- What happens when the sidebar is resized to the minimum (260px) with many sections open? → Sections should scroll vertically.
- What happens when the viewport is very narrow (e.g., 1024px) and both sidebars are open? → Each sidebar clamps to its compact min width (180px), viewport gets remaining space.
- What happens to saved sidebar width when upgrading from v0.4.7 to v0.5.0? → Settings file is forward-compatible; missing fields use defaults.

## Requirements

### Functional Requirements

- **FR-001**: Right sidebar max width MUST be at least 1080px (increased from 720px).
- **FR-002**: Right sidebar default width MUST be 360px (increased from 320px).
- **FR-003**: Sidebar width MUST be persisted in `viewer_settings.json` and restored on launch.
- **FR-004**: Each section in the right sidebar MUST be collapsible via its header.
- **FR-005**: PM4 overlay controls MUST be accessible from the "World Tools" section of the right sidebar.
- **FR-006**: "PM4 CK24 Bounds" checkbox MUST render one merged bounding box per CK24 object.
- **FR-007**: "PM4 Bounds" checkbox MUST render per-sub-object bounding boxes (existing behavior).
- **FR-008**: Version displayed in the About dialog and window title MUST be `0.5.0`.
- **FR-009**: Training-specific UI elements MUST NOT appear in the right sidebar.
- **FR-010**: The left sidebar MUST remain locked with its current panel set (Navigator, File Browser, World Maps, Runtime Stats).

### Key Entities

- **ShellPanel**: A UI panel in the sidebar system. Right sidebar contains: Viewer Settings, Selection, World Tools, Utilities.
- **Pm4OverlayObject**: A PM4 overlay object with CK24 grouping, sub-object partitioning, and bounding box data.
- **ViewerSettings**: Persisted UI state including sidebar widths, section open/closed state, and overlay toggles.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All text in the right sidebar is fully readable at 600px+ width without truncation.
- **SC-002**: Sidebar resize操作 is smooth (no frame drops) from 260px to 1080px.
- **SC-003**: PM4 CK24 Bounds shows one box per CK24 object (not per sub-object).
- **SC-004**: Version 0.5.0 appears in the About dialog.
- **SC-005**: No training-related controls appear in the right sidebar.

## Assumptions

- The existing ImGui-based sidebar infrastructure (splitter, width control, section headers) is reused.
- The left sidebar remains unchanged — only the right sidebar is simplified.
- Saved settings from v0.4.7 are forward-compatible with v0.5.0.
- PM4 overlay controls in the PM4 workbench panel are moved into the "World Tools" section of the right sidebar, not duplicated.
- The "Utilities" section retains diagnostic tools but removes training-specific commands.
