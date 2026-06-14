# Feature Specification: PM4 Workbench Cleanup and UI Simplification

**Feature Branch**: `005-pm4-workbench-cleanup`

**Created**: 2026-05-21

**Status**: Draft

**Input**: User description: "Complete cleanup of MdxViewer's PM4 workbench, which is meant to be a window not a panel, and simplify the use of the UI to filter data."

## Context

The PM4 workbench in MdxViewer is currently embedded as a panel in the right sidebar. The user wants it to be a **floating window** (like the MCNK Explorer) that can be moved, resized, and positioned independently. The UI also needs simplification — too many controls, too much text, and the data filtering is unclear.

### Known Uncertainty

**CK24Type** is a real byte value extracted from `MSUR.PackedParams` bits 24-31. However, the semantic labels ("WMO", "M2 Interior", "M2 Exterior") applied to those values came from WoWRollback code that may contain hallucinations. The type byte itself is real; the interpretation is unverified.

## User Scenarios & Testing

### User Story 1 — Floating PM4 Workbench Window (Priority: P1)

As a developer, I need the PM4 workbench to be a floating window that I can move, resize, and position independently of the right sidebar.

**Why this priority**: The PM4 workbench is the primary tool for PM4 data analysis. Being stuck in the sidebar limits screen real estate and makes it hard to use alongside the 3D viewport.

**Independent Test**: Open MdxViewer, enable PM4 overlay, open the PM4 workbench as a floating window. Verify it can be moved, resized, and positioned anywhere on screen.

**Acceptance Scenarios**:

1. **Given** MdxViewer is running with PM4 overlay enabled, **When** I click "PM4 Workbench" in the menu or toolbar, **Then** a floating window appears with all PM4 controls.
2. **Given** the PM4 workbench window is open, **When** I drag it by its title bar, **Then** it moves to the new position.
3. **Given** the PM4 workbench window is open, **When** I drag its edge/corner, **Then** it resizes smoothly.
4. **Given** the PM4 workbench window is open, **When** I close it and reopen it, **Then** its position and size are restored from saved settings.

---

### User Story 2 — Simplified PM4 Filtering (Priority: P1)

As a developer, I need simple, clear controls to filter which PM4 objects are visible in the viewport.

**Why this priority**: The current UI has too many controls and the filtering logic is unclear. Simple toggles for object types would be more useful than the current complex panel.

**Independent Test**: Open PM4 workbench, toggle object type filters. Verify only matching objects appear in the viewport.

**Acceptance Scenarios**:

1. **Given** the PM4 workbench is open, **When** I toggle "Show WMOs", **Then** only WMO-type PM4 objects (CK24 type 0x42/0x43) are visible.
2. **Given** the PM4 workbench is open, **When** I toggle "Show M2s", **Then** only M2-type PM4 objects (CK24 type 0x40/0x41) are visible.
3. **Given** the PM4 workbench is open, **When** I toggle "Show Nav Mesh", **Then** only terrain/nav-mesh objects (CK24=0) are visible.
4. **Given** the PM4 workbench is open, **When** I toggle "Show All", **Then** all PM4 objects are visible regardless of type.
5. **Given** the PM4 workbench is open, **When** I look at the filter controls, **Then** each filter shows a count of matching objects.

---

### User Story 3 — Simplified PM4 Overlay Controls (Priority: P2)

As a developer, I need the PM4 overlay rendering controls (solid fill, x-ray, bounds, etc.) to be in the floating workbench window, not scattered across the sidebar.

**Why this priority**: Currently PM4 rendering controls are in the right sidebar. Moving them to the floating workbench keeps all PM4 tools in one place.

**Independent Test**: Open PM4 workbench, toggle overlay rendering options. Verify the viewport updates accordingly.

**Acceptance Scenarios**:

1. **Given** the PM4 workbench is open, **When** I toggle "Solid Fill", **Then** PM4 surfaces render with solid fill in the viewport.
2. **Given** the PM4 workbench is open, **When** I toggle "X-Ray", **Then** PM4 surfaces render without depth testing.
3. **Given** the PM4 workbench is open, **When** I toggle "CK24 Bounds", **Then** one merged bounding box per CK24 group appears.
4. **Given** the PM4 workbench is open, **When** I toggle "MPRL Refs", **Then** position reference markers appear.

---

### User Story 4 — CK24Type Uncertainty Handling (Priority: P2)

As a developer, I need the PM4 workbench to clearly label which data is verified and which is uninterpreted.

**Why this priority**: The CK24Type byte is real but its semantic labels are unverified. The UI should not present unverified labels as facts.

**Independent Test**: Open PM4 workbench, select a PM4 object. Verify the type display shows the raw hex value, not an unverified semantic label.

**Acceptance Scenarios**:

1. **Given** a PM4 object is selected, **When** I look at its type display, **Then** it shows `type=0x42` (raw hex), not "WMO" (unverified label).
2. **Given** a PM4 object is selected, **When** I hover over the type field, **Then** a tooltip explains "CK24 type byte from MSUR.PackedParams bits 24-31. Semantic meaning unverified."

---

### User Story 5 — Right Sidebar Cleanup (Priority: PM4-specific)

As a developer, I need the right sidebar to be cleaned up now that PM4 controls are in a floating window.

**Why this priority**: With PM4 workbench as a floating window, the right sidebar has less content and should be simpler.

**Independent Test**: Verify the right sidebar no longer contains PM4-specific controls. Verify it only has Viewer Settings, Selection, and Utilities sections.

**Acceptance Scenarios**:

1. **Given** the PM4 workbench is a floating window, **When** I look at the right sidebar, **Then** there is no "World Tools" section (PM4 controls moved to floating window).
2. **Given** the right sidebar, **When** I look at the remaining sections, **Then** they are: Viewer Settings, Selection, Utilities.

---

### Edge Cases

- What happens when the PM4 workbench window is dragged off-screen? → ImGui clamps to viewport bounds.
- What happens when multiple PM4 workbench windows are opened? → Only one instance allowed; opening a second focuses the existing one.
- What happens when PM4 overlay is disabled while the workbench is open? → Workbench window stays open but controls are greyed out.

## Requirements

### Functional Requirements

- **FR-001**: PM4 workbench MUST be a floating ImGui window (not a sidebar panel).
- **FR-002**: PM4 workbench window position and size MUST be persisted in viewer settings.
- **FR-003**: PM4 object type filters MUST show: Show All, Show WMOs, Show M2s, Show Nav Mesh.
- **FR-004**: Each filter MUST show a count of matching visible objects.
- **FR-005**: PM4 overlay rendering controls (solid fill, x-ray, bounds, CK24 bounds, MPRL refs, centroids) MUST be in the floating workbench.
- **FR-006**: CK24 type display MUST show raw hex values, not unverified semantic labels.
- **FR-007**: Right sidebar MUST NOT contain PM4-specific controls after cleanup.
- **FR-008**: Version MUST remain 0.5.0.

### Key Entities

- **Pm4WorkbenchWindow**: Floating ImGui window containing all PM4 analysis controls.
- **Pm4TypeFilter**: Toggle for filtering PM4 objects by CK24 type byte.
- **Pm4OverlayControls**: Rendering options (solid fill, x-ray, bounds, etc.).

## Success Criteria

### Measurable Outcomes

- **SC-001**: PM4 workbench is a floating window that can be moved and resized.
- **SC-002**: PM4 type filters correctly hide/show objects by CK24 type.
- **SC-003**: Right sidebar has no PM4-specific controls.
- **SC-004**: CK24 type display shows raw hex, not semantic labels.
- **SC-005**: All PM4 controls are accessible from the floating workbench window.

## Assumptions

- The existing ImGui window infrastructure (floating windows, settings persistence) is reused.
- The MCNK Explorer floating window is the reference implementation for floating window behavior.
- CK24Type byte values are real; only the semantic labels are uncertain.
- The PM4 overlay rendering pipeline in WorldScene.cs is unchanged — only the UI controls move.
