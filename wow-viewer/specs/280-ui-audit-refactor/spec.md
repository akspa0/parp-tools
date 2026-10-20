# Feature Specification: UI Audit and Pre-Alpha-Style Refactor

**Feature Branch**: `280-ui-audit-refactor`

**Created**: 2026-10-20

**Status**: Draft

**Input**: User description: "Full UI audit and pre-alpha-style refactor for wow-viewer: bottom-bar menu buttons, keep FPS/stats bar, options toolbar as a sub-menu nuzzled to the chat, consistent fonts/styles across all sidebars, remove dead/duplicate UI, verify functionality behind every control, separate Editor/Viewer/Archaeology surfaces, rename WC3 data support to 'WC3 Beta'."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Audit every UI surface and its functionality (Priority: P1)

An operator opens the viewer and walks every visible surface — top menu, bottom bar, left navigator, right inspector, workbench tabs, popups, and status bars — and records for each control whether it is reachable, what it does, and whether it is duplicated or dead. The audit produces a single inventory that names the owner of each surface and the action behind each control.

**Why this priority**: The operator cannot trust the UI until every control is known to do something. The audit is the evidence base for every later removal or move.

**Independent Test**: The inventory lists every visible control with a status of `works`, `dead`, `duplicate`, or `misplaced`, and a reviewer can spot-check any row against the running viewer.

**Acceptance Scenarios**:

1. **Given** the viewer is running with a world loaded, **When** the operator opens each top-level surface, **Then** every control is present in the inventory with an owner and a status.
2. **Given** a control marked `dead`, **When** the operator clicks it, **Then** the inventory's claim is confirmed (no visible effect, no file written, no state change).
3. **Given** a control marked `duplicate`, **When** the operator compares it to its twin, **Then** both are listed with the same action and a recommended survivor.

---

### User Story 2 - Bottom-bar navigation replaces scattered tabs (Priority: P1)

The operator uses a single bottom bar as the primary navigation: menu buttons for the major destinations, the FPS/stats readout kept in place, and the options toolbar moved into a sub-menu that opens next to the chat. The top menu bar is reduced to the few global actions that do not belong in the bottom bar.

**Why this priority**: The current UI spreads navigation across a top menu, a bottom bar, a left navigator, and several tab strips. Consolidating navigation is the largest real-estate and clarity win.

**Independent Test**: With the new bar, the operator can reach Viewer, Editor, Archaeology, and the options sub-menu from the bottom bar alone, and the FPS/stats readout is still visible.

**Acceptance Scenarios**:

1. **Given** the viewer is running, **When** the operator looks at the bottom bar, **Then** the major destinations are visible as menu buttons and the FPS/stats readout is present.
2. **Given** the operator clicks the options button, **When** the sub-menu opens, **Then** it is anchored next to the chat and contains the render/overlay toggles that used to live in the bottom toolbar.
3. **Given** the operator switches destination, **When** the destination changes, **Then** the left and right sidebars show that destination's surfaces and no other destination's surfaces.

---

### User Story 3 - One visual language across all sidebars (Priority: P2)

The operator sees the same font, text size, spacing, header style, and control style in every sidebar, popup, and workbench page. No surface uses a bespoke section style.

**Why this priority**: The current mix of old and new styling makes the tool look unfinished and makes controls hard to scan.

**Independent Test**: A reviewer opens every sidebar and confirms that section headers, body text, buttons, and spacing match the shared widget set.

**Acceptance Scenarios**:

1. **Given** any two sidebars, **When** the operator compares their section headers and body text, **Then** they use the same font, size, and spacing.
2. **Given** a new section is added, **When** it is drawn, **Then** it uses the shared section-header and body-text primitives rather than a local style.

---

### User Story 4 - Destination-specific surfaces (Priority: P2)

The operator sees only the surfaces that belong to the current destination: Viewer emphasizes the Inspector, Editor shows editor menus and editor tools, and Archaeology shows its own analysis surfaces. Cross-destination shortcuts are explicit and few.

**Why this priority**: The current UI mixes editor, viewer, and archaeology controls in the same panels, which is the main source of confusion.

**Independent Test**: Switching destination changes the visible sidebar content, and no editor-only control appears in Viewer or Archaeology.

**Acceptance Scenarios**:

1. **Given** the Viewer destination, **When** the operator opens the right sidebar, **Then** the Inspector and its PM4/selection surfaces are present and editor-only tools are absent.
2. **Given** the Editor destination, **When** the operator opens the right sidebar, **Then** editor menus and tools are present and viewer-only inspection surfaces are absent.
3. **Given** the Archaeology destination, **When** the operator opens the right sidebar, **Then** archaeology analysis surfaces are present and unrelated editor/viewer tools are absent.

---

### User Story 5 - WC3 Beta naming (Priority: P3)

The operator sees the WC3 data support labelled **WC3 Beta** everywhere it appears, so the scope is clear: only clients without proper modern tooling are supported.

**Why this priority**: The label is a small change but it sets the correct expectation for the feature's support level.

**Independent Test**: A search of the UI strings finds no "WC3" label without the "Beta" qualifier.

**Acceptance Scenarios**:

1. **Given** the file browser or data-source picker, **When** the operator looks for WC3 support, **Then** it is labelled "WC3 Beta".
2. **Given** the documentation that mentions WC3 support, **When** the operator reads it, **Then** it uses "WC3 Beta".

---

### Edge Cases

- A control that is dead but is the only path to a feature: the audit must mark it `dead` and the refactor must either wire it or remove the feature from the UI, not leave a dead entry point.
- A control that is duplicated across destinations: the refactor must keep one authoritative home and replace the other with a navigation shortcut.
- A destination with no content yet: the refactor must show an explicit empty state, not a blank panel.
- A persisted layout from the old UI: the refactor must migrate or reset it without crashing.
- A user with a small window: the bottom bar and options sub-menu must remain usable and must not cover the chat or the status bars.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The audit MUST inventory every visible UI control with its surface, owner, action, and status (`works`, `dead`, `duplicate`, `misplaced`).
- **FR-002**: The audit MUST identify every duplicate control and name the single authoritative survivor.
- **FR-003**: The bottom bar MUST expose the major destinations as menu buttons.
- **FR-004**: The bottom bar MUST keep the FPS/stats readout visible.
- **FR-005**: The options toolbar MUST move into a sub-menu anchored next to the chat.
- **FR-006**: The top menu bar MUST be reduced to global actions that do not belong in the bottom bar.
- **FR-007**: Every sidebar MUST use the shared section-header, body-text, and control primitives.
- **FR-008**: The Viewer destination MUST emphasize the Inspector and MUST NOT show editor-only tools.
- **FR-009**: The Editor destination MUST show editor menus and editor tools and MUST NOT show viewer-only inspection surfaces.
- **FR-010**: The Archaeology destination MUST show its own analysis surfaces and MUST NOT show unrelated editor/viewer tools.
- **FR-011**: Every control that remains visible MUST have a working action; dead controls MUST be removed or wired.
- **FR-012**: WC3 data support MUST be labelled "WC3 Beta" in the UI and documentation.
- **FR-013**: The refactor MUST preserve the existing PM4, population, terrain, and model functionality.
- **FR-014**: The refactor MUST provide an explicit empty state for any destination with no content.
- **FR-015**: The refactor MUST migrate or reset persisted layouts from the old UI without crashing.

### Key Entities

- **UI Surface**: A named region of the viewer (top menu, bottom bar, left navigator, right inspector, workbench page, popup, status bar).
- **UI Control**: A button, checkbox, slider, combo, or menu item with an owner surface and an action.
- **Destination**: Viewer, Editor, or Archaeology; determines which surfaces are visible.
- **Audit Row**: One control's surface, owner, action, status, and recommended disposition.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of visible controls appear in the audit inventory with a status.
- **SC-002**: 0 dead controls remain visible after the refactor.
- **SC-003**: 0 duplicate controls remain without a named authoritative survivor.
- **SC-004**: The operator can reach Viewer, Editor, Archaeology, and the options sub-menu from the bottom bar in at most two clicks each.
- **SC-005**: Every sidebar uses the shared section-header and body-text primitives; 0 bespoke section styles remain.
- **SC-006**: Switching destination changes the visible sidebar content; 0 editor-only controls appear in Viewer or Archaeology.
- **SC-007**: The FPS/stats readout remains visible in every destination.
- **SC-008**: The options sub-menu opens next to the chat and does not cover the chat or the status bars at the default window size.
- **SC-009**: 0 UI strings label WC3 support without "Beta".
- **SC-010**: The existing PM4, population, terrain, and model workflows still run after the refactor.

## Assumptions

- The operator's pre-alpha reference is a layout and interaction reference, not a pixel-perfect art requirement.
- The existing shared widget set (`SharedUiWidgets`) is the intended styling base; the refactor extends it rather than inventing a second system.
- The bottom bar is the primary navigation; the top menu keeps only global actions such as file open, settings, and help.
- The options sub-menu is anchored to the chat bar because the chat bar is already the bottom-of-screen control strip.
- The audit is a document plus a checklist, not a runtime diagnostic overlay.
- WC3 Beta support is limited to clients without proper modern tooling; broader WC3 support is out of scope.
- The refactor is staged: audit first, then navigation, then styling, then destination separation, then naming.
