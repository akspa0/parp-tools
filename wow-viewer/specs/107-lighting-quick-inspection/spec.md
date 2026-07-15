# Feature Specification: Lighting Quick Controls and Confident Hover Inspection

**Feature Branch**: `107-lighting-quick-inspection`

**Created**: 2026-07-15

**Status**: Implementing

**Input**: Make lighting and LIT controls obvious in the Quick tab, use a safe FogEnd to prevent distant terrain fog artifacts, and show precise hover inspection only for confident matches.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Control the active lighting where the scene is controlled (Priority: P1)

A world viewer user can see the active LIT state and change time-of-day and fog from Quick without hunting through a separate utility panel.

**Independent Test**: Load a lit world, open Tools > Quick, adjust time/fog, and reach the detailed LIT inspector in one action.

**Acceptance Scenarios**:

1. **Given** a terrain world, **When** the user opens Quick, **Then** current lighting source, time, fog range, and a link to detailed LIT evidence are visible.
2. **Given** a LIT or DBC fog sample, **When** its FogEnd changes, **Then** rendering culls beyond a bounded safety margin instead of exposing distant blended terrain.

### User Story 2 - Trust hover inspection (Priority: P1)

A user sees a hover card only when one scene asset is a confident match; ambiguous nearby brush matches do not pop up a misleading exact asset path.

**Independent Test**: Exercise a single ray hit and a multi-candidate brush hit; the first displays one precise card while the second suppresses the card.

**Acceptance Scenarios**:

1. **Given** one nearest valid ray hit, **When** it is hovered, **Then** the card identifies that exact placement and path.
2. **Given** multiple brush candidates or an unresolved candidate, **When** it is hovered, **Then** the card is suppressed and click inspection remains available.

### Edge Cases

- No terrain world: Quick explains that lighting controls require terrain.
- Invalid/missing fog data: existing safe defaults apply and far plane remains finite.
- A user disables hover cards: no card is shown regardless of confidence.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Quick MUST expose active time of day, FogStart, FogEnd, LIT override state, and direct navigation to detailed Lighting/LIT inspection.
- **FR-002**: The scene far plane MUST be derived from the active fog end plus a small padding and MUST NOT impose the existing 6000-unit minimum that defeats valid short fog ranges.
- **FR-003**: Hover cards MUST require a confident single scene-ray hit; brush-only, multi-candidate, and unresolved hits MUST not display an exact-path card.
- **FR-004**: Click inspection behavior remains available when hover-card display is suppressed.
- **FR-005**: Fog and hover confidence behavior MUST be covered by focused automated tests or pure helper tests.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Quick exposes all active lighting controls in one panel and reaches detailed LIT status in one action.
- **SC-002**: A valid FogEnd below 6000 produces a far plane within the defined padding rather than 6000.
- **SC-003**: Ambiguous hover candidates produce zero hover cards; a single ray hit produces one precise card.

## Assumptions

- Existing LIT/DBC evaluation remains the source of FogEnd; this feature fixes visibility consumption and control discoverability, not the data readers.
- The detailed Lighting utility remains the authority for full diagnostics; Quick is its scene-control entry point.
