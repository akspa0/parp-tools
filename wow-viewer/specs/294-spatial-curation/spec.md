# Feature Specification: Spatial Donor Transforms and Manual PM4 Curation

**Feature Branch**: `294-spatial-curation`

**Created**: 2026-11-05

**Status**: Specified — ready for planning

**Input**: User description: “I need to be able to rotate things not by 90 degrees, but a custom amount, and some way to do this in the viewer UI itself, the ability to grab the whole donor map tiles as a single object and then place it properly in place, would be great. If we could do the same with objects, then we could do REAL human curation where we place objects by hand and pair the relationships properly that way.”

## User Scenarios & Testing

### User Story 1 — Transform a donor tile group (Priority: P1)

As a map restorer, I want to select multiple donor tiles and manipulate them as one group so that I can align a donor region to its target without adjusting every tile independently.

**Why this priority**: The current donor picker supports selecting tiles, while the layer transform workflow is limited to exact quarter-turns. Group manipulation is the foundation for efficient spatial alignment.

**Independent Test**: Select a known set of donor tiles, move the selection, and verify every tile retains its relative position while the group moves as one unit.

**Acceptance Scenarios**:

1. **Given** a donor map with several selected tiles, **When** the user moves the group, **Then** every selected tile moves together and the unselected tiles remain unchanged.
2. **Given** a selected group, **When** the user previews and applies a transform, **Then** the transform origin and resulting placement are visible before acceptance.
3. **Given** the group has been transformed and saved, **When** the layer project is reopened, **Then** the same selected tiles and transform are restored.

---

### User Story 2 — Apply an arbitrary rotation amount (Priority: P1)

As a map restorer, I want to rotate a selected donor group by a user-chosen angle rather than only by 90-degree steps so that irregular coastlines and terrain features can be aligned more accurately.

**Why this priority**: The screenshot shows a layer rotation control whose choices are quarter turns. That is insufficient for the operator’s requested hand-alignment workflow.

**Independent Test**: Apply a non-quarter-turn value such as 17 degrees to a selected group, inspect the preview and placement, then undo and redo the operation.

**Acceptance Scenarios**:

1. **Given** a selected donor group, **When** the user enters a rotation angle that is not a multiple of 90 degrees, **Then** the preview reflects that angle and the UI reports the applied value.
2. **Given** an arbitrary-angle preview, **When** the user cancels it, **Then** the selected group returns to its prior state without changing its source.
3. **Given** the user accepts the transform, **When** the project is saved and reopened, **Then** the group’s angle and placement are retained.

---

### User Story 3 — Place and pair objects by hand (Priority: P1)

As a human curator, I want to place or reposition an object in the viewer and explicitly pair it with the corresponding PM4 hull so that curated object relationships are visible and reviewable rather than inferred only by automation.

**Why this priority**: PM4 hulls provide the structural reference for restoration. Direct manipulation plus explicit pairing gives the operator control over cases where automated matching is uncertain or wrong.

**Independent Test**: Select an object and PM4 hull, create a pair, inspect the relation in the workbench, and verify it survives save/reopen without creating a duplicate placement.

**Acceptance Scenarios**:

1. **Given** a selectable object and a PM4 hull, **When** the user creates a pair, **Then** the viewer clearly identifies both members of the relationship.
2. **Given** a selected object, **When** the user moves or rotates it by hand, **Then** the resulting placement and any linked hull remain visibly associated.
3. **Given** an object is already paired, **When** the user attempts to create the same pair again, **Then** the viewer reports the existing relation and does not silently duplicate it.
4. **Given** a curated pair, **When** the user saves and reopens the curation project, **Then** the object placement and pair are restored while source map files remain unchanged.

## Edge Cases

- A selected tile group spans the edge of the 64×64 map or contains gaps between selected tiles.
- The transform would move tiles outside the map extent or cause multiple donor tiles to claim the same target.
- The rotation angle is zero, negative, outside the normal revolution, or has fractional degrees.
- The selected pivot lies outside the selected group or the group contains only one tile.
- A transformed terrain feature crosses chunk/tile boundaries or includes holes, textures, liquids, placements, and height data with different resampling needs.
- An object has no resolved bounds, is already paired, or shares a hull with a different object.
- The user cancels, undoes, redoes, or reloads after a partially completed transform or pair operation.
- A requested save target cannot preserve the transform or relationship without rewriting source map data.

## Requirements

### Functional Requirements

- **FR-001**: Users MUST be able to select multiple donor tiles and treat that selection as one transformable group.
- **FR-002**: Users MUST be able to translate the selected donor group while preserving the relative arrangement of its members.
- **FR-003**: Users MUST be able to specify a rotation angle in degrees that is not restricted to 90-degree increments.
- **FR-004**: The viewer MUST show the transform origin, angle, and resulting group placement before the user accepts the operation.
- **FR-005**: Users MUST be able to cancel, undo, and redo group transforms without changing the source donor data.
- **FR-006**: The existing exact quarter-turn workflow MUST remain available for transformations that can be performed exactly on the map grid.
- **FR-007**: Arbitrary-angle donor-group transforms MUST be non-destructive: the source donor data remains unchanged and the transform is stored as project state; this feature does not resample or materialize terrain channels.
- **FR-008**: Users MUST be able to manually move and rotate an object in the viewer and explicitly associate it with a selected PM4 hull.
- **FR-009**: Manual curation MUST support repositioning existing placements and placing new objects selected from the client asset catalog.
- **FR-010**: The viewer MUST display the members and state of a manual object-to-hull relationship and MUST prevent an identical relationship from silently creating a duplicate.
- **FR-011**: The curation project MUST persist group transforms, hand-authored object placements, and PM4 hull associations. Saving this curation state MUST NOT write modified placements or terrain into source map files.
- **FR-012**: The viewer MUST preserve existing pairings and make pair removal or replacement an explicit, reviewable user action.

### Key Entities

- **Donor Tile Group**: A user-selected set of donor map tiles, retaining each tile’s identity and relative placement while manipulated as one group.
- **Group Transform**: A user-authored translation and arbitrary-angle rotation, including its origin and preview/accepted state.
- **Manual Object Placement**: A user-positioned object with a visible source identity and world placement.
- **PM4 Hull Association**: An explicit, reviewable relationship between a manual object placement and a PM4 hull; multiple distinct objects may be associated with a hull where the curation model permits it.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A user can place a selected group with a non-quarter-turn angle (including a fractional-degree value) and complete preview, accept, cancel, undo, and redo without losing the original donor selection.
- **SC-002**: After a group transform, all selected tiles retain their relative arrangement and save/reopen to the same transform state.
- **SC-003**: A user can manually place or reposition an object, create and inspect its PM4 hull association, and save/reopen the supported curation state without silent duplicate relationships.
- **SC-004**: Curation save/reopen preserves every accepted group transform, hand-authored placement, and hull association while leaving source donor data and source map files unchanged.

## Assumptions

- The donor group workflow extends the current donor tile picker and phase-layer authoring experience; it does not replace the existing exact-grid quarter-turn path.
- Angles are expressed in degrees. A default group pivot at the selected group’s center is assumed unless the operator requests a different pivot control.
- Arbitrary-angle group transforms are project-level and non-destructive; no terrain resampling or map-file materialization is included.
- Manual object curation can reposition existing placements or create a placement from the client asset catalog; placements and pair relationships persist in the curation project only.
- Manual pairing is explicit and reversible; automatic matching may suggest candidates but cannot silently create or replace a user-authored relationship.
- The feature does not modify protected format readers or `AlphaWdtWriter`, and does not write curated placements back to source map files.
- Runtime, visual, and real-client proof remains operator-owned.

## Clarifications Resolved — 2026-11-05

- Arbitrary-angle donor transforms remain non-destructive project state; they do not resample or materialize terrain data.
- Hand placement supports both moving existing placements and creating placements from the client asset catalog.
- Transforms, placements, and PM4 associations save in the curation project only; source map files are not modified.
