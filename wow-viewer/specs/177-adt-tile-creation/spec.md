# Feature Specification: ADT Tile Creation

**Feature Branch**: `177-adt-tile-creation`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [176](../176-object-transfer/spec.md).

## Scope

Create a new ADT tile in an **existing** map — empty, or seeded from an existing tile — saved to the
output directory and registered in the map's tile index so both the client and the viewer find it.
Object transfer is most useful when there is somewhere new to transfer *to*.

## User Story - Create new ADT tiles (Priority: P2)

The user creates a tile at an unoccupied coordinate and it behaves like any other tile.

**Independent Test**: Create a tile at an unoccupied coordinate, transfer objects into it, save, and
confirm it loads with its objects in a fresh session and its existence is reflected in the map's tile
index.

**Acceptance Scenarios**:

1. **Given** an unoccupied coordinate, **When** the user creates a tile, **Then** a valid ADT is
   produced that loads without error in the viewer **and in an independent tool**.
2. **Given** an existing tile as seed, **When** a new tile is created from it, **Then** the new tile's
   terrain matches the seed and its placements are copied or omitted per the user's choice.
3. **Given** a tile is created, **When** the map is reloaded, **Then** the tile index reflects it and
   the viewer streams it like any other tile.
4. **Given** the coordinate is occupied, **When** creation is attempted, **Then** it is refused with
   the existing tile named; overwriting requires explicit confirmation.
5. **Given** a map whose era uses a monolithic WDT rather than split ADTs, **When** the tile is saved,
   **Then** it is written in that era's correct form.

### Edge Cases

- A coordinate the map's WDT index cannot represent.
- Creating a tile in a map currently being streamed.
- A seed tile from a different era than the target map.
- Interrupted creation leaving a tile registered but unwritten, or written but unregistered.

## Requirements

### Functional Requirements

- **FR-001**: Create an ADT tile at an unoccupied coordinate, empty or seeded from an existing tile.
- **FR-002**: Created tiles are valid for the map's era and load in the viewer **and** an independent
  tool.
- **FR-003**: Tile creation updates the map's tile index so the tile is discoverable.
- **FR-004**: Creation at an occupied coordinate requires explicit confirmation.
- **FR-005**: Creation is a single undoable Editor Operation covering both the tile file and the index
  update — partial state is not permitted.
- **FR-006**: Uses the existing core writers; adds no serializer.
- **FR-007**: Output to the configured output directory only; no game install, no Blizzard container
  (**Constitution VII**).

## Success Criteria

- **SC-001**: A newly created tile loads in the viewer **and** in at least one independent tool.
- **SC-002**: A created tile is streamed by the viewer on reload without special handling.
- **SC-003**: An interrupted creation leaves either a complete tile+index pair or neither — never one
  without the other.
- **SC-004**: Creating a tile in an alpha-era map and in a split-ADT-era map both produce the correct
  form for that era.

## Out of Scope

- **Creating new maps** — new WDT with a new map entry and DBC map registration. This spec creates
  tiles within an existing map.
- Generating terrain. Seeded tiles **copy** terrain; synthesis is out of scope.

## Assumptions

- Terrain data (heights, textures, holes) is copied, not synthesized, when seeding from another tile.
- The map's era determines the output form; an era the plugin does not support is refused rather than
  guessed at.
