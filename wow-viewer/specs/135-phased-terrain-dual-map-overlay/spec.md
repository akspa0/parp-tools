# Feature Specification: Phased Terrain Dual-Map Overlay

**Feature Branch**: `135-phased-terrain-dual-map-overlay`  
**Created**: 2026-08-07  
**Status**: Draft  
**Input**: User request: "for 4.x maps, we need a feature that can swap in 2 maps of tiles at once, without unloading one maps' tiles, for phased terrain. For instance, We can have the azeroth map loaded, but also want to load in tiles from Gilneas and Gilneas2, as well as possibly loading even more child maps in from GilneasPhase1 or GilneasPhase2. In order for this to work, we need to be able to load from two map folders at the same time, and simply swap out the tiles that the 2nd map has, from the first map. Later versions of WoW include many parent map id's and cosmetic map id's as well as other types of overlay tiles, but we only have to worry about supporting two for now. use speckit"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Secondary Overlay Map Terrain Patch (Priority: P1)

As a WoW map researcher or viewer user, when I load a primary terrain map (such as `Azeroth`), I want to specify an active secondary overlay map folder (such as `Gilneas`, `Gilneas2`, `GilneasPhase1`, or `GilneasPhase2`), so that sparse phase data present in the secondary map folder patches the primary map at matching tile/chunk coordinates, while retaining parent terrain and parent-owned liquids everywhere the phase map does not provide a patch.

**Why this priority**: Core requirement for inspecting Cataclysm 4.x phased terrain where phase-specific ADTs provide sparse changes over continent terrain locally.

**Independent Test**: Can be tested by loading map `Azeroth` with overlay map `Gilneas` configured. Tiles at Gilneas coordinates (e.g. tile row/col $(48, 20)$) load from `World\Maps\Gilneas\Gilneas_20_48.adt`, while surrounding Eastern Kingdoms tiles load from `World\Maps\Azeroth\Azeroth_X_Y.adt`.

**Acceptance Scenarios**:

1. **Given** primary map `Azeroth` and secondary overlay map `Gilneas` are configured, **When** a phase MCNK exists at a matching coordinate, **Then** that MCNK's terrain/layer/alpha payload and phase placements are merged into the parent tile, while the parent MCNK's liquid remains authoritative.
2. **Given** primary map `Azeroth` and secondary overlay map `Gilneas` are configured, **When** `Gilneas` has no tile or MCNK patch at a coordinate, **Then** the corresponding parent tile/chunk remains loaded from `Azeroth`.
3. **Given** a phased tile is loaded from `Gilneas`, **Then** the rest of the resident world tiles from `Azeroth` remain loaded without unloading or resetting camera state.

---

### User Story 2 - Dynamic Overlay Map Switching in Viewer UI (Priority: P2)

As a viewer user, I want a control in the World / Terrain UI panel to select or type a secondary overlay map name (or clear it), so that I can switch phase overlays (e.g. from `Gilneas` to `Gilneas2`) on the fly while staying in the world.

**Why this priority**: Enables interactive phase comparison without re-opening or re-instantiating the viewer world scene.

**Independent Test**: In `ViewerApp`, open `Azeroth`, set secondary map overlay to `Gilneas`, verify phased tiles load. Change overlay to `Gilneas2`, verify affected tiles refresh to `Gilneas2` geometry/textures.

**Acceptance Scenarios**:

1. **Given** a world scene loaded with primary map `Azeroth`, **When** the user enters `Gilneas` into the Secondary Overlay Map input field and clicks Apply, **Then** only the tiles present in `Gilneas` are invalidated and re-streamed/re-loaded.
2. **Given** an active secondary overlay map, **When** the user clears the secondary overlay map input, **Then** the phased tiles revert back to the primary map's ADTs.

---

### User Story 3 - Companion Split ADT File Support for Overlay Maps (Priority: P3)

As a viewer developer, I want secondary overlay map tile resolution to respect Cataclysm split ADT companion files (`_tex0.adt` and `_obj0.adt`), so that overlay map textures and placements load correctly when the secondary map uses split ADT profiles.

**Why this priority**: Cataclysm 4.x client builds store textures in `_tex0.adt` and placement definitions in `_obj0.adt`.

**Independent Test**: Load a 4.x split ADT overlay map (e.g. `Gilneas` in 4.3.4), verify `Gilneas_20_48_tex0.adt` and `Gilneas_20_48_obj0.adt` are read from the overlay map directory for texture and object placement payload data.

**Acceptance Scenarios**:

1. **Given** a 4.x split ADT overlay map with `_tex0.adt` and `_obj0.adt` companion files, **When** loading an overlay tile, **Then** companion files are fetched from `World\Maps\<OverlayMapName>\` matching the overlay tile base path.

---

## Edge Cases

- **Overlay Map Missing/Invalid**: If the specified secondary overlay map directory does not exist or contains no valid tiles, the system logs a warning and cleanly falls back to primary map tiles for all coordinates.
- **WDT Header Mismatch**: The primary map WDT determines the global coordinate grid and tile availability mask (`MAIN` chunk). Overlay tiles use the primary grid coordinates $(tileX, tileY)$.
- **Concurrent Streaming**: When an overlay map is toggled mid-flight while tiles are streaming in background threads, in-flight background loads for affected tiles complete safely or are invalidated before GPU mesh upload.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `ITerrainAdapter` / `StandardTerrainAdapter` MUST support an optional `OverlayMapName` property or parameter representing a secondary map folder.
- **FR-002**: When `OverlayMapName` is non-null/non-empty, `StandardTerrainAdapter.TileExists(tileX, tileY)` MUST return `true` if either the primary map or the secondary overlay map has an ADT tile at $(tileX, tileY)$.
- **FR-003**: `StandardTerrainAdapter.LoadTileWithPlacements(tileX, tileY)` MUST parse the parent tile first, merge sparse overlay MCNKs by chunk coordinate, remap overlay MTEX indices into the merged tile texture table, and preserve parent liquid data.
- **FR-004**: `TerrainManager` MUST provide a method `SetOverlayMap(string? overlayMapName)` that updates the terrain adapter's overlay map setting, invalidates tile cache entries for tiles affected by the overlay map, and triggers tile re-streaming/re-upload.
- **FR-005**: `WorldScene` MUST expose `SecondaryOverlayMap` and forward overlay map updates to `TerrainManager`.
- **FR-006**: `ViewerApp` UI (World Scene / Terrain panel) MUST display an input field for `Secondary Overlay Map` allowing users to type or clear an overlay map name and apply it to the active scene.
- **FR-007**: Overlay map loading MUST preserve parent terrain/liquid data not covered by a phase patch, merge phase terrain normals and texture/alpha layers at matching MCNK coordinates, and retain both parent and phase placements.
- **FR-008**: Overlay map resolution MUST operate without unloading or resetting resident primary map tiles outside the overlay map's footprint.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Swapping a secondary overlay map (e.g. `Azeroth` + `Gilneas`) patches only the affected MCNKs (e.g. the Gilneas region), preserves parent liquids, and leaves 100% of surrounding resident continent terrain loaded and rendered without visual disruption.
- **SC-002**: Tile load time for overlay tiles is identical to standard ADT tile load time (< 50ms per tile on SSD).
- **SC-003**: Reverting or clearing the secondary overlay map restores primary map tiles at those coordinates in a single frame update pass.

---

## Assumptions

- Phased terrain overlay in 4.x operates on a 2-map composition model (Primary Continent Map + Secondary Phase Overlay Map). Multi-layered (> 2 maps) phase cascades are out of scope for this initial implementation.
- Both primary and secondary map ADTs share the standard WoW $64 \times 64$ tile coordinate grid.
- Game client data sources (`MPQ` or loose disk folders) contain `World\Maps\<OverlayMapName>\<OverlayMapName>_{x}_{y}.adt` files.
