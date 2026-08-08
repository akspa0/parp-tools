# Spec 137: Phased Minimap Overlay & Consistent Minimap Teleport Interaction

## Overview

1. **Phased Minimap Tiles**: When a secondary phased overlay map (e.g. `Gilneas`, `Gilneas2`, `GilneasPhase1`) is active in the world scene, the minimap panel should display the secondary map's minimap tile BLP textures for tiles that exist in the overlay, falling back to the base map's minimap tile BLP textures for unaffected tiles.
2. **Consistent Triple-Click Teleportation**: Currently, clicking on the small dockable minimap requires a 3-click confirmation to armed-teleport (`MinimapTeleportMode.Armed`), whereas clicking on the fullscreen minimap (`M` key overlay) immediately teleports on a single click (`MinimapTeleportMode.Immediate`). Teleport interaction should be consistent across both small and fullscreen minimap modes using armed 3-click confirmation (or consistent configurable teleport behavior).

## User Stories

### User Story 1 - Phased Minimap Tile Overlay (Priority: P1)

As a viewer user, I want the minimap panel to show tile textures from the active secondary overlay map (when available), so that the minimap matches the phased world terrain currently rendered on screen.

**Acceptance Criteria**:
1. When `WorldScene.SecondaryOverlayMap` (or `TerrainManager.OverlayMapName`) is set, `MinimapHelpers` queries `MinimapRenderer` for the secondary overlay map's tile BLP texture first.
2. If the secondary map tile BLP texture is found and loaded, it renders on the minimap surface for that tile.
3. If no secondary map tile BLP texture exists for a tile, the minimap seamlessly falls back to the primary base map's tile BLP texture.

### User Story 2 - Consistent Triple-Click Teleport (Priority: P1)

As a viewer user, I want clicking on the fullscreen minimap to use the same 3-click armed teleport confirmation as the small minimap panel, so that accidental single clicks on the fullscreen map do not unintendedly warp the camera across the world.

**Acceptance Criteria**:
1. Fullscreen minimap uses `MinimapTeleportMode.Armed` matching the small dockable minimap panel.
2. Clicking a tile on either minimap arms the teleport; 3 consecutive clicks on the same tile confirm and execute the teleportation.
