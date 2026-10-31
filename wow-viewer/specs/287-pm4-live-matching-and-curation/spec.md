# Feature Specification: Spec 287 — Live PM4 Object Duplication & Matching, Burgundy-Red MCCV Holes, and Texture Layers Workbench

**Owner**: Epic 250 (Map Reconstruction, Composition & Editor Platform) & Epic 253 (PM4/PD4 Navmesh Research)  
**Origin**: Operator prompt (2026-10-31) regarding:
1. Live PM4 object duplication and magnetic matching in the viewer (as shown in `development.wdt` tile 35_38 where single M2 objects like `DRAGONBLIGHT_CLIFF_01.M2` have dozens of unmatched red PM4 doodad collision hulls from Blizzard level-designer copy-pasting).
2. Deriving chunk terrain holes from burgundy-red MCCV / minimap colors representing underside fog holes.
3. Exposing a Photoshop-style terrain layers toolbar with per-layer opacity, alpha mask overlay blend intensity, and decomp-backed tooltip glossaries (`H:\wowdecomps\4.3.4\wowdecomp15595-w15595`).
4. Fixing skyboxes to render across the full day cycle (removing the night-only gate in `SceneAtmosphere.cs`) and enabling zone audio/music.
5. Saving curated merged object placements to a Zarr v60 dataset archive.

**Status**: Draft / Planned (2026-10-31)

---

## 1. Executive Summary & Problem Statement

### 1.1 PM4 Object Duplication & Magnetic Matching (`development.wdt`)
In early and reconstructed maps like `development.wdt`, ADT object placements are not 100% original Blizzard placements—they were hand-placed or partially reconstructed in 2021 before modern VLMs or PM4 analyzers existed. Conversely, the **PM4 navmesh collision files contain authentic Blizzard ground truth**:
- On `development.wdt` tile `35_38`, an original cliff doodad (`DRAGONBLIGHT_CLIFF_01.M2`, Index 6, UniqueId 1118548) exists in the ADT.
- Immediately adjacent to it, spanning the ridge, are dozens of red wireframe PM4 doodad collision meshes (`Ck24 == 0`, dominant group `0x03`).
- These red meshes represent identical copies of `DRAGONBLIGHT_CLIFF_01.M2` placed, scaled, and rotated by Blizzard level designers. The 2021 ADT placed only one, leaving the rest completely empty.
- The viewer currently lacks any mechanism to select an existing model and automatically clone/duplicate it across all unmatched PM4 doodad collision hulls on the tile.

### 1.2 Burgundy-Red MCCV / Minimap Hole Detection
When Blizzard rendered minimaps or baked MCCV (vertex colors: 145 BGRA vertices per chunk) over areas where holes existed in the terrain mesh, the "underside of terrain fog" rendered through the hole as a characteristic solid burgundy-red color.
- Rather than manually guessing or hand-drawing holes, the presence of this solid burgundy color in MCCV serves as an empirical indicator of terrain holes.
- To avoid noise, detection must require a cluster of **2 or more contiguous chunks** sharing this solid color, and only when the color is unmixed with surface texture diffuse.

### 1.3 Texture Layers Toolbar & Blending Options
Currently, terrain layer toggles are buried deep inside the "Options" bottom-tab panel (`ViewerChromeService.Options.cs`).
- The terrain shader (`TerrainRenderer.cs`) only supports binary layer visibility (`uShowLayerMask`).
- When "Show Alpha Mask" is enabled, it completely replaces the terrain color with grayscale alpha; the user cannot blend the alpha mask over the painted tileset to inspect alignment.
- Texture layer flags (`MCLY.Flags`) and `EffectId` are opaque to users. Decompilation of the 4.3.4 client (`H:\wowdecomps\4.3.4\wowdecomp15595-w15595`) provides exact definitions for all bits (animation direction/speed, overbright/specular masking, alpha RLE, cubemap reflections, and `GroundEffectTexture.dbc` clutter).

### 1.4 Skybox & Audio Immersion
- In `SceneAtmosphere.cs` (line 361), active `LightSkybox.dbc` models are gated behind `if (_skyDome.NightVisibility > 0.001f)`. During daytime, `NightVisibility` is 0, so the client skybox model is never rendered.
- `_skyDome.UpdateFromLighting(...)` overwrites DBC/LIT zenith and horizon colors with hardcoded day/night constants.
- Status bar displays "AUDIO: ON", but zone music and ambient sound are not hooked to camera position.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Live PM4 Object Duplication & Magnetic Matching (Priority: P1)
As an archaeologist or world builder curating `development.wdt`, I want to select a model (such as `DRAGONBLIGHT_CLIFF_01.M2`) and automatically duplicate it to fill all nearby unmatched red PM4 doodad meshes, matching their position, scale, and rotation.

- **AC-101**: When an M2 or WMO object is selected in the viewer, the Inspector offers an action: **"Fill PM4 Doodad Meshes with Model"**.
- **AC-102**: The resolver scans all unmatched PM4 doodad collision clusters on the tile (or within user radius).
- **AC-103**: For each unmatched cluster, it computes:
  - Placement position attracted to the PM4 collision centroid.
  - Scale factor derived from the ratio of the PM4 collision extent to the template model's base extent.
  - Planar rotation aligned to the PM4 cluster's dominant axis.
- **AC-104**: Instantiated duplicates receive distinct sequential UniqueIds, are added to `_tileMdxInstances`, and appear immediately in the viewport without reloading.
- **AC-105**: The user can preview, accept, or discard the duplicated placements before saving.
- **AC-106**: Curated placements can be saved directly to a project-managed output or Zarr v60 archive.

### User Story 2: Derive Chunk Holes from Burgundy-Red MCCV / Minimap (Priority: P1)
As a level restorer, I want an option to analyze MCCV vertex colors and automatically set chunk terrain hole flags for clusters of chunks displaying the distinctive burgundy-red underside fog color.

- **AC-201**: The Editor Terrain Tools / Archaeology panel provides an option: **"Derive Holes from Burgundy MCCV"**.
- **AC-202**: The detector samples MCCV vertex colors across each chunk (145 vertices) or sampled minimap tiles.
- **AC-203**: A chunk is classified as burgundy-red candidate if its mean color matches the underside fog chroma threshold ($R > 80, G < 30, B < 30$) with low color variance ($\sigma < 15$).
- **AC-204**: An isolation filter enforces that only groups of **2 or more contiguous candidate chunks** receive `HoleMask = 0xFFFF` (or full 64-bit hole mask); isolated single chunks are discarded.
- **AC-205**: Chunks whose surface has significant texture diffuse contribution (mixed paint) are excluded.
- **AC-206**: The operation updates `TerrainChunkData.HoleMask` and rebuilds the chunk mesh indices, rendering visible holes immediately in the viewport.

### User Story 3: Photoshop-Style Texture Layers Workbench & Decomp Tooltips (Priority: P2)
As a terrain artist, I want a dedicated Layers toolbar with per-layer opacity sliders, alpha mask overlay blend intensity, and decomp-backed tooltips explaining texture effect flags.

- **AC-301**: A top/bottom toolbar toggle opens a floating/docked **Terrain Layers** panel showing resident layers (0 to 3, up to 7).
- **AC-302**: Each layer displays its texture filename, `.blp` thumbnail, resolution, and layer flags.
- **AC-303**: Each layer includes an opacity slider ($0.0 \dots 1.0$), passed to the terrain shader via `uLayerOpacity[8]` uniform to blend layers interactively.
- **AC-304**: An **"Alpha Mask Overlay"** slider ($0.0 \dots 1.0$) blends the raw MCAL alpha mask over the shaded diffuse texture in real time, allowing visual inspection of texture boundaries.
- **AC-305**: Hovering over layer flags displays accurate tooltips verified against 4.3.4 decomp:
  - `0x07`: UV Animation Direction vector.
  - `0x38`: UV Animation Speed (1.0 to 64.0 units/sec).
  - `0x40`: UV Animation Enabled.
  - `0x80`: Specular Masking / Overbright Disable (`spec[i] = 0.0f`).
  - `0x100`: MCAL Alpha Map Present.
  - `0x200`: RLE Compressed Alpha Map.
  - `0x400`: Environment / Cubemap Skybox Reflection (`hasEnvMap`).
  - `0x8000`: Dynamic Texture (`CMap::GetDynamicTexture`).
  - `EffectId`: Ground Clutter Doodad definition from `GroundEffectTexture.dbc`.

### User Story 4: Skybox Full-Day Rendering & Audio Immersion (Priority: P2)
As a user exploring the world, I want skybox models to be visible at noon as well as midnight, and zone music to play based on camera location.

- **AC-401**: In `SceneAtmosphere.cs`, the active `LightSkybox.dbc` model renders continuously throughout the full day/night cycle, scaling opacity or ambient tint naturally rather than vanishing at daybreak.
- **AC-402**: Authored DBC/LIT zenith, horizon, and fog colors are preserved on `SkyDomeRenderer` rather than overwritten by hardcoded day/night constants.
- **AC-403**: Moving the camera into a new area/zone queries `AreaTable.dbc` and `SoundEntries.dbc`, streaming zone background music and ambience.

---

## 3. Architectural Constraints & Non-Goals

- **AGENTS.md §10 (God-Class Freeze)**: No new members in `WorldScene.cs` or `ViewerApp.cs`. All duplication logic, MCCV hole detection, and layer presentation reside in owned service classes (`Pm4DoodadDuplicationService`, `MccvHoleDetectionService`, `TerrainLayersPanelService`).
- **AGENTS.md §4 (Do Not Break Format Readers)**: ADT/WMO/M2 readers remain untouched. New placement duplication operates on runtime scene structures and exports through existing writer interfaces.
- **AGENTS.md §9 (Receipts Required)**: Verification requires real command outputs, test passes, and UI reference verifier green status.
