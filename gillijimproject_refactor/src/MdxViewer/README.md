# AlphaWoW Viewer

A high-performance .NET 9 / OpenGL 3.3 world viewer for **World of Warcraft Alpha 0.5.3** and **Lich King 3.3.5** game data. Renders monolithic Alpha WDTs, standard split ADTs, WMO world objects, MDX/M2 models, and DBC-driven overlays (Area POIs, Taxi Paths) with high fidelity.

## Key Features

### 🌍 Terrain & World
- **Alpha WDT Support** — Full support for Alpha 0.5.3 monolithic WDT files with 256 MCNK chunks per tile.
- **Standard WDT+ADT (3.3.5)** — Renders standard split ADTs (root, obj, tex) from MPQ or loose files.
- **AOI Streaming** — Area-of-Interest based async tile streaming for seamless world traversal.
- **MCSH Shadows** — 64×64 shadow bitmasks applied across all terrain layers.
- **Liquids (MCLQ/MLIQ)** — Ghidra-verified liquid rendering for both terrain (MCLQ) and WMO (MLIQ) with proper type detection (Water, Ocean, Magma, Slime).

### 🏛️ WMO Rendering (v14, v16 & v17)
- **4-Pass Transparency** — Correct sorting for Opaque → Doodads → Liquids → Transparent layers.
- **100% Doodad Load Rate** — Robust asset resolution using case-insensitive MPQ searching and `.mdx`/`.mdl` extension swapping.
- **Doodad Sets** — Full support for switching between internal WMO doodad configurations.
- **Orientation Fix** — Fixed 180° rotation and coordinate mapping for Alpha WMOs.

### 📦 MDX/M2 Models
- **Two-Pass Rendering** — Opaque pass followed by depth-sorted transparent pass.
- **Blend Modes 0-6** — Implementation of all 7 standard WoW blend modes (Opaque, AlphaKey, Alpha, Additive, etc.).
- **DBCD Texture Resolution** — Resolves character, creature, and item textures using 4 DBC tables:
  - `CreatureModelData.dbc`
  - `CreatureDisplayInfo.dbc`
  - `CreatureDisplayInfoExtra.dbc`
  - `ItemDisplayInfo.dbc`

### 🗺️ Minimap & Overlays
- **Minimap** — Camera-centered minimap with BLP tile textures, scroll-wheel zoom, and double-click teleport.
- **Area POIs** — DBC-driven point-of-interest markers rendered as 3D pins and minimap dots.
- **Taxi Paths** — Flight path visualization from TaxiPath/TaxiPathNode DBC data, rendered as 3D lines and minimap overlays. Click a node or route in the sidebar to isolate it.
- **Area Names** — Live area name display from AreaTable.dbc with MapID validation.
- **Batched Rendering** — All overlay geometry (pins, lines) drawn in a single GPU draw call for performance.

## Coordinate System (Ghidra Verified)

The viewer uses a unified coordinate system derived from WoW's raw file data:
- **WoW Coords**: X=North, Y=West, Z=Up (Right-handed).
- **Renderer Coords**: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`, `rendererZ = wowZ`.
- **Winding**: Direct3D CW winding is reversed to OpenGL CCW winding during GPU upload.
- **Placement**: Models receive a 180° Z-rotation to align with renderer basis.

## Controls

| Input | Action |
|-------|--------|
| **WASD** | Move camera (North/South/East/West) |
| **Q / E** | Move camera Vertical (Down / Up) |
| **Right Drag** | Look around (Yaw / Pitch) |
| **Scroll Wheel** | Adjust camera speed (viewport) / Minimap zoom (minimap) |
| **Double-Click Minimap** | Teleport camera to clicked location |
| **Day/Night Slider** | Adjust world lighting (Terrain/WMO) |

## Requirements

- **Runtime**: .NET 9.0 SDK
- **OS**: Windows (Native `StormLib.dll` for MPQ access)
- **GPU**: OpenGL 3.3+ capable hardware

## Build & Run

```bash
cd src/MdxViewer
dotnet build
dotnet run -- path/to/world.wdt
```

## Architecture

The viewer is built on a modular "Adapter" pattern:
- **IDataSource** — Abstraction for MPQ archives (`MpqDataSource`) or loose files.
- **ITerrainAdapter** — Unified interface for Alpha (`AlphaTerrainAdapter`) and LK (`StandardTerrainAdapter`) terrain.
- **ReplaceableTextureResolver** — DBCD-backed service for mapping dynamic texture IDs to BLP paths.
- **WorldAssetManager** — Centralized caching for WMO and MDX/M2 geometry.
- **BoundingBoxRenderer** — Batched line/pin renderer for overlays (1 draw call for all markers).
- **TaxiPathLoader** — Loads TaxiNodes, TaxiPath, and TaxiPathNode DBC data for flight path visualization.
- **AreaTableService** — MapID-aware area name lookups from AreaTable.dbc.

## Supported Formats

| Format | Version | Status |
|--------|---------|--------|
| **WDT** | Alpha / LK | ✅ Fully supported |
| **ADT** | Alpha / LK | ✅ Fully supported |
| **WMO** | v14, v16, v17 | ✅ Fully supported |
| **MDX** | v1300+ | ✅ Supported (Rendering Quality WIP) |
| **M2** | v264+ | 🔧 Partial Support |
| **DBC** | Alpha 0.5.3 | ✅ AreaTable, AreaPOI, TaxiPath, TaxiPathNode, Map |
| **GLB** | Export | ✅ MDX/WMO Export |
