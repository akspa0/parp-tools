# MdxViewer User Guide

A high-performance 3D world viewer for **World of Warcraft Alpha 0.5.3**, **0.6.0**, and **Lich King 3.3.5** game data. Renders terrain, buildings, models, liquids, and DBC-driven overlays.

---

## Getting Started

### Running MdxViewer

**From a release ZIP:**
```
ParpToolsWoWViewer
```

**From source:**
```
cd gillijimproject_refactor/src/MdxViewer
dotnet run --project .\MdxViewer.csproj
```

The normal startup flow is in-app and explicit:

1. Use `File > Open Game Folder (MPQ)...`
2. Pick the client build in the build-selection dialog
3. Load worlds from the left sidebar
4. Use the right sidebar sections for `Inspect`, `Terrain`, `PM4`, `World`, and `Diagnostics`

### v0.4.7.1 patch highlights

- Taxi controls are back in the active right-sidebar object workflow instead of being stranded behind older navigation-only paths.
- Taxi ride capture now supports smoother actor heading, freelook while attached to the actor, and direct ffmpeg-backed route recording from the same sidebar surface.
- World-object selection is now sticky across camera movement and scene rebuilds, so inspection does not drop the current object just because the scene refreshed.
- Standalone WMO group inspection keeps groups loaded while moving the camera and uses explicit highlighted labels instead of labeling everything at once.

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--full-load` | Load all map tiles at startup instead of streaming (high memory usage) |
| `--verbose` | Enable detailed logging output |
| `--game-path <clientRoot>` | Open a base client directly on startup |
| `--build <buildVersion>` | Pin the client build instead of using the build picker |
| `--world <path-or-virtual-path>` | Open a world or loose asset after startup configuration |

### System Requirements

- **GPU**: OpenGL 3.3+ capable hardware
- **OS**: Windows x64 is the primary release target; cross-platform publish builds also exist, but should still be treated as compatibility output rather than broad runtime signoff
- **Data**: WoW game directory with MPQ archives (Alpha 0.5.3, 0.6.0, or 3.3.5)

---

## Camera Controls

| Input | Action |
|-------|--------|
| **W / A / S / D** | Move camera forward / left / backward / right |
| **Q / E** | Move camera down / up |
| **Right-click + Drag** | Look around (yaw / pitch) |
| **Scroll Wheel** (viewport) | Adjust camera movement speed |
| **Scroll Wheel** (minimap) | Zoom minimap in / out |
| **Triple-click same tile** (minimap) | Teleport camera to the clicked tile |

---

## Features

### 1. World Viewer (Main Viewport)

The main viewport renders the 3D world with terrain, buildings (WMOs), doodad models (MDX/M2), and liquid surfaces.

- **AOI Streaming** — Tiles load and unload dynamically as the camera moves. The active viewer now keeps a ranked near-field of detailed ADTs around the camera and uses WDL fallback for distance terrain.
- **Frustum Culling** — Only geometry visible to the camera is rendered.
- **Fog** — Distance-based fog blends objects into the horizon for a natural look.
- **Day/Night Slider** — Adjusts the scene lighting from dawn to dusk.

### 2. Minimap

A camera-centered minimap panel shows BLP minimap tiles with:
- Current camera position indicator
- Scroll-wheel zoom
- Triple-click the same visible tile to teleport the camera there
- Area POI markers (if DBC data is available)

### 3. Terrain Rendering

Supports three terrain formats:
- **Alpha 0.5.3** — Monolithic WDT files containing all map chunks
- **Alpha 0.6.0** — Split ADT files with reversed FourCC encoding
- **LK 3.3.5** — Standard split ADT files (root + _obj0 + _tex0)

Terrain features:
- Multi-layer texture splatting with alpha maps
- MCSH shadow map overlays (baked shadows from the original data)
- Chunk and tile grid overlays (toggle in UI)
- Topographical contour lines (toggle in UI, adjustable interval)

### 4. WMO Rendering (Buildings & Structures)

World Map Objects are rendered with full fidelity:
- **4-pass transparency**: Opaque geometry → Doodads → Liquids → Transparent geometry
- **Doodad sets**: Switch between interior decoration configurations
- **WMO liquids**: Water, ocean, magma, and slime with correct type detection
- **Doodad culling**: Distance-based (500 units) with nearest-first priority, capped at 64 visible doodads per WMO

Supports WMO format versions v14 (Alpha 0.5.3), v16 (Alpha 0.6.0), and v17 (3.3.5).

### 5. MDX/M2 Model Rendering

Models (trees, creatures, props, etc.) are rendered with:
- **Blend modes 0–7**: Opaque, AlphaKey, Alpha, additive, modulate, modulate 2x, and BlendAdd are mapped into the local renderer's supported material families
- **Alpha cutout**: Hard discard for tree canopies and foliage
- **Sphere environment mapping**: Reflective surfaces
- **Unshaded flag**: Fullbright materials that ignore lighting
- **DBC texture resolution**: Automatically resolves creature/item textures via CreatureModelData, CreatureDisplayInfo, and ItemDisplayInfo DBC tables

### 6. Liquid Rendering

Three liquid systems are supported:
- **MCLQ** (Alpha 0.5.3 / 0.6.0) — Per-vertex sloped heights for waterfalls, absolute world Z coordinates
- **MH2O** (3.3.5) — Per-tile liquid with sub-rect heightmaps and visibility masks
- **MLIQ** (WMO interior) — WMO group liquids with automatic type detection (water, ocean, magma, slime)

Liquid types are color-coded:
- **Water**: Blue
- **Ocean**: Deep blue
- **Magma/Lava**: Orange-red
- **Slime**: Green

### 7. DBC Overlays

When DBC data is available in the game directory, the viewer provides:
- **Area Names** — Live display of the current area name from AreaTable.dbc
- **Area POIs** — Point-of-interest markers rendered as 3D pins and minimap dots
- **Taxi Paths** — Flight path visualization from TaxiPath/TaxiPathNode DBC data
- **Zone Lighting** — Ambient, fog, and sky colors driven by Light.dbc + LightData.dbc

### 7.1 Taxi ride workflow

- Select a taxi route from the active taxi controls in the right sidebar.
- Enable the route actor and attach the ride camera in either `Cockpit` or `Chase` mode.
- Use right mouse drag while attached to freelook around the moving actor instead of forcing a rigid camera heading.
- `Record Selected Route Video` streams frames directly to ffmpeg for `.mp4` or `.mov` output.
- Practical requirement: the ffmpeg executable path must be valid in the capture controls before route video recording can succeed.

### 8. Debug & Visualization Tools

Toggle these from the UI panel:
- **Chunk Grid** — Cyan grid showing MCNK chunk boundaries (33.3 unit spacing)
- **Tile Grid** — Orange grid showing ADT tile boundaries (533.3 unit spacing)
- **Alpha Mask View** — Grayscale visualization of terrain texture alpha maps
- **Shadow Map View** — Shows baked MCSH shadow data
- **Contour Lines** — Topographical contour lines with adjustable interval (major lines every 5th interval)
- **Wireframe Mode** — Toggle wireframe rendering for terrain

### 9. Capture Automation

Use the live viewer capture tools when debugging renderer regressions, especially M2 material, texture, and animation issues.

- **Quick Capture** — `Tools -> Capture Current View (No UI)` or `Tools -> Capture Current View (With UI)` saves the current framebuffer immediately.
- **Shot Points** — `Tools -> Capture Automation...` lets you save named camera positions with map name, build version, yaw, pitch, and FOV.
- **Batch Evidence Loop** — The capture window can filter saved shots to the current map/build and queue a whole set of before/after screenshots for the same viewpoints.
- **Output Path** — Captures are written under `output/captures/<map>/<build>/...` by default.
- **Taxi Route Video** — The taxi sidebar can record the selected route directly through ffmpeg, using the same capture settings and output root.
- **Recommended Regression Workflow** — Save stable shot points for problematic assets or zones, run `No UI` captures before a change, repeat after the change, and compare the resulting PNGs against probe output.

---

## Asset Catalog

The Asset Catalog is an integrated browser for exploring NPC and GameObject assets from an alpha-core database.

### Setup

1. Open the Asset Catalog panel from the menu
2. Point it to your alpha-core installation root (the directory containing `etc/databases/`)
3. Click **Connect** to load the database

### Features

- **Search & Filter** — Search by name, filter by type (Creature / GameObject), filter to only entries with models or spawns
- **Preview** — Select an entry to see its model path and metadata
- **Load in Viewer** — Double-click any entry to load its 3D model in the main viewport
- **GLB Export** — Export individual models or batch-export entire categories to GLB format for use in Blender, Unity, Unreal, or other 3D tools

### Batch Export

Select multiple entries and click **Export Selected** to batch-export models as GLB files. The exporter:
- Resolves textures from MPQ archives
- Handles replaceable texture IDs via DBC lookup
- Outputs standard glTF Binary (.glb) files with embedded textures

---

## Map Converter (WoWMapConverter)

The Map Converter is a companion CLI tool for converting map data between WoW versions.

### Conversion Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert Alpha 0.5.3 WDT/ADT → LK 3.3.5 format |
| `convert-lk-to-alpha` | Convert LK 3.3.5 ADT → Alpha 0.5.3 format |
| `convert-wmo` | Convert WMO v14 (Alpha) → v17 (3.3.5) |
| `convert-wmo-to-alpha` | Convert WMO v17 (3.3.5) → v14 (Alpha) |
| `convert-mdx` | Convert Alpha MDX → LK M2 |
| `convert-m2-to-mdx` | Convert LK M2 → Alpha MDX |
| `wmo-info` | Dump WMO structure information |
| `analyze` | Analyze map data and generate reports |

### Usage

```bash
# Convert an Alpha map to 3.3.5 format
WoWMapConverter convert --in path/to/alpha/map --out path/to/output

# Convert a WMO from Alpha to 3.3.5
WoWMapConverter convert-wmo --in path/to/alpha.wmo --out path/to/output.wmo

# Convert an M2 model back to Alpha MDX format
WoWMapConverter convert-m2-to-mdx --in path/to/model.m2 --out path/to/output.mdx
```

These converters handle the significant format differences between WoW versions, including reversed FourCC encoding, different chunk layouts, material format changes, and coordinate system variations.

---

## ML Dataset

ML Dataset is the user-facing name for the open interchange format used for WoW map data, designed primarily for:

1. **Machine Learning** — Structured, normalized terrain data suitable for training ML models (heightmaps, alpha maps, shadow maps, depth maps, liquid masks)
2. **Preservation** — Archiving hobbyist map work in a non-proprietary format that doesn't depend on WoW client tools
3. **Cross-version portability** — A common representation that can be imported/exported to any WoW version

### What's in an ML Dataset?

An ML dataset is a directory of standardized image and metadata files per map tile:

| Layer | Format | Description |
|-------|--------|-------------|
| **Heightmap** | 16-bit PNG | Per-vertex terrain elevation (normalized) |
| **Alpha Maps** | 8-bit PNG | Per-layer texture blend weights (up to 4 layers) |
| **Shadow Map** | 8-bit PNG | Baked shadow data from MCSH chunks |
| **Depth Map** | 16-bit PNG | Camera-relative depth for ML tasks |
| **Liquid Mask** | 8-bit PNG | Water/ocean/magma/slime presence and type |
| **Minimap** | RGB PNG | Baked minimap tile images |
| **Metadata** | JSON | Tile coordinates, texture references, area IDs, layer info |

### MK Commands

```bash
# Export a WoW map to ML dataset
WoWMapConverter ml-export --in path/to/game --map Azeroth --out ml_output/

# Decode an ML dataset back to viewable images
WoWMapConverter ml-decode --in ml_output/ --out decoded/

# Bake MK layers into composite images
WoWMapConverter ml-bake --in ml_output/ --out baked/

# Bake heightmaps only (for ML training data)
WoWMapConverter ml-bake-heightmap --in ml_output/ --out heightmaps/

# Synthesize new terrain from MK data
WoWMapConverter ml-synth --in ml_output/ --out synthesized/

# Batch export multiple maps using a config file
WoWMapConverter ml-batch --config batch_config.json
```

### Batch Export Config

For exporting multiple maps at once, create a JSON config file:

```json
{
  "GamePath": "path/to/game",
  "OutputRoot": "mk_datasets/",
  "Maps": ["Azeroth", "Kalimdor", "development"],
  "Layers": ["heightmap", "alpha", "shadow", "liquid", "minimap"],
  "TileSize": 256
}
```

### Loading ML Dataset in MdxViewer

MdxViewer can load ML datasets directly for visualization:
```
MdxViewer.exe path/to/mk_dataset/
```

The viewer detects the ML dataset structure and renders the terrain from the dataset's heightmaps and textures, allowing you to visually inspect exported data.

---

## GLB Export

MdxViewer can export loaded models to **glTF Binary (.glb)** format for use in external 3D tools:

- **MDX models** — Export with geometry, UVs, normals, and embedded textures
- **WMO buildings** — Export with all groups, materials, and textures
- Works with Blender, Unity, Unreal Engine, and any glTF-compatible tool

---

## Troubleshooting

### Black screen / no terrain
- Verify the game path points to a valid WoW installation with MPQ archives
- Check the console output for version detection messages
- Try `--verbose` for detailed loading logs

### Models appear untextured (magenta)
- The model's textures may not be found in the MPQ archives
- For creature models, DBC data is needed for texture resolution
- Check console for "texture not found" messages

### Low FPS / stuttering
- AOI streaming is enabled by default — tiles load in the background as you move
- Reduce camera speed to allow tiles to load before you reach them
- Close other GPU-intensive applications
- The viewer targets OpenGL 3.3 for broad compatibility

### Out of memory
- AOI streaming (default) keeps only nearby tiles loaded
- Avoid `--full-load` on large maps (it loads everything at once)
- The viewer unloads tiles as you move away from them

---

## Supported Formats

| Format | Versions | Status |
|--------|----------|--------|
| **WDT** | 0.5.3, 0.6.0, 3.3.5 | Fully supported |
| **ADT** | 0.5.3, 0.6.0, 3.3.5 | Fully supported |
| **WMO** | v14, v16, v17 | Fully supported |
| **MDX** | v1300+ (Alpha) | Fully supported |
| **M2** | v264+ (3.3.5) | Partial support |
| **DBC** | 0.5.3, 0.6.0, 3.3.5 | AreaTable, AreaPOI, TaxiPath, Light, Map |
| **BLP** | v1, v2 | Fully supported |
| **VLM** | Current | Fully supported |
| **GLB** | Export only | MDX and WMO export |

---

## License

Pending review of third-party library licenses. This tool is intended for preservation, research, and educational purposes.
