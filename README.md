# parp-tools
Tools for analyzing World of Warcraft files

## MdxViewer - WoW World Viewer

A high-performance 3D world viewer supporting WoW Alpha 0.5.3, 0.6.0, and LK 3.3.5 game data with ADT/WDT/WMO/MDX/M2 rendering, format conversion, and asset export.

### Quick Start

1. **Open Game Folder**: `File > Open Game Folder` - Select your WoW client directory
2. **Load a Map**: Double-click a map from the "World Maps" list in the left sidebar
3. **Navigate**: Use WASD to fly, mouse to look around, scroll wheel to zoom minimap

### Controls

#### Camera Navigation
- **W/A/S/D** - Move forward/left/back/right
- **Q/E** - Move up/down
- **Shift** - 5x speed boost
- **Mouse Drag** - Look around (right-click and drag)

#### Minimap
- **Location**: Top of right sidebar
- **M Key** - Toggle fullscreen minimap overlay
- **Scroll Wheel** - Zoom in/out (1x to 32x)
- **Click & Drag** - Pan the minimap view
- **Double-Click** - Teleport to clicked location
- **Reset Pan Button** - Return to camera-centered view

#### UI Panels
- **Left Sidebar**: File browser, World Maps list
- **Right Sidebar**: Minimap, Model Info, Terrain Controls, World Objects
- **Toolbar**: Visibility toggles (Terrain, Doodads, WMOs, Liquids, etc.)

### Features

- **Multi-Version Support**: Alpha 0.5.3, 0.6.0, LK 3.3.5
- **Terrain Rendering**: Full ADT/WDT support with liquids (MCLQ/MLIQ)
- **WMO Rendering**: v14/v16/v17 with doodads, liquids, transparency
- **MDX/M2 Rendering**: Animated models with textures, specular, environment maps
- **Format Conversion**: WMO v14↔v17, MDX↔M2 bidirectional conversion
- **GLB Export**: Visual and collision mesh export with materials
- **Asset Catalog**: Browse/export creatures and gameobjects with screenshots
- **Alpha-Core Integration**: NPC/GameObject spawn visualization from SQL dumps

### Export Options

- **File > Export GLB (Visual)** - Export WMO with textures (excludes collision)
- **File > Export GLB (Collision Only)** - Export collision/portal meshes only
- **Tools > WMO Converter** - Convert between v14 and v17 formats
- **View > Asset Catalog** - Batch export creatures/gameobjects with screenshots

### Supported File Formats

- **Terrain**: WDT, ADT (Alpha 0.5.3 monolithic, 0.6.0+ split)
- **Models**: WMO (v14/v16/v17), MDX (v1300), M2 (v264+)
- **Textures**: BLP (v0/v1), with DBC replaceable texture resolution
- **Data**: DBC files, Alpha-Core SQL dumps, minimap tiles

---

## WoWRollback

Sink objects below the ground that match specific ranges of layers of uniqueID values.

## Libraries
- gillijimproject_refactor/src/gillijimproject-csharp - A C# port of mjollna/gillijimproject with many enhancements (AreaTable remapping for 3.3.5 without modding DBCs)

----

# Archived projects

AlphaWDTAnalysisTool - The proof of concept patch tool for AreaID patching. Rolled in to WoWRollback as a library.

PM4Tool - PM4 parsing tools

parpToolbox - Additional PM4 parsing tools

PM4FacesTool - current experimental toolkit for parsing PM4 files. Will be rolled into WoWRollback.

parpToolbox/src/PM4Rebuilder - A complete tool for parsing PM4 files into exportable objects. Includes lots of analysis tooling too.

parpToolbox - Base library for PM4/PD4/WMO reading, with tons of tests and analysis tooling for understanding the PM4 file format.

PM4Tool/WoWToolbox - a PM4/PD4 file decoder with support for reading from ADT and WMO assets for identifying data in the PM4 or PD4 file formats. Currently a working proof of concept. Supports leaked development pm4 files and the cataclysm-era split ADT files for analysis. We use WotLK 3.3.5 assets (wmo/m2) for mesh comparison.

ProjectArcane - Extemely work-in-progress re-write, based on the wowdev wiki documentation

WCA\WCAnalyzer - A multi-tool built off ModernWoWTools/Warcraft.NET for parsing ADT files from version 0.6 through 11.x and dumping useful data from them, as well as parsing PM4 and PD4 files for useful data.


----
This project is not an official Blizzard Entertainment product and is not affiliated with or endorsed by World of Warcraft or Blizzard Entertainment.


