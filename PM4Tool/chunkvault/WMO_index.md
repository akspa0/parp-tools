# WMO Format Documentation

## Related Documentation
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections
- [M2 Format](M2_index.md) - Referenced model format
- [ADT Format](ADT_index.md) - Parent terrain format

## Overview
The WMO (World Map Object) format is used for complex, static objects in the World of Warcraft game environment. These objects include buildings, caves, bridges, and other large structures that make up the game world. WMO files were introduced in the original World of Warcraft release (2004) and continue to be used in modern versions of the game, albeit with additional features and extensions added over time.

A WMO consists of a root file and one or more group files. The root file contains global information about the object, while each group file contains the actual geometry and rendering information for a section of the object.

## File Structure
WMO files follow a chunk-based format, where each chunk begins with a 4-character ID followed by a 4-byte size value. The content of each chunk depends on its type, as defined by the ID.

### Root File (.wmo)
The root file contains global information about the WMO, such as materials, textures, doodad sets, and group definitions.

### Group Files (_xxx.wmo)
Group files contain the actual geometry and rendering data for a section of the WMO, including vertices, normals, texture coordinates, and rendering batches.

## Chunk Documentation Status

### Root File Chunks

| ID    | Name        | Description                            | Status     | Documentation |
|-------|-------------|----------------------------------------|------------|---------------|
| MVER  | Version     | File version information               | Documented | [chunks/WMO/Root/MVER.md](chunks/WMO/Root/MVER.md) |
| MOHD  | Header      | WMO header with global properties      | Documented | [chunks/WMO/Root/MOHD.md](chunks/WMO/Root/MOHD.md) |
| MOTX  | Textures    | Texture filenames                      | Documented | [chunks/WMO/Root/MOTX.md](chunks/WMO/Root/MOTX.md) |
| MOMT  | Materials   | Material definitions                   | Documented | [chunks/WMO/Root/MOMT.md](chunks/WMO/Root/MOMT.md) |
| MOGN  | Group Names | Names of WMO groups                    | Documented | [chunks/WMO/Root/MOGN.md](chunks/WMO/Root/MOGN.md) |
| MOGI  | Group Info  | Information about WMO groups           | Documented | [chunks/WMO/Root/MOGI.md](chunks/WMO/Root/MOGI.md) |
| MOSB  | Skybox      | Skybox filename                        | Documented | [chunks/WMO/Root/MOSB.md](chunks/WMO/Root/MOSB.md) |
| MOPV  | Portal Vertices | Portal vertex positions            | Documented | [chunks/WMO/Root/MOPV.md](chunks/WMO/Root/MOPV.md) |
| MOPT  | Portal Info | Portal definitions                     | Documented | [chunks/WMO/Root/MOPT.md](chunks/WMO/Root/MOPT.md) |
| MOPR  | Portal References | Portal references by groups      | Documented | [chunks/WMO/Root/MOPR.md](chunks/WMO/Root/MOPR.md) |
| MOVV  | Visible Vertices | Visible vertex positions          | Documented | [chunks/WMO/Root/MOVV.md](chunks/WMO/Root/MOVV.md) |
| MOVB  | Visible Blocks | Visible block definitions           | Documented | [chunks/WMO/Root/MOVB.md](chunks/WMO/Root/MOVB.md) |
| MOLT  | Lights      | Light definitions                      | Documented | [chunks/WMO/Root/MOLT.md](chunks/WMO/Root/MOLT.md) |
| MOLP  | Light Parameters | Point light parameters            | Documented | [chunks/WMO/O012_MOLP.md](chunks/WMO/O012_MOLP.md) |
| MOLV  | Light Volumes | Light volume definitions             | Documented | [chunks/WMO/O013_MOLV.md](chunks/WMO/O013_MOLV.md) |
| MOLS  | Light Names | Light name definitions                 | Documented | [chunks/WMO/Root/MOLS.md](chunks/WMO/Root/MOLS.md) |
| MODS  | Doodad Sets | Doodad set definitions                 | Documented | [chunks/WMO/Root/MODS.md](chunks/WMO/Root/MODS.md) |
| MODN  | Doodad Names | Doodad filenames                      | Documented | [chunks/WMO/Root/MODN.md](chunks/WMO/Root/MODN.md) |
| MODD  | Doodad Definitions | Doodad placements               | Documented | [chunks/WMO/Root/MODD.md](chunks/WMO/Root/MODD.md) |
| MFOG  | Fog         | Fog definitions                        | Documented | [chunks/WMO/Root/MFOG.md](chunks/WMO/Root/MFOG.md) |
| MCVP  | Convex Volume Planes | Convex volume plane definitions | Documented | [chunks/WMO/Root/MCVP.md](chunks/WMO/Root/MCVP.md) |

**Status: 18/18 Root Chunks Documented (100%)**

### Group File Chunks

| ID    | Name        | Description                            | Status     | Documentation |
|-------|-------------|----------------------------------------|------------|---------------|
| MVER  | Version     | File version information               | Documented | [chunks/WMO/Group/MVER.md](chunks/WMO/Group/MVER.md) |
| MOGP  | Group Header| WMO group header with properties       | Documented | [chunks/WMO/Group/MOGP.md](chunks/WMO/Group/MOGP.md) |
| MOPY  | Materials   | Material IDs for triangles             | Documented | [chunks/WMO/Group/MOPY.md](chunks/WMO/Group/MOPY.md) |
| MOVI  | Indices     | Vertex indices for triangles           | Documented | [chunks/WMO/Group/MOVI.md](chunks/WMO/Group/MOVI.md) |
| MOVT  | Vertices    | Vertex positions                       | Documented | [chunks/WMO/Group/MOVT.md](chunks/WMO/Group/MOVT.md) |
| MONR  | Normals     | Vertex normals                         | Documented | [chunks/WMO/Group/MONR.md](chunks/WMO/Group/MONR.md) |
| MOTV  | TexCoords   | Texture coordinates                    | Documented | [chunks/WMO/Group/MOTV.md](chunks/WMO/Group/MOTV.md) |
| MOBA  | Batches     | Rendering batch definitions            | Documented | [chunks/WMO/Group/MOBA.md](chunks/WMO/Group/MOBA.md) |
| MOLR  | Light Refs  | Light references                       | Documented | [chunks/WMO/Group/MOLR.md](chunks/WMO/Group/MOLR.md) |
| MODR  | Doodad Refs | Doodad references                      | Documented | [chunks/WMO/Group/MODR.md](chunks/WMO/Group/MODR.md) |
| MOBN  | BSP Tree    | BSP tree nodes for collision           | Documented | [chunks/WMO/Group/MOBN.md](chunks/WMO/Group/MOBN.md) |
| MOBR  | BSP Faces   | BSP face indices                       | Documented | [chunks/WMO/Group/MOBR.md](chunks/WMO/Group/MOBR.md) |
| MOCV  | Vertex Colors | Vertex colors for lighting           | Documented | [chunks/WMO/Group/MOCV.md](chunks/WMO/Group/MOCV.md) |
| MLIQ  | Liquid      | Liquid surface definitions             | Documented | [chunks/WMO/Group/MLIQ.md](chunks/WMO/Group/MLIQ.md) |
| MLVR  | Liquid Render | Liquid render vertex data            | Documented | [chunks/WMO/G015_MLVR.md](chunks/WMO/G015_MLVR.md) |
| MORI  | Render indices | Render indices for rendering          | Documented | [chunks/WMO/Group/MORI.md](chunks/WMO/Group/MORI.md) |

**Status: 15/15 Group Chunks Documented (100%)**

## Implementation Status
✅ **Fully Implemented** - Core parsing system and all chunks are implemented

### Core Components
- ✅ `WmoFile` - Main WMO file parser
- ✅ `WmoGroupFile` - Group file parser
- ✅ `WmoRootFile` - Root file parser
- ✅ `WmoLightFile` - Light data parser
- ✅ `WmoFileSet` - Manager for related WMO files

### Features
- ✅ Group file support
- ✅ FileDataID support (8.1+)
- ✅ Asset reference tracking
- ✅ Validation reporting
- ⏳ Visualization (Planned)
- ⏳ Collision detection (Planned)
- ⏳ Editing tools (Planned)

## Implemented Chunks

### Root File Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/WMO/Root/MVER.md](chunks/WMO/Root/MVER.md) |
| MOHD | ✅ | Header | [chunks/WMO/Root/MOHD.md](chunks/WMO/Root/MOHD.md) |
| MOTX | ✅ | Texture names | [chunks/WMO/Root/MOTX.md](chunks/WMO/Root/MOTX.md) |
| MOMT | ✅ | Materials | [chunks/WMO/Root/MOMT.md](chunks/WMO/Root/MOMT.md) |
| MOGN | ✅ | Group names | [chunks/WMO/Root/MOGN.md](chunks/WMO/Root/MOGN.md) |
| MOGI | ✅ | Group info | [chunks/WMO/Root/MOGI.md](chunks/WMO/Root/MOGI.md) |
| MOSB | ✅ | Skybox | [chunks/WMO/Root/MOSB.md](chunks/WMO/Root/MOSB.md) |
| MOPV | ✅ | Portal vertices | [chunks/WMO/Root/MOPV.md](chunks/WMO/Root/MOPV.md) |
| MOPT | ✅ | Portal info | [chunks/WMO/Root/MOPT.md](chunks/WMO/Root/MOPT.md) |
| MOPR | ✅ | Portal references | [chunks/WMO/Root/MOPR.md](chunks/WMO/Root/MOPR.md) |
| MOVV | ✅ | Visible vertices | [chunks/WMO/Root/MOVV.md](chunks/WMO/Root/MOVV.md) |
| MOVB | ✅ | Visible blocks | [chunks/WMO/Root/MOVB.md](chunks/WMO/Root/MOVB.md) |
| MOLT | ✅ | Lights | [chunks/WMO/Root/MOLT.md](chunks/WMO/Root/MOLT.md) |
| MOLP | ✅ | Light parameters | [chunks/WMO/O012_MOLP.md](chunks/WMO/O012_MOLP.md) |
| MOLV | ✅ | Light volumes | [chunks/WMO/O013_MOLV.md](chunks/WMO/O013_MOLV.md) |
| MOLS | ✅ | Light names | [chunks/WMO/Root/MOLS.md](chunks/WMO/Root/MOLS.md) |
| MODS | ✅ | Doodad sets | [chunks/WMO/Root/MODS.md](chunks/WMO/Root/MODS.md) |
| MODN | ✅ | Doodad names | [chunks/WMO/Root/MODN.md](chunks/WMO/Root/MODN.md) |
| MODD | ✅ | Doodad definitions | [chunks/WMO/Root/MODD.md](chunks/WMO/Root/MODD.md) |
| MFOG | ✅ | Fog | [chunks/WMO/Root/MFOG.md](chunks/WMO/Root/MFOG.md) |
| MCVP | ✅ | Convex volume planes | [chunks/WMO/Root/MCVP.md](chunks/WMO/Root/MCVP.md) |

### Group File Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/WMO/Group/MVER.md](chunks/WMO/Group/MVER.md) |
| MOGP | ✅ | Group header | [chunks/WMO/Group/MOGP.md](chunks/WMO/Group/MOGP.md) |
| MOPY | ✅ | Material info | [chunks/WMO/Group/MOPY.md](chunks/WMO/Group/MOPY.md) |
| MOVI | ✅ | Indices | [chunks/WMO/Group/MOVI.md](chunks/WMO/Group/MOVI.md) |
| MOVT | ✅ | Vertices | [chunks/WMO/Group/MOVT.md](chunks/WMO/Group/MOVT.md) |
| MONR | ✅ | Normals | [chunks/WMO/Group/MONR.md](chunks/WMO/Group/MONR.md) |
| MOTV | ✅ | Texture coordinates | [chunks/WMO/Group/MOTV.md](chunks/WMO/Group/MOTV.md) |
| MOBA | ✅ | Render batches | [chunks/WMO/Group/MOBA.md](chunks/WMO/Group/MOBA.md) |
| MOLR | ✅ | Light references | [chunks/WMO/Group/MOLR.md](chunks/WMO/Group/MOLR.md) |
| MODR | ✅ | Doodad references | [chunks/WMO/Group/MODR.md](chunks/WMO/Group/MODR.md) |
| MOBN | ✅ | BSP nodes | [chunks/WMO/Group/MOBN.md](chunks/WMO/Group/MOBN.md) |
| MOBR | ✅ | BSP faces | [chunks/WMO/Group/MOBR.md](chunks/WMO/Group/MOBR.md) |
| MOCV | ✅ | Vertex colors | [chunks/WMO/Group/MOCV.md](chunks/WMO/Group/MOCV.md) |
| MLIQ | ✅ | Liquid | [chunks/WMO/Group/MLIQ.md](chunks/WMO/Group/MLIQ.md) |
| MLVR | ✅ | Liquid render vertices | [chunks/WMO/G015_MLVR.md](chunks/WMO/G015_MLVR.md) |
| MORI | ✅ | Render indices | [chunks/WMO/Group/MORI.md](chunks/WMO/Group/MORI.md) |

Total Progress: 34/34 chunks implemented (100%)

## Implementation Notes
- Root file must be loaded before group files
- Group files can be loaded on demand
- BSP tree used for collision and culling
- Portal system for visibility optimization
- Doodad references link to M2 models

## File Structure
```
example.wmo           - Root file
example_000.wmo      - First group
example_001.wmo      - Second group
...
example_xxx.wmo      - Additional groups
```

## Next Steps
1. Implement visualization system
2. Add collision detection
3. Create editing tools
4. Add format conversion utilities

## References
- [WMO Documentation](../docs/WMO.md)
- [WMO Group Documentation](../docs/WMO_Group.md)

## Key Concepts

### Portal System
WMO models use portals to define connections between groups and optimize rendering and culling. Portals are planar polygons that act as windows between different sections of the model. When a portal is crossed, the connected group becomes visible.

### Doodad System
Doodads are smaller objects (like furniture, decorations, or environmental elements) that can be placed within a WMO. The root file contains references to the M2 model files that define these doodads, along with their placement information.

### Group Structure
WMO models are divided into groups, each with its own geometry and rendering properties. Groups can represent different rooms, floors, or sections of a building or structure. This division allows for more efficient rendering and culling.

### Material System
WMO materials define how surfaces look and react to lighting. They include texture references, blend modes, shader flags, and other properties that affect the visual appearance of the model.

## Example Applications
- **WoW Model Viewer**: View and explore WMO models outside the game
- **Map Editors**: Create and modify WMO models for custom maps or modifications
- **Game Clients**: Render WMO models in custom game engines or server emulators
- **Asset Extraction**: Extract textures, models, and other assets from WMO files 