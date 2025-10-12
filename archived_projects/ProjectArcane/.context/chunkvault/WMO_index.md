# WMO Format Documentation

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

| ID    | Name        | Description                            | Status     |
|-------|-------------|----------------------------------------|------------|
| MVER  | Version     | File version information               | Documented |
| MOHD  | Header      | WMO header with global properties      | Documented |
| MOTX  | Textures    | Texture filenames                      | Documented |
| MOMT  | Materials   | Material definitions                   | Documented |
| MOGN  | Group Names | Names of WMO groups                    | Documented |
| MOGI  | Group Info  | Information about WMO groups           | Documented |
| MOSB  | Skybox      | Skybox filename                        | Documented |
| MOPV  | Portal Vertices | Portal vertex positions            | Documented |
| MOPT  | Portal Info | Portal definitions                     | Documented |
| MOPR  | Portal References | Portal references by groups      | Documented |
| MOVV  | Visible Vertices | Visible vertex positions          | Documented |
| MOVB  | Visible Blocks | Visible block definitions           | Documented |
| MOLT  | Lights      | Light definitions                      | Documented |
| MODS  | Doodad Sets | Doodad set definitions                 | Documented |
| MODN  | Doodad Names | Doodad filenames                      | Documented |
| MODD  | Doodad Definitions | Doodad placements               | Documented |
| MFOG  | Fog         | Fog definitions                        | Documented |
| MCVP  | Convex Volume Planes | Convex volume plane definitions | Documented |

**Status: 18/18 Root Chunks Documented (100%)**

### Group File Chunks

| ID    | Name        | Description                            | Status     |
|-------|-------------|----------------------------------------|------------|
| MVER  | Version     | File version information               | Documented |
| MOGP  | Group Header| WMO group header with properties       | Documented |
| MOPY  | Materials   | Material IDs for triangles             | Documented |
| MOVI  | Indices     | Vertex indices for triangles           | Documented |
| MOVT  | Vertices    | Vertex positions                       | Documented |
| MONR  | Normals     | Vertex normals                         | Documented |
| MOTV  | TexCoords   | Texture coordinates                    | Documented |
| MOBA  | Batches     | Rendering batch definitions            | Documented |
| MOLR  | Light Refs  | Light references                       | Documented |
| MODR  | Doodad Refs | Doodad references                      | Documented |
| MOBN  | BSP Tree    | BSP tree nodes for collision           | Documented |
| MOBR  | BSP Faces   | BSP face indices                       | Documented |
| MOCV  | Vertex Colors | Vertex colors for lighting           | Documented |
| MLIQ  | Liquid      | Liquid surface definitions             | Documented |

**Status: 14/14 Group Chunks Documented (100%)**

## Implementation Status
- **Root File Parser**: Not implemented
- **Group File Parser**: Not implemented
- **Rendering System**: Not implemented
- **Collision System**: Not implemented

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

## Next Steps
1. Complete documentation of all remaining chunks
2. Implement parsers for documented chunks
3. Create a rendering system for WMO visualization
4. Develop collision detection based on BSP tree structure
5. Create tools for WMO editing and creation 