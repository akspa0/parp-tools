# WMO (World Model Object) Format Documentation

## Overview
The WMO (World Model Object) format is used to store complex 3D models in World of Warcraft, typically representing buildings, dungeons, and other large structures that cannot be efficiently represented as terrain or individual models. WMO files consist of a root file and multiple group files, enabling efficient rendering of complex structures with proper occlusion culling and level of detail management.

## Historical Significance
WMO files have been a part of World of Warcraft since its original release, providing a way to create detailed and complex structures beyond what was possible with terrain or standalone models. The format has evolved over time but maintains backward compatibility with older game versions.

## File Structure
WMO files are split into two types of files:
1. **Root File** (.wmo): Contains global information about the model, including materials, textures, doodad definitions, and group information.
2. **Group Files** (_###.wmo): Each group file represents a discrete portion of the model, allowing for efficient occlusion culling and selective rendering.

### Main Chunks (Root File)

| Chunk ID | Name | Status | Description |
|----------|------|--------|-------------|
| MVER | Version | Documented | Specifies the version of the WMO format |
| MOHD | Header | Documented | Contains global information about the WMO model |
| MOTX | Texture Names | Documented | Lists filenames of textures used in the model |
| MOMT | Materials | Documented | Defines materials used in the model |
| MOGN | Group Names | Documented | Names for the groups that make up the model |
| MOGI | Group Info | Documented | Information about each group in the model |
| MOSB | Skybox | Documented | Defines the skybox model for this WMO |
| MOPV | Portal Vertices | Documented | Vertices that define portals between groups |
| MOPT | Portal Definitions | Documented | Defines portals that connect different groups |
| MOPR | Portal References | Documented | Maps portals to the groups they connect |
| MOVV | Visible Vertices | Documented | Vertices used for visibility testing |
| MOVB | Visible Blocks | Documented | Defines blocks used for visibility determination |
| MOLT | Lights | Documented | Defines light sources within the model |
| MODS | Doodad Sets | Documented | Sets of doodads that can be shown/hidden together |
| MODN | Doodad Names | Documented | Filenames of doodad models used |
| MODD | Doodad Definitions | Documented | Placement information for doodads |
| MFOG | Fog | Documented | Defines fog volumes within the model |
| MCVP | Convex Volume Planes | Documented | Planes that make up convex volumes for visibility testing |

### Group File Chunks

| Chunk ID | Name | Status | Description |
|----------|------|--------|-------------|
| MVER | Version | Documented | Specifies the version of the group file |
| MOGP | Group Header | Documented | Contains information specific to this group and wraps other chunks |
| MOPY | Material Info | Documented | Material information for each triangle |
| MOVI | Indices | Documented | Vertex indices defining triangles |
| MOVT | Vertices | Documented | Vertex positions |
| MONR | Normals | Documented | Vertex normals |
| MOTV | Texture Coordinates | Documented | Texture mapping coordinates |
| MOBA | Render Batches | Documented | Defines batches for efficient rendering |
| MOLR | Light References | Documented | References to lights affecting this group |
| MODR | Doodad References | Documented | References to doodads in this group |
| MOBN | BSP Tree | Documented | Binary space partitioning tree for the group |
| MOBR | BSP Face References | Documented | Face references for the BSP tree |
| MOCV | Vertex Colors | Documented | Vertex color information |
| MLIQ | Liquid | Documented | Defines liquid surfaces in the group |

## Format Relationships
WMO files interact with several other formats in the World of Warcraft client:
- **M2 Files**: Doodads (decorative objects) in WMOs are M2 models
- **BLP Files**: Textures referenced by the WMO are stored in BLP format
- **ADT Files**: World terrain files can reference WMOs for placement in the world
- **WDT Files**: World definition files list all WMOs used in a map

## Implementation Status
- **Documentation Progress**: 17/17 root chunks documented (100%), 14/14 group chunks documented (100%)
- **Implementation Progress**: 0/31 chunks implemented (0%)

### Implementation Plan
1. ~~Complete documentation of all group file chunks~~ (Completed March 2025)
2. Implement parser for root file chunks
3. Implement parser for group file chunks
4. Create utility functions for WMO manipulation and rendering

## Key Concepts

### Portal System
WMO files use a portal system to define connectivity between different groups, enabling efficient visibility determination and occlusion culling. Portals are defined by:
- Portal vertices (MOPV)
- Portal definitions (MOPT)
- Portal references (MOPR)

This system allows the engine to determine which groups are potentially visible from any given viewpoint, significantly improving rendering performance.

### Doodad System
WMOs can include numerous decorative objects (doodads) that enhance visual detail. The doodad system consists of:
- Doodad names (MODN)
- Doodad definitions (MODD)
- Doodad sets (MODS)

This system allows for efficient management of decorative objects, including the ability to selectively show or hide sets of doodads based on game conditions.

### Group Structure
WMOs are divided into groups, each stored in a separate file. This structure enables:
- Efficient occlusion culling
- Level of detail management
- Selective rendering

Groups typically represent logical divisions of the model, such as rooms or sections of a building.

### Material System
The material system defines the visual appearance of the WMO surfaces:
- Texture references (MOTX)
- Material definitions (MOMT)

Materials can include multiple texture layers, various blend modes, and special effects such as environment mapping.

## Example Applications
- Buildings and structures in cities and towns
- Dungeons and raids
- Ships, bridges, and other large constructions
- Unique terrain features that cannot be represented through the terrain system

## Next Steps
1. Implement parsers for the documented chunks
   - Begin with root file parser
   - Implement group file parser with BSP tree support
2. Develop collision detection system using the BSP tree structure
3. Create a rendering system for WMO visualization
   - Support indoor/outdoor lighting transitions
   - Implement portal-based visibility determination
   - Support doodad and liquid rendering
4. Create tools for WMO editing and conversion
   - Develop export capabilities for modern modeling applications
   - Create validation tools to ensure format compliance 