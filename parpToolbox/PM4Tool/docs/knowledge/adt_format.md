# ADT (Azeroth Doodad Table) File Format

## Overview
ADT files contain terrain and object information for map tiles in World of Warcraft. They follow a chunked structure similar to WDT files.

## Map Organization
- A map tile is split into 16x16 = 256 map chunks
- Each map chunk is represented by an MCNK chunk
- Chunks contain terrain, textures, models, and other object data

## Coordinate System
- WoW uses a right-handed coordinate system
- Positive X-axis points north, positive Y-axis points west
- Z-axis is vertical height, with 0 being sea level
- The origin is at the center of the map

## Main Chunks

### MVER Chunk
- Contains version information
- Usually contains a single uint32 value

### MHDR Chunk
- Map header with offsets to other chunks
- Contains flags for map features
- Key flags:
  - `mhdr_MFBO` (0x1) - Contains an MFBO chunk
  - `mhdr_northrend` (0x2) - Used for northrend maps

### MCIN Chunk
- Pointers to MCNK (map chunk) chunks and their sizes
- Pre-Cataclysm only, as Cataclysm introduced split files

### MTEX Chunk
- List of textures used for texturing terrain
- Contains zero-terminated strings with paths to textures

### MMDX and MMID Chunks
- MMDX: List of filenames for M2 models (doodads)
- MMID: List of offsets into the MMDX chunk

### MWMO and MWID Chunks
- MWMO: List of filenames for WMOs (world map objects)
- MWID: List of offsets into the MWMO chunk

### MDDF Chunk
- Placement information for doodads (M2 models)
- Contains position, rotation, scale, and flags

### MODF Chunk
- Placement information for WMOs
- Contains position, rotation, extents, and flags

### MH2O Chunk (WotLK+)
- Liquid data (water, lava, etc.)
- Replaced the older MCLQ sub-chunk

### MCNK Chunk
- Map chunk data (contains terrain, textures, etc.)
- Each map tile has 256 MCNK chunks (16x16 grid)
- Contains various sub-chunks for different data types

## Split Files (Cataclysm+)
Starting with Cataclysm, ADTs are split into multiple files:
- `.adt` (root)
- `_tex%d.adt` (textures, where %d is the level of detail)
- `_obj%d.adt` (objects, where %d is the level of detail)
- `_lod.adt` (LOD data, added in Legion)

## Resources
- [wowdev wiki ADT_v18](https://wowdev.wiki/ADT)
- Warcraft.NET implementation in `/docs/libs/Warcraft.NET/Warcraft.NET/Files/ADT/` 