# WDT (World Doodad Table) File Format

## Overview
WDT files specify which map tiles are present in a World of Warcraft world and can also reference a "global" World Map Object (WMO). They follow a chunked file structure similar to ADT files.

## Purpose
- Defines which map tiles (ADTs) exist in a world map
- Can reference a global WMO for the entire map
- Contains flags that affect map rendering

## Main Chunks

### MVER Chunk
- Contains version information
- Usually contains a single uint32 value

### MPHD Chunk
- Contains map header information and flags
- Modern versions (8.1.0+) include file data IDs for related files
- Key flags:
  - `wdt_uses_global_map_obj` (0x0001) - Use global map object definition
  - `adt_has_mccv` (0x0002) - ADTs have vertex colors
  - `adt_has_big_alpha` (0x0004) - Uses specific terrain shaders
  - `adt_has_height_texturing` (0x0080) - Uses height-based texturing

### MAIN Chunk
- Map tile table containing 64x64 = 4096 entries
- Each entry has flags indicating if a tile exists and other properties
- Key flags:
  - `Flag_HasADT` - Indicates an ADT exists for this tile
  - `Flag_AllWater` - Indicates tile is all water (no land)

### MAID Chunk (Modern, 8.1.0+)
- Map file data ID table for 64x64 tiles
- Contains FileDataIDs for:
  - Root ADT
  - Object ADTs
  - Texture ADTs
  - LOD ADT
  - Map textures
  - Minimap textures

### MWMO Chunk (Optional)
- Contains a filename for a global WMO
- Only present if the map uses a global WMO

### MODF Chunk (Optional)
- Contains placement information for the global WMO
- Only present if the map uses a global WMO
- Includes position, orientation, and extent information

## Additional Files (Modern Versions)

### _occ.wdt (WoD+)
- Contains occlusion information for the map
- Key chunks:
  - MVER - Version
  - MAOI - Tile information
  - MAOH - Occlusion heightmap

### _lgt.wdt (WoD+)
- Contains lighting information for the map
- Key chunks:
  - MVER - Version
  - MPLT - Point lights (pre-Legion)
  - MPL2 - Point lights (Legion+)
  - MSLT - Spot lights (Legion+)

### _fogs.wdt (Legion+)
- Contains fog information for the map
- Key chunks:
  - MVER - Version
  - VFOG - Volumetric fog data

### _mpv.wdt (BfA+)
- Contains particulate volume information
- Key chunks:
  - MVER - Version
  - PVPD, PVMI, PVBD - Particle volume data

## Alpha WDT Format Considerations
- The Alpha WDT format likely lacks many modern features
- Probably uses a simpler structure with fewer chunks
- Focus should be on MVER, MPHD, and MAIN chunks initially
- May have different flag values or interpretation

## Resources
- [wowdev wiki WDT](https://wowdev.wiki/WDT)
- Warcraft.NET implementation in `/docs/libs/Warcraft.NET/Warcraft.NET/Files/WDT/` 