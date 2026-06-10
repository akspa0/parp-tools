# World of Warcraft File Format Documentation

This directory contains documentation for various World of Warcraft file formats, sourced from [wowdev.wiki](https://wowdev.wiki/).

## Available Documentation

- [ADT](ADT_v18.md) - Terrain and object information for map tiles
- [ADT/v22](ADT_v22.md) - Temporary version 22 of the ADT format
- [ADT/v23](ADT_v23.md) - Temporary version 23 of the ADT format
- [WDT](WDT.md) - World map tile specifications
- [M2](M2.md) - Model objects (doodads, players, monsters, etc.)
- [Common Types](Common_Types.md) - Common data types used across WoW file formats
- [PD4](PD4.md) - Server-side supplementary files for WMOs
- [PM4](PM4.md) - Server-side supplementary files for ADTs
- [WLQ](WLQ.md) - Water level data (liquid quantities)
- [WLM](WLM.md) - Water level data (magma)
- [WLW](WLW.md) - Water level data (water)
- [WDL/v18](WDL_v18.md) - Low-resolution world heightmap
- [WMO](WMO.md) - World Model Objects


## File Format Relationships

- **WDT files** specify which map tiles are present in a world and can reference global WMO objects
- **ADT files** contain terrain and object information for individual map tiles
- **M2 files** describe model objects like doodads, players, and monsters

## Coordinate Systems

World of Warcraft uses a right-handed coordinate system where:
- The positive X-axis points north
- The positive Y-axis points west
- The Z-axis is vertical height, with 0 being sea level
- The origin of the coordinate system is in the center of the map

See the individual documentation files for more detailed information about each file format.