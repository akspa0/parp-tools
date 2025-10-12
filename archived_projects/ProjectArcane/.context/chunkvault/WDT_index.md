# WDT (World Data Table) Format

## Overview
The World Data Table (WDT) format is a file structure used to define the overall map layout and global objects in the World of Warcraft client. WDT files serve as an index to ADT terrain files and contain global object references. 

There are two main versions of the WDT format:
1. **Modern WDT**: A reference/index file that points to separate ADT terrain files (one ADT per 64x64 grid cell).
2. **Alpha WDT**: A self-contained format that embedded all map data directly, including terrain data that would later be split into separate ADT files.

## File Versions

| Version | Name | Description |
|---------|------|-------------|
| Alpha (~1999) | Pre-release | Self-contained format with embedded terrain data. Contains chunks with different naming conventions (MAOT, MAOI, MAOH, etc.) |
| 18 (Vanilla) | Original | Initial retail release format, featuring a grid-based structure with references to separate ADT files |
| 22 (Cataclysm) | Updated | Added MD21 chunk for enhanced M2 doodad handling |
| 23+ (Later) | Modern | Minor additions to support newer game features |

## Modern WDT Chunks

| Chunk ID | Name | Description | Status |
|----------|------|-------------|--------|
| W001 | [MVER](chunks/WDT/W001_MVER.md) | Map Version | ✅ Documented |
| W002 | [MPHD](chunks/WDT/W002_MPHD.md) | Map Header | ✅ Documented |
| W003 | [MAIN](chunks/WDT/W003_MAIN.md) | Main Map Index | ✅ Documented |
| W004 | [MWMO](chunks/WDT/W004_MWMO.md) | Map WMO Names | ✅ Documented |
| W005 | [MODF](chunks/WDT/W005_MODF.md) | Map WMO Placement | ✅ Documented |
| W006 | [MWID](chunks/WDT/W006_MWID.md) | Map WMO Index | ✅ Documented |
| W007 | [MDDF](chunks/WDT/W007_MDDF.md) | Map Doodad Placement | ✅ Documented |
| W008 | [MDNM](chunks/WDT/W008_MDNM.md) | Map Doodad Names | ✅ Documented |
| W009 | [MDID](chunks/WDT/W009_MDID.md) | Map Doodad Index | ✅ Documented |

## Alpha WDT Chunks

| Chunk ID | Name | Description | Status |
|----------|------|-------------|--------|
| WA01 | [MAOT](chunks/WDT/WA01_MAOT.md) | Map Object Table | ✅ Documented |
| WA02 | [MAOI](chunks/WDT/WA02_MAOI.md) | Map Object Information | ✅ Documented |
| WA03 | [MAOH](chunks/WDT/WA03_MAOH.md) | Map Object Header | ✅ Documented |
| WA04 | [MOTX](chunks/WDT/WA04_MOTX.md) | Map Object Texture | ✅ Documented |
| WA05 | [MOBS](chunks/WDT/WA05_MOBS.md) | Map Object BSP | ✅ Documented |
| WA06 | [MODR](chunks/WDT/WA06_MODR.md) | Map Object Directory | ✅ Documented |

## Common Patterns

Several patterns are repeated across both Modern and Alpha WDT formats:

1. **Version Identification**:
   - Both formats begin with a version chunk (MVER in modern, embedded in MAOH for Alpha)

2. **String Storage**:
   - Strings are stored as null-terminated arrays in dedicated chunks
   - Referenced via offsets from other chunks
   - Modern: MWMO/MWID for WMO names, MDNM/MDID for doodad names
   - Alpha: MOTX for all texture and object names

3. **Object Placement**:
   - Coordinates are defined in world space
   - Modern: MODF for WMO placement, MDDF for doodad placement
   - Alpha: MAOI contains embedded object data

4. **Spatial Organization**:
   - Modern: Grid-based map cells (16×16 grid)
   - Alpha: BSP tree-based spatial organization (MOBS)

## Key Architectural Differences

The fundamental architectural difference between Alpha and Modern WDT formats:

1. **Alpha WDT Format (Self-Contained)**:
   - All map data is embedded in a single file
   - Includes terrain data that later became separate ADT files
   - Uses BSP trees (MOBS) for spatial organization
   - Hierarchical directory structure (MODR) for object organization
   - Direct indexing of embedded objects (MAOT/MAOI)

2. **Modern WDT Format (Reference-Based)**:
   - Acts primarily as an index/reference file
   - References external ADT files for terrain data (one per grid cell)
   - Only contains global object placements that span multiple ADT cells
   - Grid-based organization (MAIN chunk) for terrain tiles
   - Separate handling for WMO objects (MWMO/MODF) and doodads (MDNM/MDDF)

This shift from a self-contained to a modular approach improved memory management and allowed for more efficient loading of large worlds.

## Version Differences

The evolution from Alpha to Modern formats represents a shift from:
- A monolithic, fully embedded approach
- To a modular, reference-based approach

This change enabled:
1. More efficient memory usage (only loading needed map regions)
2. Simplified updates (changing single ADT files instead of entire maps)
3. More specialized handling of different object types
4. Better scalability for larger world sizes

## Implementation Status

- Modern WDT Chunks: 9/9 documented (100%)
- Alpha WDT Chunks: 6/6 documented (100%)
- **Total**: 15/15 documented (100%)

## Next Steps

1. **Parser Implementation**:
   - Implement parsers for Modern WDT format
   - Implement parsers for Alpha WDT format (for research/historical purposes)
   - Create hybrid parser that can detect and handle both formats

2. **Architectural Comparison**:
   - Create detailed architectural comparison between formats
   - Document the evolution from Alpha to Modern for historical context

3. **Documentation Expansion**:
   - Add visual diagrams showing the structure of both formats
   - Include sample data for educational purposes 