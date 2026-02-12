# ADT v22 Format Index

## Overview
The ADT v22 format is a transitional version of the ADT (Area Data Table) format that appeared only in the Cataclysm beta. It was never used in the final release, which continued to use v18. The v22 format introduced a different naming convention and structure compared to v18, but was ultimately abandoned.

## Main Chunks

| ID | Chunk | Status | Description |
|----|-------|--------|-------------|
| C001 | [AHDR](chunks/ADT_v22/C001_AHDR.md) | ✅ Documented | Header with grid structure information |
| C002 | [AVTX](chunks/ADT_v22/C002_AVTX.md) | ✅ Documented | Vertex height data |
| C003 | [ANRM](chunks/ADT_v22/C003_ANRM.md) | ✅ Documented | Normal vectors for terrain lighting |
| C004 | [ATEX](chunks/ADT_v22/C004_ATEX.md) | ✅ Documented | Texture filenames |
| C005 | [ADOO](chunks/ADT_v22/C005_ADOO.md) | ✅ Documented | M2 and WMO model filenames |
| C006 | [ACNK](chunks/ADT_v22/C006_ACNK.md) | ✅ Documented | Map chunk data container |

## ACNK Subchunks

| ID | Subchunk | Status | Description |
|----|----------|--------|-------------|
| S001 | [ALYR](chunks/ADT_v22/S001_ALYR.md) | ✅ Documented | Texture layer information |
| S002 | [AMAP](chunks/ADT_v22/S002_AMAP.md) | ✅ Documented | Alpha map for textures |
| S003 | [ASHD](chunks/ADT_v22/S003_ASHD.md) | ✅ Documented | Shadow map |
| S004 | [ACDO](chunks/ADT_v22/S004_ACDO.md) | ✅ Documented | Object definitions |

## Format Relationships
- Related to ADT v18: Completely reorganized version of the same terrain data that was ultimately abandoned
- Related to ADT v23: Predecessor with similar structure but fewer features, also only in beta

## Key Differences from ADT v18
1. **New Naming Convention**: Uses A-prefixed chunks (AHDR, AVTX) instead of M-prefixed (MHDR, MCVT)
2. **Simplified Structure**: Fewer overall chunks with more consolidated data
3. **Vertex Organization**: Separated outer and inner vertices instead of interleaved approach
4. **Model References**: Combined M2 and WMO references into a single ADOO chunk
5. **Chunk Hierarchy**: Different organization of subchunks within parent chunks

## Implementation Notes
- This format was only used in the Cataclysm beta and was never used in any final release
- All retail ADT files, even split files from Cataclysm onwards, use v18 format
- Documentation exists primarily for historical understanding of format evolution
- There is no need to implement this format for parsing retail game files
- The recommended implementation approach is to focus on v18 format, which is used for all retail ADTs

## Documentation Status
- Main chunks: 6/6 (100%)
- Subchunks: 4/4 (100%)
- Total: 10/10 (100%)

## Historical Significance
This format represents an interesting glimpse into how Blizzard was experimenting with reorganizing terrain data during Cataclysm development. While ultimately not used, the A-prefix naming convention and some structural ideas show the evolution of their format design thinking.

## Key Design Experiments in v22

Some of the most interesting experimental design choices in the v22 format include:

1. **Unified Object Management**: Combining both M2 doodads and WMO objects into a single system (ADOO/ACDO), which simplifies object loading and management.

2. **Separated Vertex Data**: Moving vertex data from subchunks inside MCNK to standalone AVTX chunk, potentially allowing more efficient memory usage and shared vertex data.

3. **Non-Uniform Scaling**: Supporting per-axis scaling in object placement, enabling more versatile model transformations.

4. **Embedded Alpha Maps**: Including alpha maps directly within texture layer definitions rather than in separate chunks, which could improve loading performance.

5. **Reduced Redirection**: Fewer levels of indirection when referencing data, which potentially simplifies parsing and improves performance.

While v22 was not adopted for the final Cataclysm release, these design experiments may have influenced future development decisions for World of Warcraft's terrain system. 