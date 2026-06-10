# MSUR Fields Analysis

## Overview
This document provides detailed information about the MSUR chunk fields used in the PM4 grouping tool and explains why the MSUR raw fields grouping strategy produces the most coherent object assemblies.

## MSUR Chunk Structure
The MSUR (Surface) chunk contains entries with the following key fields:

| Field Name | Offset | Type | Description |
|------------|--------|------|-------------|
| FlagsOrUnknown_0x00 | 0x00 | uint | Primary grouping field, appears to define object type/category |
| Unknown_0x02 | 0x02 | ushort | Secondary grouping field, further subdivides objects |
| MsviFirstIndex | 0x04 | uint | First index in the MSVI chunk |
| IndexCount | 0x08 | uint | Number of indices for this surface |
| SurfaceGroupKey | 0x0C | byte | Legacy grouping identifier (not as effective) |

## Why MSUR Raw Fields Work Best

Through extensive testing and comparison of multiple grouping strategies, we've discovered that MSUR raw fields provide the most semantically meaningful grouping of PM4 geometry for these reasons:

1. **Horizontal Slicing**: MSUR fields appear to define horizontal slices of objects, which corresponds to how complex objects like buildings are actually constructed in the game world.

2. **Consistent Object Boundaries**: The `FlagsOrUnknown_0x00` field consistently identifies object categories across different PM4 files.

3. **Logical Subdivision**: The `Unknown_0x02` field creates logical subdivisions within each object category, likely corresponding to floors or sections of buildings.

4. **Superior to Alternatives**:
   - SurfaceGroupKey alone is too generic and doesn't capture object structure
   - ParentIndex-based methods fail due to cross-tile vertex references
   - Vertex connectivity fails due to missing vertices from adjacent tiles
   - MPRR sentinel values don't correspond directly to coherent objects

## Field Value Patterns

The typical patterns observed in MSUR fields are:

### FlagsOrUnknown_0x00
- Values in range 0-32 appear to be building exteriors
- Values in range 33-64 appear to be building interiors
- Values in range 65-128 appear to be terrain elements
- Higher values may indicate special objects or world elements

### Unknown_0x02
- Within each FlagsOrUnknown_0x00 group, this field typically increases sequentially
- Jumps in values often indicate new floors or major architectural changes
- Values of 0 often indicate base elements

## Future Research

Additional research should focus on:

1. Correlation between MSUR fields and MPRL/MSLK chain data
2. Spatial relationship of objects with similar field values
3. Understanding how these fields relate to global tile loading
4. Mapping field values to specific in-game object types

## Example Values

| Object Type | FlagsOrUnknown_0x00 | Unknown_0x02 | Notes |
|-------------|---------------------|--------------|-------|
| Building Exterior | 22 | 0-12 | Main exterior walls and details |
| Building Interior | 36 | 0-8 | Interior rooms and corridors |
| Terrain | 67 | 0-2 | Ground surfaces |
| Special Object | 144 | 0-4 | Unique world elements |

These patterns have been observed across multiple PM4 files and show consistent meaning across different world regions.
