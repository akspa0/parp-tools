# ChunkVault Expansion Plan

This document outlines the plan for expanding the ChunkVault with documentation for additional file formats beyond ADT.

## File Formats to Document

1. **ADT Versions**
   - ADT_v22.md (Version differences from v18)
   - ADT_v23.md (Version differences from v22)

2. **WDT (World Data Table)**
   - Primary world map file for World of Warcraft maps
   - Multiple versions with fundamentally different architectures:
     - Alpha.md (Alpha version WDT format): Contains ADT terrain data directly embedded within the WDT file
     - WDT.md (Modern WDT format): References separate ADT files instead of containing terrain data directly

3. **WDL (World Data Low-resolution)**
   - Contains low-resolution height map for distant terrain

4. **WMO (World Map Object)**
   - Complex structures like buildings, caves, etc.
   - Multiple chunk types and groups

5. **M2 (Model Format)**
   - Character and object models
   - Animation data and rendering information

6. **PM4 (Player Model)**
   - Player-specific model format
   - Character customization data

7. **PD4 (Player Data)**
   - Player-specific data format
   - Character information storage

## Documentation Approach

For each file format, we will:

1. **Create Index File**: Create a format-specific index.md in the chunkvault directory
2. **Extract Chunks**: Extract all chunk definitions from the documentation
3. **Follow Template**: Use the same documentation template as ADT chunks
4. **Reference Dependencies**: Document relationships with chunks from other formats
5. **Implementation Examples**: Provide C# implementation examples
6. **Version Differences**: Clearly document architectural differences between versions

## File Structure

```
chunkvault/
├── index.md                   (Main index)
├── ADT_index.md               (ADT chunks index)
├── WDT_index.md               (WDT chunks index - includes both modern and Alpha formats)
├── WDL_index.md               (WDL chunks index)
├── WMO_index.md               (WMO chunks index)
├── M2_index.md                (M2 model chunks index)
├── PM4_index.md               (PM4 player model chunks index)
├── PD4_index.md               (PD4 player data chunks index)
├── chunks/
│   ├── ADT/                   (existing ADT chunks)
│   ├── WDT/                   (WDT chunks, including Alpha version)
│   ├── WDL/                   (WDL chunks)
│   ├── WMO/                   (WMO chunks)
│   ├── M2/                    (M2 model chunks)
│   ├── PM4/                   (PM4 player model chunks)
│   ├── PD4/                   (PD4 player data chunks)
│   └── common/                (common types)
└── relationships/
    ├── ADT_relationships.md   (existing ADT relationships)
    ├── WDT_relationships.md   (WDT relationships)
    ├── WMO_relationships.md   (WMO relationships)
    ├── M2_relationships.md    (M2 relationships)
    └── cross_format.md        (Cross-format relationships)
```

## Prioritization

We should document these formats in the following order:

1. **WDT**: Central to world structure
   - Modern WDT format: References separate ADT files
   - Alpha WDT format: Contains ADT terrain data directly embedded within it
2. **WDL**: Complements ADT for distant terrain rendering
3. **WMO**: Complex structures placed in the world
4. **M2**: Model format for creatures and objects
5. **PM4/PD4**: Player-specific formats

## Estimated Chunks per Format

- WDT (Modern): ~5-8 chunks
- WDT (Alpha): ~6-8 chunks (includes chunks that contain terrain data)
- WDL: ~3-5 chunks
- WMO: ~15-20 chunks
- M2: ~20-25 chunks
- PM4: ~10-15 chunks
- PD4: ~10-15 chunks

Total: ~69-96 additional chunks to document

## Common Types

We should continue expanding our common types as needed when documenting new formats:

- Matrix types
- Quaternion types
- Color formats
- Specialized vectors
- Animation structures

## Timeline Estimate

Based on our previous pace of documenting ADT chunks:

- WDT (including Alpha): 2-3 days (more complex due to architectural differences)
- WDL: 1 day
- WMO: 2-3 days
- M2: 2-3 days
- PM4: 1-2 days
- PD4: 1-2 days

Total: ~9-14 days to complete full documentation

## Special Considerations for Alpha WDT

The Alpha WDT format requires special attention because:

1. It combines what would later be split into separate WDT and ADT files
2. It uses different chunk names (MAOT, MAOI, MOTX) than modern formats
3. It represents a fundamentally different architecture for storing map data
4. It provides historical context for how the format evolved
5. Special parsing logic will be needed to extract the terrain data

## Next Steps

1. Update main chunkvault/index.md to reflect expansion plan
2. Create directory structure for new formats
3. Finish modern WDT documentation
4. Document Alpha WDT format with special attention to how it contains ADT data
5. Create architectural comparison between Alpha and modern formats
6. Proceed in priority order through other formats 