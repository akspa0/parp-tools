# World Format Documentation Index

## Format Overview

| Format | Status | Main Chunks | Subchunks | Notes |
|--------|--------|-------------|-----------|-------|
| ADT v18 | ✅ Complete | 30/30 (100%) | 45/45 (100%) | Complete documentation |
| ADT v22 | ✅ Complete | 16/16 (100%) | 0/0 (N/A) | Complete documentation |
| ADT v23 | ✅ Complete | 8/8 (100%) | 0/0 (N/A) | Complete documentation |
| WDT | 📝 Planned | 0/6 (0%) | 0/0 (N/A) | Planned for future |
| WDL | 📝 Planned | 0/4 (0%) | 0/0 (N/A) | Planned for future |
| WMO | ✅ Complete | 18/18 (100%) | 14/14 (100%) | Complete documentation |
| M2 | 📝 Planned | 0/20 (0%) | 0/0 (N/A) | Planned for future |
| DBC/DB2/ADB | 📝 Planned | 0/3 (0%) | 0/0 (N/A) | Database formats planned |

## Recent Updates

- **2025-03-24**: Completed documentation of all WMO format chunks (32/32)
- **2023-06-25**: Began documentation of WMO format
- **2023-06-20**: Completed documentation of all ADT v23 format chunks (8/8)
- **2023-06-15**: Completed documentation of ADT v22 format (16/16)
- **2023-06-10**: Completed documentation of ADT v18 format (75/75)
- **2023-06-01**: Created ChunkVault structure and initial format organization

## Implementation Progress

| Format | Documented | Implemented | Notes |
|--------|------------|-------------|-------|
| ADT v18 | 75/75 (100%) | 75/75 (100%) | Complete implementation |
| ADT v22 | 16/16 (100%) | 16/16 (100%) | Complete implementation |
| ADT v23 | 8/8 (100%) | 8/8 (100%) | Complete implementation |
| WDT | 0/6 (0%) | 0/6 (0%) | Planned for Q4 2023 |
| WDL | 0/4 (0%) | 0/4 (0%) | Planned for Q4 2023 |
| WMO | 32/32 (100%) | 32/32 (100%) | Complete implementation |
| M2 | 0/20 (0%) | 0/20 (0%) | Planned for Q1 2024 |
| DBC/DB2/ADB | 0/3 (0%) | 0/3 (0%) | Planned for Q1 2024 |

## Format Status

| Format | Documented Chunks | Implemented | Status | Documentation |
|--------|------------------|-------------|---------|---------------|
| ADT v18 | 45/45 (30 main, 15 sub) | ✅ 45/45 | Complete | [ADT_index.md](ADT_index.md), [mcnk_structure.md](mcnk_structure.md) |
| ADT v22 | 16/16 | 📝 0/16 | Documented | [ADT_v22_index.md](ADT_v22_index.md) |
| ADT v23 | 12/12 | 📝 0/12 | Documented | [ADT_v23_index.md](ADT_v23_index.md) |
| WMO | 33/33 | ✅ 33/33 | Complete | [WMO_index.md](WMO_index.md) |
| M2 | 29/29 (25 core, 4 supp) | ✅ 29/29 | Complete | [M2_index.md](M2_index.md), [chunks/M2Supplementary/](chunks/M2Supplementary/) |
| MDX | 24/24 | ✅ 24/24 | Complete | [MDX_index.md](MDX_index.md) |
| WDT | 15/15 | ✅ 15/15 | Complete | [WDT_index.md](WDT_index.md) |
| WDL | 8/8 | ✅ 8/8 | Complete | [WDL_index.md](WDL_index.md) |
| PM4 | 14/14 | ✅ 14/14 | Complete | [PM4_PD4_index.md](PM4_PD4_index.md), [PM4_PD4_relationships.md](PM4_PD4_relationships.md) |
| PD4 | 10/10 | ✅ 10/10 | Complete | [PM4_PD4_index.md](PM4_PD4_index.md), [PM4_PD4_relationships.md](PM4_PD4_relationships.md) |
| WDB | 5/5 | ✅ Using DBCD | Complete | [WDB_index.md](WDB_index.md) |
| DBC/DB2 | 6/6 | ✅ Using DBCD | Complete | [DBC_index.md](DBC_index.md) |

## Common Systems

### Parsing Infrastructure
- [Common Parsing System](common/parsing.md) - Core chunk parsing implementation
- [Common Types](common/types.md) - Shared data structures and types
- [Relationships](relationships.md) - Format relationships and dependencies
- [Validation](validation.md) - Validation rules and requirements

## Format Documentation

### Core Formats
- [ADT Format](ADT_index.md) - Terrain file format
  - [MCNK Structure](mcnk_structure.md) - Detailed terrain chunk documentation
  - [ADT v22](ADT_v22_index.md) - Version 22 format changes
  - [ADT v23](ADT_v23_index.md) - Version 23 format changes
- [WMO Format](WMO_index.md) - World map object format
- [M2 Format](M2_index.md) - Model format
  - [Supplementary Formats](chunks/M2Supplementary/) - BONE, PHYS, SKEL, SKIN documentation
- [MDX Format](MDX_index.md) - Model exchange format

### World Data
- [WDT Format](WDT_index.md) - World template format
- [WDL Format](WDL_index.md) - Low detail terrain format
- [PM4/PD4 Format](PM4_PD4_index.md) - Server-side supplementary formats
  - [Format Relationships](PM4_PD4_relationships.md) - PM4/PD4 dependencies
- [WDB Format](WDB_index.md) - World database cache format
- [DBC Format](DBC_index.md) - Client database format

### Binary Formats
- [Binary Formats](binary_formats/index.md) - Non-chunked file formats (BLP, TEX, LIT, WFX)

## Status Legend
✅ Complete (fully implemented)
🔄 In Progress
📝 Documented Only
⏳ Planned
❌ Not Started 