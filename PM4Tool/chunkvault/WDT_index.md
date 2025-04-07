# WDT Format Documentation

## Related Documentation
- [ADT Format](ADT_index.md) - Child terrain format
- [WDL Format](WDL_index.md) - Low detail terrain format
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections

## Implementation Status
✅ **Fully Implemented** - Core parsing system and all chunks are implemented

### Core Components
- ✅ `WdtFile` - Main WDT file parser
- ✅ `WdtMapTile` - Map tile management
- ✅ `WdtFlags` - Tile flag handling
- ✅ `WdtAssetLoader` - Asset reference system

### Features
- ✅ Map tile tracking
- ✅ FileDataID support (8.1+)
- ✅ Asset reference tracking
- ✅ Validation reporting
- ⏳ Map visualization (Planned)
- ⏳ Editing tools (Planned)

## Implemented Chunks

### Main Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/WDT/MVER.md](chunks/WDT/MVER.md) |
| MAIN | ✅ | Main tile table | [chunks/WDT/MAIN.md](chunks/WDT/MAIN.md) |
| MAID | ✅ | Map file IDs | [chunks/WDT/MAID.md](chunks/WDT/MAID.md) |
| MPHD | ✅ | Map header | [chunks/WDT/MPHD.md](chunks/WDT/MPHD.md) |
| MODF | ✅ | Map object definitions | [chunks/WDT/MODF.md](chunks/WDT/MODF.md) |
| MAOF | ✅ | Map area offsets | [chunks/WDT/MAOF.md](chunks/WDT/MAOF.md) |
| MAOC | ✅ | Map area compression | [chunks/WDT/MAOC.md](chunks/WDT/MAOC.md) |
| MARE | ✅ | Map area records | [chunks/WDT/MARE.md](chunks/WDT/MARE.md) |
| MAHO | ✅ | Map hole information | [chunks/WDT/MAHO.md](chunks/WDT/MAHO.md) |
| MAIE | ✅ | Map instance effects | [chunks/WDT/MAIE.md](chunks/WDT/MAIE.md) |
| MNAM | ✅ | Map names | [chunks/WDT/MNAM.md](chunks/WDT/MNAM.md) |
| MPLT | ✅ | Map palette | [chunks/WDT/MPLT.md](chunks/WDT/MPLT.md) |
| MAEX | ✅ | Map extras | [chunks/WDT/MAEX.md](chunks/WDT/MAEX.md) |
| MWMO | ✅ | Map WMO names | [chunks/WDT/MWMO.md](chunks/WDT/MWMO.md) |
| MWID | ✅ | Map WMO indices | [chunks/WDT/MWID.md](chunks/WDT/MWID.md) |

Total Progress: 15/15 chunks implemented (100%)

## Implementation Notes
- WDT files define the map structure
- Each tile can reference an ADT file
- WDL files provide low detail terrain
- FileDataIDs used in modern versions

## File Structure
```
<MapName>.wdt        - World definition table
<MapName>_xx_yy.adt  - Referenced terrain tiles
<MapName>.wdl        - Low detail terrain
```

## Next Steps
1. Implement map visualization
2. Add editing tools
3. Create format conversion utilities
4. Add validation tools

## References
- [WDT Format Specification](../docs/WDT.md)
- [Map Format Overview](../docs/Map_Formats.md) 