# PM4/PD4 Format Documentation

## Related Documentation
- [ADT Format](ADT_index.md) - Related terrain format
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections
- [PM4/PD4 Relationships](PM4_PD4_relationships.md) - Format-specific relationships

## Implemented Chunks

### PM4 Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MVER | ✅ | Version information | [chunks/PM4/M001_MVER.md](chunks/PM4/M001_MVER.md) |
| MSPV | ✅ | MSP vertices | [chunks/PM4/M002_MSPV.md](chunks/PM4/M002_MSPV.md) |
| MSPI | ✅ | MSP indices | [chunks/PM4/M003_MSPI.md](chunks/PM4/M003_MSPI.md) |
| MSLK | ✅ | Links | [chunks/PM4/M004_MSLK.md](chunks/PM4/M004_MSLK.md) |
| MSVT | ✅ | Vertices | [chunks/PM4/M005_MSVT.md](chunks/PM4/M005_MSVT.md) |
| MSVI | ✅ | Vertex indices | [chunks/PM4/M006_MSVI.md](chunks/PM4/M006_MSVI.md) |
| MSUR | ✅ | Surface definitions | [chunks/PM4/M007_MSUR.md](chunks/PM4/M007_MSUR.md) |
| MSCN | ✅ | Normal vectors | [chunks/PM4/M008_MSCN.md](chunks/PM4/M008_MSCN.md) |
| MSHD | ✅ | Header | [chunks/PM4/M009_MSHD.md](chunks/PM4/M009_MSHD.md) |
| MPRL | ✅ | Position data | [chunks/PM4/M010_MPRL.md](chunks/PM4/M010_MPRL.md) |
| MPRR | ✅ | Reference data | [chunks/PM4/M011_MPRR.md](chunks/PM4/M011_MPRR.md) |
| MDBH | ✅ | Destructible building header | [chunks/PM4/M012_MDBH.md](chunks/PM4/M012_MDBH.md) |
| MDOS | ✅ | Object data | [chunks/PM4/M013_MDOS.md](chunks/PM4/M013_MDOS.md) |
| MDSF | ✅ | Structure data | [chunks/PM4/M014_MDSF.md](chunks/PM4/M014_MDSF.md) |

Total PM4 Progress: 14/14 chunks implemented (100%)

### PD4 Chunks
| Chunk | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| MCRC | ✅ | CRC data | [chunks/PD4/P001_MCRC.md](chunks/PD4/P001_MCRC.md) |
| MVER | ✅ | Version information | Same as PM4 |
| MSPV | ✅ | MSP vertices | Same as PM4 |
| MSPI | ✅ | MSP indices | Same as PM4 |
| MSLK | ✅ | Links | Same as PM4 |
| MSVT | ✅ | Vertices | Same as PM4 |
| MSVI | ✅ | Vertex indices | Same as PM4 |
| MSUR | ✅ | Surface definitions | Same as PM4 |
| MSCN | ✅ | Normal vectors | Same as PM4 |
| MSHD | ✅ | Header | Same as PM4 |

Total PD4 Progress: 10/10 chunks implemented (100%)

## Implementation Notes
- Server-side only formats
- Potentially used for NPC pathfinding
- PM4 contains mesh shape point data
- PD4 shares many chunks with PM4

## File Structure
```
<MapName>_xx_yy.pm4        - PM4 data
<MapName>_xx_yy.pd4        - PD4 data
<MapName>_xx_yy.adt  - Referenced terrain tiles
```

## Next Steps
1. Implement path visualization
2. Add editing tools
3. Create format conversion utilities
4. Add validation tools

## References
- [PM4 Format Specification](docs/wowdev.wiki/PM4.md)
- [PD4 Format Specification](docs/wowdev.wiki/PD4.md)