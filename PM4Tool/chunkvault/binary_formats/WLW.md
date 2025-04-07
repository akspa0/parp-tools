# Water Level Water (WLW) Format Documentation

## Overview
The WLW format contains heightmap data for water bodies in World of Warcraft. Each water body (except oceans) has a corresponding WLW file that defines its height information. The format uses a grid-based structure to represent water surface heights.

## File Structure

### Header (12 bytes)
| Offset | Type | Name | Description | ID |
|--------|------|------|-------------|-----|
| 0x00 | char[4] | magic | "LIQ*" identifier | W001 |
| 0x04 | uint16 | version | Format version (0, 1, or 2) | W002 |
| 0x06 | uint16 | flags | Always 1, likely version flags | W003 |
| 0x08 | uint16 | liquidType | Version ≤1: LiquidType enum, Version 2: DB/LiquidType ID | W004 |
| 0x0A | uint16 | padding | Padding (or liquidType might be uint32) | W005 |

### Block Count (4 bytes)
| Offset | Type | Name | Description | ID |
|--------|------|------|-------------|-----|
| 0x0C | uint32 | blockCount | Number of water blocks | W006 |

### Water Block Structure (Variable Size)
Each block is composed of:
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| vertices | C3Vector[16] | 16 vertices in a 4x4 grid (z-up) | W007 |
| coord | C2Vector | Internal coordinates | W008 |
| data | uint16[0x50] | Additional block data | W009 |

### Optional Block2 Structure
Only seen in 'world/maps/azeroth/test.wlm':
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| count | uint32 | Number of block2 structures | W010 |
| _unk00 | C3Vector | Unknown vector | W011 |
| _unk0C | C2Vector | Unknown coordinates | W012 |
| _unk14 | byte[0x38] | 4 floats then zero-filled | W013 |

### Version 1+ Additional Data
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| unknown | byte | Usually 1 | W014 |

## LiquidType Enumeration
| Value | Type | Description |
|-------|------|-------------|
| 0 | Still | Standing water |
| 1 | Ocean | Ocean water |
| 2 | Unknown | Used by 'Shadowmoon Pools 02.wlm' |
| 4 | River | Slow/river water |
| 6 | Magma | Magma/lava |
| 8 | Fast | Fast flowing water |

## Height Map Structure
The 16 vertices in each block form a 4x4 grid arranged from lower right to upper left:

```
15 14 13 12
11 10  9  8
 7  6  5  4
 3  2  1  0
```

When combined, these blocks create a complete water surface heightmap.

## Implementation Status
- ⏳ Documentation Complete
- 🔲 Parser Implementation
- 🔲 Height Map Generation
- 🔲 Conversion to MH20/MCLQ

## Related Formats
- WLM (Water Level Magma) - Magma variant
- WLQ (Water Level Quality?) - Companion format
- MCLQ (Map Chunk Liquid) - ADT liquid format
- MH2O (Map Height 2.0) - Modern ADT liquid format 