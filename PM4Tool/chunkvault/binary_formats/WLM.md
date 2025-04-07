# Water Level Magma (WLM) Format Documentation

## Overview
The WLM format is a specialized variant of the WLW format specifically used for magma/lava liquid surfaces in World of Warcraft. It shares the exact same structure as WLW but with a fixed liquid type of magma (6).

## File Structure
The WLM format follows the identical structure as WLW:

### Header (12 bytes)
| Offset | Type | Name | Description | ID |
|--------|------|------|-------------|-----|
| 0x00 | char[4] | magic | "LIQ*" identifier | M001 |
| 0x04 | uint16 | version | Format version (0, 1, or 2) | M002 |
| 0x06 | uint16 | flags | Always 1, likely version flags | M003 |
| 0x08 | uint16 | liquidType | Always 6 (Magma) | M004 |
| 0x0A | uint16 | padding | Padding (or liquidType might be uint32) | M005 |

### Block Count (4 bytes)
| Offset | Type | Name | Description | ID |
|--------|------|------|-------------|-----|
| 0x0C | uint32 | blockCount | Number of magma blocks | M006 |

### Magma Block Structure (Variable Size)
Each block is composed of:
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| vertices | C3Vector[16] | 16 vertices in a 4x4 grid (z-up) | M007 |
| coord | C2Vector | Internal coordinates | M008 |
| data | uint16[0x50] | Additional block data | M009 |

### Optional Block2 Structure
Only seen in 'world/maps/azeroth/test.wlm':
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| count | uint32 | Number of block2 structures | M010 |
| _unk00 | C3Vector | Unknown vector | M011 |
| _unk0C | C2Vector | Unknown coordinates | M012 |
| _unk14 | byte[0x38] | 4 floats then zero-filled | M013 |

### Version 1+ Additional Data
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| unknown | byte | Usually 1 | M014 |

## Height Map Structure
Identical to WLW, the 16 vertices in each block form a 4x4 grid arranged from lower right to upper left:

```
15 14 13 12
11 10  9  8
 7  6  5  4
 3  2  1  0
```

## Implementation Status
- ⏳ Documentation Complete
- 🔲 Parser Implementation
- 🔲 Height Map Generation
- 🔲 Conversion to MH20/MCLQ

## Related Formats
- WLW (Water Level Water) - Base water format
- WLQ (Water Level Quality?) - Companion format
- MCLQ (Map Chunk Liquid) - ADT liquid format
- MH2O (Map Height 2.0) - Modern ADT liquid format

## Key Differences from WLW
1. LiquidType is always set to 6 (Magma)
2. Used exclusively for magma/lava surfaces
3. May have different rendering properties in-game
4. Often found in volcanic or underground areas

## Implementation Notes

### Key Differences from WLW
1. **Fixed Liquid Type**
   - Always uses liquid type 6 (Magma)
   - No variation in liquid type allowed
   - Simplifies parsing and validation

2. **Rendering Considerations**
   - May require special effects for magma
   - Different texture and animation settings
   - Unique particle effects
   - Heat distortion effects

### Validation Requirements
1. **Header Validation**
   - Standard WLW header checks
   - Additional check for liquidType = 6
   - Version compatibility verification

2. **Data Validation**
   - Same block validation as WLW
   - Magma-specific property validation
   - Temperature/effect parameter checks

### Usage Context
- Used for lava pools and flows
- Volcanic areas and magma chambers
- Underground lava rivers
- Specific dungeon/raid effects

### Best Practices
1. **Implementation**
   - Reuse WLW parsing code
   - Add magma-specific validation
   - Implement special effects
   - Consider performance optimizations

2. **Error Handling**
   - Validate liquid type strictly
   - Provide clear error messages
   - Handle version differences
   - Maintain format compatibility 