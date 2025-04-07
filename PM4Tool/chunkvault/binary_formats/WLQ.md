# Water Level Quality (WLQ) Format Documentation

## Overview
The WLQ format is a companion format to WLW/WLM files, containing additional liquid quality and type information. Each WLQ file corresponds to a WLW file at the same path and filename (with .wlq extension). The format provides extended liquid properties and potentially quality settings for the liquid surface.

## File Structure

### Header (32 bytes)
| Offset | Type | Name | Description | ID |
|--------|------|------|-------------|-----|
| 0x00 | char[4] | magic | "LIQ2" identifier | Q001 |
| 0x04 | uint16 | version | Format version | Q002 |
| 0x06 | uint16 | flags | Always 1 | Q003 |
| 0x08 | byte[4] | padding | Always 0 | Q004 |
| 0x0C | uint32 | liquidType | Liquid type identifier | Q005 |
| 0x10 | uint16[9] | unknown | Unknown data | Q006 |
| 0x22 | uint32 | blockCount | Number of blocks (matches WLW) | Q007 |

### Liquid Types
| Value | Type | Description |
|-------|------|-------------|
| 0 | River | River/stream water |
| 1 | Ocean | Ocean water |
| 2 | Magma | Magma/lava |
| 3 | Slime | Slime liquid |

### Block Structure
Each block is 360 bytes and follows the same format as WLW blocks:
| Component | Type | Description | ID |
|-----------|------|-------------|-----|
| vertices | C3Vector[16] | 16 vertices in a 4x4 grid (z-up) | Q008 |
| coord | C2Vector | Internal coordinates | Q009 |
| data | uint16[0x50] | Additional block data | Q010 |

## Implementation Status
- ⏳ Documentation Complete
- 🔲 Parser Implementation
- 🔲 Quality Data Integration
- 🔲 Conversion to MH20/MCLQ

## Related Formats
- WLW (Water Level Water) - Base water format
- WLM (Water Level Magma) - Magma variant
- MCLQ (Map Chunk Liquid) - ADT liquid format
- MH2O (Map Height 2.0) - Modern ADT liquid format

## Key Differences from WLW/WLM
1. Different magic number ("LIQ2" vs "LIQ*")
2. Extended header with additional fields
3. Different liquid type enumeration
4. Always paired with a corresponding WLW/WLM file
5. May contain quality or rendering properties

## Usage Notes
1. Each WLQ file must have a corresponding WLW/WLM file
2. Block count must match the corresponding WLW/WLM file
3. Used by the client to determine liquid rendering properties
4. May affect visual quality or behavior of liquid surfaces 