# MODF — WMO Placement Entries

## Summary
Placement records for WMO instances in ADT object data.

## Parent Chunk
Root-level ADT object section.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Fully mapped; matches LK-era structure |
| 0.7.0.3694 | Confirmed count formula in `FUN_006bd840`: `modfCount = chunkSize >> 6` |

## Structure — Build 0.7.0.3694 (high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | nameId | Index into WMO name table |
| 0x04 | uint32 | uniqueId | Unique instance ID |
| 0x08 | float[3] | position | Position vector |
| 0x14 | float[3] | rotation | Rotation vector |
| 0x20 | float[6] | extents | Axis-aligned bounding extents |
| 0x38 | uint16 | flags | Placement flags |
| 0x3A | uint16 | doodadSet | Active doodad set |
| 0x3C | uint16 | nameSet | Name set index |
| 0x3E | uint16 | padding | Reserved/padding |

Entry size: `0x40` (64) bytes.

## Confidence
- **High**
