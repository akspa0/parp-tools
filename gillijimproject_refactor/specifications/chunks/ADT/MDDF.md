# MDDF — Doodad Placement Entries

## Summary
Placement records for M2/MDX doodads referenced by ADT object data.

## Parent Chunk
Root-level ADT object section.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Fully mapped; binary-compatible with LK-era layout |
| 0.6.0.3592 | No conflicting changes observed |
| 0.7.0.3694 | Inferred stable layout |

## Structure — Build 0.7.0.3694 (high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | nameId | Index into model name table |
| 0x04 | uint32 | uniqueId | Unique instance ID |
| 0x08 | float[3] | position | Position vector |
| 0x14 | float[3] | rotation | Rotation vector |
| 0x20 | uint16 | scale | Fixed-point scale (`scale / 1024.0`) |
| 0x22 | uint16 | flags | Placement flags |

Entry size: `0x24` (36) bytes.

## Confidence
- **High**
