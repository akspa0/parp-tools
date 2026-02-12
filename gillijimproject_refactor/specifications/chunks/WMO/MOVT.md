# MOVT — Group Vertices

## Summary
Stores group vertex positions for a WMO group.

## Parent Chunk
`MOGP`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_006c1a10`: `MOVT` checked after `MOVI` |

## Structure — Build 0.7.0.3694 (confirmed)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[3] * n | vertices | XYZ vertex list |

Stride: 12 bytes (`count = chunkSize / 0x0C`).

## Confidence
- **High**
