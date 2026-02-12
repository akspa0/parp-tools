# MOVT — Group Vertices

## Summary
Stores group vertex positions for a WMO group.

## Parent Chunk
`MOGP`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | WMO parser lineage implies group geometry chunks in monolithic file |
| 0.7.0.3694 | Inferred same v14 group model |

## Structure — Build 0.7.0.3694 (inferred, medium-high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[3] * n | vertices | XYZ vertex list |

Stride expected: 12 bytes per vertex.

## Confidence
- **Medium-High**
