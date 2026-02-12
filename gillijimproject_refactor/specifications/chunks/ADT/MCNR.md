# MCNR — Vertex Normals

## Summary
Stores per-vertex terrain normals for one MCNK.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Confirmed sequential 3-byte normal reads |
| 0.7.0.3694 | Inferred continuity from 0.6.0 |

## Structure — Build 0.7.0.3694 (inferred, high confidence)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | int8[145][3] | normals | X/Y/Z normal triplets |

## Size
- Expected payload size: `145 * 3 = 435` bytes.

## Ghidra Notes
- 0.6.0 processing function: `FUN_006a7490`.
- Values are converted to float using a scale constant after byte decode.

## Confidence
- **High**
