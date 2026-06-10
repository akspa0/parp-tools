# MCNR — Vertex Normals

## Summary
Stores per-vertex terrain normals for one MCNK.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed sequential 145x3-byte decode in `FUN_006afe90` |

## Structure — Build 0.7.0.3694 (confirmed)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | int8[145][3] | normals | X/Y/Z normal triplets |

## Size
- Expected payload size: `145 * 3 = 435` bytes.

## Ghidra Notes
- `FUN_006afe90` runs fixed loop count `0x91` (145 normals).
- Reads signed bytes triplets and scales each component by `_DAT_00827ff0`.

## Confidence
- **High**
