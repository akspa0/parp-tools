# MOTV — Group Texture Coordinates

## Summary
UV coordinates for group vertices.

## Parent Chunk
`MOGP`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_006c1a10` required group sequence |

## Structure — Build 0.7.0.3694 (confirmed)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[2] * n | uv | UV coordinate list |

Stride: 8 bytes (`count = chunkSize >> 3`).

## Open Questions
- Single vs multiple `MOTV` blocks per group in this build.
- Possible V-flip behavior.

## Confidence
- **High**
