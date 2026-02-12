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

## Runtime behavior (0.7.0.3694)

- `FUN_006c1a10` wires `MOTV` to `group+0xC4` with UV count at `group+0x11C`.
- UV data is used by textured batch rendering in conjunction with `MOBA` material selection and texture handles resolved from root material tables.
