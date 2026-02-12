# MOTV — Group Texture Coordinates

## Summary
UV coordinates for group vertices.

## Parent Chunk
`MOGP`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Requires direct confirmation for dual-MOTV behavior; baseline UV role is inferred |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | float[2] * n | uv | UV coordinate list |

Stride expected: 8 bytes per UV entry.

## Open Questions
- Single vs multiple `MOTV` blocks per group in this build.
- Possible V-flip behavior.

## Confidence
- **Medium**
