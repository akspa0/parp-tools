# MCLY — Texture Layer Entries

## Summary
Defines texture layers used by one MCNK and links into alpha map data.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Layer-loop usage confirmed |
| 0.6.0.3592 | Offset table and presence validated |
| 0.7.0.3694 | Inferred from continuity |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | textureId | Texture table index |
| 0x04 | uint32 | flags | Layer flags (`???` exact per-bit map in 0.7) |
| 0x08 | uint32 | ofsMCAL | Offset into alpha payload |
| 0x0C | uint32 | effectId | Ground effect / extra index (`???`) |

Entry size is expected to remain `0x10` bytes.

## Confidence
- **Medium**
