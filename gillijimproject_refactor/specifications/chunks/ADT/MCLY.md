# MCLY — Texture Layer Entries

## Summary
Defines texture layers used by one MCNK and links into alpha map data.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Presence/offset confirmed in `FUN_006af6f0`; per-layer loop in `FUN_006af0f0` |

## Structure — Build 0.7.0.3694 (confirmed shape)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | textureId | Texture table index |
| 0x04 | uint32 | flags | Layer flags (`???` exact per-bit map in 0.7) |
| 0x08 | uint32 | ofsMCAL | Offset into alpha payload |
| 0x0C | uint32 | effectId | Ground effect / extra index (`???`) |

Entry size is expected to remain `0x10` bytes.

## Ghidra Notes
- `FUN_006af0f0` walks layer entries with `0x10` stride for `nLayers` iterations.
- `nLayers` read from MCNK header offset `+0x0C`.

## Confidence
- Entry stride/count handling: **High**
- Bit semantics: **Medium**
