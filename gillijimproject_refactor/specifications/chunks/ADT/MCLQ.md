# MCLQ — Terrain Liquid Data

## Summary
Legacy terrain liquid payload referenced by MCNK; present in early builds before MH2O takeover.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Inline liquid representation documented |
| 0.6.0.3592 | `MCLQ` validated as MCNK subchunk and parsed in packed instances |
| 0.7.0.3694 | Inferred same family as 0.6.0; direct confirmation pending |

## Structure — Build 0.7.0.3694 (inferred, medium-high confidence)

### Chunk-level
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MCLQ` |
| 0x04 | uint32 | size | Payload byte size |
| 0x08 | byte[] | payload | Packed liquid instances |

### Per-liquid-instance (0.6 lineage)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x000 | float | minHeight | Minimum liquid height |
| 0x004 | float | maxHeight | Maximum liquid height |
| 0x008 | byte[0x288] | vertexRegion | Likely 9x9 vertex-related records (`???` exact field split) |
| 0x290 | byte[64] | tileFlags | 8x8 tile/flags area |
| 0x2D0 | uint32 | extra | Trailing value (`???`) |

Per-instance stride (observed in 0.6): `0x2D4` bytes.

## Version Differences
- **0.5.3**: inline liquid model heavily coupled to MCNK internals.
- **0.6.0**: explicit `MCLQ` token validation in MCNK subchunk table.
- **0.7.0**: expected to still support `MCLQ` while beta transitions continue.

## Ghidra Notes
- 0.6.0 functions: `FUN_006a6d00` (validation), `FUN_006a6960` (instance parsing).

## Confidence
- Presence/parsing path: **High**
- Inner record semantics: **Medium**
