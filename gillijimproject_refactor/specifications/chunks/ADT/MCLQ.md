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
| 0.7.0.3694 | Confirmed in `FUN_006af6f0` (token check) + `FUN_006af340` (instance decode) |

## Structure — Build 0.7.0.3694 (confirmed layout shape)

### Chunk-level
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MCLQ` |
| 0x04 | uint32 | size | Payload byte size |
| 0x08 | byte[] | payload | Packed liquid instances |

### Per-liquid-instance (0.7 parser)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x000 | float | minHeight | Minimum liquid height |
| 0x004 | float | maxHeight | Maximum liquid height |
| 0x008 | void* | vertexRegionPtr | Runtime pointer assigned to `instance + 0x008` |
| 0x290 | void* | tileRegionPtr | Runtime pointer assigned to `instance + 0x290` |
| 0x2D0 | uint32 | extra | Copied from `instance + 0x2D0` |

Per-instance stride (observed in 0.7): `0x2D4` bytes (`0xB5` dwords).

## Version Differences
- **0.5.3**: inline liquid model heavily coupled to MCNK internals.
- **0.6.0**: explicit `MCLQ` token validation in MCNK subchunk table.
- **0.7.0**: expected to still support `MCLQ` while beta transitions continue.

## Ghidra Notes
- `FUN_006af6f0` validates `MCLQ` via header offset `+0x60`.
- `FUN_006af340` iterates liquid-type bits (`0x04,0x08,0x10,0x20`) and consumes packed instances.

## Confidence
- Presence/parsing path: **High**
- Inner subfields not named by decompiler: **Medium**
