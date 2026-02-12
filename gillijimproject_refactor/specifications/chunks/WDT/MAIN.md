# MAIN — Tile Presence/Index Table

## Summary
64x64 tile table indicating available map tiles and associated metadata.

## Parent Chunk
Root-level WDT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_006987e0`: reads MAIN payload size `0x8000` |

## Structure — Build 0.7.0.3694 (confirmed size, inferred entry semantics)

### Chunk payload
- Entry count: `64 * 64 = 4096`
- Reported payload size lineage: `0x8000` bytes
- Implied entry stride: `8` bytes

### Likely entry layout
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | Tile existence/flags |
| 0x04 | uint32 | value | Reserved/index/async (`???` exact semantic in 0.7) |

## Notes
- In `FUN_006987e0`, parse flow is `MVER -> MPHD -> MAIN` then optional `MWMO`/`MODF` when MPHD bit0 is set.

## Confidence
- Payload size: **High**
- Per-field semantics in 0.7: **Medium/Unknown**
