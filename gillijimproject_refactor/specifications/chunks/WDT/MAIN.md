# MAIN — Tile Presence/Index Table

## Summary
64x64 tile table indicating available map tiles and associated metadata.

## Parent Chunk
Root-level WDT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | 16-byte monolithic entries documented |
| 0.6.0.3592 | Loader reads `0x8000`; indicates 8-byte entry regime |
| 0.7.0.3694 | Inferred likely 8-byte entry continuity |

## Structure — Build 0.7.0.3694 (inferred, medium confidence)

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
- 0.6.0 shows separate ADT file loading by tile naming, so `MAIN` is primarily presence/index metadata.
- Exact tile index formula for 0.7 (`x*64+y` vs `y*64+x`) should be verified directly in 0.7 decompile.

## Confidence
- Payload size trend: **High**
- Per-field semantics in 0.7: **Medium/Unknown**
