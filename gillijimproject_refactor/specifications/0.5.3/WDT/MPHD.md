# MPHD — WDT Header/Global Flags

## Summary
Global WDT header chunk controlling map-wide behavior and offsets/flags.

## Parent Chunk
Root-level WDT

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Parser assertion string `iffChunk.token=='MPHD'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32 | flags? | likely global map flags |
| 0x04 | ??? | offsetsOrCounts? | unresolved in this pass |

## Ghidra Notes
- **Function address**: `???`
- **Evidence**: assertion string at `0x0089FC6C`
- **Parser pattern**: validated as mandatory early root chunk near `MVER`/`MAIN`

## Confidence
- **Medium** for presence/importance, **Low** for exact layout

## Unknowns
- Exact bit definitions
- Whether this build carries extra monolithic-map indicators in MPHD
