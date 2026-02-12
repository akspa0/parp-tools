# MH2O — New Liquid System Chunk

## Summary
Post-Alpha liquid system chunk that eventually replaces or coexists with MCLQ.

## Parent Chunk
Root-level ADT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Not part of documented Alpha terrain path |
| 0.6.0.3592 | No confirmation of MH2O parser usage in collected reports |
| 0.7.0.3694 | **Unknown** — transition window; requires direct Ghidra confirmation |

## Structure — Build 0.7.0.3694

Not yet confirmed from direct 0.7.0.3694 parser code in this repo snapshot.

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ??? | ??? | Requires direct decompile trace |

## Open Questions
- Does 0.7.0.3694 contain an active `MH2O` parse path? NO, MH2O is in 3.0.0+ only!
- If yes, do `MCLQ` and `MH2O` coexist in the same build/runtime path? 3.x supports both MCLQ and MH2O due to not recompiling all the maps until much later (6.x era)
- What is the exact header and per-chunk liquid instance table in this build?

## Confidence
- **Low** (placeholder pending direct Ghidra extraction)
