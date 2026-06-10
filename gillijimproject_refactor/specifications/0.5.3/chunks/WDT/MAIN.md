# MAIN — Tile Presence/Index Table

## Summary
Primary map tile presence/index table for WDT root.

## Parent Chunk
Root-level WDT chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk.token=='MAIN'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ???[] | tileEntries | Tile presence/index entries (`???` exact stride/fields) |

## Notes
- Role matches later MAIN semantics, but 0.5.3 entry stride and flags remain unconfirmed.
- MAIN participates in root parse before ADT-like embedded chunk handling in monolithic flow.

## Confidence
- Presence and high-level role: **High**
- Entry layout: **Low-Medium**
