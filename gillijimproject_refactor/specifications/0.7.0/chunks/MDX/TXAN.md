# TXAN — Texture Animation Data

## Summary
Texture animation support chunk used by animation setup.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_00421310` queries token `0x4e415854` (`TXAN`).

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | TXAN payload bytes |
| 0x04 | uint32 | entryCount | Entry count consumed into shared state |

## Confidence
- Presence/count wiring: **High**
- Entry internals: **Low/Medium**
