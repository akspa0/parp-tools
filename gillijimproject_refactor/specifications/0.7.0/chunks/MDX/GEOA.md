# GEOA — Geoset Animations

## Summary
Optional geoset animation block tied to GEOS parsing.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_0044d730` checks optional token `0x414f4547` (`GEOA`).

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total GEOA payload bytes |
| 0x04 | uint32 | entryCount | Number of geoset animation entries |
| 0x08 | byte[] | entries | Variable-size records |

## Confidence
- Presence and framing: **High**
- Entry semantics: **Medium**
