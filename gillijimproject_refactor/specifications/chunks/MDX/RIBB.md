# RIBB — Ribbon Emitters

## Summary
Ribbon/trail emitter definitions and animation tracks.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_0044a180` token `0x42424952` (`RIBB`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total RIBB payload bytes |
| 0x04 | uint32 | emitterCount | Number of ribbon emitters |
| 0x08 | byte[] | emitters | Variable-size records; each starts with `bytesThisEmitter` |

## Confidence
- Framing: **High**
- Full inner map: **Medium**
