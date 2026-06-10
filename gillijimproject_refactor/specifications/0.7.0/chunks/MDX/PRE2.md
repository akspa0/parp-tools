# PRE2 — Particle Emitter Type 2

## Summary
Advanced particle emitter data and tracks.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_00447bf0` token `0x32455250` (`PRE2`) |

## Structure — Build 0.7.0.3694 (inferred, medium)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total PRE2 payload bytes |
| 0x04 | uint32 | emitterCount | Number of emitters |
| 0x08 | byte[] | emitters | Variable-size records; each starts with `bytesThisEmitter` |

## Confidence
- **High (framing), Medium (inner fields)**
