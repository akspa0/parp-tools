# PRE2 — Particle Emitter Type 2

## Summary
Advanced particle emitter data and tracks.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | `MdxReadEmitters2` stage confirmed |
| 0.7.0.3694 | Inferred continuity |

## Structure — Build 0.7.0.3694 (inferred, medium)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | inclusiveSize | Record size |
| ... | struct | nodeAndEmitterData | Emitter parameters + animation tracks (`???` exact offsets) |

## Confidence
- **Medium**
