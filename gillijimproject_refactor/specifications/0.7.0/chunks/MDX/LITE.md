# LITE — Light Records

## Summary
Model light definitions.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_00449330` queries token `0x4554494c` (`LITE`).

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total LITE payload bytes |
| 0x04 | uint32 | lightCount | Number of light records |
| 0x08 | byte[] | lights | Variable-size records; each starts with `bytesThisEmitter` |

## Confidence
- Framing: **High**
- Per-light semantics: **Medium**
