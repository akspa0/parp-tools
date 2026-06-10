# HTST — Hit Test Data

## Summary
Optional hit-test shape data consumed when load flag `0x20` is set.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_00421440` queries token `0x54535448` (`HTST`).
- `FUN_00421310` incorporates HTST count into bone/matrix path when enabled.

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total HTST payload bytes |
| 0x04 | uint32 | shapeCount | Number of hit-test shapes |
| 0x08 | byte[] | shapes | Variable-size records by shape type tag |

## Confidence
- **High (framing), Medium (shape internals)**
