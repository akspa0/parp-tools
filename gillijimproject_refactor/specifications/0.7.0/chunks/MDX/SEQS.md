# SEQS — Animation Sequences

## Summary
Sequence table used for animation timing/extents.

## Parent Chunk
Root-level MDX chunk stream.

## Build 0.7.0.3694 Evidence
- `FUN_00421a00` queries token `0x53514553` (`SEQS`).

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | sectionBytes | Total SEQS payload bytes |
| 0x04 | uint32 | sequenceCount | Number of sequence entries |

## Confidence
- Presence and count wiring: **High**
- Per-sequence inner fields: **Medium**
