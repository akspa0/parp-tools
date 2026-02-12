# BONE — Skeleton/Bone Nodes

## Summary
Bone hierarchy and transform animation tracks for model deformation.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_00421310` token `0x454e4f42` (`BONE`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ??? | ??? | Node/bone records and key tracks; direct map pending |

## Ghidra Notes
- `FUN_00421310` reads `BONE` count from chunk header `+4`.
- When load flag `0x100` is set, bone processing is bypassed and count is forced to 1.
- This function also adjusts counts with `HTST` and reads `TXAN` count.

## Confidence
- Presence: **High**
- Count semantics: **High**
- Full record map: **Low**
