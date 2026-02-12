# MLIQ — WMO Liquid Data

## Summary
Liquid definition chunk for WMO groups; distinct from terrain `MCLQ`.

## Parent Chunk
`MOGP`

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed optional parse in `FUN_006c1c60` when `MOGP.flags & 0x1000` |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MLIQ` |
| 0x04 | uint32 | size | Payload size |
| 0x08 | uint32 | field0 | Copied to group +0xE0 |
| 0x0C | uint32 | field1 | Copied to group +0xE4 |
| 0x10 | uint32 | field2 | Copied to group +0xE8 |
| 0x14 | uint32 | field3 | Copied to group +0xEC |
| 0x18 | uint32 | field4 | Copied to group +0xF0 |
| 0x1C | uint32 | field5 | Copied to group +0xF4 |
| 0x20 | uint32 | field6 | Copied to group +0xF8 |
| 0x24 | uint16 | field7 | Copied to group +0xFC |
| 0x26 | byte[] | vertexRegion | Base pointer stored at group +0x100 |

Derived pointer formula from parser:
- `liquidDataPtr = mliq + 0x26`
- `liquidExtraPtr = liquidDataPtr + (field1 * field0 * 8)`

## Confidence
- Presence/role: **High**
- Layout skeleton and pointer math: **High**
- Field semantics: **Medium**
