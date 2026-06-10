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
| 0x08 | uint32 | xVerts | Copied to `group +0xE0`; X sample width for liquid grid |
| 0x0C | uint32 | yVerts | Copied to `group +0xE4`; Y sample height for liquid grid |
| 0x10 | uint32 | xTiles | Copied to `group +0xE8`; tile width for flag map |
| 0x14 | uint32 | yTiles | Copied to `group +0xEC`; tile height for flag map |
| 0x18 | float | liquidBaseX | Copied to `group +0xF0`; world-space liquid origin X |
| 0x1C | float | liquidBaseY | Copied to `group +0xF4`; world-space liquid origin Y |
| 0x20 | float | liquidBaseZ | Copied to `group +0xF8`; reference height |
| 0x24 | uint16 | liquidType | Copied to `group +0xFC`; type/material selector |
| 0x26 | byte[] | liquidData | Base pointer stored at `group +0x100` |

Derived pointer formula from parser:
- `liquidDataPtr = mliq + 0x26`
- `liquidExtraPtr = liquidDataPtr + (field1 * field0 * 8)`

With named fields:
- `liquidDataPtr = mliq + 0x26`
- `liquidFlagsPtr = liquidDataPtr + (xVerts * yVerts * 8)`

## Runtime behavior (0.7.0.3694)

- `FUN_006a3d90` and `FUN_006a3e60` sample this data for liquid presence/height/type queries.
- Liquid tile flags come from `group+0x104` (`liquidFlagsPtr`) and use bit tests (`0x40`, low nibble/type bits).
- Height interpolation reads from `group+0x100` (`liquidDataPtr`) and uses grid dimensions/origin in `group+0xE0..0xF4`.
- If `MOGP.flags & 0x1000` is not set, liquid query path exits early.

## Confidence
- Presence/role: **High**
- Layout skeleton and pointer math: **High**
- Field semantics: **High**
