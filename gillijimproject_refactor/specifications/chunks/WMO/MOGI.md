# MOGI — WMO Group Info Records

## Summary
Per-group metadata table used to instantiate root-side group objects.

## Parent Chunk
Root WMO file.

## Build 0.7.0.3694 evidence
- Parsed in `FUN_006c11a0` after `MOGN`.
- Count formula: `count = chunkSize >> 5` (32-byte stride).

## Structure
- Fixed-size records, 32 bytes each.

## Structure (usage-derived map, 0.7.0.3694)

`FUN_006aaac0`, `FUN_006aa8e0`, and `FUN_006aaa30` expose how each 0x20-byte record is consumed:

| Offset | Type | Name | Runtime meaning |
|---|---|---|---|
| `+0x00` | uint32 | flags | Group behavior/classification bits (queried by `FUN_006aaac0`) |
| `+0x04` | float | bboxMinX | Local AABB min X |
| `+0x08` | float | bboxMinY | Local AABB min Y |
| `+0x0C` | float | bboxMinZ | Local AABB min Z |
| `+0x10` | float | bboxMaxX | Local AABB max X |
| `+0x14` | float | bboxMaxY | Local AABB max Y |
| `+0x18` | float | bboxMaxZ | Local AABB max Z |
| `+0x1C` | uint32 | nameOrId | Not consumed in traced rendering path (likely name/group metadata index) |

Derived at runtime:
- Center/radius from min/max (`FUN_006aa8e0`) for culling/sphere tests.
- AABB bulk copy (`FUN_006aaa30`) for visibility/query math.

## Confidence
- Presence/order/stride: **High**
- Per-field names: **Medium**

## Runtime behavior (0.7.0.3694)

- `FUN_00699c10` iterates per-group records from root group-info storage and builds per-group runtime defs.
- Group flags from these records are read through `FUN_006aaac0(groupIndex)` and used to classify group state/pathing.
- Group AABB values are copied and used by visibility/frustum culling before draw-list insertion in the world loop.
- In `FUN_00699c10`, flag test `(flags & 0x48)` routes group-def state (`0x08` vs `0x10` bit assignment), affecting downstream pass behavior.
