# MOGP — WMO Group Header and Container

## Summary
Defines one WMO group and contains subordinate geometry/material/liquid chunks.

## Parent Chunk
Root-level WMO group region.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Direct parser evidence in `FUN_006c1570`, `FUN_006c17f0`, `FUN_006c1a10`, `FUN_006c1c60` |

## File-level context (0.7.0.3694)

- Groups are loaded from separate files, generated from root WMO path by replacing extension with `_%03d` in `FUN_006c1570`.
- Group file starts with `MVER(version=0x10)` followed by `MOGP` (`FUN_006c17f0`).

## Structure — Build 0.7.0.3694 (confirmed subchunk contract)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | Group flags (used by optional subchunk gates in `FUN_006c1c60`) |
| ... | ... | ... | Additional group header fields copied in `FUN_006c17f0` |

### Required subchunk order (from `FUN_006c1a10`)
1. `MOPY` (count = size >> 2)
2. `MOVI` (count = size >> 1)
3. `MOVT` (count = size / 12)
4. `MONR` (count = size / 12)
5. `MOTV` (count = size >> 3)
6. `MOBA` (count = size >> 5)

### Optional chunks (from `FUN_006c1c60`, flag-gated)

- `MOLR` when `flags & 0x200`
- `MODR` when `flags & 0x800`
- `MOBN` then `MOBR` when `flags & 0x1`
- `MPBV` then `MPBP` then `MPBI` then `MPBG` when `flags & 0x400`
- `MOCV` when `flags & 0x4`
- `MLIQ` when `flags & 0x1000`

## Confidence
- Subchunk order/count formulas: **High**
- Complete field map: **Medium**
