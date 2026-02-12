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

### Header length and usage-derived field map

`FUN_006c17f0` starts subchunk parsing at `piVar1 + 0x16`, confirming a **68-byte MOGP header** before `MOPY`.

| MOGP data offset | Runtime field | Evidence | Behavioral meaning |
|---|---|---|---|
| `+0x00` | `group+0xB0` | `FUN_006c17f0` | Name/ID offset into root group-name blob (`MOGN`) |
| `+0x08` | `group+0x0C` | `FUN_006c17f0`, `FUN_006c1c60` | Group flags controlling optional chunks and render/query classification |
| `+0x0C..0x23` | `group+0x10..0x24` | `FUN_006c17f0` | Group local AABB min/max copy |
| `+0x24` (u16) | `group+0x28` | `FUN_006c17f0` | Portal-start-like index used by portal/visibility recursion paths |
| `+0x2A` (u16) | `group+0x2C` | `FUN_006c17f0` | Portal-count-like value used as loop bound in portal adjacency traversal |
| `+0x2C..0x2F` | `group+0x38..0x3B` | `FUN_006c17f0` | Small-count metadata (batch/portal auxiliary counts) |
| `+0x30..0x33` | `group+0x3C` | `FUN_006c17f0` | Additional small-count/control field |
| `+0x34` | `group+0x30` | `FUN_006c17f0` | Group-level metadata field (not fully named) |
| `+0x38` | `group+0x34` | `FUN_006c17f0` | Group-level metadata field (not fully named) |
| `+0x3C` | `group+0x130` | `FUN_006c17f0` | Group identifier/index used by downstream systems |

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

## Runtime behavior (0.7.0.3694)

- `FUN_006c17f0` copies `MOGP` header fields into the runtime group object and links the group into its parent map object.
- `FUN_006c1a10` wires required geometry arrays (`MOPY/MOVI/MOVT/MONR/MOTV/MOBA`) into fixed runtime offsets used by rendering/query code.
- `FUN_006c1c60` conditionally wires optional arrays (`MOLR/MODR/MOBN/MOBR/MOCV/MLIQ`) that drive lights, doodads, BSP tests, colors, and liquid behavior.
- Group flag bits influence vertex-format sizing (`group+0xD4`) and optional parse/render branches.
- Portal graph traversal paths (`FUN_006ab560`, `FUN_006ab730`) consume group header fields around `group+0x28/+0x2C` as adjacency range/bounds.
