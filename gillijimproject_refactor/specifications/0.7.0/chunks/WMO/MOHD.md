# MOHD — WMO Root Header

## Summary
Global header for WMO root metadata (counts, bounds, flags).

## Parent Chunk
Root-level WMO chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_006c11a0` root WMO parser sequence |

## Structure — Build 0.7.0.3694

`FUN_006c11a0` resolves MOHD first and then maps subsequent chunk offsets via cumulative walking.

Confirmed root chunk walk after `MOHD` in this build:
`MOTX -> MOMT -> MOGN -> MOGI -> MOPV -> MOPT -> MOPR -> MOLT -> MODS -> MODN -> MODD -> MFOG -> [MCVP optional]`

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ??? | ??? | Field-level MOHD internals not named in this pass |

## Field semantics from runtime usage (0.7.0.3694)

The parser mostly derives counts from chunk sizes (`MOMT/MOGI/...`) rather than trusting `MOHD` count fields.
However, several `MOHD` fields are explicitly consumed:

| MOHD offset | Runtime use | Evidence | Meaning |
|---|---|---|---|
| `+0x1C` | copied to `mapObj + 0x184` | `FUN_006c0f70` | Root-level color/lighting control value used when building group/light state (high confidence it is ambient-like color) |
| `+0x24..+0x38` (6 dwords) | copied to `mapObj + 0x18C..0x1A0` | `FUN_006c0f70` | Root AABB min/max used by `FUN_006aa6f0`/`FUN_006aa850` to compute center/radius and culling bounds |

### Behavioral notes
- `FUN_006aa6f0` computes world-space center/radius from `mapObj + 0x18C..0x1A0` (the copied `MOHD` bounds).
- `FUN_006aa850` exposes those six AABB values as a fast copy for downstream render/query code.
- This means `MOHD` contributes directly to top-level map-object culling and broad-phase spatial tests.

## Confidence
- Presence/order: **High**
- Per-field semantics: **Medium** (for offsets listed above)
