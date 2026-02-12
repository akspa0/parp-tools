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

## Confidence
- Presence/order: **High**
- Per-field semantics: **Low**
