# MCNK — Map Chunk Header and Subchunk Table

## Summary
Primary terrain chunk container in ADT files; owns offsets to terrain, layers, alpha, and liquids.

## Parent Chunk
Root-level ADT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Full Alpha header mapping available |
| 0.6.0.3592 | Transitional build; MCNK layout remains Alpha-like |
| 0.7.0.3694 | **Direct Ghidra-confirmed in `FUN_006af6f0` + `FUN_006af0f0`** |

## Structure — Build 0.7.0.3694 (confirmed where listed)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | MCNK flags; includes liquid/shadow related bits |
| 0x04 | uint32 | indexX | Chunk index X |
| 0x08 | uint32 | indexY | Chunk index Y |
| 0x0C | uint32 | nLayers | Layer count (`FUN_006af0f0` loop bound) |
| 0x10 | uint32 | layerCount | Number of texture layers |
| 0x14 | uint32 | ofsMCVT | Offset to `MCVT` |
| 0x18 | uint32 | ofsMCNR | Offset to `MCNR` |
| 0x1C | uint32 | ofsMCLY | Offset to `MCLY` |
| 0x20 | uint32 | ofsMCRF | Offset to `MCRF` |
| 0x24 | uint32 | ofsMCAL | Offset to `MCAL` |
| 0x2C | uint32 | ofsMCSH | Offset to `MCSH` |
| 0x34 | uint32 | flags2 | Additional flags/metadata (`???` exact semantics) |
| 0x3C | uint16 | holesOrHoleCount | Holes-related value (`???` exact interpretation in 0.7) |
| 0x58 | uint32 | ofsMCSE | Offset to `MCSE` |
| 0x60 | uint32 | ofsMCLQ | Offset to `MCLQ` |
| 0x68 | float | posX | Chunk world position X |
| 0x6C | float | posY | Chunk world position Y |
| 0x70 | float | posZ | Chunk world position Z |

## Ghidra 0.7.0.3694 Notes
- `FUN_006af6f0` validates: `MCNK`, `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`.
- `FUN_006af0f0` copies runtime fields from header offsets:
	- `+0x34` (flags2-like), `+0x3C` (16-bit holes/holeCount), `+0x68/+0x6C/+0x70` (position).
	- Layer loop uses `*(header + 0x0C)` with `0x10` stride per `MCLY` entry.

## Confidence
- Core offset table and validations: **High**
- Unnamed semantic labels: **Medium**
