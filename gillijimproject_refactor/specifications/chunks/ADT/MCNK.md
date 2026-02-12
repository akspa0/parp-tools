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
| 0.7.0.3694 | **Inferred from 0.6.0 + transitional continuity** |

## Structure — Build 0.7.0.3694 (inferred, high confidence for core fields)

| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | MCNK flags; includes liquid/shadow related bits |
| 0x04 | uint32 | indexX | Chunk index X |
| 0x08 | uint32 | indexY | Chunk index Y |
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

## Version Differences
- **0.5.3 → 0.6.0**: layout remains largely compatible; offset table preserved.
- **0.6.0 → 0.7.0**: no contradictory evidence found; treated as same pending direct decompile.

## Ghidra Notes
- 0.6.0 parser lineage: `FUN_006a6d00`, `FUN_006a6710`.
- Validation checks observed for `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCAL`, `MCSH`, `MCSE`, `MCLQ`.

## Confidence
- Core offset table: **High**
- Secondary field semantics: **Medium**
