# MAIN — Tile Presence / Map Grid Table

## Summary
Core WDT table chunk used to describe which map tiles/substructures are present.

## Parent Chunk
Root-level WDT

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Parser assertion string `iffChunk.token=='MAIN'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | tileTable | Presence/flags table for map cells (exact row width unresolved) |

## Ghidra Notes
- **Function address**: `???`
- **Evidence**: assertion string at `0x0089FC54`
- **Parser pattern**: root WDT token validation before table consumption

## Confidence
- **Medium** for chunk role, **Low** for field-level offsets

## Unknowns
- Exact entry size and whether flags include alpha-era streaming hints
- Whether MAIN directly maps to 64x64 in this build or transitional dimensions
