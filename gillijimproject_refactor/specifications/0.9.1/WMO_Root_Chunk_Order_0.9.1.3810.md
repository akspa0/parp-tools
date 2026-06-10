# WMO Root Chunk Order — Build 0.9.1.3810 (macOS)

## Summary
`CMapObj::CreateDataPointers()` reveals the expected root-level WMO chunk walk for this build, with strict token assertions and endian conversion.

## Build
`0.9.1.3810`

## Required root sequence observed
1. `MVER` (must be `0x11`)
2. `MOHD`
3. `MOTX`
4. `MOMT`
5. `MOGN`
6. `MOGI`
7. `MOSB`
8. `MOPV`
9. `MOPT`
10. `MOPR`
11. `MOVV`
12. `MOVB`
13. `MOLT`
14. `MODS`
15. `MODN`
16. `MODD`
17. `MFOG`
18. Optional trailing `MCVP` (if present)

## Key structural hints
- `MOMT` entries are treated as 0x40-byte records.
- `MOGI` entries are treated as 0x20-byte records.
- `MOPT` entries are treated as 0x14-byte records.
- `MOPR` entries are treated as 0x08-byte records.
- `MOLT` entries are treated as 0x30-byte records.
- `MODS` entries are treated as 0x20-byte records.
- `MODD` entries are treated as 0x28-byte records.
- `MFOG` entries are treated as 0x30-byte records.

These are direct from `ConvertArrayToBinary<T>(..., size/divisor)` usage in the decompile.

## Related to guide priorities
- Strong WMO structural evidence is now available for `MOGP`, `MOTV`, `MLIQ`, and root chunk ordering.
- No direct `MH2O` references were observed in this binary during this pass.

## Ghidra Notes
- `CMapObj::CreateDataPointers()` (decompile captured from `0x0029ace8` call path output).
- Token mismatch paths consistently use `_SErrDisplayErrorFmt` with FourCC-aware formatting.

## Confidence
- **High** on chunk order and fixed-size array divisors.
- **Medium** on unresolved semantic naming for some `MOHD` field members.
