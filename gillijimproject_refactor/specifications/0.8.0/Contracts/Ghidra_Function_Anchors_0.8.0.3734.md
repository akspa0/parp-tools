# Ghidra Function Anchors — 0.8.0.3734

## Scope
Address-only anchor map for parser-relevant functions in `WoW.exe` build `0.8.0.3734`.

## ADT anchors
- `0x006c7220` — ADT root parser (`MVER`/`MHDR` checks, root chunk offset wiring, `MDDF`/`MODF` count derivation)
- `0x006b8f90` — MCNK required subchunk pointer resolution (`MCVT/MCNR/MCLY/MCRF/MCSH/MCAL/MCLQ/MCSE`)
- `0x006b8be0` — MCLQ layer parse (`4` slots, stride `0xB5` dwords)

## WMO anchors
- `0x006ca8b0` — root WMO file load entry
- `0x006cac40` — root WMO strict chunk sequence parser (`MVER=0x10` path)
- `0x006cb290` — group root parser (`MVER` + `MOGP` checks)
- `0x006cb4b0` — required group chunk parser (`MOPY/MOVI/MOVT/MONR/MOTV/MOBA`)
- `0x006cb700` — optional group chunk parser (`MOLR/MODR/MOBN/MOBR/MPB*/MOCV/MLIQ`)

## MDX anchors
- `0x00422620` — model async completion path (`MDLX` signature gate; dispatches to deeper reader)
- `0x006bbd10` — world/model MDX section parser (`TEXS`, `GEOS`, geoset subchunks)
- `0x0044e380` — geoset parse core (`VRTX/NRMS/UVAS` and consistency checks)
- `0x0044ea20` — geoset continuation parser (`GNDX/MTGC/MATS/BIDX/BWGT`)

## Notes
- Function names are compiler-generated (`FUN_*`) in this program; behavior labels are from decompiled control flow and token assertions.
- These anchors are intended to seed profile deltas and implementation patching, not full-file reverse engineering.
