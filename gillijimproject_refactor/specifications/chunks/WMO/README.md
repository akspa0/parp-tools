# WMO Chunks — Build 0.7.0.3694

This folder documents WMO chunks for **0.7.0.3694** from direct Ghidra decompilation of the loaded client.

## Core parser evidence (0.7.0.3694)

- Root parser: `FUN_006c11a0`
- Group parser entry: `FUN_006c17f0`
- Required group subchunk sequence: `FUN_006c1a10`
- Optional group subchunk gates: `FUN_006c1c60`
- Group file name synthesis/loading: `FUN_006c1570`

## Early v17 transition behavior confirmed in this build

- Root and groups are split: root file loads first, groups are loaded from suffixed files using `_%03d` replacement in `FUN_006c1570`.
- Group files are validated as `MVER(version=0x10)` then `MOGP` (`FUN_006c17f0`).
- Group payload layout is already close to later v17-style chunked groups (`MOPY/MOVI/MOVT/MONR/MOTV/MOBA` plus flag-gated optionals).
- Version value is still checked as `0x10` in both root and group files.

## Chunks documented

### Root file chunks
- `MVER.md`
- `MOHD.md`
- `MOTX.md`
- `MOMT.md`
- `MOGN.md`
- `MOGI.md`
- `MOPV.md`
- `MOPT.md`
- `MOPR.md`
- `MOLT.md`
- `MODS.md`
- `MODN.md`
- `MODD.md`
- `MFOG.md`
- `MCVP.md`

### Group file chunks
- `MOGP.md`
- `MOPY.md`
- `MOVI.md`
- `MOVT.md`
- `MONR.md`
- `MOTV.md`
- `MOBA.md`
- `MOLR.md`
- `MODR.md`
- `MOBN.md`
- `MOBR.md`
- `MPBV.md`
- `MPBP.md`
- `MPBI.md`
- `MPBG.md`
- `MOCV.md`
- `MLIQ.md`
