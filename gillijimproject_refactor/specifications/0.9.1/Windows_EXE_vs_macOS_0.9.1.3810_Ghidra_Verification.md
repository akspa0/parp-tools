# Windows EXE vs macOS Binary Verification — WoW 0.9.1.3810

## Scope
Cross-check of previously documented macOS findings against `WoW.exe` (same build `0.9.1.3810`) in Ghidra.

## Conclusion
The Windows EXE behavior **matches** the macOS findings for ADT `MCNK/MCLQ`, WMO root/group parsing, optional `MLIQ` gating, and absence of `MH2O` references.

---

## 1) ADT MCNK + MCLQ parity

### MCNK subchunk offset assertions (EXE)
Function: `FUN_006d7710` (stripped; corresponds to mac `CMapChunk::CreatePtrs`)

Verified required subchunks and header offsets:
- `MCVT` at `header+0x14`
- `MCNR` at `header+0x18`
- `MCLY` at `header+0x1C`
- `MCRF` at `header+0x20`
- `MCAL` at `header+0x24`
- `MCSH` at `header+0x2C`
- `MCSE` at `header+0x58`
- `MCLQ` at `header+0x60`

This matches macOS decompilation exactly.

### MCLQ liquid construction parity (EXE)
Function: `FUN_006d7500` (stripped; corresponds to mac `CMapChunk::CreateLiquids`)

Observed:
- Iterates 4 liquid slots
- Uses `MCNK.flags` mask sequence (`0x4`, then left-shift per slot)
- Per-slot data block pointer advances by `0xC9 dwords` (`0x324` bytes)
- Field wiring equivalent to mac path:
  - `+0x10` <= block dword 0
  - `+0x14` <= block dword 1
  - `+0x18` <= block +2 dwords
  - `+0x1C` <= block +0xA4 dwords
  - `+0x20` <= block +0xB4 dwords
  - `+0x24` <= block +0xB5 dwords
- Calls vertex build helper `FUN_006b1380` (stripped; corresponds to mac `CChunkLiquid::CreateVertices`)

### Liquid vertex builder parity (EXE)
Function: `FUN_006b1380`

Observed:
- Reads source records with stride 8 bytes
- Uses `*(float *)(source + 4)` for height component
- Emits a 9×9 grid behavior consistent with mac path

---

## 2) WMO Group parity (MOGP/MOTV/MLIQ)

### Group entry and required sequence (EXE)
Functions:
- `FUN_006e8650` (stripped; corresponds to mac `CMapObjGroup::Create`)
- `FUN_006e8870` (stripped; corresponds to mac `CMapObjGroup::CreateDataPointers`)

Verified sequence:
1. `MVER` (`0x11`)
2. `MOGP`
3. `MOPY`
4. `MOVI`
5. `MOVT`
6. `MONR`
7. `MOTV`
8. `MOBA`
9. optional section

`MOTV` handling remains single required chunk in this parser path, matching mac findings.

### Optional chunks + MLIQ gating (EXE)
Function: `FUN_006e8ae0` (stripped; corresponds to mac `CreateOptionalDataPointers`)

Verified flag-gated optional chunks:
- `0x0001` -> `MOBN` + `MOBR`
- `0x0004` -> `MOCV`
- `0x0200` -> `MOLR`
- `0x0400` -> `MPBV` + `MPBP` + `MPBI` + `MPBG`
- `0x0800` -> `MODR`
- `0x1000` -> `MLIQ`
- `0x20000` -> `MORI` + `MORB`

`MLIQ` (`'MLIQ'`) parsing and field extraction pattern matches mac findings (dimension fields, base vector, pointers into vertex/tile arrays).

---

## 3) WMO Root order parity

### Root parser (EXE)
Function: `FUN_006e7f80` (stripped; corresponds to mac `CMapObj::CreateDataPointers`)

Verified root chunk order:
- `MVER`, `MOHD`, `MOTX`, `MOMT`, `MOGN`, `MOGI`, `MOSB`, `MOPV`, `MOPT`, `MOPR`, `MOVV`, `MOVB`, `MOLT`, `MODS`, `MODN`, `MODD`, `MFOG`, optional `MCVP`

Record-size divisors match mac notes:
- `MOMT >> 6`, `MOGI >> 5`, `MOPT / 0x14`, `MOPR >> 3`, `MOLT / 0x30`, `MODS >> 5`, `MODD / 0x28`, `MFOG / 0x30`

---

## 4) MH2O check

String search in EXE for `MH2O`: **no results**.

This matches expected era behavior and the updated research-guide scope guard (no `MH2O` hunts pre-3.0.0 alpha).

---

## Notes on symbols/stripping
- EXE function names are mostly stripped (`FUN_xxxxxxxx`), but assert/source-path strings remain rich enough to align confidently with mac functions.
- Source-path anchors present in EXE include:
  - `C:\build\buildWoW\WoW\Source\WorldClient\MapChunk.cpp`
  - `C:\build\buildWoW\WoW\Source\WorldClient\MapObjRead.cpp`

## Confidence
- **High** overall parity for all previously documented 0.9.1 mac findings.
