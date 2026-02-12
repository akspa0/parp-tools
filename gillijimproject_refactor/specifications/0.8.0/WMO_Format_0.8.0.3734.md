# WMO Format Analysis — WoW 0.8.0.3734

## Summary
WMO root parsing is strict and ordered in this build. Group files are loaded separately (`..._NNN.wmo`) and group chunk order is also strict (`MOPY -> MOVI -> MOVT -> MONR -> MOTV -> MOBA`, then optional blocks by flags).

## Build
- **Build**: `0.8.0.3734`
- **Source confidence**: High

---

## Root WMO Parsing (Main .wmo)

### Root parser entry
- **Read path**: `FUN_006ca8b0` → `FUN_006caa10`
- **Chunk scanner**: `FUN_006cac40`

### Verified root chunk sequence in `FUN_006cac40`
`MVER` (version check `0x10`) then:
1. `MOHD`
2. `MOTX`
3. `MOMT`
4. `MOGN`
5. `MOGI`
6. `MOPV`
7. `MOPT`
8. `MOPR`
9. `MOLT`
10. `MODS`
11. `MODN`
12. `MODD`
13. `MFOG`
14. Optional `MCVP`

### Derived count formulas from root parser
- `MOMT`: `size >> 6`  (0x40-byte entries)
- `MOGI`: `size >> 5`  (0x20-byte entries)
- `MOPV`: `size / 0x0c`
- `MOPT`: `size / 0x14`
- `MOPR`: `size >> 3`
- `MOLT`: `size / 0x30`
- `MODS`: `size >> 5`
- `MODD`: `size / 0x28`
- `MFOG`: `size / 0x30`
- Optional `MCVP`: `size >> 4`

### Group object allocation
- `FUN_006caa10` copies root header fields and allocates `groupCount` objects from parsed count (`param_1 + 0x15c` source).

---

## Group WMO Parsing (`..._NNN.wmo`)

### Group file loader
- `FUN_006cb010` builds group filename suffix `_%03d` and opens group file.
- Parses sync via `FUN_006cb290` or async callback `FUN_006cb200`.

### Group root parser (`FUN_006cb290`)
- Requires:
  - `MVER` with version `0x10`
  - next chunk `MOGP`
- Copies group header fields and then dispatches to subchunk parser.

### Group subchunk parser (`FUN_006cb4b0`)
Mandatory order:
1. `MOPY`
2. `MOVI` (`size >> 1` index count)
3. `MOVT` (`size / 0x0c` vertex count)
4. `MONR` (`size / 0x0c` normal count)
5. `MOTV` (`size >> 3` uv count)
6. `MOBA` (`size >> 5` batch count)

Then optional section parser: `FUN_006cb700`
- Optional by flags in `MOGP` header:
  - `MOLR` (flag `0x200`)
  - `MODR` (flag `0x800`)
  - `MOBN` + `MOBR` (flag `0x01`)
  - `MPBV` + `MPBP` + `MPBI` + `MPBG` (flag `0x400`)
  - `MOCV` (flag `0x04`)
  - `MLIQ` (flag `0x1000`)

### `MLIQ` handling in WMO group
From `FUN_006cb700`:
- Reads scalar fields from `param_2[2..12]` region.
- Sets two payload pointers:
  - data A at `chunk + 0x26`
  - data B at `chunk + 0x26 + width*height*8`
- Implies first liquid grid payload is `width * height * 8` bytes.

---

## Confidence
- **High** for root/group chunk order and size-derived entry counts.
- **Medium** for semantic naming of some header fields (offset meanings need additional field-level mapping).