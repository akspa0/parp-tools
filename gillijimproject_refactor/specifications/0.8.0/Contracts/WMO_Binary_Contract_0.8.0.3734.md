# WMO Binary Contract — 0.8.0.3734 (Ghidra)

## Scope
Binary-derived WMO root/group/liquid contract for WoW.exe build `0.8.0.3734`.

---

## 1) Function Map

- `0x006ca8b0` — WMO root file load entry
- `0x006cac40` — strict root chunk parser
- `0x006cb290` — group root parser (`MVER` + `MOGP`)
- `0x006cb4b0` — required group chunk parser
- `0x006cb700` — optional group chunk parser including `MLIQ`

Confidence: **High** for function roles.

---

## 2) Root WMO Contract (W1)

From `FUN_006cac40`:

- Requires `MVER` and version `0x10`.
- Enforces ordered root chunk decode chain:
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
  14. optional `MCVP`

### Root divisors/constants
- `MOMT`: `size >> 6` (`0x40`)
- `MOGI`: `size >> 5` (`0x20`)
- `MOPV`: `size / 0x0C`
- `MOPT`: `size / 0x14`
- `MOPR`: `size >> 3` (`0x08`)
- `MOLT`: `size / 0x30`
- `MODS`: `size >> 5` (`0x20`)
- `MODD`: `size / 0x28`
- `MFOG`: `size / 0x30`
- `MCVP` (if present): `size >> 4` (`0x10`)

Confidence: **High**
Contradictions: none observed.

---

## 3) Group Required Contract (W2)

From `FUN_006cb290` + `FUN_006cb4b0`:

- Group file requires `MVER` with version `0x10`, then `MOGP`.
- Required group chunk order:
  1. `MOPY`
  2. `MOVI`
  3. `MOVT`
  4. `MONR`
  5. `MOTV`
  6. `MOBA`

### Group divisors/constants
- `MOPY`: `size >> 2`
- `MOVI`: `size >> 1`
- `MOVT`: `size / 0x0C`
- `MONR`: `size / 0x0C`
- `MOTV`: `size >> 3` (`0x08`)
- `MOBA`: `size >> 5` (`0x20`)

Confidence: **High**

---

## 4) Group Optional Gates + `MLIQ` (W3)

From `FUN_006cb700` using `groupFlags = *(this+0x0C)`:

- `flags & 0x00000200` -> `MOLR`
- `flags & 0x00000800` -> `MODR`
- `flags & 0x00000001` -> `MOBN` then `MOBR`
- `flags & 0x00000400` -> `MPBV` then `MPBP` then `MPBI` then `MPBG`
- `flags & 0x00000004` -> `MOCV`
- `flags & 0x00001000` -> `MLIQ`

### `MLIQ` structure wiring
When `flags & 0x1000`:
- token must be `MLIQ`
- header words copied into group fields:
  - `+0xE0 = param_2[2]`
  - `+0xE4 = param_2[3]`
  - `+0xE8 = param_2[4]`
  - `+0xEC = param_2[5]`
  - `+0xF0 = param_2[6]`
  - `+0xF4 = param_2[7]`
  - `+0xF8 = param_2[8]`
  - `+0xFC = (short)param_2[9]`
- payload pointers:
  - sample block start: `chunk + 0x26`
  - secondary block start: `chunk + 0x26 + (E4 * E0 * 8)`

Confidence: **High** for structure/pointers; **Medium** for semantic names of header fields.

---

## 5) Implementation-Ready `IWmoProfile` Seeds

```text
ProfileId: WmoProfile_080_3734
BuildRange: [0.8.0.3734, 0.8.0.3734]

RootChunkPolicy:
  RequiredRootChunks:
    [MVER(version=0x10), MOHD, MOTX, MOMT, MOGN, MOGI, MOPV, MOPT, MOPR, MOLT, MODS, MODN, MODD, MFOG]
  OptionalRootChunkGates:
    MCVP: trailing optional

GroupChunkPolicy:
  RequiredGroupChunks:
    [MOPY, MOVI, MOVT, MONR, MOTV, MOBA]
  RequiredGroupRecordSizes:
    MOVI=0x02, MOVT=0x0C, MONR=0x0C, MOTV=0x08, MOBA=0x20
  OptionalGroupChunkGates:
    MOLR: flags&0x200
    MODR: flags&0x800
    MOBN/MOBR: flags&0x1
    MPBV/MPBP/MPBI/MPBG: flags&0x400
    MOCV: flags&0x4
    MLIQ: flags&0x1000

LiquidPolicy:
  Chunk: MLIQ
  SampleStride: 0x08
  SampleStartOffset: 0x26
  SecondaryStartFormula: 0x26 + (xVerts*yVerts*8)
```

---

## 6) Open Unknowns

1. Exact semantic meaning of `MLIQ` header fields copied from `[2..9]`.
2. Whether any additional optional chunk gates exist in alternate 0.8 paths not reached by this parser.
3. Cross-link of `MLIQ` header values to final material/render type selection in downstream renderer.

Impact severity:
- (1) visual artifact risk
- (2) parser compatibility risk
- (3) visual/perf tuning risk
