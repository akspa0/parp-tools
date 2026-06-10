# Parser Profile — 0.8.0.3734 (Binary-Derived)

## Purpose
Build a function-level parser contract for `0.8.0.3734` from live Ghidra evidence, with enough detail to implement strict profile-gated parsing in viewer code paths.

---

## Profile IDs
- `AdtProfile_080_3734`
- `WmoProfile_080_3734`
- `MdxProfile_080_3734_Provisional`

Build range:
- Exact build: `0.8.0.3734`

Fallback policy:
- Unknown `0.8.0.x` should remain on `*_080x_Unknown` until validated.

---

## A) ADT Contract (high confidence)

## A1. Root parse chain and strictness
- `FUN_006c7220` enforces root contract:
  - `MVER` required.
  - `MHDR` required immediately after `MVER`.
  - Root chunks resolved from `MHDR` offsets + `+0x10` payload addressing.
  - Token assertions for `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`.

## A2. Root chunk record sizing
- `MDDF` count = `chunkSize / 0x24` (`FUN_006c7220`).
- `MODF` count = `chunkSize >> 6` (`FUN_006c7220`, i.e. `/0x40`).

## A3. MCNK required subchunk contract
- `FUN_006b8f90` hard-asserts required subchunks from MCNK header offsets:
  - `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`.
- Recovered MCNK header offsets used in this chain:
  - `+0x14` `MCVT`
  - `+0x18` `MCNR`
  - `+0x1C` `MCLY`
  - `+0x20` `MCRF`
  - `+0x24` `MCAL`
  - `+0x2C` `MCSH`
  - `+0x58` `MCSE`
  - `+0x60` `MCLQ`

## A4. MCLQ layout and liquid slots
- `FUN_006b8be0` configures up to 4 liquid slots from MCNK flags `0x04/0x08/0x10/0x20`.
- Per-slot stride in dwords is `0xB5` (`181` dwords = `0x2D4` bytes).
- Slot pointers wired as:
  - sample block pointer from `+2 dwords` (`+0x08`)
  - tile/aux block pointer from `+0xA4 dwords` (`+0x290`)
  - tail scalar from `+0xB4 dwords` (`+0x2D0`)

## A5. Unknowns still open
- MCIN entry-size/count proof function is not extracted in this pass.
- Full runtime semantics for all MCLQ sample lanes beyond the verified layout remain partial.
- No confirmed MH2O path in the extracted ADT parser chain for this build.

---

## B) WMO Contract (high confidence root/group)

## B1. WMO root parser and strict token order
- Root load path includes `FUN_006ca8b0` (file load) and `FUN_006cac40` (root parse sequence).
- `FUN_006cac40` enforces:
  - `MVER` required with version `0x10`.
  - Required root sequence:
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

## B2. Root record divisors
- `MOMT`: `/0x40`
- `MOGI`: `/0x20`
- `MOPV`: `/0x0C`
- `MOPT`: `/0x14`
- `MOPR`: `/0x08`
- `MOLT`: `/0x30`
- `MODS`: `/0x20`
- `MODD`: `/0x28`
- `MFOG`: `/0x30`
- optional `MCVP`: `/0x10`

## B3. Group parser and optional gates
- Group parse chain: `FUN_006cb290` -> `FUN_006cb4b0` -> `FUN_006cb700`.
- `FUN_006cb290` enforces `MVER(0x10)` then `MOGP`.
- `FUN_006cb4b0` required group sequence:
  - `MOPY`, `MOVI`, `MOVT`, `MONR`, `MOTV`, `MOBA`.
- Group divisors:
  - `MOPY`: `>>2`
  - `MOVI`: `>>1`
  - `MOVT`: `/0x0C`
  - `MONR`: `/0x0C`
  - `MOTV`: `>>3`
  - `MOBA`: `>>5` (`/0x20`)

## B4. Group optional gates (`FUN_006cb700`)
- `0x00000200` -> `MOLR`
- `0x00000800` -> `MODR`
- `0x00000001` -> `MOBN` then `MOBR`
- `0x00000400` -> `MPBV` -> `MPBP` -> `MPBI` -> `MPBG`
- `0x00000004` -> `MOCV`
- `0x00001000` -> `MLIQ`

## B5. Group MLIQ internal layout
- On `MLIQ` gate, fields at group offsets are assigned from `param_2[2..9]`.
- Sample pointer starts at `chunk + 0x26`.
- Secondary mask/flag region starts at:
  - `chunk + 0x26 + (xVerts * yVerts * 8)`.
- Confirms 8-byte sample semantics and split sample/mask regions.

---

## C) MDX Contract (improved coverage, still provisional)

## C1. Loader entry and magic
- `FUN_00422620` validates top-level magic `MDLX` (`0x584C444D`) in async completion path.
- `FUN_006bbd10` also hard-checks `MDLX` in world/model path.

## C2. Section constraints in `FUN_006bbd10`
- `TEXS` (`0x53584554`):
  - count computed as `sectionBytes / 0x10C`
  - strict relation `sectionBytes == count * 0x10C`
  - hard expectation in this path: `count == 1`
- `GEOS` (`0x534F4547`) parse chain enforces:
  - `VRTX`
  - `NRMS` with `numNormals == numVertices`
  - `UVAS` with hard expectation `count == 1`
  - `PTYP` -> `PCNT` -> `PVTX`

## C3. Geoset subchain deep parse
- `FUN_0044e380` handles geoset core arrays (`VRTX`, `NRMS`, `UVAS`) and consistency checks.
- `FUN_0044ea20` enforces tail sequence:
  - `GNDX` -> `MTGC` -> `MATS` -> `BIDX` -> `BWGT`

## C4. Unknowns still open
- Full top-level required/optional MDX chunk order under dispatcher `FUN_00421700` is not fully recovered in this pass.
- Animation sequence/keyframe compression details remain unresolved.
- Other loader variants may differ from world/model strictness (`TEXS` count policy) and need targeted proof.

---

## D) Cross-domain stability assessment (0.8.0.3734 vs 0.9.1.3810 baseline)

- ADT: confirmed MCLQ stride/slot contract differs (`0x2D4` vs `0x324` baseline assumption).
- WMO: root `MVER` and root/group structural expectations differ materially from 0.9.1 baseline.
- MDX: strict legacy constraints (`TEXS` count `1`, `UVAS` count `1`) require explicit build-profile gating.

---

## E) Immediate implementation guidance

1. Add exact-build registry entries for `0.8.0.3734` using extracted contracts.
2. Keep ADT MCLQ parsing profile-specific (`0x2D4` stride), no silent reuse of `0.9.x` assumptions.
3. Gate WMO root/group parsing by `MVER=0x10` and recovered chunk/divisor contract.
4. Keep MDX profile provisional, but enforce recovered strict size/count checks in profile mode.
5. Surface unresolved items through diagnostics counters rather than speculative coercion.
