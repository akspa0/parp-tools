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

### Refined `MLIQ` field map (runtime-backed)
From `FUN_006cb700`, `FUN_006c0810`, `FUN_00695ea0`, `FUN_006c05b0`:

- `group + 0xE0`: `xVerts`
- `group + 0xE4`: `yVerts`
- `group + 0xE8`: `xTiles`
- `group + 0xEC`: `yTiles`
- `group + 0xF0/+0xF4/+0xF8`: liquid origin/placement basis
- `group + 0xFC`: material selector (used as index into root material table)
- `group + 0x100`: pointer to vertex grid payload (`xVerts * yVerts * 8` bytes)
- `group + 0x104`: pointer to tile payload (starts immediately after vertex grid)

Vertex payload use:
- Runtime consumes each vertex element as 8-byte stride and reads `+4` as height-like Z (`FUN_006c0810`, `FUN_00695ea0`).

Tile payload use:
- Runtime reads one byte per tile from `group + 0x104`.
- `tileType = tileByte & 0x0F`.
- `0x0F` means no liquid tile (skip).
- High bit (`tileByte < 0`) gates an additional mode in index generation (`FUN_006c05b0`) via world flag `DAT_00ebd3cc`.

High-bit gate detail:
- `DAT_00ebd3cc` is not constant; it is set from traversal/render pass state (`FUN_006be4d0`, `FUN_006be710`).
- During portal recursion, pass parity contributes `param_4 & 1` into `DAT_00ebd3cc`.
- In `FUN_006c05b0`, high-bit tiles are emitted only when that pass-state bit allows it.

Practical effect: signed tile bytes can make liquid polygons appear/disappear depending on the active visibility traversal side/pass, which can look like type instability if a tool rewrites tile-byte high bits.

---

## WMO Liquid Type Resolution (Why it “changes”)

### Where type actually comes from
In this build, WMO liquid draw path is:
- `FUN_006be4d0` (group draw) → `FUN_006c0740` (liquid dispatcher)
- `FUN_006c0740` chooses renderer path from `FUN_006ae130(group)`.

`FUN_006ae130` returns the **first non-empty tile nibble** from `group+0x104` (`tileByte & 0x0F`).

So effective liquid type is derived from per-tile `MLIQ` bytes, not just one global header constant.

### Dispatch behavior
`FUN_006c0740` branches by that nibble class:
- `0/4/8` → `FUN_006c0810` (or `FUN_006c0ab0` for indoor/flagged groups)
- `2/3/6/7` → `FUN_006c0d40`
- others: no matching liquid draw branch

This means changing tile nibble values can switch the entire rendering path and perceived liquid type.

### Cross-version mismatch explanation
If a WMO was authored/exported with a different liquid nibble convention (or toolchain remapped `MLIQ` tile bytes), WoW 0.8.0.3734 will reinterpret those bytes with the legacy nibble rules above.

Common symptom:
- Same geometry, different apparent liquid class/behavior between file versions, because the tile nibble domain changed while this client expects classic `MVER=0x10` semantics.

Additional symptom source:
- If export/import pipelines alter tile-byte sign/high-bit usage, visibility-pass gating can differ even when low nibble stays the same.

Secondary factor:
- `group+0xFC` influences material lookup in `FUN_006c0810`, so visual presentation can also differ even when tile nibble class is unchanged.

---

## Converter Normalization Rules (`MLIQ`)

Use this profile when writing files intended to behave like WoW `0.8.0.3734`.

### 1) Structural invariants
- Keep `MVER == 0x10` for this compatibility target.
- Keep `MOGP` liquid-present flag (`0x1000`) and actual `MLIQ` chunk presence in sync.
- Enforce grid relationship before write:
  - `xVerts = xTiles + 1`
  - `yVerts = yTiles + 1`

### 2) Tile byte canonicalization
For each tile byte `b` in payload B:

- Preserve low nibble exactly if known-valid.
- Treat `type = b & 0x0F`; canonical empty is `0x0F`.
- If `type` is outside known renderer classes for this build, remap by policy:
  - Strict mode: fail validation.
  - Compat mode: map unknown to nearest supported class set (`0/4/8` or `2/3/6/7`) and log.

Recommended write canonical form:
- `outType = inType & 0x0F`
- `outHi = inByte & 0x80` only if portal/pass-aware semantics are intentionally preserved.
- `outByte = outType | outHi | (inByte & 0x70 if you intentionally preserve extra bits)`

### 3) High-bit safety mode
Because high bit participates in pass-dependent gating in this client:

- For deterministic offline export/preview pipelines, default `outHi = 0` for all non-empty tiles.
- Provide opt-in `preserveHighBit` mode for archival round-trip fidelity.
- Never randomize high bit; keep it data-driven and deterministic.

### 4) Material/type consistency
- Validate `group+0xFC` (material selector) against root `MOMT` count.
- If you remap tile type classes, verify the target material set supports the intended visual family.

### 5) Writer-side validation checklist
Before final write, assert:
- payload A size is exactly `xVerts * yVerts * 8`
- payload B size is at least `xTiles * yTiles`
- each tile nibble is either `0x0F` or in accepted class domain
- `MLIQ` chunk size fields and data pointers are self-consistent

### 6) Recommended normalization modes
- `lossless-roundtrip`: preserve all tile bits and material index, only fix structural corruption.
- `compat-0.8.0` (recommended default): preserve low nibble, zero high bit, enforce grid/count invariants.
- `strict-legacy`: reject files with unsupported nibble patterns or inconsistent flags/chunk presence.

---

## Confidence
- **High** for root/group chunk order and size-derived entry counts.
- **High** for `MLIQ` tile-type extraction (`tile & 0x0F`), dispatch classes, and dual-payload interpretation.
- **Medium** for precise meaning of all non-height bytes/words in payload A and high-bit tile behavior details.