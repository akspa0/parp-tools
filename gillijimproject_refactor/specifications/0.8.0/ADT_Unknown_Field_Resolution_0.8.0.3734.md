# ADT Unknown Field Resolution — 0.8.0.3734 (Ghidra)

## Scope
Resolve unknown ADT fields in build `0.8.0.3734` by tracing actual runtime consumers.

## Build Fingerprint
- Assertion string: `World of WarCraft: Assertions Enabled Build (build 3734)`
- Source-path strings present (`MapArea.cpp`, `MapChunk.cpp`), but function symbols are stripped (`FUN_xxxxxxxx`).

---

## A) MHDR field mapping (root ADT)
From `FUN_006c7220` (MapArea create path):

- `MHDR + 0x04` -> `ofsMCIN`
- `MHDR + 0x08` -> `ofsMTEX`
- `MHDR + 0x0C` -> `ofsMMDX`
- `MHDR + 0x10` -> `ofsMMID`
- `MHDR + 0x14` -> `ofsMWMO`
- `MHDR + 0x18` -> `ofsMWID`
- `MHDR + 0x1C` -> `ofsMDDF`
- `MHDR + 0x20` -> `ofsMODF`

### Unknown still
- `MHDR + 0x00` is not consumed in this path (likely flags/reserved for this era).

---

## B) MCNK unknown/header fields
From `FUN_006b8990`, `FUN_006b8f90`, `FUN_006b99d0`, `FUN_006b8c80`:

- `MCNK + 0x10` = doodad reference count (**confirmed**)
- `MCNK + 0x38` = mapobject reference count (**confirmed**)
- `MCNK + 0x5C` = sound emitter count (`MCSE` entry loop bound) (**confirmed**)
- `MCNK + 0x34` copied into chunk persistent field (`+0x160`) -> likely `areaId` (**inferred**)
- `MCNK + 0x3C` low 16-bit copied to chunk field (`+0x80`) -> likely hole/aux bitmask (**inferred, medium confidence**)

### Count-field confirmation
This build clearly uses multiple MCNK fields as true counters, reinforcing the pattern that historically “unknown” fields in older ADT headers are often counts.

---

## C) MCLQ unknowns resolved (0.8.0 layout)
From `FUN_006b8be0` + liquid query consumers (`FUN_006a8aa0`, `FUN_006a8c80`, `FUN_006a91b0`):

### Per-layer record shape in 0.8.0
- Parser walks 4 layers gated by MCNK flags `0x04/0x08/0x10/0x20`.
- Layer stride is `0x2D4` bytes.
- In-record map:
  - `+0x000` -> scalar A
  - `+0x004` -> scalar B
  - `+0x008` -> 9x9 sample block (`81 * 8` bytes)
  - `+0x290` -> 8x8 tile flags block (64 bytes)
  - `+0x2D0` -> mode/count scalar
  - `+0x2D4` -> next layer start

### Runtime semantics now confirmed
- Tile map source: layer pointer at object `+0x10` (parser assigns from `record + 0x290`).
- Tile addressing: `(y & 7) * 8 + (x & 7)`.
- Empty tile marker: `(tileByte & 0x0F) == 0x0F`.
- Behavior class: `tileByte & 0x03`.
- Additional flag exposure:
  - `(tileByte & 0x40)` returned by one query path.
  - `(tileByte >> 7)` returned by richer query path.
- Height interpolation uses sample `+4` in each 8-byte sample cell.

### Unknown still
- Exact semantic of sample lane `+0` in each 8-byte sample remains unresolved.
- Exact semantic names of scalars at `record+0x000`, `+0x004` remain unresolved in this build.

---

## D) Confidence
- **High**: MHDR offset table, MCNK count fields (`0x10/0x38/0x5C`), MCLQ tile/height runtime semantics.
- **Medium-High**: `MCNK+0x34` as areaId.
- **Medium**: `MCNK+0x3C` hole/aux mask inference; unresolved sample lane `+0` meaning.
