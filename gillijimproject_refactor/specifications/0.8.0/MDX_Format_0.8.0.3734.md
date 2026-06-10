# MDX Format Analysis — WoW 0.8.0.3734

## Summary
The model loader expects classic MDX signature `'MDLX'` (stored/checked as `0x584C444D` in little-endian). Parsing is chunk-structured, with strong assertions around geoset subchunks and fixed size relations.

## Build
- **Build**: `0.8.0.3734`
- **Source confidence**: High for validated chunks below

---

## Entry / Header Expectations

### Loader entry points
- `FUN_00422620` (async model completion path)
- `FUN_006bbd10` (world/model data path)

### Header assertion
- `*((ULONG*)fileData) == 'XLDM'` (little-endian check for `MDLX`)
- String evidence at `0x0085861c`

---

## Observed MDX Chunk Expectations

## In `FUN_006bbd10`
- Looks up `TEXS` (`0x53584554`) and expects section size divisible by `0x10c`.
- Looks up `GEOS` (`0x534f4547`), with nested geoset chunk checks:
  - `VRTX` (`0x58545256`)
  - `NRMS` (`0x534d524e`)
  - `UVAS` (`0x53415655`) — expected count `1` in this path
  - `PTYP` (`0x50595450`)
  - `PCNT` (`0x544e4350`)
  - `PVTX` (`0x58545650`)

## In `FUN_0044e380` / `FUN_0044ea20` (detailed geoset parsing)
- Mandatory geoset sequence starts with:
  - `VRTX` → vertex array
  - `NRMS` → normals, must match vertex count
- Optional/next sequence includes:
  - `UVAS` (multiple UV set support logic present)
  - Then geoset metadata/index sections parsed via:
    - `GNDX` (`0x58444e47`)
    - `MTGC` (`0x4347544d`)
    - `MATS` (`0x5354414d`)
    - `BIDX` (`0x58444942`)
    - `BWGT` (`0x54475742`)

### Count / size relations observed
- Vertex and normal counts are actively cross-checked.
- Several arrays are resized dynamically but guarded by strict assertions.
- UV-related path indicates per-vertex UV payload handling and supports more than one texture coordinate channel in some paths.

---

## Practical Implications for 0.8.0
- MDX is fully chunk-driven with strict sanity checks.
- Loader is sensitive to ordering and count consistency.
- Geoset chunk validation is robust enough to reject malformed/partial chunks quickly.

## Unknowns / Next Steps
- Full map of all top-level MDX chunks (`SEQS`, `BONE`, etc.) in this build is not yet fully enumerated from these specific functions.
- Additional decompilation around `FUN_00421700` and directly called model section readers would complete the full schema.

## Confidence
- **High** for `MDLX` signature and geoset-related chunk expectations above.
- **Medium** for comprehensive full-file chunk catalog (not fully enumerated yet).