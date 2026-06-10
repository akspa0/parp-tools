# Stability Matrix — 0.6.0/0.7.0 Baseline vs 0.8.0/0.9.x Drift

## Scope
Operational stability assumptions for map loading/parsing across early builds.

This matrix is used to drive per-build parser profiles in MdxViewer.

---

## Baseline Policy
- `0.6.0` and `0.7.0` are treated as one stable family.
- Everything after `0.7.0` is treated as unknown-by-default until binary evidence confirms structure.
- No expansion-level assumptions; decisions are build-level.

---

## A) Stability Matrix

| Build family | ADT root/MCNK | MCLQ layout | WMOv17 behavior | MDX behavior | Status |
|---|---|---|---|---|---|
| 0.6.x | baseline | baseline | unknown/legacy mix | baseline-ish | stable (working assumption) |
| 0.7.x | baseline-compatible | baseline-compatible | unknown/legacy mix | baseline-ish | stable (working assumption) |
| 0.8.0.3734 | partially unstable on some maps | confirmed stride `0x2D4`, unresolved lanes remain | unknown | unknown | unstable/partial |
| 0.9.0.x | unstable/unknown | unknown (must verify, do not assume 0.9.1 parity) | unknown | **extension/container divergence starts: `.mdx` not sufficient discriminator** | unknown/high risk |
| 0.9.1.3810 | unstable/unknown on some maps | confirmed stride `0x324`, flow block expanded | likely still moving target | **mixed parser era: profile must branch by binary signature/version, not extension** | unstable/partial |

---

## B) Known Confirmed ADT Facts

## 0.8.0.3734 (confirmed from existing docs)
- MHDR practical offset chain includes: `MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`.
- MCNK counters confirmed:
  - `+0x10` doodad refs
  - `+0x38` mapobject refs
  - `+0x5C` sound emitter count
- MCLQ layer stride: `0x2D4`.
- MCLQ tile semantics confirmed (8x8 tiles, low nibble `0xF` empty marker).
- MCLQ sample height lane at sample `+4` confirmed.

## 0.9.1.3810 (confirmed from existing docs)
- ADT root parse is strict token + MHDR-offset driven in native chain.
- Placement chain semantics confirmed:
  - `nameId -> MMID/MWID[nameId] -> MMDX/MWMO + byteOffset`.
- MDDF record size `0x24`; MODF record size `0x40`.
- MCLQ layer stride: `0x324` (delta `+0x50` from 0.8.0).

---

## C) High-Risk Unknowns (must prove, not assume)

1. Exact `0.9.0.x` MCLQ stride and extra region layout.
2. Whether 0.9.0/0.9.1 subchunk strictness differs (required vs optional sets).
3. WMOv17 group/root chunk schema drift across 0.8->0.9.x.
4. MDX skeletal/animation/material chunk drift across 0.8->0.9.x.
5. Any byteswap/endianness mode gates that changed between these builds.
6. Exact adoption point where `.mdx` extension begins carrying `MD20`-family content in this branch.

---

## D) Required Evidence Per Build (minimum)

For each build candidate (`0.8.0`, `0.9.0`, `0.9.1`):
1. ADT root create function with MHDR offset usage.
2. MCIN entry struct and chunk pointer derivation.
3. MCNK subchunk contract and consumed header fields.
4. MCLQ per-layer field map (full offsets/types).
5. Placement chain (`MMID/MWID`, `MDDF/MODF`) with record sizes.
6. WMO root/group parse contract (required chunks and optional gates).
7. MDX parse contract (core geometry/material/anim chunks consumed).
8. Extension/container discriminator proof (`.mdx`/`.mdl` path vs root magic gate and version gate).

---

## E) MdxViewer Impact

Implementation policy:
- One parser profile for `0.6/0.7` baseline family.
- Per-build profiles for `0.8+` until proven to share exact schemas.
- Unknown fields/chunks in unstable builds must trigger:
  1) explicit warning counters,
  2) safe skip/degrade paths,
  3) no silent reinterpretation.

Primary targets:
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MdxViewer/Formats/Mdx/*` and `src/MdxViewer/Rendering/MdxRenderer.cs`

---

## F) Acceptance Criteria

This matrix is considered complete when:
- each build family has confidence-labeled status,
- unknown fields are tracked with explicit owners/tasks,
- parser profile boundaries are unambiguous,
- map-load failures can be mapped to a build-family row and chunk-family column.
