# M2/MDX Transitional Spec — 0.12.0.3988 (Ghidra-derived)

## Scope
This document captures the transitional model-container contract observed in `WoW.exe` build `0.12.0.3988`, focusing on parser-relevant binary invariants.

Baseline comparison target:
- `MdxProfile_091_3810_Provisional` (legacy `MDLX`-centric assumptions)

Result:
- Primary runtime model parse path in this build is `MD20` typed-offset/count-table parsing, not `MDLX` chunk-seek parsing.
- Version `0x100` is **pre-modern** relative to `v264+` (`0x108+`); this build sits in early/transitional M2 era.

---

## Evidence anchors (core)
- Container gate + top-level section validator chain: `FUN_0071e1f0`
- Loader path into validator: `FUN_00716890 -> FUN_007096f0 -> FUN_0071e8c0 -> FUN_0071e9a0 -> FUN_0071e1f0`
- Model async load entry: `FUN_0071e810`
- Post-validate initialization path: `FUN_0071eb10`

---

## 1) Container identity and version contract
From `FUN_0071e1f0`:
- Required root magic: `0x3032444D` (`MD20`)
- Required version: `0x100`
- If either fails, function returns parse-fail (`0`)

Version-era interpretation:
- `0x100` = decimal `256`
- `0x108` = decimal `264`
- Therefore, this build is not in the `v264+` modern branch; it is in an earlier M2 contract branch.

Contract implication:
- In this build family, extension-based routing to MDX is unsafe.
- Container magic must drive parser-family selection first.

---

## 2) Top-level MD20 span policy
`FUN_0071e1f0` enforces strict in-buffer relocation semantics:
- For each `{count, offset}` style pair:
  - `offset <= fileSize`
  - `offset + count*stride <= fileSize` (or `+ count` for byte arrays)
  - If `count == 0`, relocated pointer is set to `0`
  - If `count != 0`, relocated pointer becomes `base + offset`

Primary fixed checks at top level include (direct in `FUN_0071e1f0`):
- `*4`
- `*0x44`
- `*2`
- `*4`
and then a long sequence of nested typed validators.

---

## 3) Typed validator primitives (reused contracts)
These helpers are repeatedly used by higher-level validators:
- `FUN_00720430`: `stride=0x08`
- `FUN_0071f650`: `stride=0x04`
- `FUN_0071f320`: `stride=0x02`
- `FUN_0071f420`: `stride=0x0C`
- `FUN_00720d90`: `stride=0x10`
- `FUN_00720e10`: `stride=0x02`
- `FUN_00720e90`: `stride=0x01`
- `FUN_00720f10`: `stride=0x04`
- `FUN_00720f90`: `stride=0x24`
- `FUN_0071f3a0`: `stride=0x18`

All share strict bounds + pointer relocation behavior described above.

---

## 4) Major nested section records (transitional map)
The `FUN_0071e1f0` section chain validates the following record families:

- `FUN_0071f4a0`: record `0x6C`
- `FUN_0071f6d0`: record `0x30`
- `FUN_0071f750`: record `0x2C` (contains nested `FUN_0071e110`)
- `FUN_0071f800`: record `0x38`
- `FUN_0071f990`: record `0x10`
- `FUN_0071fa60`: record `0x1C`
- `FUN_0071fb80`: record `0x1C`
- `FUN_0071fca0`: record `0x54`
- `FUN_0071fef0`: record `0x30`
- `FUN_00720020`: record `0x2C`
- `FUN_00720110`: record `0xD4`
- `FUN_007204b0`: record `0x7C`
- `FUN_00720660`: record `0xDC`
- `FUN_00720940`: record `0x1F8`

High-confidence detail:
- `FUN_007204b0` explicitly validates `count * 0x7C` and includes nested sub-span checks (`*8`, `*4`, `*0x0C`, plus nested helper validators), making `0x7C` a hard structural contract in this build.

---

## 5) Nested `FUN_0071e110` sub-structure
`FUN_0071f750` calls `FUN_0071e110` per `0x2C` entry.
`FUN_0071e110` validates sequential subspans with fixed multipliers:
- `*2`
- `*2`
- `*4`
- `*0x20`
- then `FUN_0071f3a0` (`*0x18`)

This indicates `0x2C` entries are composite nodes with at least five typed child spans.

---

## 6) Loader behavior summary (implementation impact)
Observed runtime behavior:
- File is loaded and asynchronously parsed (`FUN_0071e810` / `FUN_0071e9a0`)
- Structural validation gates initialization (`FUN_0071e1f0` must pass)
- Initialization then derives active region/state and allocates runtime arrays (`FUN_0071eb10`)

Extension gate/normalization behavior (this exact build):
- `FUN_00721860` enforces model extension policy before load.
- It accepts only a small extension set and rewrites legacy extension(s) to the internal M2 extension token before opening/rehashing path.
- Non-accepted extensions fail with `"Model2: Invalid file extension"` (`0x0083fb48`).
- Successful path continues into `FUN_0071e810` -> `FUN_0071e9a0` -> `FUN_0071e1f0`.

Implication for tooling:
- Any converter/parser treating this build as legacy `MDLX` chunk stream risks hard parse fail or silent misdecode.

---

## 7) MDX compatibility guidance for `0.12.0.3988`
Required routing policy:
1. Probe root magic.
2. If `MD20` + version `0x100`, use transitional M2 typed-table parser.
3. Do not force legacy `MDLX` chunk profile for this build.

Recommended profile additions:
- New exact profile id: `M2Profile_012_3988` (or equivalent)
- Enforce strict typed-span validation
- Encode proven section strides listed above

---

## 8) ADT/WMO side findings relevant to model transition work
- WMO root chain is strict and fully recovered in this build:
  - `FUN_006c4d00`: `MOHD -> MOTX -> MOMT -> MOGN -> MOGI -> MOSB -> MOPV -> MOPT -> MOPR -> MOVV -> MOVB -> MOLT -> MODS -> MODN -> MODD -> MFOG`
  - Optional `MCVP` via `FUN_006c4cc0`
- ADT liquid path remains `MCLQ`-centric in recovered chain; no direct `MH2O` token evidence found in primary ADT parser anchors.

---

## 9) Transitional semantic map (best-effort labels)
This map assigns human-meaning labels to validator groups using:
- structural stride signatures,
- call-site behavior (`FUN_0071eb10` initialization),
- known Model2 RTTI/type symbols embedded in this binary (`M2ModelRegion`, `M2ModelBone`, `M2ModelSequence`, `M2ModelParticle`, etc.),
- and cross-build consistency from `3.0.1.8303` contracts.

### High-confidence semantic anchors
- `FUN_0071f750` (`stride=0x2C`): **skin/profile-like selector set**
  - Evidence: `FUN_0071eb10` chooses active entry from this family based on runtime budget, then allocates/copies region-style arrays from selected entry.
- `FUN_00720110` (`stride=0xD4`): **effect/complex component A (legacy-style)**
  - Evidence: heavy nested span graph; aligns with known profile split where pre-`0x108` branch uses larger effect-like layout.
- `FUN_007204b0` (`stride=0x7C`): **effect/complex component B (alternate/compact)**
  - Evidence: explicit `*0x7C` validation and nested subspans; matches established `EffectLikeBStride` pattern.
- `FUN_0071e1f0` top-level `*0x44` table: **sequence/track-like control table family**
  - Evidence: repeated use as an early gating table and downstream animation-state setup patterns.

### Medium-confidence semantic anchors
- `FUN_0071f4a0` (`stride=0x6C`): **core model node family (bone/region/attach-adjacent)**
- `FUN_0071f800` (`stride=0x38`): **render-batch/material-adjacent set**
- `FUN_0071fca0` (`stride=0x54`): **event/call/attachment-adjacent set**
- `FUN_00720660` (`stride=0xDC`) and `FUN_00720940` (`stride=0x1F8`): **high-complexity runtime/scene wiring families**

### Still provisional
- Exact 1:1 field names for every member inside each record family remain provisional until each family is correlated with immediate runtime consumer writes/reads in `CM2Model` render/update paths.

---

## 10) Confidence and remaining unknowns
Confidence:
- Container/version gate: **High**
- Section stride map (structural): **High**
- Extension normalization + MD20-only route evidence: **High**
- Semantic naming of each section (animation vs texture vs geoset-equivalent labels): **Medium-High**

Remaining unknowns:
- Full per-field names/types inside each large record family (`0xD4`, `0xDC`, `0x1F8`) remain incomplete.
- No concrete `MDLX` parser entry has been recovered for this build so far; current evidence strongly indicates Model2/MD20-only route for runtime model loading.

---

## 11) Practical implementation checklist
- Add exact build route for `0.12.0.3988` in model profile resolver.
- Prefer `MD20` typed parser for this build; keep extension mismatch diagnostics.
- Preserve strict bounds checks and null-pointer relocation semantics.
- Surface diagnostics counters when profile fallback/routing mismatch occurs.
- Encode this build as pre-modern (`Version=0x100`) and avoid `v264+` assumptions (do not apply `>=0x108` layout branches).
