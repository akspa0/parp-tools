# Implementation Patch Checklist — 0.5.3

## 1) Terrain source routing (WDT-primary)
- [ ] Add explicit build/profile gate for `0.5.3` terrain source mode: `WdtPrimary`.
- [ ] In terrain load entrypoint, bypass unconditional ADT-first path when `WdtPrimary`.
- [ ] Add guarded fallback policy: only attempt ADT parse when WDT path is unavailable and diagnostics mark fallback reason.
- [ ] Emit `UnsupportedProfileFallbackCount` when ADT fallback is used under `0.5.3`.

## 2) Profile registry updates
- [ ] Add `AdtProfile_053_WdtPrimary` (or equivalent) with explicit non-primary ADT semantics.
- [ ] Add `WmoProfile_053` with:
  - [ ] `StrictGroupChunkOrder = true`
  - [ ] `EnableMliqGroupLiquids = true`
  - [ ] `EnablePortalOptionalBlocks = true`
- [ ] Add `MdxProfile_053` with:
  - [ ] strict `MODL` section size check (`0x175`)
  - [ ] strict sequence/global section boundary checks (`SEQS`/`GLBS` cursor equality invariants)

## 3) Placement behavior
- [ ] Add/verify WDT-level placement ingestion path for optional `MODF` encountered during `LoadWdt`-style map bootstrap.
- [ ] Ensure placement source precedence is profile-aware (`WDT` source first for 0.5.3).
- [ ] Keep ADT placement reader isolated behind non-0.5.3 gates unless explicitly enabled.

## 4) Diagnostics contract
- [ ] Wire counters in 0.5.3 path:
  - [ ] `InvalidChunkSignatureCount`
  - [ ] `InvalidChunkSizeCount`
  - [ ] `MissingRequiredChunkCount`
  - [ ] `UnknownFieldUsageCount`
  - [ ] `UnsupportedProfileFallbackCount`
- [ ] Ensure every log event includes: build id, profile id, file path, chunk family (`ADT|WMO|MDX`).

## 5) Validation pass
- [ ] Run loader against known 0.5.3 maps with very large WDT files.
- [ ] Confirm terrain appears without ADT hard dependency.
- [ ] Confirm WMO group liquids (`MLIQ`) render when group flags permit.
- [ ] Confirm MDX models with valid `MODL` size load and invalid size fails are diagnosed.
