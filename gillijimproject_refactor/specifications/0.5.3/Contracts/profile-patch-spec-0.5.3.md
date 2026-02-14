# 0.5.3 Profile Patch Spec (Code-Ready)

## Purpose
Translate proven 0.5.3 binary contracts into exact code edits for profile selection and parser routing.

## Evidence baseline
- WDT-primary terrain pipeline proven (`CMap::Load -> LoadWdt -> PrepareUpdate -> PrepareArea/PrepareChunk -> CMapArea::Load/CMapChunk::Load`).
- ADT-like chunk contracts still active inside WDT-backed blobs (`MHDR/MCIN/MCNK` asserts).
- MDX canonical dispatch order proven via `BuildModelFromMdxData` and `BuildSimpleModelFromMdxData`.

---

## 1) `FormatProfileRegistry` edits
File: `src/MdxViewer/Terrain/FormatProfileRegistry.cs`

### 1.1 Add explicit terrain source mode
Add a new profile field to `AdtProfile`:
- `public required bool WdtPrimaryTerrainSource { get; init; }`

Set values:
- Existing profiles: `WdtPrimaryTerrainSource = false`
- New 0.5.3 profile: `WdtPrimaryTerrainSource = true`

### 1.2 Add dedicated 0.5.3 ADT contract profile
Add:
- `AdtProfile053WdtPrimary`

Exact values:
- `ProfileId = "AdtProfile_053_WdtPrimary"`
- `McinEntrySize = 0x10`
- `MclqLayerStride = 0x324`
- `MclqTileFlagsOffset = 0x290`
- `MddfRecordSize = 0x24`
- `ModfRecordSize = 0x40`
- `UseMhdrOffsetsOnly = true`
- `EnableMh2oFallbackWhenNoMclq = false`
- `WdtPrimaryTerrainSource = true`

### 1.3 Add dedicated 0.5.3 WMO profile
Add:
- `WmoProfile053`

Exact values:
- `ProfileId = "WmoProfile_053"`
- `StrictGroupChunkOrder = true`
- `EnableMliqGroupLiquids = true`
- `EnablePortalOptionalBlocks = true`

### 1.4 Add dedicated 0.5.3 MDX profile
Add:
- `MdxProfile053`

Add new `MdxProfile` fields:
- `public required int ModlSectionSize { get; init; }`
- `public required bool SequenceSectionCursorStrict { get; init; }`
- `public required bool GlobalSequenceCursorStrict { get; init; }`

Set for `MdxProfile053`:
- `ProfileId = "MdxProfile_053"`
- `RequiresMdlxMagic = true`
- `TextureRecordSize = 0x10C`
- `TextureSectionSizeStrict = true`
- `GeosetHardFailIfMissing = false`
- `ModlSectionSize = 0x175`
- `SequenceSectionCursorStrict = true`
- `GlobalSequenceCursorStrict = true`

For non-0.5.3 profiles:
- Keep current values; use conservative defaults:
  - `ModlSectionSize = 0` (disabled strict check)
  - `SequenceSectionCursorStrict = false`
  - `GlobalSequenceCursorStrict = false`

### 1.5 Resolver mapping
In all three resolver methods, add explicit match before broad major/minor branches:
- `if (string.Equals(buildVersion, "0.5.3", StringComparison.OrdinalIgnoreCase)) return <Profile053>;`

---

## 2) `StandardTerrainAdapter` routing edits
File: `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`

### 2.1 Gate ADT-first assumptions
In load flow (`LoadTileWithPlacements` and/or `ParseAdt` call site):
- If `_adtProfile.WdtPrimaryTerrainSource == true`, route through WDT-backed terrain source first.
- Do **not** treat missing standalone ADT bytes as fatal for 0.5.3.

### 2.2 Preserve ADT chunk contract parser
Keep existing `MHDR/MCIN/MCNK` parsing logic intact (it still matches runtime contracts), but source bytes from WDT-extracted buffers for 0.5.3 path.

### 2.3 MH2O behavior
For 0.5.3 profile:
- Keep `EnableMh2oFallbackWhenNoMclq = false`
- Never auto-promote MH2O fallback when MCLQ parse count is zero in this profile.

### 2.4 Placement source precedence
For 0.5.3:
- Prefer WDT-driven placement source when available.
- Only use ADT-root placement extraction as guarded fallback.

---

## 3) Diagnostics wiring
Primary file:
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`

If counters live elsewhere, add pass-through there as needed.

### Required counters
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

### Required log context per emit
- `buildId=0.5.3`
- `profileId=<resolved profile id>`
- `filePath=<active source file>`
- `chunkFamily=<ADT|WMO|MDX>`

### Trigger rule for fallback counter
Increment `UnsupportedProfileFallbackCount` when:
- 0.5.3 path falls back from WDT-primary route to alternate route due to unsupported/absent data.

---

## 4) MDX contract encoding notes
Dispatcher order for 0.5.3 should be documented in code comments or adjacent profile docs as canonical behavior:
- Complex path root: `BuildModelFromMdxData`
- Simple path root: `BuildSimpleModelFromMdxData`

Minimum strict checks to enforce at profile layer:
- `MODL` section size exact `0x175`
- Sequence section cursor equality checks
- Global sequence section cursor equality checks

---

## 5) Minimal patch sequence (recommended)
1. Add new profile fields/types (`WdtPrimaryTerrainSource`, MDX strictness fields).
2. Add `*_053` profile instances.
3. Add resolver mapping for `0.5.3`.
4. Add 0.5.3 route gate in terrain adapter.
5. Wire diagnostics counters/context.
6. Validate with one known large-WDT 0.5.3 map load.

---

## 6) Acceptance criteria
- Build `0.5.3` resolves to `AdtProfile_053_WdtPrimary`, `WmoProfile_053`, `MdxProfile_053`.
- Terrain loads without standalone ADT file dependency.
- WMO group liquids (`MLIQ`) remain enabled and stable.
- MDX parser rejects invalid `MODL` section sizes and logs strict section mismatch diagnostics.
- Required diagnostics counters increment with required context.
