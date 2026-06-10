# Build Delta Report — 0.5.3

## Baseline compared against
- AdtProfile: AdtProfile_060_070_Baseline
- WmoProfile: WmoProfile_060_070_Baseline
- MdxProfile: MdxProfile_060_070_Baseline

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.5.3 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Terrain container source | ADT-centric tile parse path (`MHDR -> MCIN -> MCNK`) expected via standalone ADT files | Build runtime is WDT-primary: `LoadWdt` reads `MVER -> MPHD -> MAIN` and streams area/chunk payloads from WDT offsets into ADT-like parsers | `0x0067fe19 CMap::LoadWdt` reads `MAIN(0x10000)`; `0x00684a30 PrepareArea -> 0x006aaab0 CMapArea::Load -> 0x006aad30 Create`; `0x00684be0 PrepareChunk -> 0x00698940 CMapChunk::Load -> 0x00698e10 Create` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, `LoadTileWithPlacements` flow) | hard parse fail | High |
| ADT file usage policy | Baseline often assumes external `.adt` unit files as terrain source | No standalone `.adt` file dependency observed in active load path; ADT chunk contracts still apply but are embedded in WDT-backed area/chunk blobs | `0x0067f986 CMap::Load`: opens `%s\\%s.wdt` then `LoadWdt`; `0x006aad30 CMapArea::Create` asserts `MHDR/MCIN`; `0x00698e10 CMapChunk::Create` asserts `MCNK` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`ResolveAdtProfile` for 0.5.3) and adapter routing | wrong geometry | High |
| Placement source | Baseline ADT placement chain via ADT root `MDDF/MODF` | WDT path includes optional `MODF` at world-level parse in `LoadWdt` | `0x0067fe19`: after `MAIN`, reads chunk header and if token == `MODF` reads `0x40` map obj def and creates map object def link | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`CollectPlacementsViaMhdr`) | wrong placement | Medium |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.5.3 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Group token enforcement | Baseline WMO profile is non-strict | Group create path hard-asserts `MOGP` token | `0x006af05a CMapObjGroup::Create`: `if (*(int*)param_1 != 'MOGP') error` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`WmoProfile_060_070_Baseline.StrictGroupChunkOrder`) | hard parse fail | High |
| Optional group liquid gating | Baseline WMO 060/070 profile disables group liquids | `MLIQ` is parsed when `flags & 0x1000` | `0x006af77b CMapObjGroup::CreateOptionalDataPointers`: `if (flags & 0x1000) assert 'MLIQ'` then decode dimensions/pointers | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`EnableMliqGroupLiquids`) | visual artifact | High |
| Portal/optional block chain | Baseline profile marks portal optional blocks disabled | Optional blocks chain (`MPBV -> MPBP -> MPBI -> MPBG`) is flag-gated and enforced when present | `0x006af77b`: under `flags & 0x400`, sequential token assertions for MPB* blocks | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`EnablePortalOptionalBlocks`) | wrong geometry | Medium |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.5.3 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MODL section size policy | Baseline profile does not encode strict MODL section byte-size gate | Parser rejects MODL section unless size is exactly `0x175` | `0x007b3091 MDL::ReadBinModelGlobals`: `if (param_2 != 0x175) ... "Invalid MODL section"` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`MdxProfile` fields; missing explicit MODL size contract) | hard parse fail | High |
| Sequence/global-seq section invariants | Baseline profile lacks explicit strictness flags for `SEQS/GLBS` section-length consistency | Parser validates exact data cursor agreement for sequence/global-seq payloads | `0x00756b6f AnimAddSequences`: seeks `SEQS` (`0x53514553`) and `GLBS` (`0x53424C47`), asserts end-pointer equalities | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` and MDX reader policy surface | visual artifact | Medium |
| Model extension fallback | Baseline docs/profiles do not explicitly encode `.mdx/.mdl` fallback behavior | Alternate filename helper toggles extension between `.mdx` and `.mdl` | `0x0078bb98 PickAlternateFilename`: `param2==0 -> .mdx`, `param2==1 -> .mdl` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (asset resolution assumptions) | wrong geometry | Medium |

## Required profile edits
- Add a 0.5.3 terrain source mode indicating WDT-root terrain path (non-ADT-primary) and prevent unconditional ADT parser entry for that build.
- Add WMO 0.5.3 profile settings: `StrictGroupChunkOrder=true`, `EnableMliqGroupLiquids=true`, `EnablePortalOptionalBlocks=true`.
- Extend MDX profile contract with `ModlSectionSize=0x175` (strict) and sequence/global-section strict cursor checks.
- Add explicit 0.5.3 placement-source policy note: support WDT-level `MODF` handling in addition to ADT-era placement extraction assumptions.
- Ensure diagnostics wiring increments and logs for this build/profile path: `InvalidChunkSignatureCount`, `InvalidChunkSizeCount`, `MissingRequiredChunkCount`, `UnknownFieldUsageCount`, `UnsupportedProfileFallbackCount` with build/profile/file/chunk-family context.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/MdxViewer/Terrain/WorldAssetManager.cs` (if terrain source routing depends on asset-open path)

## Open unknowns
- None remaining from this baseline-diff pass; canonical MDX dispatcher order and WDT runtime handoff are now proven in `open-unknowns-resolution-0.5.3.md`.
