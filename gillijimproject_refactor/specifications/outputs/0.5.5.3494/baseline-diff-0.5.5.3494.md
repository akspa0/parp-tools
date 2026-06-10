# Build Delta Report — 0.5.5.3494

## Baseline compared against
- AdtProfile: AdtProfile_060_070_Baseline
- WmoProfile: WmoProfile_060_070_Baseline
- MdxProfile: MdxProfile_060_070_Baseline

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.5.5.3494 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root offset policy | Non-strict scan allowed (`UseMhdrOffsetsOnly=false`) | Offset-driven contract from `MHDR` offsets (`MHDR -> MCIN/MTEX/MDDF/MODF`) | `0x00210348` `CMapArea::Create`: asserts `MHDR`, `MCIN`, `MTEX`, `MDDF`, `MODF`; computes chunk pointers from MHDR offsets | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, MHDR/MCIN dispatch path) | hard parse fail | High |
| `MCLQ` layer stride | `0x2D4` | `0x324` | `0x002151fc` `CMapChunk::Create`: per-layer pointer increments by `+0x324`; liquid block copies align to that layout | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`MclqLayerStride`), `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`Extract/parse liquid path`) | visual artifact | High |
| `MCLQ` decode mode policy | Single-layout assumption | Dual decode path selected by mode bit | `0x0024267c` `CMapObjGroup::CreateOptionalDataPointers` `MLIQ` parse: branch on `(**(byte **)(this + 0x160) & 4)` with two decode routines | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (liquid decode flow policy plumbing) | visual artifact | Medium |
| `MH2O` usage | Fallback enabled in baseline profile | No direct `MH2O` proof in parser anchors; behavior unresolved | No `MH2O` parser anchor discovered in ADT chain (`CMapArea::Create`, `CMapChunk::Create`, `CMapChunk::SyncLoad`) | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`EnableMh2oFallbackWhenNoMclq`) | unknown | Low |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.5.5.3494 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root required chunk order | Baseline profile not strict | Strict root contract enforced | `0x00244428` `CMapObj::CreateDataPointers` asserts ordered chain: `MOHD -> MOTX -> MOMT -> MOGN -> MOGI -> MOPV -> MOPT -> MOPR -> MOLT -> MODS -> MODN -> MODD -> MFOG`, optional trailing `MCVP` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Group required chunks | Baseline profile non-strict | Strict group contract includes `MOPY`, `MOVT`, `MONR`, `MOTV`, `MOLV`, `MOIN`, `MOBA` | `0x0024338c` `CMapObjGroup::CreateDataPointers`; `0x00243c40` `CMapObjGroup::Create` asserts `MOGP` before data-pointer parse | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Optional group gates | Optional behavior undefined in baseline profile | Bit-gated optional blocks are active | `0x0024267c` `CMapObjGroup::CreateOptionalDataPointers`: gates `MOLR(0x200)`, `MODR(0x800)`, `MOBN/MOBR(0x1)`, `MPBV/MPBP/MPBI/MPBG(0x400)`, `MOCV(0x4)`, `MLIQ(0x1000)` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` | wrong geometry | High |
| `MLIQ` decode semantics | Group liquids disabled in baseline profile | Group liquids enabled; two decode modes selected by mode bit | `0x0024267c` `MLIQ` token assert + mode branch `(**(byte **)(this + 0x160) & 4)`; converts either 32-bit lanes or 16+16+32 layout | `src/MdxViewer/Rendering/WmoRenderer.cs` | visual artifact | High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.5.5.3494 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Binary magic requirement | Required but not evidenced at baseline contract level | Enforced `MDLX` magic gate | `0x000fab48` `MDLFileBinaryLoad`: swapped header must equal `0x584C444D` (`MDLX`) else fail (`"File is not a binary model file"`) | `src/MdxViewer/Formats/Mdx/*` | hard parse fail | High |
| Texture section strictness | `TextureSectionSizeStrict=false` | Strict integral texture record sizing | `0x000b1568` `MdxReadTextures`: seeks `TEXS` (`0x53584554`) and validates `sectionBytes % 0x10C == 0`; `0x000fe93c` `MDL::ReadBinTextures` enforces same | `src/MdxViewer/Formats/Mdx/*`, `src/MdxViewer/Rendering/MdxAnimator.cs` | hard parse fail | High |
| `MODL` globals/flag semantics | Not pinned in baseline profile | `MODL` required for globals; flag byte at `+0x174` affects load flags | `0x000b1b00` `MdxLoadGlobalProperties`: seek `MODL` (`0x4C444F4D`), read flag byte `*(iVar1 + 0x174)` | `src/MdxViewer/Formats/Mdx/*`, `src/MdxViewer/Rendering/MdxAnimator.cs` | visual artifact | High |
| Geoset/material/animation ordering | Baseline assumptions loose | Exact top-level read order unresolved from current anchors | `MdxReadGeosets` (`0x000b13c0`) and `MdxReadMaterials` wrapper only; need dispatcher proof for definitive order | `src/MdxViewer/Formats/Mdx/*` | unknown | Medium |

## Required profile edits
- Add `AdtProfile_055_3494`:
  - `UseMhdrOffsetsOnly = true`
  - `McinEntrySize = 0x10`
  - `MclqLayerStride = 0x324`
  - `MclqTileFlagsOffset = 0x290` (provisional; retain until contradicted)
  - `MddfRecordSize = 0x24`
  - `ModfRecordSize = 0x40`
  - `EnableMh2oFallbackWhenNoMclq = false` (provisional; pending MH2O proof)
- Add `WmoProfile_055_3494`:
  - `StrictGroupChunkOrder = true`
  - `EnableMliqGroupLiquids = true`
  - `EnablePortalOptionalBlocks = true`
  - Root chunk sequence contract per `CMapObj::CreateDataPointers`
  - Group required sequence includes `MOIN` and `MOBA`
- Add `MdxProfile_055_3494`:
  - `RequiresMdlxMagic = true`
  - `TextureRecordSize = 0x10C`
  - `TextureSectionSizeStrict = true`
  - `GeosetHardFailIfMissing = false` (no hard-fail evidence in current anchors)

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

## Open unknowns
- Confirm ADT `MH2O` policy for `0.5.5.3494` in non-primary map paths (proof target: ADT loader path that handles post-MCLQ fallback).
- Extract definitive MDX top-level binary dispatch order (proof target: function coordinating `ReadBinVersion/ReadBinModelGlobals/ReadBinGeosets/ReadBinMaterials/...`).
- Verify whether `MCLQ` tile flags offset differs from retained `0x290` under all liquid mode combinations (proof target: ADT chunk liquid decode function with explicit offset constants).
