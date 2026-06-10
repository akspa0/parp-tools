# Build Delta Report — 0.12.0.3988

## Baseline compared against
- AdtProfile: AdtProfile_091_3810
- WmoProfile: WmoProfile_091_3810
- MdxProfile: MdxProfile_091_3810_Provisional

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.12.0.3988 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root + placements contract | `MVER->MHDR` strict, MHDR offsets, `MCIN` required, `MDDF=0x24`, `MODF=0x40` | same (no parser-affecting delta observed) | `FUN_006c2d90`: checks `MVER(0x4D564552)`, `MHDR(0x4D484452)`, `MCIN(0x4D43494E)` and computes counts via `/0x24` and `>>6` | src/MdxViewer/Terrain/StandardTerrainAdapter.cs (`ParseAdt`) and src/MdxViewer/Terrain/FormatProfileRegistry.cs (`AdtProfile0913810`) | wrong placement (if contradicted later) | High |
| MCNK subchunk gate + MCLQ layout | Required `MCVT/MCNR/MCLY/MCRF/MCSH/MCAL/MCLQ/MCSE`; MCLQ stride `0x324` | same | `FUN_006b4940`: asserts all required subchunks; `FUN_006b4730`: advances `puVar3 += 0xC9` dwords (= `0x324`) and wires fixed offsets | src/MdxViewer/Terrain/StandardTerrainAdapter.cs (`ParseAdt` MCNK+liquid handling), src/MdxViewer/Terrain/FormatProfileRegistry.cs (`MclqLayerStride`) | visual artifact | High |
| MH2O runtime policy | Baseline profile treats MH2O unsupported for this era path | unknown in this binary pass (no direct MH2O consumer proven) | No direct `MH2O` consumer reached in recovered ADT chain (`FUN_006c2d90 -> FUN_006b44d0 -> FUN_006b4730/47e0`) | src/MdxViewer/Terrain/StandardTerrainAdapter.cs (fallback decisions around liquid parsing) | visual artifact | Medium |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.12.0.3988 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Group header/version gate | `MVER=0x11`, `MOGP` required in group path | same | `FUN_006c53a0`: asserts `*piVar1==0x4D564552`, `piVar1[2]==0x11`, `piVar1[3]==0x4D4F4750` (`MOGP`) | src/MdxViewer/Terrain/FormatProfileRegistry.cs (`WmoProfile0913810`) and WMO converter/renderer path per profile architecture | hard parse fail | High |
| Group chunk sequence + optional gates | Baseline: strict group order with optional gates | same for proven sequence; optionality confirmed by bit masks | `FUN_006c55c0`: ordered `MOPY->MOVI->MOVT->MONR->MOTV->MOBA`; `FUN_006c5830`: conditional blocks for `MOLR/MODR/MOBN+MOBR/MPBV+MPBP+MPBI+MPBG/MOCV/MLIQ/MORI+MORB` via flag tests (`0x200,0x800,0x1,0x400,0x4,0x1000,0x20000`) | src/MdxViewer/Terrain/FormatProfileRegistry.cs (`StrictGroupChunkOrder`, portal/lq flags) and WMO conversion/rendering modules | hard parse fail | High |
| Root required chunk order completeness | Baseline root list includes full MOHD..MFOG contract | same; full root chain recovered | `FUN_006c4d00` + `FUN_006c4bf0`: strict ordered checks `MOHD->MOTX->MOMT->MOGN->MOGI->MOSB->MOPV->MOPT->MOPR->MOVV->MOVB->MOLT->MODS->MODN->MODD->MFOG`, optional `MCVP` via `FUN_006c4cc0` | WMO root loader path (profile architecture target modules) | hard parse fail | High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.12.0.3988 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Container identity | Baseline `MdxProfile_091_3810` still models `MDLX` contract (`RequiresMdlxMagic=true`) | changed: runtime parser explicitly requires `MD20` root for this load path | `FUN_0071e1f0`: `if (*param_3 != 0x3032444D) return 0;` (`MD20`) | src/MdxViewer/ViewerApp.cs (`DetectContainerKindFromHeader`, model routing) and src/MdxViewer/Terrain/FormatProfileRegistry.cs (`ResolveMdxProfile` / model-profile routing) | hard parse fail | High |
| Model version gate | Baseline MDX-era assumptions (chunk-seek) | changed: `MD20` version required as `0x100` in this path | `FUN_0071e1f0`: `if (param_3[1] != 0x100) return 0;` | src/MdxViewer/ViewerApp.cs (routing diagnostics), model profile resolution | hard parse fail | High |
| Geometry/material stream contract | Baseline MDX profile fields (`TextureRecordSize=0x10C`, GEOS assumptions) | changed: MD20 typed offset/count table with many fixed record strides | `FUN_0071e1f0`: typed table chain with validated record strides (`0x6C`, `0x30`, `0x2C`, `0x38`, `0x10`, `0x1C`, `0x54`, `0xD4`, `0x7C`, `0xDC`, `0x1F8`); nested validators (`FUN_0071f650/420/320`, `FUN_00720430`, `FUN_00720f10/90`, `FUN_00720e10/90`) | src/MdxViewer/Terrain/FormatProfileRegistry.cs (`M2Profile*` fields) and MD20/M2 decode path | wrong geometry | High |
| Legacy MDLX chunk-seek behavior | Baseline MDX parser expects MDLX chunk container | unknown applicability for this build; not evidenced as primary path | No `MDLX` signature gate found in recovered 0.12 model load chain (`FUN_00716890 -> FUN_007096f0 -> FUN_0071e8c0 -> FUN_0071e9a0 -> FUN_0071e1f0`) | src/MdxViewer/Formats/Mdx/* and routing guardrails | hard parse fail | Medium |

## Required profile edits
- Add exact build mapping for `0.12.0.3988` to avoid fallback to legacy `MdxProfile_*`.
- Route `0.12.0.3988` models to an explicit `M2Profile` (MD20, version `0x100`) instead of `MdxProfile` MDLX-first assumptions.
- Add/confirm a dedicated profile ID for this build family (e.g., `M2Profile_012_3988`) with typed table semantics and validated stride fields (`0x6C`, `0x30`, `0x2C`, `0x38`, `0x10`, `0x1C`, `0x54`, `0xD4`, `0x7C`, `0xDC`, `0x1F8`).
- Keep ADT/WMO profile values aligned with 0.9.1-equivalent contracts unless contradicted by future proof.
- Ensure diagnostics counters are emitted in this routing path: `InvalidChunkSignatureCount`, `InvalidChunkSizeCount`, `MissingRequiredChunkCount`, `UnknownFieldUsageCount`, `UnsupportedProfileFallbackCount` with build/profile/file/family context.

## Implementation targets
- src/MdxViewer/Terrain/FormatProfileRegistry.cs
  - `ResolveMdxProfile(string? buildVersion)`
  - `ResolveModelProfile(string? buildVersion)`
  - new `M2Profile_012_3988` constant (or equivalent exact-build profile)
- src/MdxViewer/ViewerApp.cs
  - container identity routing (`MDLX` vs `MD20`) and mismatch diagnostics path
- src/MdxViewer/Terrain/StandardTerrainAdapter.cs
  - verify diagnostics context includes build/profile for ADT family when profile fallback occurs

## Open unknowns
- ADT MH2O policy for this build (enabled/ignored/fallback); no direct `MH2O` token/string path recovered in ADT parser chain (`FUN_006c2d90`, `FUN_006b4940`, `FUN_006b44d0`, `FUN_006b4730`). Proof target: isolate any alternative liquid path beyond `MCLQ`/`MCSE` in adjacent ADT loaders.
- MD20 semantic labeling (human names for each typed section) remains partial even though structural contracts are now mapped; proof target: correlate `FUN_0071e1f0` section slots with runtime consumers/renderer fields.
