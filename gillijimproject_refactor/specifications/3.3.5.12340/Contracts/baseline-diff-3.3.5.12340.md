# Build Delta Report — 3.3.5.12340

## Baseline compared against
- AdtProfile: AdtProfile_091_3810
- WmoProfile: WmoProfile_091_3810
- MdxProfile: MdxProfile_091_3810

## ADT deltas (only changed/unknown)
| Item | Baseline | 3.3.5.12340 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MCNK/root pointer policy | `RequireStrictTokenOrder=true` and explicit token validation chain | Appears offset-table driven in tile callback (no obvious per-subchunk token assert in this pass) | `0x007d6ef0`: derives chunk pointers via fixed offsets (`+0x04,+0x08,+0x0C,+0x10,...,+0x2C`) and assigns `+8`-adjusted pointers directly | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, MCNK validation path) | wrong geometry | Medium |
| Optional root aux block gate | `MH2O disabled` in baseline profile | Optional block gated by MCNK flag bit and root offset slot; with ADT MHDR naming context, slot neighborhood maps to `mfbo/mh2o/mtxf`, and observed branch behavior aligns better with non-liquid auxiliary geometry path | `0x007d6ef0`: `if ((*pbVar1 & 1) != 0) param_1+0xb0 = pbVar1 + *(int *)(pbVar1+0x24) + 8` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`AdtProfile` flags), `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`TryParseMh2oFromRootAdt`) | visual artifact | Medium |

## WMO deltas (only changed/unknown)
| Item | Baseline | 3.3.5.12340 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root/group table parsing divisors | Baseline docs define required chunks/gates; no contradiction expected | Record counts derived by fixed divisors in root data-pointer builder (confirming schema-style coupling; exact chunk labels partly unresolved in this pass) | `0x007d7470`: count derivations `/0x0C`, `/0x14`, `/0x30`, `>>5`, `>>3`, `>>2`; optional tail check `0x4d435650` (`MCVP`) | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` (`ParseWmoV14Internal`, `ParseMogp`) | wrong geometry | Medium |
| Group flag normalization behavior | Not explicitly called out in baseline profile | If no name/string block, group flag bit `0x00040000` is cleared across groups | `0x007d7470`: when name block missing, iterates group entries and `*puVar1 = *puVar1 & 0xfffbffff` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` (MOGP flag handling) | visual artifact | Medium |

## MDX deltas (only changed/unknown)
| Item | Baseline | 3.3.5.12340 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root magic contract | MDX profile framed around MDX-era chunk assumptions | Runtime model loader is `MD20`-centric (M2), not classic MDX root-chunk contract | `0x0083cf00`: requires `*piVar1 == 0x3032444d` (`MD20`) and then dispatches typed section readers | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`M2Profile*`, `ResolveModelProfile`) | hard parse fail | High |
| Version acceptance window | Baseline MDX profile does not encode strict M2 version window | M2 header version appears constrained to `0x108` in this path | `0x0083cf00`: version check `0x107 < piVar1[1] && piVar1[1] < 0x109` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`M2Profile3018303.MinSupportedVersion/MaxSupportedVersion`) | hard parse fail | High |
| Extension resolution behavior | Baseline profile does not specify extension coercion | Loader accepts `.m2`, rewrites `.mdx`/`.mdl` to `.m2`, rejects others | `0x0081c390`: invalid extension branch; `.mdx`/`.mdl` normalized to `.m2`; logs `"Model2: Invalid file extension"` | `src/WoWMapConverter/WoWMapConverter.Core/Formats/FormatDetector.cs`, `src/MdxViewer/Terrain/WorldAssetManager.cs` | hard parse fail | High |
| Skin profile sidecar policy | Baseline MDX profile provisional on texture/material details | Explicit `%02d.skin` sidecar generation/load path exists | `0x00835a80` builds `%02d.skin`; `0x00838490` validates skin profile sections | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`M2Profile` skin/effect stride fields) | visual artifact | High |

## Required profile edits
- Add explicit `3.3.5.12340` dispatch entries in `FormatProfileRegistry` for ADT/WMO/MDX/M2 instead of relying only on `3.0.x` fallback.
- Prefer `M2Profile_335_12340` (or equivalent exact-build alias) with validated `RequiredRootMagic=MD20` and version window evidence from `0x0083cf00`.
- Keep ADT liquid optional-block semantics profile-gated until the `+0x24` block is proven as MH2O in a dedicated proof pass.

## Named-field crosswalk updates (from wowdev references)
- M2 late header pairs (`+0x36..+0x4c`) now map to named fields: `collisionIndices`, `collisionPositions`, `collisionFaceNormals`, `attachments`, `attachmentIndicesById`, `events`, `lights`, `cameras`, `cameraIndicesById`, `ribbon_emitters`, `particle_emitters`, optional `textureCombinerCombos`.
- M2 mid-table pairs (`+0x16..+0x26`) now map to named fields: `texture_weights`, `texture_transforms`, `textureIndicesById`, `materials`, `boneCombos`, `textureCombos`, `textureCoordCombos`, `textureWeightCombos`, `textureTransformCombos`.
- ADT MHDR root offsets for WotLK naming context: `mfbo` (`0x24`), `mh2o` (`0x28`), `mtxf` (`0x2C`), used as naming anchors for further proof passes.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`ResolveAdtProfile`, `ResolveWmoProfile`, `ResolveMdxProfile`, `ResolveModelProfile`)
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, `TryParseMh2oFromRootAdt`)
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` (`ParseWmoV14Internal`, `ParseMogp`)
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/FormatDetector.cs`

## Resolved unknowns
- ADT MCNK signature validation path is `0x007c64b0 -> 0x007c3a10` (not `0x007d7020`): `0x007c3a10` performs bounded subchunk scanning and FourCC dispatch (`MCVT/MCNR/MCLY/MCRF/MCSH/MCAL/MCLQ/MCCV/MCSE`).
- ADT optional root slot around `+0x24/+0x28` in `0x007d6ef0` resolves to MapArea auxiliary path in this branch (non-MH2O consumption): `+0x28` branch allocates from `.\\MapArea.cpp` and dispatches `0x007d4f10` mesh-style consumer path.
- WMO `0x007d7470` chunk label mapping is now resolved by size/order plus `MCVP` literal check: `MOGI(0x20) -> MOSB(string) -> MOPV(0x0C) -> MOPT(0x14) -> MOPR(0x08) -> MOVV(0x0C) -> MOVB(0x04) -> MOLT(0x30) -> MODS(0x20) -> MODN(blob) -> MODD(0x28) -> MFOG(0x30) -> optional MCVP(0x10)`.
- Diagnostics counters are now implemented in build path code:
	- `InvalidChunkSignatureCount`, `InvalidChunkSizeCount`, `MissingRequiredChunkCount`, `UnknownFieldUsageCount`, `UnsupportedProfileFallbackCount` in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`.
	- `MpqSectorTableInvalidCount`, `MpqPatchedDeleteHitCount` in `src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`.
	- `M2TableValidationRejectCount`, `M2ConversionFailureCount` in `src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`.
	- Shared sink: `src/WoWMapConverter/WoWMapConverter.Core/Diagnostics/Build335Diagnostics.cs`.