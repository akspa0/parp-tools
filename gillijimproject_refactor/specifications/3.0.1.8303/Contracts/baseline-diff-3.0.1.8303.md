# Build Delta Report — 3.0.1.8303

## Baseline compared against
- AdtProfile: AdtProfile_091_3810
- WmoProfile: WmoProfile_091_3810
- MdxProfile: MdxProfile_091_3810

## ADT deltas (only changed/unknown)
| Item | Baseline | 3.0.1.8303 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MCNK trailing optional subchunk at header `+0x58` | `MCSE` expected/mapped in existing field map | Runtime check gates `MCVS` (`0x4D435653`) at `+0x58`; non-match zeroes pointer | `0x0072E510`: `piVar3 = *(...+0x58); if (*piVar3 == 0x4D435653) ... else *(...+0x5C)=0` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`/MCNK optional handling via parser library) | visual artifact | High |
| MCNK liquid layer stride contract | `0x324` (4 packed layers max; tile flags at `+0x290`) | Confirmed unchanged in this build | `0x0072EAB0`: per-layer advance `puVar4 = puVar4 + 0xC9` dwords (`0x324` bytes); tile flags pointer `puVar4 + 0xA4` (`0x290`) | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ExtractMclq`) | wrong geometry | High |
| ADT root/placement core contracts | Strict MHDR-offset rooted parse; `MDDF=0x24`, `MODF=0x40` | Confirmed unchanged in this build | `0x0073C710`: asserts `MVER->MHDR->MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`; counts via `/0x24` and `>>6` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, `CollectPlacementsViaMhdr`) | hard parse fail | High |
| MH2O policy | Baseline profile marks disabled for 0.9.1 path | No direct `MH2O` token evidence found in this focused pass; unresolved for 3.0.1.8303 | string scan: no `MH2O` hit in current program pass | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseMh2o` fallback gate) | visual artifact | Low |

## WMO deltas (only changed/unknown)
| Item | Baseline | 3.0.1.8303 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Group required chunk chain | `MOPY->MOVI->MOVT->MONR->MOTV->MOBA` strict | Confirmed unchanged in this build | `0x0073EFB0` token assertions and element divisors (`>>1`, `/0xC`, `>>3`, `/0x18`) | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` | wrong geometry | High |
| Optional group gates | `MOLR/MODR/MOBN+MOBR/MPB*/MOCV/MLIQ/MORI+MORB` by flag bits | Confirmed unchanged in this build | `0x0073EBD0` gates on `0x200/0x800/0x1/0x400/0x4/0x1000/0x20000` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` | hard parse fail | High |
| Root version/order validation breadth | `MVER==0x11`, strict root chain (full list in baseline docs) | Partial in this pass: `MVER==0x11` and immediate `MOGP` requirement confirmed; full root-chain re-proof pending | `0x0073F630`: `MVER`, version `0x11`, `MOGP` asserts | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` | hard parse fail | Medium |

## MDX deltas (only changed/unknown)
| Item | Baseline | 3.0.1.8303 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Model pipeline family | `MdxProfile_091_3810` assumptions (MDX-profiled handling) | Active runtime path is `M2`-centric (`M2Cache.cpp`/`M2Model.cpp`) in this binary; direct `GEOS/TEXS/SEQS` token proof not recovered in focused pass | `0x0077D3C0` (`M2Cache.cpp` load path, extension normalization to `.m2`), `0x00792F80` (`M2Model.cpp` heavy load/animate path) | `src/MdxViewer/Formats/Mdx/*`, `src/MdxViewer/Rendering/MdxRenderer.cs`, `src/MdxViewer/Rendering/MdxAnimator.cs` | wrong geometry | Medium |
| Required model chunk-order contract | Baseline provisional | Unknown for 3.0.1.8303 (needs dedicated parser-dispatch trace) | Missing direct token assertions for `GEOS/TEXS/MODL/SEQS` in this pass | same as above | hard parse fail | Low |

## Required profile edits
- Add ADT optional-subchunk discriminator for MCNK header offset `+0x58`: accept `MCVS` for `3.0.1.8303` profile (do not force `MCSE` for this build).
- Keep `MclqLayerStride=0x324`, `MclqTileFlagsOffset=0x290`, `MddfRecordSize=0x24`, `ModfRecordSize=0x40` unchanged for this build profile.
- Add a build-scoped MDX/M2 uncertainty flag in profile metadata (provisional) to force diagnostics instead of silent coercion.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`ResolveAdtProfile`/profile constants for `3.0.1.8303`)
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (MCNK optional subchunk handling alignment + diagnostic counters)
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` (no structural changes expected; keep strict gates)
- `src/MdxViewer/Formats/Mdx/*` (only if dedicated MDX/M2 dispatch proof shows contract drift)

## Open unknowns
- Verify whether `MCSE` is fully replaced by `MCVS` in all ADT variants for `3.0.1.8303` (proof task: trace all callers of `0x0072E510` and MCNK subchunk consumers).
- Prove/deny MH2O usage in this build’s terrain path (proof task: find any parser function asserting `MH2O` token or using MHDR MH2O offset).
- Extract model-file chunk dispatcher for `3.0.1.8303` to map `GEOS/TEXS/MODL/SEQS` equivalents with hard addresses (proof task: follow load callback chain from `0x0079BC70` into shared parse routines).

## Diagnostics notes
Ensure the following counters are surfaced for this build profile and include context (`build=3.0.1.8303`, `profileId`, `filePath`, `chunkFamily`):
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`
