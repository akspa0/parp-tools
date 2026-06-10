# Build Delta Report — 0.9.0.3807

## Baseline compared against
- AdtProfile: 091_3810
- WmoProfile: 091_3810
- MdxProfile: 091_3810

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.9.0.3807 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MH2O policy consumer proof | `Enabled=false` (no active MH2O path in baseline profile) | Unknown in sampled chain (no direct MH2O token/consumer recovered in this pass) | `FUN_006e6220` validates root offsets/chunks (`MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`), `FUN_006d7590` validates MCNK subchunks including `MCLQ` and no MH2O check in this chain | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, MH2O fallback block) | visual artifact | Medium |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.9.0.3807 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root required chunk order | Strict root contract (MVER/MOHD/.../MFOG) | Same token order and strict sequence confirmed | `FUN_006e7e00` + strict matcher `FUN_006e7cf0`: `MVER(0x11)->MOHD->MOTX->MOMT->MOGN->MOGI->MOSB->MOPV->MOPT->MOPR->MOVV->MOVB->MOLT->MODS->MODN->MODD->MFOG`, optional `MCVP` via `FUN_006e7dc0` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`; `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| MLIQ header field naming | Provisional naming in baseline | Still unknown semantic labels; structural layout confirmed | `FUN_006e8960`: reads `param_2[2..9]`, sample base `+0x26`, sample extent `dim*dim*8`, secondary mask block follows | `src/MdxViewer/Rendering/WmoRenderer.cs` | visual artifact | Medium-High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.9.0.3807 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Top-level section dispatcher | Provisional in baseline | Dispatcher recovered with two branch paths | `FUN_0042a6a0` calls section readers (`MODL`, `TEXS`, `MTLS`, `GEOS`, `ATCH`, `PRE2`, `BONE/HTST`, `SEQS`, `PIVT`, `RIBB`, `LITE`, `CLID`, `CAMS`) with load-flag gates; alt path in `FUN_0042b4d0` | `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`; `src/MdxViewer/Rendering/MdxRenderer.cs` | hard parse fail | Medium-High |
| GEOS requiredness semantics | `GeosetSectionSeek=GEOS`, `GeosetHardFailIfMissing=false` (provisional) | Unknown strictness (assert path present, but runtime fail/continue behavior not fully proven) | `FUN_006da220` seeks `GEOS` (`0x534F4547`) via `FUN_00783c80`, asserts if missing (`line 0x1e2` path) | `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`; `src/MdxViewer/Rendering/MdxRenderer.cs` | wrong geometry | Medium |
| Animation compression/rotation policy | Provisional in baseline | Still unknown | Geoset animation key readers confirmed (`FUN_007a2410`, `FUN_007a2280`, `FUN_007a2a00`) but no complete top-level sequence/compression contract extracted | `src/MdxViewer/Rendering/MdxAnimator.cs`; `src/MDX-L_Tool/Formats/Mdx/MdxModels.cs` | visual artifact | Low |
| Texture replaceable/UV/wrap semantics | Provisional in baseline | Still unknown | TEXS section record size/strictness confirmed (`FUN_006da220`, `FUN_00453930`, `FUN_00453a90`: `TEXS` size `% 0x10C == 0`), but replaceable/UV/wrap branch semantics not fully recovered | `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`; `src/MdxViewer/Rendering/MdxRenderer.cs` | visual artifact | Medium |

## Required profile edits
- No mandatory ADT field delta discovered relative to `091_3810` in sampled parse chain.
- Add explicit `WmoProfile_090_3807` using recovered strict root order and group optional gates.
- Keep `MdxProfile_090x_Unknown` (or introduce `MdxProfile_090_3807_Provisional`) until animation compression and texture replaceable/UV semantics are proven.
- Optional: add explicit build alias mapping (`0.9.0.3807`) to current `*_090x_Unknown` profiles for deterministic diagnostics labeling.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (explicit build mapping and profile id emission)
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (ADT MH2O/MCLQ diagnostics clarity)
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` (WMO root contract gating once proven)
- `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs` (MDX section contract knobs)
- `src/MdxViewer/Rendering/MdxAnimator.cs` (animation policy isolation)

## Open unknowns
- MH2O consumer presence/absence in ADT runtime chain — prove via xrefs to `MH2O` token or liquid fallback callsites beyond `FUN_006d7130` path.
- MDX sequence/keyframe compression and interpolation semantics — dispatcher is recovered, but decode policy details are still unresolved.
- MDX replaceable texture/UV/wrap branch semantics — prove in TEXS consumer call path downstream of `FUN_006da220`/`FUN_00453930`.