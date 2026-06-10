# Build Delta Report — 0.11.3925

## Baseline compared against
- AdtProfile: 091_3810
- WmoProfile: 091_3810
- MdxProfile: 091_3810_Provisional

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.11.3925 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MCIN entry stride/indexing | Historical expectation `entrySize=0x10` | **resolved: direct runtime indexing uses `index * 0x10`** | `FUN_006b3be0` (`0x006b3be0`): computes `iVar9 = y * 0x10 + x`; reads MCIN entry at `*(param_1+0x684) + 8 + iVar9*0x10`; loads chunk via `*(entryBase + iVar9*0x10) + rawAreaData` and calls `FUN_006a23d0` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, chunk materialization/indexing path) | wrong geometry | High |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.11.3925 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| `MLIQ` header field roles (`+0xF4..+0x10C`) | Previously partially inferred | **resolved (except `+0x10C` final semantic)** | Parse/store at `FUN_006b5260` (`0x006b5260`): `+0xF4,+0xF8,+0xFC,+0x100,+0x104,+0x108,+0x10C`; data ptr `+0x114=chunk+0x26`; mask ptr `+0x118=+0x114 + (f4*f8*8)`. Consumers: `FUN_006962f0` uses `+0xF4/+0xF8` as vertex grid dims and `+0x104/+0x108` as XY origins with per-vertex height from sample `(+4)`; `FUN_006aa0c0` uses `+0xFC/+0x100` as mask/tile dims and validates against `(f4*f8)`; `FUN_00678c40` and `FUN_006aa250` consume mask low-nibble liquid types; `FUN_006aa390` uses `*(ushort*)(+0x110)` as liquid/material class index (`*0x40` table stride) | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`WmoProfile` liquid contracts), WMO group liquid parser/renderer paths | visual artifact | High |
| `MLIQ` dword at `+0x10C` usage | Unknown | **no read observed in traced runtime consumers (likely reserved/unused in this build path)** | Parsed/stored in `FUN_006b5260` (`0x006b5260`) but not read in expanded consumer graph: `FUN_006962f0`, `FUN_006aa0c0`, `FUN_00678c40`, `FUN_006aa250`, `FUN_006aa390`, `FUN_006aa610`, `FUN_006aa930`, `FUN_00679000`, `FUN_006905f0`, `FUN_006907d0`, `FUN_0067b4f0`, `FUN_0067bc50` | WMO group liquid read/query/render paths | visual artifact | Medium-High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.11.3925 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root magic + version contract | Provisional MDX-era assumptions | **changed: runtime model loader is M2 (`MD20`) with strict version `0x100`** | `FUN_0070d500` (`0x0070d500`): `*param_3 == 0x3032444d` (`MD20`) and `param_3[1] == 0x100`; hard fail (`return 0`) otherwise; validates many count/offset tables via `FUN_0070e*` chain | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`ResolveModelProfile`, M2 profile selection) | hard parse fail | High |
| Extension normalization policy | MDX profile does not encode extension coercion | **changed: `.mdx` and `.mdl` inputs are rewritten to `.m2` before shared-model load** | `FUN_00710890` (`0x00710890`): compares extension against `DAT_00860bd0`/`DAT_00860bc8` and copies `DAT_00860bc4` into suffix prior to `FUN_0070db30`; loader path is `CM2Shared`/`CM2Model` (`FUN_00706a40`) | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`ResolveModelProfile`), `src/MdxViewer/Terrain/WorldAssetManager.cs` | hard parse fail | High |

## Required profile edits
- Add explicit build mapping for `0.11.3925` in `ResolveModelProfile` to an early-M2 contract (new alias profile suggested: `M2Profile_011_3925`).
- Add explicit contract note for `0.9+`: extension is advisory only (`.mdx` may carry non-`MDLX` containers); parser selection must key on root magic + version.
- In that profile, enforce:
  - `RequiredRootMagic = MD20`
  - `MinSupportedVersion = 0x100`
  - `MaxSupportedVersion = 0x100`
  - `UseTypedOffsetCountTable = true`
  - `StrictSpanValidation = true`
  - nested fixed-stride table contract (at minimum): `0x2C`, `0x30`, `0x38`, `0x54`, `0x6C`, `0x7C`, `0xD4`, `0xDC`, `0x1F8`
  - extension coercion policy for `.mdx`/`.mdl` -> `.m2` in parser-facing diagnostics context.
- No ADT/WMO profile-field edits required from this pass (no contradictory deltas found in sampled parser chain).

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
  - `ResolveModelProfile`
  - (if needed) `ResolveMdxProfile` to avoid selecting MDX contract for `0.11.3925` assets that are model2-backed
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - diagnostics/fallback plumbing when model profile resolution falls back

## Open unknowns
- No blocking parser unknowns remain for `0.11.3925`; `MLIQ +0x10C` is currently treated as observed-unused/reserved in traced runtime paths and does not require profile enforcement.
