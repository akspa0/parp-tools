# Build Delta Report — 0.10.3892

## Baseline compared against
- AdtProfile: 091_3810
- WmoProfile: 091_3810
- MdxProfile: 091_3810_Provisional

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.10.3892 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MH2O fallback policy | Disabled for 0.9.1 profile (`EnableMh2oFallbackWhenNoMclq=false`) | **resolved: keep disabled (no evidence of MH2O parse path in 0.10.3892 chain)** | ADT parse requires `MCLQ` at `FUN_006e5e60` (`0x006e5e60`); liquid creation consumes only MCLQ layers at `FUN_006e5c50` (`0x006e5c50`); no `MH2O`/`2OHM` token strings found in program strings | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`EnableMh2oFallbackWhenNoMclq`), `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`TryBuildLiquidFromMh2o`) | visual artifact | High |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.10.3892 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Group `MLIQ` tail semantics (+0xF0..+) | 0.9.1 docs mark part of MLIQ interpretation as provisional | **partially resolved (layout proven, semantic names still provisional)** | `FUN_006f70f0` (`0x006f70f0`): gated by `flags & 0x1000`; stores header words to `+0xF0..+0x108`; sample ptr `+0x110 = chunk+0x26`; sample bytes `f0*f4*8`; mask ptr `+0x114 = sampleEnd`; mask bytes `f8*fc`; low16 of `param_2[9]` stored at `+0x10C` | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`WmoProfile*.EnableMliqGroupLiquids`), WMO group parser consumers | visual artifact | Medium-High |
| Group `MLIQ` field-role mapping | Provisional naming only | **mostly resolved from downstream consumers** | `FUN_006ed910`/`FUN_006edb90`/`FUN_006edeb0` consume `+0xF0,+0xF4` as vertex grid dims (`nVerts=f0*f4`), `+0xF8,+0xFC` as tile/mask dims, `+0x100,+0x104` as XY origin offsets, `+0x110` as 8-byte vertex stream start, `+0x114` as tile-mask stream; `+0x10C` used as liquid-type/material lookup index (`*0x40 + area->materialTable`) | WMO group liquid rendering/query consumers | visual artifact | High |
| Group `MLIQ` `+0x108` semantic | Unproven in prior pass | **resolved as position-Z/base-Z header field (not required by sampled render path)** | In `FUN_006f70f0` (`0x006f70f0`), fields `+0x100/+0x104/+0x108` are loaded contiguously from `param_2[6..8]` (float triplet pattern matching X/Y/Z origin header layout); downstream liquid mesh builders consume X/Y from `+0x100/+0x104` and per-vertex heights from stream, leaving `+0x108` as unused-by-path Z origin/base | WMO group liquid reader/renderer pipeline | visual artifact | Medium-High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.10.3892 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MDX top-level section dispatch policy | `MdxProfile_091_3810_Provisional` (order partly provisional) | **resolved: tag-dispatch loop (no hardcoded strict top-level order)** | `FUN_00793800` (`0x00793800`) verifies `MDLX`, loops section headers, dispatches each by tag via `FUN_00794250`; unknown tags produce warning + seek skip | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`MdxProfile*`) | hard parse fail | High |
| TEXS record/strict-size confirmation for 0.10 | Baseline `TextureRecordSize=0x10C`, `TextureSectionSizeStrict=true` | **resolved: same as baseline** | `FUN_007b8e40` (`0x007b8e40`) requires `sectionSize % 0x10C == 0`; per-record read = `uint` + `0x104` bytes + `uint` (stride `0x10C`) | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`TextureRecordSize`, `TextureSectionSizeStrict`) | visual artifact | High |

## Required profile edits
- Add build dispatch aliases for `0.10.3892` -> provisional profiles reusing 0.9.1 values until unknowns are proven:
  - `AdtProfile_010_3892_Provisional` (mirror `AdtProfile_091_3810`)
  - `WmoProfile_010_3892_Provisional` (mirror `WmoProfile_091_3810`)
  - `MdxProfile_010_3892_Provisional` (mirror `MdxProfile_091_3810_Provisional`)
- Deep-dive confirmations now available:
  - ADT MCIN entry indexing is `index * 0x10` in live 0.10 path (`FUN_006f5ae0`, `0x006f5ae0`; `iVar9*0x10 + area->mcin + 8`).
  - MDX root parser enforces `MDLX` magic (`FUN_00793800`, `0x00793800`) and dispatches by section tag (`FUN_00794250`, `0x00794250`).
  - MDX `TEXS` strict record stride remains `0x10C` (`FUN_007b8e40`, `0x007b8e40`).
- Final deep-dive resolution:
  - ADT `MH2O` fallback remains disabled for `0.10.3892` (no observable consumer/dispatch evidence).
  - WMO `MLIQ +0x108` maps to header Z/base position (currently non-critical for sampled consumer path).

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
  - `ResolveAdtProfile`
  - `ResolveWmoProfile`
  - `ResolveMdxProfile`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `ParseAdt`
  - `TryBuildLiquidFromMh2o`

## Open unknowns
- None (all prior unknowns resolved to implementation-ready profile decisions with confidence noted above).
