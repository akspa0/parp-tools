# Build Delta Report — 0.8.0.3734

## Baseline compared against
- AdtProfile: AdtProfile_091_3810
- WmoProfile: WmoProfile_091_3810
- MdxProfile: MdxProfile_091_3810_Provisional

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.8.0.3734 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| MCLQ per-layer stride | `0x324` (`0xC9 dwords`) | `0x2D4` (`0xB5 dwords`) | `0x006b8be0`: `puVar4 = puVar4 + 0xb5` and in-layer pointers at `+0x08`, `+0x290`, `+0x2D0` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`MCLQ` decode path) | wrong geometry / visual artifact | High |
| MCLQ flow block policy | Dual-flow mode (`+0x2D0` mode-dependent) | No confirmed dual-flow decode branch in parser write-up; only scalar at `+0x2D0` captured | `0x006b8be0`: scalar copy from `*(puVar4+0xB4)` with no mode split in this function | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` + `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` | visual artifact | Medium |
| MCIN entry-size proof in this pass | `0x10` | not directly re-proved in current extracted function set | ADT root at `0x006c7220` verifies `MCIN` token but this pass did not re-extract downstream count/divisor function | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` | unknown | Medium |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.8.0.3734 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root `MVER` required value | `0x11` | `0x10` | `0x006cac40`: `if (piVar1[2] != 0x10) assert` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`; `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Root required chunk set | Includes `MOSB`, `MOVV`, `MOVB` in baseline profile | Root parser enforces `MOHD,MOTX,MOMT,MOGN,MOGI,MOPV,MOPT,MOPR,MOLT,MODS,MODN,MODD,MFOG` (+ optional `MCVP`) | `0x006cac40` strict `FUN_006cab30` chain | `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Group `MOBA` divisor | `/0x18` in baseline profile | `>>5` (`/0x20`) | `0x006cb4b0`: `*(uint *)(param_1 + 0x120) = size >> 5` | `src/MdxViewer/Rendering/WmoRenderer.cs` | wrong geometry | High |
| Group optional gate set | Includes `MORI/MORB` gate (`0x20000`) in baseline | No `0x20000` branch observed in this build’s optional parser | `0x006cb700` shows gates `0x200,0x800,0x1,0x400,0x4,0x1000` only | `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail / wrong geometry | Medium |
| `MLIQ` decode policy | Mode bit (`firstMaskByte & 0x04`) with dual decode mode in baseline | Direct pointer wiring only in group parser (`+0x26` data, then `xVerts*yVerts*8`) in this extracted function | `0x006cb700`: sets `group+0x100` and `group+0x104` from fixed offset math | `src/MdxViewer/Rendering/WmoRenderer.cs`; converter liquid adapters | visual artifact | Medium |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.8.0.3734 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| `TEXS` texture count policy | `sectionBytes % 0x10C == 0` (multi allowed) | hard check `numTextures == 1` and `sectionBytes == 1*0x10C` | `0x006bbd10`: `uVar11 = sectionBytes/0x10c; if (uVar11 != 1) assert` | `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`; `src/MdxViewer/Rendering/MdxAnimator.cs` | hard parse fail | High |
| `GEOS` UV set policy | baseline profile treated as provisionally flexible | hard check `UVAS` count == `1` in this function path | `0x006bbd10`: after `SAVU/UVAS` token, `if (piVar1[1] != 1) assert` | `src/MDX-L_Tool/Formats/Mdx/*` | hard parse fail / visual artifact | High |
| Full top-level chunk order | partial/provisional in baseline | not fully re-derived in this pass | `0x00422620` confirms `MDLX` gating and dispatch via `FUN_00421700`, but no complete chunk-order matrix extracted here | `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs` | unknown | Medium |

## Required profile edits
- Add `AdtProfile_080_3734.MclqPolicy.LayerStride = 0x2D4`.
- Add `AdtProfile_080_3734.MclqPolicy` in-layer offsets (`SampleBase=+0x08`, `TileFlags=+0x290`, `Scalar2=+0x2D0`).
- Add `WmoProfile_080_3734.RootChunkPolicy.RequiredRootChunks` with `MVER=0x10` and 0.8 root order from `0x006cac40`.
- Add `WmoProfile_080_3734.GroupChunkPolicy.RequiredGroupChunks` where `MOBA` divisor uses `0x20` entry-size semantics.
- Add `WmoProfile_080_3734.GroupChunkPolicy.OptionalGroupChunkGates` excluding `MORI/MORB` unless separately proven for this build.
- Add `MdxProfile_080_3734.MaterialPolicy.TextureRecordSize=0x10C` plus `ExpectedTextureCount=1` (strict mode).
- Add `MdxProfile_080_3734.GeometryPolicy.ExpectedUvasSetCount=1` for this strict legacy path.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

## Open unknowns
- ADT MCIN entry-size/count derivation for `0.8.0.3734` (prove in MCIN consumer function adjacent to `0x006c7220`).
- Whether any 0.8 path supports dual-flow-style `MCLQ` behavior beyond the scalar captured at `+0x2D0` (prove in liquid update/render functions).
- Full MDX top-level chunk required/optional order for this build via dispatcher at/under `0x00421700`.
- Precise WMO `MLIQ` field semantics at header words copied in `0x006cb700` (`+0xE0..+0xFC`) beyond structural mapping.
