# Build Delta Report — 0.7.0.3694

## Baseline compared against
- AdtProfile: AdtProfile_060_070_Baseline
- WmoProfile: WmoProfile_060_070_Baseline
- MdxProfile: MdxProfile_060_070_Baseline

## ADT deltas (only changed/unknown)
| Item | Baseline | 0.7.0.3694 | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| ADT root parse strictness | Baseline implementation may tolerate loose/scan behavior in some paths | Strict `MVER -> MHDR` and MHDR-offset driven token chain enforced | `0x00283ac0` `CMapArea::Create`: asserts `MVER`, `MHDR`, then `MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`; pointers computed from MHDR offsets (`+0x14..+0x30`) | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt` root dispatch), `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`UseMhdrOffsetsOnly`) | hard parse fail | High |
| MCNK required subchunk contract | Baseline may not hard-gate all required chunks | Required chain is strict and offset-mapped in header | `0x0028540c` `CMapChunk::CreatePtrs`: asserts `MCNK`, then `MCVT(+0x14)`, `MCNR(+0x18)`, `MCLY(+0x1C)`, `MCRF(+0x20)`, `MCSH(+0x2C)`, `MCAL(+0x24)`, `MCLQ(+0x60)`, `MCSE(+0x58)` | `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (MCNK subchunk validation/offset map) | hard parse fail | High |
| MCLQ layer stride for 0.7.0.3694 | Baseline family assumes 0.6/0.7 compatible but lacked direct proof in prior pass | Per-layer liquid block is confirmed `0xB5` dwords (`0x2D4` bytes), with sample/flag/tail offsets matching legacy profile | `0x0028601c` `CMapChunk::CreateLiquids`: per-layer advance `puVar18 = puVar18 + 0xB5`; block field pointers set to `+0x08` (samples), `+0x290` (`+0xA4` dwords) and tail scalar at `+0x2D0` (`+0xB4` dwords) | `src/MdxViewer/Terrain/FormatProfileRegistry.cs` (`MclqLayerStride`, `MclqTileFlagsOffset`), `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (liquid decode) | visual artifact | High |

## WMO deltas (only changed/unknown)
| Item | Baseline | 0.7.0.3694 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Root required chunk order and divisors | Baseline implementation may not fully enforce legacy root ordering/divisors | Strict root sequence and per-chunk divisors are enforced | `0x002891fc` `CMapObj::CreateDataPointers`: `MVER(0x10) -> MOHD -> MOTX -> MOMT(/0x40) -> MOGN -> MOGI(/0x20) -> MOPV(/0x0C) -> MOPT(/0x14) -> MOPR(/0x08) -> MOLT(/0x30) -> MODS(/0x20) -> MODN -> MODD(/0x28) -> MFOG(/0x30)`; optional trailing `MCVP(/0x10)` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Group required chunk order/divisors | Baseline may be looser on required group sequence | Strict required group chain and record divisors | `0x0028fcac` `CMapObjGroup::CreateDataPointers`: `MOPY(/0x04) -> MOVI(/0x02) -> MOVT(/0x0C) -> MONR(/0x0C) -> MOTV(/0x08) -> MOBA(/0x20)` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs` | hard parse fail | High |
| Optional group gates and MLIQ decode mode | Baseline optional policy not fully pinned | Bit-gated optional chain + dual MLIQ decode branch | `0x0029016c` `CMapObjGroup::CreateOptionalDataPointers`: gates `MOLR(0x200)`, `MODR(0x800)`, `MOBN/MOBR(0x1)`, `MPBV/MPBP/MPBI/MPBG(0x400)`, `MOCV(0x4)`, `MLIQ(0x1000)`; `MLIQ` sample base `+0x26`; decode branch on first sample/mask bit `& 4` | `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs` | wrong geometry | High |

## MDX deltas (only changed/unknown)
| Item | Baseline | 0.7.0.3694 | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|
| Binary magic and section-bound enforcement | Baseline-ish MDX assumptions | Loader hard-requires binary magic and bounded section reads | `0x0018eb2c` (`FUN_0018eb2c`): requires `0x584C444D` (`MDLX`), rejects otherwise (`"File is not a binary model file."`); enforces `sectionLen <= bytesRemaining` and aborts on overrun | `src/MdxViewer/Formats/Mdx/*`, `src/MdxViewer/Rendering/MdxAnimator.cs` | hard parse fail | High |
| MDX top-level section dispatch map | Baseline-ish assumptions were under-specified | Explicit section-dispatch map is present and unknown sections are safely skipped | `0x0018eec8` `MDL::CallBinReadHandler`: explicit token map (`VERS`, `MODL`, `SEQS`, `GEOS`, `MTLS`, `TEXS`, `GLBS`, `LITE`, `BONE`, `HELP`, `ATCH`, `PIVT`, `PREM`, `PRE2`, `RIBB`, `EVTS`, `CAMS`, `GEOA`, `TEXA`, `CLID`, `HTST`) and warning+skip for unknown section tags | `src/MdxViewer/Formats/Mdx/*` | hard parse fail | High |
| Geoset/texture strictness details | Baseline-ish, not fully enumerated in prior pass | Strict size/shape checks are confirmed for key sections | `0x00190d04` `ReadBinTextures`: requires `sectionBytes % 0x10C == 0`; `0x0016e24c`: enforces `VRTX`, `NRMS`, and `numVertices == numNormals` with `UVAS` lane checks; `0x0018f404` `ReadBinModelGlobals`: requires `MODL` section size `0x175` | `src/MdxViewer/Formats/Mdx/*`, `src/MdxViewer/Rendering/MdxAnimator.cs` | visual artifact | High |

## Required profile edits
- Keep `AdtProfile_060_070_Baseline` as family baseline, but ensure strict flags are explicit in profile fields (not implicit behavior):
  - `UseMhdrOffsetsOnly = true`
  - required MCNK subchunk gate list and header offset map include `MCLQ=0x60`, `MCSE=0x58`
  - `MclqLayerStride = 0x2D4`
  - `MclqTileFlagsOffset = 0x290`
- Ensure `WmoProfile_060_070_Baseline` encodes strict root/group order and divisors shown at `0x002891fc` and `0x0028fcac`.
- Ensure `WmoProfile_060_070_Baseline` optional gate map includes `0x1/0x4/0x200/0x400/0x800/0x1000` and `MLIQ` mode branch behavior.
- Add/confirm MDX profile guardrails:
  - `RequiresMdlxMagic = true`
  - section length bound checks enabled in strict mode.
- Add/confirm MDX dispatcher token map (from `CallBinReadHandler`) and retain warning+skip behavior for unknown sections.

## Implementation targets
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Rendering/MdxAnimator.cs`

## Open unknowns
- Confirm whether `MH2O` participates in native ADT path for this exact build or remains absent from active terrain loader chain. Current evidence: no `MH2O` token strings and no `MH2O`-named parser anchors in indexed function set; active terrain chain is `CMapArea::Create -> CMapChunk::CreatePtrs/CreateLiquids` with `MCLQ` usage only in captured paths.