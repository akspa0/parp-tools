# Research: Modern Client Assets

Survey date: 2026-09-16. Every item stays open until the Phase 0 coverage survey measures it on
real tier builds.

## Findings (measured)

| Finding | Evidence |
|---|---|
| No FileDataID resolution anywhere in core/viewer | 0 files match `MAID`/`SFID`/`TXID`/`GFID`/`MODI`/`FileDataId` under `src/core`, `src/viewer/WoWViewer` |
| DBCD reads WDB2–WDC5 | `libs/wowdev/DBCD/DBCD.IO/Readers/{WDB2..WDB6,WDC1..WDC5}Reader.cs` |
| Warcraft.NET has era chunk sets | `Files/M2/Chunks/{Legion,BfA,SL,DF,TWW}`, `Files/WMO/Chunks/{WoD,Legion,BfA}`, `Files/WDT/{Root,Light,Occlusion,Fog}`, `Files/ADT/{Terrain,TerrainObject,TerrainTexture,TerrainLOD}` |
| Chunked M2 already walked, but ids not resolved | `Core.IO/M2Chunked/{M2ChunkedChunkWalker,M2ChunkedModelReader,M2ModelReaderDispatcher}.cs` walk MD21 chunks; `WarcraftNetM2Adapter.cs` is the viewer path; `MopAdtChunkParser.cs` explicitly defers MDID/MHID as "later-expansion FileDataID scheme" |

## R1: Tier representative builds

- **Decision pending (operator)**: one build per tier: A (6.x–8.0), B (8.1–8.3), C (9.x+ / current).
- **Criteria**: openable through Spec 238 (local or a mirror), not heavily encrypted in world assets, with a recognizable outdoor zone and a city.

## R2: Survey before readers

- **Decision**: Build the coverage survey (Phase 0) before any reader change.
- **Rationale**: Memory lesson ("verify detector power", "a name stops the looking"). The chunk long tail must be ranked by measured occurrence and render impact, not by wiki completeness.

## R3: Warcraft.NET types vs Core.IO readers

- **Question**: The viewer already uses Warcraft.NET for M2 through an adapter, while the constitution names Core.IO as the canonical format owner.
- **Decision (provisional)**: Core.IO stays the owner of the reading surface. Where a Warcraft.NET type is already the working decode path (M2), Core.IO may wrap it behind its own model, rather than a second parallel reader appearing. It is settled per format in Phase 3 step 3 and documented.
- **Detector**: For a sample of files per tier, Core.IO output and Warcraft.NET output agree field by field. Disagreements are recorded and resolved against the file bytes.

## R4: wow.export as reference

| Path | Use |
|---|---|
| `src/js/3D/loaders/WDTLoader.js`, `ADTLoader.js` | MAID slot order, id-flagged placements, tex0 id tables |
| `src/js/3D/loaders/M2Loader.js`, `M2Generics.js`, `Skin.js`, `SKELLoader.js`, `BONELoader.js`, `ANIMLoader.js` | Chunked M2 id tables, external skeleton/anim resolution |
| `src/js/3D/loaders/WMOLoader.js`, `WMOLegacyLoader.js` | GFID/MODI vs legacy name tables (era split) |
| `src/js/3D/renderers/M2RendererGL.js`, `WMORendererGL.js`, `ShaderMapper.js`, `WMOShaderMapper.js` | Material/shader selection (hand off to Spec 198) |
| `src/js/casc/db2.js`, `dbd-manifest.js` | Table-by-id lookup and definition selection |

wow.export is MIT and a behavioral reference only; no JS is ported verbatim.

## R5: Placement file-id flags

- **Hypothesis**: MDDF flag 0x40 and MODF flag 0x8 mean "the name index is a FileDataID" (per wowdev.wiki).
- **Measure**: On a tier B build, flagged indices resolve as ids through the build (hit rate ≈100%), while the same values treated as name indices fail. That comparison is the power check.

## R6: DB2 definition selection

- **Question**: How DBCD/WoWDBDefs pick a definition for an exact build (build ranges vs layout hash), and whether the viewer's existing `IDBCProvider` path can take a table by file id.
- **Measure**: Per tier build, the table load success rate for Map/AreaTable/Light*/Liquid* tables.

## R7: Overlap with Spec 197 (height texturing)

- Spec 197 plans `MHID`/`MDID`/`MCXH` height blending from 5.0.1 Ghidra work. **Decision**: 239 reuses 197's blend implementation if it has landed. Otherwise 239 Phase 2 step 5 implements it and 197 cross-references it. Either way, only one blend implementation is kept.

## R8: Encryption impact

- **Measure**: The coverage survey counts `KeyUnavailable` results per format, so encrypted-asset gaps are attributed rather than mistaken for reader bugs (SC-002 attribution).
