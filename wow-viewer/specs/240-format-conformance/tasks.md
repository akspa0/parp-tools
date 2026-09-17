# Tasks: Format Conformance Pass

**Input**: `specs/240-format-conformance/` (spec, plan, research)
**Release**: v0.6 · **Branch**: `v0.5.4-dev` · **Measurement build**: `wow_classic_beta` 1.60.1.69876
**Tests**: included (synthetic fixtures per new field; survey counts as integration evidence)

Format: `- [ ] T### [P?] [US?] description (path)`. All paths are relative to `wow-viewer/`.

## Already done (commits d9fcbca1..251026d7)

- [x] T000a MOBA `material_id_large` honoured; batch material ids 16-bit end to end (src/core/WowViewer.Core.IO/Converters/WmoV17ToV14Converter.cs, WmoV14ToV17Converter.cs)
- [x] T000b GFID sliced per LOD, LOD 0 loaded and prefetched (WmoV17ToV14Converter.ReadGroupFileDataIds, src/viewer/WoWViewer/DataSources/CascDataSource.cs)
- [x] T000c Groups without MOBA / MOPL-only parse; failed WMOs negative-cached (WmoV17ToV14Converter.cs, src/viewer/WoWViewer/Terrain/WorldAssetManager.cs)
- [x] T000d Shader-23 base texture from texture_2 (`WmoMaterial.BaseTextureName`)
- [x] T000e Tile terrain renderer draws up to 8 layers (src/viewer/WoWViewer/Terrain/TerrainTileMeshBuilder.cs, TerrainRenderer.cs)
- [x] T000f `inspect casc wmo-survey` (tools/inspect/WowViewer.Tool.Inspect/CascCommandSupport.cs)

## Phase 1: Setup

- [ ] T001 Create evidence folder with receipt format (build identity, command, counts) in specs/240-format-conformance/evidence/README.md
- [ ] T002 [P] Rerun `inspect casc wmo-survey` on 1.60.1.69876 and save the output as specs/240-format-conformance/evidence/wmo-survey-2026-09-17.txt (research.md R2 quotes the post-fix run; it was not saved to disk)

## Phase 2: Foundational — survey library (US1, blocks all fixes)

- [ ] T003 [US1] `ChunkInventory` walker with known/skipped/unknown classification and per-format registries in src/core/WowViewer.Core.IO/Survey/ChunkInventory.cs
- [ ] T004 [US1] `ConformanceReport` model (build identity, per-format inventory, field tallies, outcomes) + JSON writer in src/core/WowViewer.Core.IO/Survey/ConformanceReport.cs
- [ ] T005 [P] [US1] Move WMO tallies (parse outcomes, material shapes, MOBA large ids, failing layouts) from CascCommandSupport into src/core/WowViewer.Core.IO/Survey/WmoConformanceSurvey.cs; add MOTV/MOCV set counts, MOHD flags, split-group header values, MOMX and MGI2 raw value distributions
- [ ] T006 [P] [US1] ADT tallies (MCNK flags incl. 0x10000, nLayers histogram, MCLY flag bits, MTXP/MHID/MTXF presence, MODF flag 0x80, MWDR/MWDS presence, `_lod.adt` presence) in src/core/WowViewer.Core.IO/Survey/AdtConformanceSurvey.cs
- [ ] T007 [P] [US1] M2 tallies via Warcraft.NET's parsed model (chunk inventory, SKID/BFID/AFID/LDV1/RPID/GPID presence and resolution, 0-section models) in src/core/WowViewer.Core.IO/Survey/M2ConformanceSurvey.cs
- [ ] T008 [P] [US1] BLP tallies (colour encoding × pixel format × alpha depth × size) in src/core/WowViewer.Core.IO/Survey/BlpConformanceSurvey.cs
- [ ] T009 [P] [US1] WDT tallies (MPHD flags, FileDataID fields, companion WDT presence) in src/core/WowViewer.Core.IO/Survey/WdtConformanceSurvey.cs
- [ ] T010 [US1] CLI `inspect casc survey --format wmo|adt|m2|blp|wdt|all [--map <wdt fdid>] [--limit n] [--out <dir>]` in tools/inspect/WowViewer.Tool.Inspect/CascCommandSupport.cs; keep `wmo-survey` as an alias
- [ ] T011 [P] [US1] Unit tests over synthetic chunk buffers for each tally in tests/WowViewer.Core.Tests/ConformanceSurveyTests.cs
- [ ] T012 [US1] SC-001: reproduce research.md R2 counts; record full-product report in specs/240-format-conformance/evidence/survey-baseline-1.60.1.md and update research.md unmeasured rows (high_res_holes, BLP formats, companions)
- [ ] T013 [US1] Document the survey command in docs/CLI-TOOLS.md and tools/README.md

## Phase 3: WMO correctness (US2)

- [ ] T014 [US2] Keep MPY2 material ids above 0xFE (16-bit face materials) in src/core/WowViewer.Core.IO/Converters/WmoV17ToV14Converter.cs and WmoV14ToV17Converter.WmoGroupData
- [ ] T015 [US2] Carry second MOTV and MOCV sets (TVERTS2/TVERTS3/CVERTS2) into the converter model in WmoV17ToV14Converter.cs
- [ ] T016 [US2] Measure the shader-23 blend: on Orgrimmar2FrontGate and 10DU_HallUldamanUprez_Main01, correlate texture_2/texture_3/color_2/flags_2 usage with MOCV alpha and UV sets; record in specs/240-format-conformance/evidence/shader23.md
- [ ] T017 [US2] Per-shader material path in src/viewer/WoWViewer/Rendering/WmoRenderer.cs: two-layer family (6, 13, 21 and 23 per T016) blends second texture by MOCV alpha; other ids use an explicit fallback logged once per shader id (FR-005)
- [ ] T018 [P] [US2] Same base/second texture selection in src/core/WowViewer.Core.Renderer/ObjectCapture/WmoObjectRenderer.cs and the GLB exporters (src/viewer/WoWViewer/Export/)
- [ ] T019 [US2] Read split-group parent/child indices (MOGP 0x40/0x42) and make children reachable through parent portals in WmoRenderer visibility
- [ ] T020 [US2] Apply MOHD `do_not_fix_vertex_color_alpha` in the vertex light build (WmoRenderer.BuildVertexLightColors)
- [ ] T021 [US2] SC-003 captures of 11DL_Dalaran, Orgrimmar2FrontGate, GoldshireInn before/after in specs/240-format-conformance/evidence/wmo-captures.md (operator witness)

## Phase 4: Terrain fields (US3)

- [ ] T022 [US3] Read `holes_high_res` (uint64) when MCNK flag 0x10000 is set in src/core/WowViewer.Core.IO/Lk/Mcnk.cs; extend TerrainChunkData with a 64-bit hole mask
- [ ] T023 [US3] 8×8 hole cells in TerrainTileMeshBuilder.BuildIndices (src/viewer/WoWViewer/Terrain/TerrainTileMeshBuilder.cs) with a synthetic test
- [ ] T024 [US3] MTXP/MHID height blending behind MPHD 0x80 in TerrainRenderer; score against authored minimaps with the synthetic-minimap scorecard before enabling by default
- [ ] T025 [P] [US3] MTXF/MCLY texture scale and MCLY layer animation in the tile shader
- [ ] T026 [P] [US3] MODF flag 0x80 + MWDR/MWDS doodad sets in src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs

## Phase 5: M2 renderer audit (US4)

Warcraft.NET already parses SKID/BFID/AFID/LDV1 and `.skel`; M2 animation works.

- [ ] T027 [US4] Per-chunk consumption audit: which Warcraft.NET-parsed chunks reach src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs and WowViewerM2RuntimeBridge.cs; record in specs/240-format-conformance/evidence/m2-chunk-consumption.md
- [ ] T028 [P] [US4] LDV1-driven skin LOD selection if T027 shows it unused
- [ ] T029 [P] [US4] Investigate the 15 "ok but 0 sections" models from map-survey; record cause in specs/240-format-conformance/evidence/m2-zero-sections.md

## Phase 6: BLP and WDT companions (US5)

- [ ] T032 [US5] If T008 finds pixel formats SereniaBLPLib lacks (BC5 = 11 or others), evaluate BLPSharp (MIT) as replacement; otherwise make unsupported formats report instead of mis-decoding (FR-009)
- [ ] T033 [P] [US5] Read MPHD FileDataID fields (lgt, occ, fogs, mpv, tex, wdl, pd4) in the WDT reader
- [ ] T034 [US5] Decide rendering for `_lgt.wdt`, `_fogs.wdt`, `_lod.adt` from T009/T006 counts; record decision in research.md

## Phase 7: Reference study (FR-011)

- [ ] T035 [P] Study WTL (MIT): CASC product/branch handling, listfile and naming (WoWNamingLib), file linking; record in specs/240-format-conformance/evidence/reference-study.md
- [ ] T036 [P] Study WoWFormatLib (no license — behaviour only): WMO LOD fallback, ADT `_lod` reader, MOBA material struct; record differences from our readers in the same file
- [ ] T036b Before each phase, recheck vendored libraries against upstream (Warcraft.NET, TACTSharp, DBCD, WoWDBDefs, wow-listfile, SereniaBLPLib/BLPSharp) and update research.md R1
- [ ] T037 Propose confirmed wiki corrections (e.g. MCMT layer note vs. 8-layer data, MGI2 layout) in specs/240-format-conformance/evidence/wiki-notes.md

## Polish

- [ ] T038 Rerun the full survey; update research.md states and SC status
- [x] T039 Update specs/STATUS.md v0.6 scope and specs/epics/active-epics.md membership

## Dependencies

Phase 2 (survey) blocks Phases 3–6. Within Phase 3, T016 blocks T017. T022 blocks T023. T027 blocks T028. Phase 7 can run any time.

## MVP

Phase 2 + T014–T017: the survey plus correct WMO materials, since WMOs are the most visible remaining defect.
