# Batch D2 — v0.6 modern data (CASC, FileDataID assets, ADT v22/v23/v26 "DAT", conformance,
modern→legacy conversion, MAI2 liquid flow, chunk survey, modern M2 cameras, DAT export)

Specs audited: 237, 238, 239, 240, 241, 243, 244, 245, 246, 247.
Method: read spec.md/tasks.md for each, cross-checked against `git log`, `docs/releases/v0.6.0-alpha.md`
and `v0.6.0-alpha2.md`, and grepped/read the real code under `src/core/WowViewer.Core.IO`,
`src/viewer/WoWViewer`, `tools/inspect`. Checkboxes were not trusted; every claim below is
grep/read-verified.

---

### 237 ADT v26 — First Reader and Renderer for a Brand-New Terrain Format
- Stated status: Draft ("fast-path wireframe next") | Tasks: 0/~55 checked (all boxes unchecked; known-stale per brief)
- Scope: detect/inventory/decode/render the brand-new DAT v26 tile format (and wiki v22/v23 siblings), from real files with no WDT/listfile.
- Verified implemented:
  - Reader/writer/slicer: `src/core/WowViewer.Core.IO/Maps/AdtAhdrReader.cs` (`IsAhdrFamily`, `TryReadVersion`, `TryReadTileLocation`, `Read`), `AdtAhdrTileSlicer.cs`, `AdtAhdrTileBuilder.cs`, `AdtAhdrWriter.cs`, `AdtAhdrAlpha.cs` (~1000 lines core logic).
  - Viewer: `src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs` (350 lines: tile discovery by ALOC, chunk fill, texture/layer/placement mapping), `src/viewer/WoWViewer/ViewerApp_CascAhdr.cs` (menu wiring).
  - CLI: `tools/inspect/WowViewer.Tool.Inspect/AdtAhdrCommandSupport.cs` implements `adt-ahdr check|objects|roundtrip|export-lk` (verb names diverge from the spec's `inventory`/`dump`/`layout-probe`, but functionally equivalent — see Checkbox accuracy).
  - Tests: `tests/WowViewer.Core.Tests/AdtAhdrObjectAndSliceTests.cs`, `AdtAhdrV23Tests.cs`.
  - Evidence: 9 dated files in `specs/237-adt-v26-terrain/evidence/` (phase0 first-look, v22/v23 real renders, roundtrip, ACDO uniqueIds, AMAP codec attempt).
  - v22 (US4/edge case) load confirmed shipped in v0.6.0-alpha2 release notes ("DAT v22 loads for the first time... v22, v23, v26 — now load and render").
  - Writer round-trips the 700-file v26 corpus byte-identically (release notes, corroborated by `roundtrip` CLI verb existing).
- Partial:
  - FR-014/FR-018 (resolve textures/models via CASC+listfile with placeholder fallback): `AhdrTerrainAdapter.cs` carries texture-name lists (`TileTextures`) but has no direct `Casc`/`FileDataId`/`Listfile` calls in that file — asset resolution appears to ride the generic `IDataSource` plumbing set up by CASC opening (Spec 238), not a special-cased FDID resolve inside the adapter. Not disproven, but not directly evidenced either.
  - Task-level class separation (`AdtAhdrInventoryReader`, `AdtAhdrLayoutProbe`, typed `AdtAhdrDiagnostic` records) was not built as separate files; the equivalent logic lives inside `AdtAhdrReader`/`AdtAhdrCommandSupport` directly. Functional coverage looks equivalent (inventory-like output in `RunCheck`), file-level task IDs (T010, T011–T033 file targets) don't map 1:1.
- Not implemented:
  - v22 `AMAP` alpha-map codec remains uncracked (evidence: `v22-amap-codec-attempt-2026-09-20.md`, "NOT CRACKED"); this is FR-007-adjacent for v22 specifically and is explicitly carried forward into spec 247 US1.
- Checkbox accuracy: N checked-but-absent = 0 (nothing is checked); N unchecked-but-present = high (T045–T052 US0 wireframe, T011–T033 US1/US2 decode, T034–T040 US3 render, most of Setup/Foundational) — essentially the whole spec shipped but is 0/N checked.
- Operator gates owed: real-client load of exported LK/Alpha files (release notes: "no output has been loaded in a 3.3.5 client or in Noggit"); operator visual gates named in tasks (T040, T050) — evidence files exist but should be operator-witnessed, not just automated.
- Open residue (spec-stated only): FR-016 (v22/v23/v26 write/convert is explicitly out of scope for this spec — see spec 247 for the one-way export); v22 AMAP codec (US1 acceptance scenario 2, still open, tracked under spec 247 FR-001/FR-002).
- Superseded by / overlaps: 241 (interchange, deferred), 247 (capture + LK export, carries the open v22 AMAP item forward).
- Disposition: ARCHIVE-COMPLETE (core spec's US0–US4 shipped and evidenced; the one open item — v22 AMAP — is already tracked as residue in spec 247, not orphaned here)
- Proposed epic theme: dat-v22-v23-v26-terrain
- Confidence: high

---

### 238 CASC Data Source (Local Install + Remote CDN)
- Stated status: Draft | Tasks: 0/41 checked (brief confirms this is known-stale; CASC shipped in v0.6.0-alpha2)
- Scope: local CASC install reads by path/id, remote CDN streaming, historical builds, encryption, build traceability, viewer/MPQ non-regression.
- Verified implemented:
  - `src/core/WowViewer.Core.IO/Casc/CascArchiveCatalog.cs`, `CascStorage.cs`, `CommunityListfile.cs`.
  - CLI `tools/inspect/WowViewer.Tool.Inspect/CascCommandSupport.cs`: `casc products|read|exists|wmo|wmo-survey|map-survey|db2|m2|adt-heights|bench` (matches the CLI verbs release notes §4 documents).
  - Viewer: `src/viewer/WoWViewer/ViewerApp_CascAhdr.cs` — "Open CASC Install (local)" and "(local + CDN fill)" flows, product picker, `CascDataSource` wiring into `_dataSource`.
  - Release notes (v0.6.0-alpha) confirm: local install + optional CDN fill, product picker, TACTSharp vendored submodule, tested against WoW: Forever `wow_classic_beta` 1.60.1.69876.
  - `IDataSource` in `src/viewer/WoWViewer/DataSources/IDataSource.cs` carries CASC-aware members (grep hit for MAID confirms id-addressed reads reached the terrain adapter layer too).
- Partial:
  - US2 (stream from CDN without local install) — release notes explicitly say "Remote-only CDN browsing without a local install is not in this build" (Known limitations). Local-first CASC (US1) is what shipped; pure-remote/CDN-only open (T021-T027) is not confirmed in release notes and its evidence files were not checked in this pass (only `specs/238.../research.md`/`quickstart.md` exist per directory listing — no `evidence/` folder was found under 238, unlike 237/239/240).
  - US3 (open by explicit historical build id, not just current) — not confirmed either way in this pass.
  - US4 (encryption / `CascKeyRing`) — not directly grepped this pass; plausible but unverified.
- Not implemented (per release notes' own "Known limitations"): remote-only CDN browsing without a local install.
- Checkbox accuracy: 0/41 checked, but US1 (local install, P1, the MVP) is clearly shipped — so this is "many unchecked-but-present" (at minimum T001–T020, T033–T037 territory).
- Operator gates owed: SC-001/SC-002 (hash-match against an independent extraction tool) and SC-004 (remote-build map load timing) are runtime proofs; no `evidence/` directory exists under `specs/238-casc-data-source/` to confirm they were ever witnessed.
- Open residue (spec-stated only): US2 CDN-only remote streaming (FR-002, explicitly called out as not in the alpha build); US3 historical builds by explicit identity (FR-002 continued); "wider asset matrix beyond wow_classic_beta 1.60.1 is unverified" (release notes, Known limitations, directly echoes SC-003's per-era-tier requirement).
- Superseded by / overlaps: 239 (consumes this), 237 (first consumer).
- Disposition: FOLD (US1 complete and load-bearing for 237/239; CDN-only remote and historical-build-by-id residue folds into the modern-data epic)
- Proposed epic theme: modern-data-casc-foundation
- Confidence: medium (no `evidence/` folder found for this spec, so remote/CDN claims rely on release-notes text rather than a receipt)

---

### 239 Modern Client Assets (Post-5.0.1, FileDataID Era)
- Stated status: Draft | Tasks: 0/36 checked
- Scope: FileDataID-era WDT/ADT/M2/WMO reading, DB2-by-id, coverage survey, on top of Spec 238.
- Verified implemented:
  - MAID (WDT tile ids): `StandardTerrainAdapter.cs`, `CascArchiveCatalog.cs`, `IDataSource.cs`.
  - SFID/TXID (chunked M2 skin/texture ids): `CascDataSource.cs`, `WowViewerM2RuntimeBridge.cs`, `M2ChunkedFileIds.cs`, `WarcraftNetM2Adapter.cs`, `FileDataIdPaths.cs`.
  - GFID/MODI (WMO group/doodad ids): `WmoV17ToV14Converter.cs`, `CascDataSource.cs`, `WorldAssetManager.cs`, `FileDataIdPaths.cs`.
  - MDID/MHID/MTXP (texture ids + height-texturing params): `StandardTerrainAdapter.cs`, `AdtTextureReader.cs`, `MopAdtChunkParser.cs`, `Mcnk.cs`, `WowFileDetector.cs`.
  - DB2-by-id: `casc db2` CLI verb in `CascCommandSupport.cs` (`RunDb2`, comment: "Spec 239: decodes a DB2 table from CASC through DBCD + WoWDBDefs"); viewer-side DB2/table consumption in `LightService.cs`, `AreaTableService.cs`, `MapDiscoveryService.cs`.
  - Release notes (v0.6.0-alpha) §2 confirm all of the above shipped and tested against `wow_classic_beta` 1.60.1.69876: 8-layer terrain, MAID/MDID map loading, MD21 native path, WMO GFID/MOBA 16-bit materials.
- Partial:
  - FR-001 (one shared `FileReferenceResolver`): the class named in tasks.md (T003/T004, `src/core/WowViewer.Core.IO/Files/FileReferenceResolver.cs`) does **not exist** — grep found zero hits outside the spec's own tasks.md/plan.md. Resolution instead happens ad hoc inside each reader/adapter (StandardTerrainAdapter, CascDataSource, etc.), which is a real but unconsolidated implementation of the same intent.
  - FR-008 (coverage survey command, `AssetCoverageSurvey`/`asset-survey` CLI): also **not found in code** — zero hits outside tasks.md/plan.md. The closer analogue that does exist is `casc wmo-survey`/`casc map-survey` (pre-dating/adjacent to this spec, formally owned by 240).
- Not implemented:
  - US4 coverage survey (T007–T010) — no `AssetCoverageSurvey` class, no `asset-survey` CLI verb.
  - Central `FileReferenceResolver` (Foundational T003–T005) — resolution is real but scattered, not the single resolver FR-001 requires.
- Checkbox accuracy: 0/36 checked; unchecked-but-present is large (US1 terrain, US2 doodads/WMOs, US3 DB2 world context are all substantially shipped per release notes + grep); unchecked-and-genuinely-absent = T003–T010 (resolver + survey).
- Operator gates owed: SC-001/SC-002/SC-005 (per-tier-build render/placement percentages, eyeball comparison against in-game/minimap reference) — release notes only claim one tier (`wow_classic_beta` 1.60.1) was tested; tiers A/B (6.x–8.3) are unconfirmed ("the wider asset matrix beyond wow_classic_beta 1.60.1 is unverified").
- Open residue (spec-stated only): FR-001 single file-reference resolver (currently scattered, not consolidated); FR-008 coverage survey command (US4, not built); tier A/B builds (SC-001/SC-003) never surveyed — only tier C (`wow_classic_beta`) is proven.
- Superseded by / overlaps: 240 (its own conformance-survey library, `casc survey`, duplicates/extends the intended asset-survey scope and is itself also unbuilt — see 240 below).
- Disposition: FOLD (terrain/doodad/DB2 reading substantially shipped and load-bearing; resolver-consolidation and coverage-survey residue folds into the modern-data conformance epic alongside 240)
- Proposed epic theme: modern-data-fileDataId-readers
- Confidence: high

---

### 240 Format Conformance Pass (WMO, ADT, M2, BLP, WDT)
- Stated status: Draft | Tasks: several pre-marked `[x]` "Already done" + rest `[ ]`
- Scope: turn the ad hoc WMO/ADT/M2 audit that already found and fixed 4 real defects into a repeatable survey-driven conformance loop.
- Verified implemented (the "Already done" block, T000a–T000f, confirmed real):
  - T000a MOBA 16-bit `material_id_large`: `WmoV17ToV14Converter.cs` — comment "Legion+ SMOBatch: bytes 0x0A-0x0B hold material_id_large" and MPY2 (uint16 flags/materialId) handling both present.
  - T000f `inspect casc wmo-survey`/`map-survey`: both CLI verbs exist and are documented in `CascCommandSupport.cs` (with Spec 239 attribution comment on map-survey).
  - Release notes (v0.6.0-alpha, commits d9fcbca1..251026d7) independently corroborate: "8 terrain layers, modern WMO materials and LOD groups" shipped.
- Not implemented (this is the dominant finding for this spec):
  - **No `src/core/WowViewer.Core.IO/Survey/` directory exists at all** — `ChunkInventory.cs`, `ConformanceReport.cs`, `WmoConformanceSurvey.cs`, `AdtConformanceSurvey.cs`, `M2ConformanceSurvey.cs`, `BlpConformanceSurvey.cs`, `WdtConformanceSurvey.cs` (T003–T013, the whole Phase 2 foundational survey library and its unifying `casc survey --format` CLI verb) are all absent.
  - **No `specs/240-format-conformance/evidence/` directory exists** (confirmed via directory listing — only plan.md/research.md/spec.md/tasks.md present). Every phase 3–7 receipt task (T012, T021, T026, T027–T029, T032–T037) is unwitnessed.
  - High-res holes (`holes_high_res`/MCNK flag `0x10000`): grepped `Mcnk.cs` directly — zero matches. T022/T023 (US3 terrain fields) not done.
  - Shader-per-material WMO rendering (two-layer shader 6/13/21/23 blending, split-group portal visibility, `do_not_fix_vertex_color_alpha`): grepped `WmoRenderer.cs` — no matches for any of these terms. T014–T020 (US2, the spec's second-highest-priority story after the survey) not done.
  - M2 chunk-consumption audit (T027), BLP pixel-format coverage decision (T032), WDT companion handling (T033/T034), reference-study writeups (T035/T036) — no evidence found.
- Checkbox accuracy: accurate for the pre-marked block (T000a–T000f, T039 all real); everything else correctly left unchecked and correctly absent — this is one of the few specs whose checkbox state matches reality.
- Operator gates owed: SC-003 (side-by-side WMO captures vs in-game reference) — not reached, since the WMO shader work it depends on isn't built.
- Open residue (spec-stated only): FR-002/FR-003 conformance survey + per-fix before/after counts (US1, Phase 2, T003–T013); FR-005/FR-006 WMO shader-correct materials + split-group portals (US2, T014–T020); FR-007 high-res holes + MTXP height blending (US3, T022–T026); M2 chunk-consumption audit (US4, T027–T029); BLP/WDT companion decisions (US5, T032–T034); wiki-correction writeup (T037).
- Superseded by / overlaps: 239 (US4 coverage-survey intent overlaps directly with this spec's Phase 2 survey library — the two specs describe the same missing capability from two angles and should be reconciled into one survey component, not built twice).
- Disposition: FOLD (small "already done" core is real and load-bearing; the survey library plus WMO/terrain/M2/BLP conformance work is substantial, real, unstarted residue — this is the biggest genuine gap in the batch)
- Proposed epic theme: modern-data-conformance-survey
- Confidence: high

---

### 241 DAT v26 as the Project Interchange Format
- Stated status: **Deferred (operator, 2026-09-17)** — spec's own header states "not pursued... kept as measured facts" | Tasks: 3 pre-marked "Already done" (from spec 237), 0/10 remaining checked
- Scope: make DAT v26 the save/edit/interchange format (lossless save, DAT→ADT build targets, harvest source).
- Verified implemented: only the "Already done" items, which are actually spec-237 deliverables re-cited here (byte-identical rewrite, ADT→DAT builder, weight↔alpha conversion) — already covered under 237 above.
- Not implemented: everything spec-241-specific — save-as-DAT editor path (US1), DAT→ADT round-trip report and Spec-234-writer integration (US2, superseded — see below), harvest-from-DAT (US3). None of these have any code; the spec's own status line says so.
- Checkbox accuracy: accurate (only pre-existing 237 work is checked; nothing else is, and nothing else exists).
- Operator gates owed: none pending — the operator already made the decision (defer) that resolves this spec's open questions.
- Open residue (spec-stated only): explicitly none intended by the operator; the spec itself says DAT stays "a format to read and study, not a save or interchange format," and directs future one-way work to spec 247 ("This spec stays deferred; one-way DAT->LK ADT export lives in 247").
- Superseded by / overlaps: 247 (the one-way DAT→LK export that operator redirected this spec's US2 intent into).
- Disposition: ARCHIVE-COLD (operator-deferred, no live residue — its only forward-looking pointer already lands in 247)
- Proposed epic theme: n/a (cold)
- Confidence: high

---

### 243 Modern-to-Legacy Map Conversion (multi-layer alpha merge, LK + Alpha outputs)
- Stated status: Draft (operator-directed 2026-09-18, "high priority — it's important and has been overlooked for too long") | Tasks: **no tasks.md exists** for this spec
- Scope: one-way modern (CASC/FileDataID) → legacy (LK v18 ADT/WDT, Alpha 0.5.3 WDT) map converter with multi-layer alpha merge, batch mode, low-touch UI, optional asset bundling.
- Verified implemented: none. Grepped `src/` and `tools/` for `ModernToLegacy`, `LayerMerge`, `MergePolicy` and modern-legacy-conversion terms — zero hits anywhere in code. `specs/243.../plan.md`, `data-model.md`, `contracts/` (cli-convert-map.md, merge-report.schema.json, service-api.md), `quickstart.md`, `checklists/requirements.md` all exist as design artifacts only; no `tasks.md`, no `evidence/` directory.
- Partial: none — this is pure unstarted design.
- Not implemented: all of FR-001 through FR-010 (the whole spec) — no converter route, no layer-merge service, no batch UI, no asset-bundling manifest.
- Checkbox accuracy: n/a (no tasks.md to check).
- Operator gates owed: all of SC-001–SC-006 are unreached (nothing built to gate).
- Open residue (spec-stated only): the entire spec is residue — FR-001 (modern→LK/Alpha route), FR-002/FR-003 (deterministic layer merge + per-tile report), FR-004 (batch mode), FR-005 (low-touch UI), FR-006 (provenance/no-overwrite), FR-007 (asset bundling), FR-008 (route validation before writing).
- Superseded by / overlaps: 244 (liquid-flow conversion is explicitly a "consumer" of this spec's LK/Alpha targets), 245 (its chunk-completeness findings are explicit inputs to this spec's merge policy — "Findings MUST be handed to Spec 243... as dated pointers or amendments").
- Disposition: KEEP-ACTIVE (operator explicitly flagged this as overlooked-and-important, plan/contracts already exist, but zero code — too large and undecided to silently fold; needs an explicit go/no-go and a tasks.md before work starts)
- Proposed epic theme: modern-to-legacy-conversion
- Confidence: high

---

### 244 Modern Liquid Directional Flow (WDT MAI2 liquidFlowTexture)
- Stated status: Draft (operator-directed 2026-09-18) | Tasks: **no tasks.md exists**
- Scope: decode WDT `MAI2.liquidFlowTexture` (R/G channel flow encoding) and surface it as viewer UI context plus a shared datum for non-UI consumers and the modern→legacy converter (243).
- Verified implemented: none. Grepped `src/` for `MAI2`, `liquidFlowTexture`, `FlowVector`, `FlowTexture` (case-insensitive) — the only hit anywhere is `MclqChunk.cs`'s pre-existing **legacy** Alpha `MclqFlowVector`, which is explicitly cited in the spec's own Context section as the *existing* legacy-side home, not new work. No MAI2 reader, no flow decode, no UI surface.
- Partial: none.
- Not implemented: all of FR-001 through FR-010 — MAI2 chunk reading, flow decode/normalize, UI liquid-cell inspection surfacing, shared flow-datum service, spec-243 conversion integration.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: all SC-001–SC-005 unreached.
- Open residue (spec-stated only): the entire spec — FR-001 (MAI2 reader), FR-003 (flow decode), FR-004 (UI context surfacing), FR-006 (shared source for UI + non-UI consumers), FR-010 (conversion-report integration with 243).
- Superseded by / overlaps: 245 (explicitly the spec that resolves MAI2's 7 unknown fields — "out of scope here," deferred to 245's completeness survey), 243 (the conversion consumer).
- Disposition: KEEP-ACTIVE (small, well-scoped, explicitly operator-directed follow-on; not started, folds naturally once 245's chunk survey exists, but distinct enough to track on its own until then)
- Proposed epic theme: modern-liquid-flow
- Confidence: high

---

### 245 Modern Chunk Completeness Survey & Legacy Build-In Feasibility
- Stated status: Draft (operator-directed 2026-09-18) | Tasks: **no tasks.md exists** ("Whether unread modern chunks can be re-expressed is an outcome of this survey" — spec itself notes no tasks generated yet)
- Scope: one authoritative inventory of every modern WDT/ADT/tex0/obj0/lod chunk, its current handling, and a per-chunk legacy-feasibility verdict (representable via alpha-mask/texture-id re-expression, or not, or unknown) for spec 243's converter to consume.
- Verified implemented: the survey deliverable itself (the inventory table) does not exist. One evidence file does exist and is directly relevant: `specs/245-modern-chunk-completeness-survey/evidence/modern-write-support-state-2026-09-20.md` — a real, measured (not merely claimed) document that: (a) confirms `MopSplitAdt` writer target is declared-but-`HasWriter=false` with zero consumers outside `MapConversionFormat.cs`; (b) confirms `MopAdtChunkParser` has exactly one public method (`ParseMtxpChunk`); (c) confirms `AdtRawChunkBlobCollector` captures unknown chunks for the ML/dataset path only, never for a writer; (d) explicitly states in its own "Not claimed" section: **"no chunk inventory was performed"** — i.e. this document is scoping/context for 245, not 245's own FR-001/FR-002 deliverable.
- Partial: the write-support-state note is a legitimate partial input to the survey (it establishes what modern *writing* currently doesn't do), but it is not the chunk-by-chunk read/ignore/feasibility inventory FR-001–FR-006 require.
- Not implemented: FR-001 (per-chunk inventory across WDT/_occ/_lgt/root ADT/_tex0/_obj0/_lod), FR-002 (name/family/count/handling/code-reference per chunk), FR-003 (documented-meaning + confidence label per ignored chunk), FR-004 (disposition per chunk), FR-005/FR-006 (legacy feasibility verdict per target, including alpha-mask/texture-id re-expression mechanisms).
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none reachable yet — the survey itself is the prerequisite gate for everything downstream (243/244).
- Open residue (spec-stated only): the entire survey — FR-001 through FR-009, explicitly named by the spec as blocking spec 243/244's downstream decisions ("Findings MUST be handed to Spec 243/244... never as duplicated implementation").
- Superseded by / overlaps: 240 ("Depends on: Spec 240... this continues it" — 240's own unbuilt Survey library, see above, is the natural implementation vehicle for 245's inventory; the two should very likely be a single conformance/completeness survey component, not two).
- Disposition: KEEP-ACTIVE (research/doc-lane spec, zero runtime risk, but its one piece of real output — the write-support-state evidence note — is genuinely useful and should not be lost; the inventory itself is real unstarted work blocking 243/244)
- Proposed epic theme: modern-data-conformance-survey (same epic as 240 — these two specs describe one missing component from two directions)
- Confidence: high

---

### 246 Modern M2 Camera Paths and Modern-Data Renderer Benchmarking
- Stated status: Draft (operator-directed 2026-09-18) | Tasks: **no tasks.md exists**
- Scope: (1) find and fix the loss point that keeps modern chunked-MD21 M2 camera tracks from reaching the existing camera-path importer/overlay-builder; (2) build a path-driven renderer benchmark for modern (CASC) maps with the same receipt shape as the existing legacy/marketing-capture benchmark.
- Verified implemented:
  - The **legacy** machinery this spec would extend is real and present: `M2CameraPath.cs`, `M2CameraPathOverlayBuilder.cs` (`CameraCount > 0` gate confirmed), `ViewerApp_CameraPaths.cs`.
  - Chunked-MD21 detection/conversion exists generally (`WarcraftNetM2Adapter.cs`: `IsMd21`, `Md21Magic`, `AllowMd21Container` profile gating) but this is generic modern-M2 plumbing (Spec 239), not camera-specific work from this spec.
- Not implemented:
  - FR-002 (the required first step: prove *where* modern cameras are lost — era dispatch vs. chunked conversion vs. nowhere) — no investigation note or evidence file exists under `specs/246-.../` beyond `checklists/requirements.md`.
  - FR-001/FR-003/FR-004 (modern camera import working end to end, coordinate-space resolution) — no modern-camera-specific code found in `M2CameraPathOverlayBuilder.cs` or the WarcraftNet M2 bridge.
  - FR-005–FR-007 (modern-data path-driven benchmark, same-shape receipt as legacy) — grepped for "modern bench"/"ModernBenchmark" across `src/`; the only hit is an unrelated UI tooltip string ("modern tabbed workbench") in `ViewerApp_Settings.cs`. `inspect casc bench` (mentioned in the spec's own Context as the nearest existing thing) measures data reads, not renderer frames, exactly as the spec describes as the gap.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: SC-001 (operator visual witness of a modern camera path playing) and SC-003 (real benchmark receipt on `wow_classic_beta`) are both unreached — nothing to gate.
- Open residue (spec-stated only): the entire spec — FR-002's required loss-point investigation is the blocking first step for everything else; FR-005 modern benchmark is separately blocked on nothing (it doesn't depend on the camera fix) but is equally unbuilt.
- Superseded by / overlaps: 242 (WMO instancing performance) is stated as depending on this spec's benchmark as its measurement vehicle ("whichever lands first, the other depends on it") — 246's benchmark deliverable is a shared dependency, not specific to 246 alone.
- Disposition: KEEP-ACTIVE (two independent, well-scoped, operator-directed asks; the benchmark half is a real blocker for spec 242's performance work and should not be silently folded away)
- Proposed epic theme: modern-data-camera-and-benchmark
- Confidence: high

---

### 247 DAT Capture & LK ADT Export
- Stated status: **self-reported and accurate** — spec header states "US3 DELIVERED 2026-09-20... US5 CODE COMPLETE 2026-09-20... unwitnessed in the viewer. US1 (v22 AMAP codec) attempted and OPEN. US2 (capture) not started." | Tasks: no tasks.md (spec.md + checklists + evidence only)
- Scope: (US1) crack v22 AMAP alpha codec; (US2) capture real rendered top-down images of loaded DAT terrain (not synthesized); (US3) export DAT folders as LK v18 ADT/WDT with a loss manifest; (US4) manifest completeness; (US5) DAT folders as Cartography layers.
- Verified implemented:
  - US3: `adt-ahdr export-lk` CLI verb confirmed in `AdtAhdrCommandSupport.cs` (`--root/--out/--map/--format lk|alpha|lk+alpha/--transpose`), matching v0.6.0-alpha2 release notes §4 exactly (measured tile/chunk/MCAL/area-id/placement counts for v22/v23/v26). Evidence: `specs/247.../evidence/us3-dat-to-lk-adt-2026-09-20.md`, `lk-writer-missing-chunks-2026-09-20.md`.
  - US5: DAT folders as Cartography layers — matches v0.6.0-alpha2 release notes §5 ("A folder of DAT files can be added as a Cartography layer..."), which itself states "*Not verified: no DAT layer has been composed on screen*" — i.e. code-complete, operator visual gate outstanding, exactly as the spec's own status line says. Evidence: `us5-dat-as-cartography-layer-2026-09-20.md`.
  - The LK-ADT-writer defects fixed in v0.6.0-alpha2 (MCSE, ofsMCCV, MCNK flag 0x40, ofsMCLV, MFBO, MTXF — release notes §2) are the direct enabling fix for this spec's US3/FR-010–FR-015 (a correct, format-complete LK writer is a precondition for a truthful export).
- Partial:
  - US2 (capture): grepped for DAT-specific capture/minimap code — none found (the only hits were false positives on "Vali-**dat**-ion"). Confirmed genuinely not started, matching the spec's own header.
- Not implemented:
  - US1 (v22 AMAP codec, FR-001/FR-002, blocking): evidence file `v22-amap-codec-attempt-2026-09-20.md` (also cited under 237) confirms attempt 1 exhausted 32 variants and remains open. This blocks faithful v22 capture/export per FR-008/SC-005, though FR-003 (blend what alpha exists rather than dropping all layers when one lacks alpha) is stated as independent and not gated on it.
  - US2 real-rendered capture (FR-005–FR-009): not started.
- Checkbox accuracy: n/a (no tasks.md; the spec's own prose status line is the tracking mechanism here and it is accurate against the code).
- Operator gates owed: US5's Cartography-layer composition has never been visually witnessed on screen (release notes' own words); US3's exported LK files have never been loaded in a real 3.3.5 client or Noggit (release notes §2, "not verified").
- Open residue (spec-stated only): US1 v22 AMAP codec (FR-001, open research, explicitly allowed to stay unresolved per the spec's own Risks section — "if FR-001 stalls, the operator decides whether to proceed with v22 marked lossy"); US2 real-rendered capture (FR-005–FR-009, not started); operator witness gates for US3 (real-client load) and US5 (on-screen composition).
- Superseded by / overlaps: 237 (shares the v22 AMAP blocker and the reader/slicer foundation), 241 (this spec is explicitly the narrow one-way case 241 deliberately excluded).
- Disposition: FOLD (US3/US5 code-complete and load-bearing; US1/US2 residue is small, explicit, and already self-tracked accurately in the spec's own header — low risk to carry forward as a short residue list rather than a full spec)
- Proposed epic theme: dat-v22-v23-v26-terrain (same epic as 237 — this spec is 237's direct sequel)
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 237 | ARCHIVE-COMPLETE | 1 (v22 AMAP codec, tracked in 247) | dat-v22-v23-v26-terrain |
| 238 | FOLD | 2 (CDN-only remote streaming; historical builds by id) | modern-data-casc-foundation |
| 239 | FOLD | 2 (central FileReferenceResolver; coverage-survey command) | modern-data-fileDataId-readers |
| 240 | FOLD | 6 (survey library; WMO shader/split-group; terrain high-res-holes/MTXP; M2 audit; BLP/WDT companions; wiki corrections) | modern-data-conformance-survey |
| 241 | ARCHIVE-COLD | 0 (operator-deferred, forward pointer already lands in 247) | n/a |
| 243 | KEEP-ACTIVE | 8 (entire spec: FR-001–FR-008) | modern-to-legacy-conversion |
| 244 | KEEP-ACTIVE | 5 (entire spec: FR-001,003,004,006,010) | modern-liquid-flow |
| 245 | KEEP-ACTIVE | 6 (entire inventory: FR-001–FR-006) | modern-data-conformance-survey |
| 246 | KEEP-ACTIVE | 2 (camera loss-point fix; modern-data benchmark) | modern-data-camera-and-benchmark |
| 247 | FOLD | 3 (v22 AMAP codec; US2 capture; operator witness gates) | dat-v22-v23-v26-terrain |
