# Batch B audit — PM4/PD4 navmesh decoding, matching, generation, server data/museum sim

Specs: 046, 065, 128, 129, 130, 149, 184, 185, 186, 187, 188, 189.
Epic context: `wow-viewer/specs/epic-pm4-restoration/epic.md` (covers 128/129/130 as a stacked chain).

---

### 046 PM4 Asset Matching
- Stated status: Active, "MAJOR BREAKTHROUGH" (fingerprint identity). Tasks: ~30 checked / 41 total
  (Phases 1-6, 8 mostly done; Phase 7 fix-the-pipeline mostly unchecked).
- Scope: fix coordinate-mismatch + segment-granularity bugs in the scalar match pipeline; build a
  Python mirror (data-harvester) for validation; `pm4 correlate-models`/`fingerprint-scan` research.
- Verified implemented: `Pm4ObjectSegmentBuilder`, `Pm4SegmentSignalExtractor`, `Pm4AssetMatchScorer`,
  `Pm4ReplacementPlacementSynthesizer`, `Pm4SegmentExportService` all present in
  `src/core/WowViewer.Core.PM4/Matching/`. CLI verbs `match-assets`, `match-report`, `correlate-models`
  confirmed in `tools/inspect/WowViewer.Tool.Inspect/Program.cs`. Python mirror at
  `data-harvester/src/harvester/pm4_asset_matching/` (models.py, scorer.py, signal_store.py,
  placement_synthesizer.py, json_import.py) with `test_pm4_asset_matching.py`.
- Partial: Phase 7 (T022-T027, coordinate-fix validation, CK24-grouped scoring, identity table) mostly
  unchecked — spec's own text says the *approach* was superseded by `correlate-models`/surface
  correlation (spec 065) rather than these tasks being finished as written.
- Not implemented: T020/T021 (`match-report` corpus run + real-3.3.5-data verification) unchecked;
  T031-T035 (expand correlation dataset, CK24 0x3E/0x3F/0xC0-0xC2 investigation, M2 collision fix,
  material-mapping doc) unchecked.
- Checkbox accuracy: accurate — unchecked items genuinely absent/unfinished; checked items' code
  verified present.
- Operator gates owed: none of the checked work claims runtime/visual proof beyond CLI output; no
  additional gate beyond what's already stated.
- Open residue (spec-stated only): T020/T021 match-report corpus verification; T031-T035 correlation
  expansion and CK24 type investigation — but the spec itself documents this scalar-scoring approach as
  superseded by spec 065's surface-triangle correlation (see 065 "Legacy / Abandoned Approaches").
- Superseded by / overlaps: 065 (surface-triangle correlation explicitly supersedes hull/footprint and
  treats `pm4 correlate-models`/`match-assets` as validation-only, not primary).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: pm4-asset-identity
- Confidence: high

---

### 065 PM4 Correlation to World Assets & Generator
- Stated status: Active (revised 2026-06-17). Tasks: ~33 checked / ~39 total (Phases 1-7 done; Phase
  8-9 mostly open).
- Scope: build WMO+PM4 surface-triangle fingerprint DBs, match via histogram intersection/F1, validate
  against ADT ground truth, then recover placements.
- Verified implemented: `Pm4SurfaceCorrelationExtractor.cs`, `Pm4SurfaceCorrelationMatcher.cs` in
  `src/core/WowViewer.Core.PM4/Services/`; CLI verbs `build-wmo-surface-db`, `extract-pm4-surfaces`,
  `match-surfaces`, `validate-matches`, `identify-models` all present in Program.cs. `Pm4Generator.cs`
  exists and was reworked per T026-T033 (WMO collision source, MSVI fix, fingerprint-based validation:
  mean score 0.462, 36/48 matched on `development_29_18`).
- Partial: P@1 remains ~1.2% despite P@3 improving to 25.3% (area-bin=1.0) — the spec's own SC-005
  full target (P@3 ≥60%, requiring normal+height) was tried in Phase 7/T023 and explicitly failed (0
  matched, P@3 regressed to 10.1%). WMO root enumeration still capped at ~502-503 of ~1985/2790 known
  roots (T036/Phase 8 unchecked) — SC-001 stretch goal (≥1900) not met.
- Not implemented: Phase 8 (listfile-based WMO enumeration, T024-T026 duplicate-numbered) unchecked;
  Phase 9 (placement recovery / MODF-MDDF regeneration from matched WMO transform, T027-T030) entirely
  unchecked — this is the spec's own stated "eventual goal."
- Checkbox accuracy: accurate.
- Operator gates owed: none beyond CLI/corpus runs already performed; placement-recovery phase would
  need real-client comparison once built.
- Open residue (spec-stated only): Phase 8 listfile enumeration (T024-T026); Phase 9 placement recovery
  (T027-T030, "the eventual goal" per spec's own Context section) — genuinely unbuilt downstream work.
- Superseded by / overlaps: shares "generation from geometry" ground with 184 (which supersedes
  `Pm4Generator`'s validation approach with a structural conformance check); shares "asset identity"
  ground with 046 (superseded) and 128 (negative-BSP, unimplemented).
- Disposition: FOLD
- Proposed epic theme: pm4-asset-identity
- Confidence: high

---

### 128 PM4 Negative-BSP Object Matching
- Stated status: Draft (spec.md only — no plan.md, no tasks.md, no data-model.md).
- Scope: replace scalar-only matching with a structural "negative space" comparison (walkable-surface
  arrangement/connectivity) between PM4 segments and candidate assets; confidence banding; accuracy
  eval against a known-placement map.
- Verified implemented: none. No `StructuralDescription`, `NegativeBsp`, or equivalent class found
  anywhere in `src/core/WowViewer.Core.PM4`. This is the spec-writing stage only.
- Partial: none — the epic (`epic-pm4-restoration/epic.md`) explicitly states "implementation not
  started" and that 128 is last in the dependency chain (130 → 129 → 128), gated on 130's grouping
  decode reaching usable quality (currently ~5% resolved per the epic's own baseline table).
- Not implemented: everything — FR-001 through FR-012, all three user stories.
- Checkbox accuracy: no tasks.md exists; nothing to misreport.
- Operator gates owed: not reached — no code to gate.
- Open residue (spec-stated only): the entire spec is open residue; it depends on 130 (object grouping)
  which itself has no implementation started (see 130 below).
- Superseded by / overlaps: depends on 130 (decode) and feeds from 129 (dataset) per the epic's
  dependency chain; also overlaps 065's surface-correlation matching in problem space (both aim at
  PM4→asset identity) though 128's structural/negative-space framing is a distinct, unbuilt approach.
- Disposition: FOLD
- Proposed epic theme: pm4-asset-identity
- Confidence: high

---

### 129 PM4 Zarr Dataset
- Stated status: Draft (spec.md only — no plan.md, no tasks.md, no data-model.md).
- Scope: a stored, queryable Zarr-family dataset for PM4 signals at map/tile/object granularity, with
  per-field coordinate space, confidence, and decode-version provenance recorded as data.
- Verified implemented: none specific to this spec's object-primary map/tile/object design. A
  `pm4_asset_matching/signal_store.py` Zarr store exists in data-harvester but is scoped to spec 046's
  segment/asset-reference signals (a different, narrower schema), not 129's three-level nested design.
  `v25/pm4_guide.py` / `v25/dataset.py` reference PM4 segments for an unrelated ML-guidance project.
- Partial: the "v50 zarr store conventions" this spec says to follow as a model exist elsewhere in the
  repo (terrain work), but no PM4-specific implementation of them exists.
- Not implemented: everything — FR-001 through FR-012, all three user stories, the object-primary
  layout decision itself.
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: not reached.
- Open residue (spec-stated only): entire spec; explicitly depends on 130's object-identity decode
  being trustworthy first (epic: "its row layout is object-primary, so it depends on object identity
  being trustworthy").
- Superseded by / overlaps: depends on 130 per epic chain; feeds 128.
- Disposition: FOLD
- Proposed epic theme: pm4-asset-identity
- Confidence: high

---

### 130 PM4 Remaining Decode — Connective Geometry and Object Identity
- Stated status: Draft. Has plan.md, research.md (Phase 0, 10 findings), data-model.md, contracts/,
  quickstart.md — but **no tasks.md**. Epic explicitly states: "Next step is `/speckit.tasks` on 130.
  Implementation: not started."
- Scope: resolve object-grouping (currently ~5% via `MSLK.GroupObjectId → MPRL.Unk04`), determine
  MSPV/MSPI vs MSCN as connective geometry, fix viewer whole-object selection, resolve MPRR.
- Verified implemented: none of Phase 2-9 (grouping-rule harness, evidence register, canonical
  object-identity service, MSPV/MSCN discriminator, MPRR domain sweep) exist as named. Pre-existing
  infra referenced by research.md as prior art does exist: `Pm4RegionObjectGrouper.cs` (CK24×MSHD
  region grouping, "G2" candidate) and viewer selection state (`Pm4SelectedObjectGraphInfo`,
  `SelectedObjectKey` tuple of tileX/tileY/ck24/objectPart, `RestoreSelectedPm4Object`,
  `BuildCk24ObjectTriangles` in `WorldScene.cs`) — but these predate 130 and are CK24-based (the "G1"
  candidate), not the corpus-measured grouping-rule outcome 130's plan calls for.
- Partial: research.md's Phase 0 findings (R1-R10c, e.g. "the 65,819/1,206,977 baseline doesn't measure
  grouping," "MSPV shares MSVT's frame," "MSCN is a co-equal candidate") are genuine measured research
  already recorded, but the plan's 9 implementation phases that would act on those findings were never
  started — confirmed by the absence of tasks.md and any grouping-rule-harness/evidence-register code.
- Not implemented: FR-001 through FR-012 (grouping-rule harness and measured rule set; canonical
  object-identity service; viewer whole-object selection per new rule; MSPV/MSCN discriminator;
  reconstruction-vs-real-asset comparison; MPRR domain sweep).
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: not reached — SC-002 (viewer selects whole objects) needs real interaction
  proof once built.
- Open residue (spec-stated only): entire spec is open residue — it is the decode the other two
  epic members (129, 128) depend on, and per the epic is the correct starting point ("130 first").
- Superseded by / overlaps: 188 and 189 (field-semantics/complete-field-map) have since done
  substantial *additional* field-level measurement beyond 130's Phase 0 (MSCN ordered-chain finding,
  MPRL axis fix, GroupObjectId-as-tiny-groups correction) — some of it verified live in code (see 188/
  189 below) — and should likely absorb/supersede 130's still-open decode questions rather than 130
  restarting independently.
- Disposition: FOLD
- Proposed epic theme: pm4-field-decode
- Confidence: high

---

### 149 PM4 Region Navigation and Audio Trigger Controls
- Stated status: Draft. Tasks: ~19 checked / 38 total (has plan/research/data-model/contracts).
- Scope: replace the PM4 correlation workbench with a region browser + camera focus; strip
  WMO/M2-matching UI from PM4 tooltips/workbench; make all audio triggers (MCNK legacy, MCSE, area
  music) default-off and individually toggleable; add an opt-in area (Zone/SubZone) overlay and speaker
  markers.
- Verified implemented: T018 (MCSE/MCNK coordinate normalization) — confirmed via code comments in
  `Pm4ObjectPositionDecoder.cs`-adjacent audio path; T032-T036 (US4 area overlay) — confirmed:
  `AreaOverlayRegion.cs` exists in `src/viewer/WoWViewer/Terrain/`, referenced from `ViewerApp.cs`,
  `WorldScene.cs`, `ViewerApp_Investigation.cs`. T038 (speaker markers) referenced as implemented in
  the 2026-08-14 checkpoint note.
- Partial: US1 (region browser) — T001 (`Pm4RegionNavigationItem`/`Pm4RegionFocusRequest` models) is
  **unchecked and confirmed absent from `src/`** (no matches anywhere); T006-T011 all unchecked. US2
  (remove correlation UI) — T012-T017 unchecked, and confirmed **not done**: `Correlation` appears 64
  times in `ViewerApp_Pm4Utilities.cs`, i.e. the correlation tab/matching UI this spec's FR-006/FR-007
  require removing is still present. US3 audio enablement — T020/T022/T024 (per-trigger enable/disable,
  master toggle, full test pass) unchecked; no `MasterWorldTrigger`/master-toggle symbol found in
  `WorldAudioRuntime.cs`.
- Not implemented: US1 entire region browser; US2 correlation-UI removal; US3's toggle/enablement layer
  (only the underlying MCNK/MCSE data normalization and passive speaker-marker overlay exist, not the
  opt-in playback gating this story is centrally about).
- Checkbox accuracy: accurate — spot-checked both directions (Pm4RegionNavigationItem absent as
  unchecked T001 predicts; AreaOverlayRegion present as checked T032-T036 predicts).
- Operator gates owed: SC-008 (Debug build + focused tests pass) recorded as partial in the 2026-08-14
  checkpoint (53/53 focused tests passing at that point, but for a narrower slice); real audible/visual
  client proof is explicitly still user-owned per the spec's own SC-008/quickstart language.
- Open residue (spec-stated only): US1 region browser (T001,T003,T006-T011); US2 correlation-UI removal
  (T005,T012-T017); US3 per-trigger/master enablement (T020,T022,T024,T020a ZoneMusic
  row-indirection); T037 (US4 focused tests); Phase 5 cross-cutting validation (T026-T031).
- Superseded by / overlaps: US2's correlation-removal goal directly overlaps 065's stance that
  ADT/scalar correlation is now validation-only, and 128/130's reframe that surface/negative-space
  identity should replace scalar matching in the UI too.
- Disposition: FOLD
- Proposed epic theme: pm4-viewer-ux-and-audio
- Confidence: high

---

### 184 PM4/PD4 Generation from Source Geometry
- Stated status: Draft (spec.md only — no plan.md, no tasks.md).
- Scope: build a conformance-check instrument (adjacency-window integrity, reciprocity, polygon-size,
  wall-edge fraction) that can score any navmesh file, then generate PD4/PM4 navmesh structure from WMO
  collision geometry that passes it — inverting the decode problem into a supervised one.
- Verified implemented: `Pm4Generator.cs` exists (`src/core/WowViewer.Core.PM4/Services/`) — but this
  is the **pre-existing, measured-as-inadequate** generator the spec's own "Measured baseline" table
  describes (always `IndexCount: 3`, 1 MSLK record, empty MSPV/MSPI, MSCN=1 point). No conformance-
  check CLI verb (`pm4 msur-window`-style scoring tool beyond the research analyzer that produced the
  spec's own cited figures) or new generation logic implementing FR-006 through FR-009 (coplanar
  merge, adjacency-run generation, reciprocity, wall-quad-on-blocked-connections) was found.
- Partial: 065's Phase 7 (T024-T033) already reworked `Pm4Generator.cs` toward WMO-collision-sourced
  fingerprint matching and reports a 0.462 mean fingerprint score — this is adjacent groundwork but not
  184's structural-conformance target (184 explicitly wants adjacency/reciprocity/wall-quad structure,
  not aggregate fingerprint score).
- Not implemented: the conformance-check instrument (US1, FR-001-003); ground-truth pairing via
  placement-Z bit-exact match (US2, FR-004a — though the underlying measurement, 844/950 objects
  88.84%, is cited as already done via a different analyzer per spec text, the pairing *tool* is not
  built); surface generation with coplanar merging (US3); adjacency+wall generation (US4); round-trip
  write/read (US5); gap report (US6).
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: not reached.
- Open residue (spec-stated only): entire spec.
- Superseded by / overlaps: explicitly built to supersede `Pm4Generator`'s validation approach used in
  065 Phase 7; shares "MSCN is unowned" finding with 189; consumes 185's terminology work (FR naming).
- Disposition: FOLD
- Proposed epic theme: pm4-generation
- Confidence: high

---

### 185 PM4/PD4 Format Documentation and Terminology Restoration
- Stated status: Draft (spec.md only — no plan.md, no tasks.md).
- Scope: correct falsified field names in `Pm4TerminologyCatalog` (`AttributeMask`→count,
  `PackedParams`/CK24→float Z), unify `Pm4MsurEntry`/`Pd4MsurEntry` naming, write per-format wiki-ready
  docs, characterize MPRR's structural grammar.
- Verified implemented: "Terminology" concept exists in code (`Pm4ResearchChunkModels.cs`,
  `Pm4ResearchAnalyzer.cs`, `Pm4Ck24ForensicsAnalyzer.cs`, `Pm4ForensicsModels.cs`) but not as a single
  dedicated `Pm4TerminologyCatalog` class file — the spec's premise ("catalog already tracks raw offset
  → local alias → confidence") is partially confirmed as scattered across these files rather than one
  catalog. `AttributeMask` remains in active use as a field/property name in code (grep across
  `src/core/WowViewer.Core.PM4`, `tools/inspect`, `src/viewer` was in progress but earlier searches in
  188/189 spec text itself confirm the name is still current in the dropdown/model as of spec 188's
  writing).
- Partial: significant *research* content behind this spec (the falsified-field measurements
  themselves, MPRR's 4n+3 block-quantization finding, MVER-is-not-a-build-string correction) is written
  up as fact in the spec's own Context section — this is evidence the measurements were done (likely by
  the same research analyzers used in 188/189), not that the documentation/renaming deliverable (FR-001
  through FR-014, the actual wiki-ready per-format docs) was produced. No `docs/architecture/pm4-*` or
  `docs/architecture/pd4-*` format-documentation file matching this spec's FR-005 was located in this
  pass (not exhaustively searched for filenames outside standard code roots).
- Not implemented (as this spec's own deliverable, distinct from the research feeding it): the
  renamed/corrected terminology catalog; the two per-format wiki documents; MPRR grammar write-up
  as a deliverable document (vs. the raw measurement already in the spec text).
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: publishing to wowdev.wiki is explicitly a user action, not in scope for the spec
  itself.
- Open residue (spec-stated only): FR-001-FR-014 — the catalog corrections and the two format
  documents are the concrete deliverables and remain the residue; much of the supporting measurement is
  arguably already done (per spec text and per 188/189's overlapping content) and should be verified/
  consolidated rather than redone.
- Superseded by / overlaps: heavy content overlap with 188 and 189 — three specs (185/188/189) all
  reference the same measurements (AttributeMask-is-a-count, CK24-is-a-float, MPRR 4n+3) with 185
  declared as owning naming/wiki, 188 owning behavior characterization, 189 owning the field-by-field
  ledger. In practice these read as one continuous research effort split across three spec files by
  date opened, not by cleanly separated scope.
- Disposition: FOLD
- Proposed epic theme: pm4-field-decode
- Confidence: medium (did not locate a dedicated `Pm4TerminologyCatalog` file or format-doc output
  files; scope boundary with 188/189 is genuinely blurry in the source text itself, not just an audit
  artifact)

---

### 186 Server Data Transformer and World Content Browsing
- Stated status: Draft (spec.md only — no plan.md, no tasks.md).
- Scope: lossless, attributed ingest of fan-server SQL dumps (multiple dialects) joined against
  existing DBC/DB2 client tables, into a queryable structured store; viewer surfaces who/what lived at
  a location.
- Verified implemented: the two pieces this spec explicitly calls "already reachable"/"one dialect
  already done" are confirmed present: `AlphaCoreDbReader.cs` and `SqlWorldPopulationService.cs` in
  `src/viewer/WoWViewer/Catalog/` and `Population/` respectively, plus DBCD/WoWDBDefs wiring (not
  re-verified in this pass but not disputed — matches other specs' references). No dialect-definition
  mechanism, entity model, lossless-reconstruction store, or diff/comparison-across-sources tooling
  (this spec's actual new scope, FR-001 through FR-018) was found anywhere in `src/`.
- Partial: none beyond the pre-existing single-dialect reader named above.
- Not implemented: everything new to this spec — lossless ingest with unmapped-field preservation
  (FR-002/003), attribution (FR-004), multi-source non-destructive merge (FR-005), diff reporting
  (FR-006), reproducible re-ingest (FR-007), dialect-plugin mechanism (FR-012/013), client-table join
  with provenance labeling in the viewer (FR-009), external/model query surface (FR-014/015/016).
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: not reached; spec explicitly notes the operator supplies sources (not this
  project's job) and model-serving runs are user-run.
- Open residue (spec-stated only): entire spec — it explicitly says it "consumes the general-purpose
  datastore that specs 179-183 own," a dependency this audit did not verify (out of this batch's
  range) and which should be checked before any epic folds this spec's residue forward.
- Superseded by / overlaps: none within this batch; 187 explicitly depends on and consumes 186's store.
- Disposition: FOLD (as pure unimplemented forward scope, pending an epic that also accounts for the
  179-183 datastore dependency, which is outside this batch)
- Proposed epic theme: server-data-museum
- Confidence: high

---

### 187 Single-Player Museum World Simulation
- Stated status: Draft (spec.md only — no plan.md, no tasks.md). Explicitly depends on 186 with "adds
  nothing to its requirements."
- Scope: a deterministic, headless, single-player world-simulation core (no network protocol, no DB at
  runtime) reading from 186's store, with model-assisted "open decisions" (never stored facts) as
  replayable recorded inputs; viewer integration; external scenario-driving adapter.
- Verified implemented: none. Spec's own "What exists today" section states this plainly:
  "`SqlWorldPopulationService` places static spawns in the viewer. There is no tick, no behaviour, no
  quest state, and no combat. This spec is entirely new construction on top of 186's store." Confirmed
  no `SimulationRun`/`DecisionSource`/`Scenario` or equivalent symbols exist in `src/`.
- Partial: none.
- Not implemented: everything — FR-001 through FR-017, all five user stories. This is the furthest
  from implementation of any spec in this batch, and is explicitly gated on 186 (itself unimplemented)
  first.
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: not reached; real-client/real-session proof and model-serving runs are
  explicitly called out as user-run in Assumptions.
- Open residue (spec-stated only): entire spec, but it cannot usefully start until 186 exists — the
  dependency chain (186 → 187) should be preserved in any epic reorganization, not flattened.
- Superseded by / overlaps: none within this batch.
- Disposition: FOLD
- Proposed epic theme: server-data-museum
- Confidence: high

---

### 188 PM4 Field Semantics and a Grouping Surface That Tests Them
- Stated status: no explicit Status line (Draft by convention); Created 2026-08-24; spec.md +
  checklists/requirements.md only, no plan/tasks. Depends on 185 for naming/wiki ownership.
- Scope: turn the viewer's PM4 grouping dropdown into a live instrument reporting cardinality/purity/
  distinctness per field against a geometry-derived reference grouping; retire/relabel four falsified
  grouping modes (Ck24Type, Ck24ObjectId, Ck24TypeVsTypeFlags, AttributeMask); give the doodad
  population (186,060 surfaces, no placement-Z key) a working grouping; sweep every remaining field.
- Verified implemented: `Pm4FieldSweepAnalyzer.cs` exists in `src/core/WowViewer.Core.PM4/Research/`
  and a `pm4 field-sweep` (or equivalent) CLI path is referenced/used by spec 189's own "Measured with
  `pm4 field-sweep`, 120 files" — confirming the FR-008/FR-009 field-characterization *sweep instrument*
  exists and is actively used. No dedicated UI-layer readout (distinct-value count / group-size
  distribution / geometric-component agreement shown live when a grouping mode is selected, per US1)
  was located as a distinct viewer feature; the existing PM4 grouping dropdown itself was not
  independently re-verified for mode retirement/relabeling (FR-002, US2) in this pass.
- Partial: the sweep/measurement *instrument* (FR-008/009/010) appears substantially built and used
  (189 cites its output directly); the *viewer-facing* half (US1 live readout, US2 mode retirement,
  US3 doodad-population grouping surfaced in UI) was not confirmed present.
- Not implemented (not confirmed): US1 live per-mode readout in the grouping dropdown; US2 actual
  retirement/relabeling of the four falsified modes in the UI; US3 doodad-population grouping exposed
  as a selectable mode.
- Checkbox accuracy: no tasks.md; nothing to misreport.
- Operator gates owed: SC-005 (readout distinguishes fields without the user being told which is which)
  and any live-viewer-session confirmation are explicitly user-run per Assumptions.
- Open residue (spec-stated only): US1 (live grouping-quality readout), US2 (mode retirement/
  relabeling in UI), US3 (doodad grouping as a usable mode) — the measurement/sweep backend (US4/FR-008
  /009/010) appears to be the part that's actually done.
- Superseded by / overlaps: near-total content overlap with 185 (falsified-name corrections) and 189
  (the same field-by-field findings, e.g. GroupObjectId/AttributeMask/_0x1C, appear verbatim-adjacent
  in both 188 and 189's "Resolved since"/"Current ledger" sections). These three specs should very
  likely be merged into one epic/spec rather than tracked separately — the audit brief's own framing
  ("open plans reconciled into a handful of new epic specs") applies directly here.
- Disposition: FOLD
- Proposed epic theme: pm4-field-decode
- Confidence: medium (UI-layer claims not independently re-verified beyond the backend sweep instrument)

---

### 189 PM4/PD4 Complete Field Map
- Stated status: "open"; Branch v0.5.3-dev; spec.md + checklists/requirements.md only, no plan/tasks.
  Explicitly: "Supersedes nothing. Spec 185 owns naming... spec 188 owns behaviour... This spec owns
  the ledger."
- Scope: an exhaustive, corpus-measured status ledger (MEASURED/PARTIAL/UNKNOWN) for every field in
  both formats, self-correcting against its own past mistakes (e.g. the `GroupObjectId` "near-unique"
  misreading it caught and fixed within the same spec).
- Verified implemented: this spec is unusual — its content **is** the deliverable (a living ledger),
  and multiple of its specific claims are independently confirmed live in code, not just asserted in
  the spec text:
  - `Pm4ObjectPositionDecoder.cs` (lines ~197-217): `MPRL.Unk04`-as-heading is confirmed **removed**,
    matching the ledger's "RESOLVED 2026-08-25 — refuted and removed" entry and FR-004/SC-004 exactly
    (code comment: "MPRL.Unk04 is NOT a heading, and this used to treat it as one... Two measurements
    refute that", `mprlHeadingMean = 0f`).
  - `pm4 field-sweep` CLI path exists (per grep of `tools/inspect/WowViewer.Tool.Inspect/Program.cs`),
    matching FR-001's "single command emits the ledger... from live data."
- Partial: the ledger table itself (MSLK/MPRL/MSUR/other-chunk status rows) is data embedded in the
  spec, not verified independently field-by-field in this pass (would require re-running `pm4
  field-sweep` against the 120-file corpus, which is a corpus-wide sweep and out of this read-only
  audit's scope per the brief). The "Known production code depending on unmeasured meaning" table's
  second row (`AdtPm4MaskBuilder` corner-relative space "never verified") was not independently checked.
- Not implemented: FR-002/FR-003 (deep characterization of `MSLK._0x04`/`_0x00`/`_0x01` against
  geometry) show as PARTIAL in the ledger itself, i.e. the spec's own status marks most rows as
  open/partial rather than closed — this is honest self-reporting, not a gap in the audit.
- Checkbox accuracy: no tasks.md; the ledger's own MEASURED/PARTIAL/UNKNOWN markers function as
  self-audited checkboxes and the one spot-checked against code (`Pm4ObjectPositionDecoder` heading
  removal) was accurate.
- Operator gates owed: corpus sweeps and viewer sessions are explicitly user-run per Assumptions.
- Open residue (spec-stated only): SC-003 (`MSLK._0x04` measured role or eliminated-hypothesis set —
  currently PARTIAL, "what a pair MEANS is open"); MPRR decode (explicitly out of scope, owned by 185);
  the `AdtPm4MaskBuilder` unverified-space risk row.
- Superseded by / overlaps: see 188 — 185/188/189 form one continuous research thread that should be
  one epic (and arguably one document) going forward rather than three cross-referencing specs.
- Disposition: FOLD
- Proposed epic theme: pm4-field-decode
- Confidence: high (the one code claim spot-checked was accurate, and the spec is explicitly
  self-auditing by design)

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 046 | ARCHIVE-SUPERSEDED | 0 (superseded by 065) | pm4-asset-identity |
| 065 | FOLD | 2 (WMO enumeration, placement recovery) | pm4-asset-identity |
| 128 | FOLD | 1 (entire spec, gated on 130) | pm4-asset-identity |
| 129 | FOLD | 1 (entire spec, gated on 130) | pm4-asset-identity |
| 130 | FOLD | 1 (entire spec — decode phases 2-9) | pm4-field-decode |
| 149 | FOLD | 4 (region browser, correlation-UI removal, audio enablement, US4 tests) | pm4-viewer-ux-and-audio |
| 184 | FOLD | 6 (all 6 user stories) | pm4-generation |
| 185 | FOLD | 3 (catalog corrections, 2 format docs) | pm4-field-decode |
| 186 | FOLD | 1 (entire spec, notes 179-183 datastore dependency) | server-data-museum |
| 187 | FOLD | 1 (entire spec, gated on 186) | server-data-museum |
| 188 | FOLD | 3 (US1 live readout, US2 mode retirement, US3 doodad grouping) | pm4-field-decode |
| 189 | FOLD | 2 (MSLK._0x04 role, AdtPm4MaskBuilder space verification) | pm4-field-decode |
