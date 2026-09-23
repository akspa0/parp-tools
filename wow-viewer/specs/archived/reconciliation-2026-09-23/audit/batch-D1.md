# Batch D1 — Legacy format readers/writers (M2/MDX eras, format profiles, MH2O/MCLQ liquids, split ADT/MoP, converter validation)

Audited against source at `I:\parp\parp-tools\wow-viewer` (branch `v0.6.0-dev`), 2026-09-23. Read-only.

---

### 104 Legacy M2 model rendering (client 1.0.0–2.4.3)
- Stated status: spec's own header already carries a 2026-09-10 "Superseded" amendment pointing to
  Spec 235. Tasks: 7/27 checked (T020–T026) — spec 235 explicitly flags these 7 as **unverified,
  pending receipt audit**, not confirmed.
- Scope: research-first recovery of the embedded-skin (pre-WotLK) M2 layout, 1.0.0 `MD20 0x100`
  classic-layout routing, and native-vs-MDX-conversion handoff for 2.x/3.0.x embedded profiles.
- Verified implemented: `M2Era100ModelReader.cs` and `M2Era100Constants.cs`
  (`src/core/WowViewer.Core.IO/M2Era100/`) exist and are real, non-stub readers; `M2ModelReaderDispatcher.cs`
  (`src/core/WowViewer.Core.IO/M2Chunked/`) validates the classic embedded layout via
  `M2Era100ModelReader.ValidateLayout` before routing — this is the T006/T007 deliverable, present in
  code despite being unchecked. `MdxMaterialRenderPolicy.cs` (`src/core/WowViewer.Core/Mdx/`) exists,
  matching checked T026.
- Partial: T020–T023's "native embedded-profile handoff, no MdxFile construction" claim is plausible
  given the dispatcher code read, but was not traced end-to-end for this report — 235's own Phase 1/2
  evidence (below) shows this route was still substantially reworked in September, suggesting 104's
  checked state was incomplete or superseded rather than simply true.
- Not implemented (per spec's own tracking): T014–T019 (1.0.0 real-render signoff, 1.12.1 layout slice,
  TBC boundary slice) — all unchecked, no evidence of a completed user-run signoff in `104/`'s own
  `evidence/` beyond what 235 separately produced.
- Checkbox accuracy: 7 checked-but-unverified (T020–T026, per 235's own audit note) + at least 2
  unchecked-but-present (T006, T007 — dispatcher/reader code exists despite `[ ]`).
- Operator gates owed: T013/T015–T017/T027 real-client visual signoff — never closed in this spec;
  partially closed since under 235 (see below).
- Open residue (spec-stated only): none carries forward under 104's own name — spec 235 §"Supersession
  notice" explicitly states it absorbs "the unimplemented residue (20 of 27 tasks)."
- Superseded by / overlaps: 235 (explicit, bidirectional acknowledgment in both spec headers).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: legacy-mdx-m2-rendering
- Confidence: high

---

### 105 Format-version profiles (1.0.0 M2 texture/lighting/animation + profile unification)
- Stated status: Draft. No `tasks.md` was ever authored (confirmed: no such file in the spec
  directory) — planning stopped at spec.md.
- Scope: three rendering pillars for a 1.0.0 M2 (texture resolution, N·L lighting, flat-key-array +
  interpolation-range animation addressing) plus reconciling `FormatProfileRegistry` vs
  `M2ModelReaderDispatcher` into one canonical M2 version→layout owner (FR-008–FR-014), deleting the
  inert `M2Profile` records (FR-010).
- Verified implemented: none of this spec's own architectural asks. `FormatProfileRegistry.cs`
  (`src/viewer/WoWViewer/Terrain/FormatProfileRegistry.cs`) still defines `M2Profile` with all five
  build entries carrying byte-identical `SkinLikeAStride=0x70/SkinLikeBStride=0x2C/EffectLikeAStride=0xD4/
  EffectLikeBStride=0x7C` — the exact "ceremony that varies nothing" the spec names (FR-010 not done;
  two competing schemes still coexist, FR-008/FR-009 not done). No `InterpolationRange` / flat-key-array
  type exists anywhere under `src/` — FR-001–FR-007's addressing-mode contract was not built.
- Partial: the **practical symptom** this spec targets (1.0.0/2.0.0-era M2 gray/frozen models) appears
  to have been independently resolved under Spec 235's September evidence receipts
  (`creature-variant-and-replaceable-texture-resolution-fix.md` for texture/replaceable-skin
  resolution; `2.0.0-animation-and-character-geoset-fix.md` for frozen-pose/sequence-offset fixes) —
  but via direct sequence-field-offset and texture-lookup-loop fixes, not via the flat-key-array /
  interpolation-range architecture 105 specifies. The user-visible outcome may now be closer to SC-000
  than the unchanged code would suggest; the architectural FRs remain untouched regardless.
- Not implemented: FR-T1–FR-T4 (texture, as literally specified with combo-mis-index diagnosis),
  FR-L1–FR-L3 (UNLIT bit audit), FR-001–FR-007 (animation addressing model), FR-008–FR-014 (profile
  unification), all of User Story 4/5.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: SC-000/SC-T1/SC-L1/SC-001 user-confirmed render checks — never run under this
  spec's own name.
- Open residue (spec-stated only): FR-008–FR-014 profile-system unification and FR-010 inert-profile
  deletion are real, unaddressed, and not claimed by 235 (235's spec text only reconciles which
  *dispatch* mechanism is authoritative for M2, not whether `FormatProfileRegistry`'s M2 half is
  deleted). FR-001–FR-007's flat-key/interpolation-range animation model is likewise not claimed
  anywhere else.
- Superseded by / overlaps: 235 explicitly calls this "narrower prior art... not superseded" and folds
  in only the texture/lighting/animation *symptom*, not the profile-unification architecture.
- Disposition: FOLD (residue: profile unification + animation addressing model carried into epic)
- Proposed epic theme: legacy-mdx-m2-rendering
- Confidence: medium — the practical-vs-architectural distinction above is inferred from evidence file
  titles/summaries, not a byte-level trace of whether 235's fixes fully satisfy 105's acceptance
  scenarios.

---

### 154 M2 reader era parity (1.x–3.0.1)
- Stated status: spec's own header already carries a 2026-09-10 "Superseded" amendment pointing to
  Spec 235. Status line says "Draft"; **no `tasks.md` was ever authored** (confirmed absent from the
  directory) — this was "planned but never task-broken," exactly as 235 describes it.
- Scope: survey-first build-by-build M2 parity across `0x100`–`0x108`, three named defects (D1 bones
  discarded for `0x100`, D2 wrong bone stride in fallback, D3 unhandled 4.0.0 camera-record crash), and
  a deferred US4 (cross-era rig comparison).
- Verified implemented: n/a directly under 154 (no tasks to check); its measured findings (the exact
  `0x100`–`0x107` broken range, `0x108`/3.3.0 as the real reference point, D1/D2/D3) are the evidence
  base 235 explicitly built its Phase 0/1 work from, and that work is independently verified real (see
  235 below).
- Not implemented: nothing to mark, since 154 itself produced no task-tracked implementation — it is a
  research/spec artifact whose findings were carried forward.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: none under 154's own name (US4 cross-era rig comparison never attempted).
- Open residue (spec-stated only): US4 "cross-era rig comparison" (0.5.3 High Elf vs Blood Elf rig) is
  explicitly named in 154's own superseded-note as "deliberately NOT carried into Spec 235's
  requirements... revisit as its own slice if asked for." This is the one piece of stated scope that
  does not currently live anywhere.
- Superseded by / overlaps: 235 (explicit).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: legacy-mdx-m2-rendering
- Confidence: high

---

### 193 Benilla 1.12.1 client reference & 1.x M2 rendering parity
- Stated status: "Active / Reference Architecture." Tasks: 0/11 checked (T101–T104, T201–T204,
  T301–T303) — genuinely 0, not a checkbox-accuracy issue.
- Scope: use the external Rust `samwhosung/benilla` 1.12.1 client as an oracle to cross-check 1.x M2
  reading, animation, and (P3) renderer batching architecture against our C# implementation.
- Verified implemented: none. No vendored/cloned Benilla source anywhere under `src/`, `libs/`, or
  `tools/` (grepped, none found). The spec's own named target file `M2ModelReader100.cs` does not
  exist — the real file is `M2Era100ModelReader.cs` (confirms the spec was authored/never updated
  against actual filenames, or filenames drifted after).
- Partial: none — this is a pure reference/methodology document; the actual comparison work (T101–T303)
  was never executed.
- Not implemented: all of Phase 1–3 (T101–T303) — byte-offset comparison, texture-combiner/material-flag
  cross-check, animation forward-kinematics comparison, and renderer-batching study.
- Checkbox accuracy: accurate (0 checked, 0 done).
- Operator gates owed: cloning/building the Rust Benilla project and running it side-by-side is
  necessarily operator/dev-environment work never scheduled.
- Open residue (spec-stated only): all of T101–T303 remain open exactly as specced; 235 cites 193 only
  as "an available external oracle," not as completed work.
- Superseded by / overlaps: 235 (US4 "fuckported"-asset parity references Benilla as a second oracle for
  1.x specifically, but does not absorb 193's own task list).
- Disposition: FOLD (residue: all 11 tasks, as an available-but-unused methodology/tool for the epic)
- Proposed epic theme: legacy-mdx-m2-rendering
- Confidence: high

---

### 197 Workspace profiles, editor mode, PM4 mouse inspection, multi-client staging & MoP 5.0.1 ADT pipeline
- Stated status: no explicit `Status:` line; tasks.md preamble says it "distinguishes implemented
  reader/runtime... work from native-evidence and native-writer work still open." Tasks: 6/23 checked
  (T116, T118, T119, T120, T121, plus T117 marked `[-]` in-progress) — but **checkbox accuracy is poor
  in both directions**, see below. Only Phase 5 (MoP ADT pipeline) matches this batch's theme; Phases
  1–4 (UI mode switcher, MK Dataset purge, PM4 click selection, multi-client staging) are reported for
  completeness since the whole spec was assigned.
- Scope: five pillars — workspace mode switcher, MK Dataset legacy purge, PM4 viewport click-selection,
  multi-client map staging, and 4.3.4–5.1 MoP split-ADT reader/runtime support informed by Ghidra
  decompilation of `WoW.exe` 5.0.1.15464.
- Verified implemented: **Phase 3 (PM4 click-selection, T109–T112, all unchecked)** — `TryHandleSceneClickSelection`
  in `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs` contains 15 PM4-specific references and is
  wired from `ViewerApp.cs`'s mouse-click handler; this is implemented despite `[ ]`.
  **Phase 4 (multi-client staging, T113–T115, all unchecked)** — `MultiClientMapStagingService.cs`
  (`src/core/WowViewer.Core.Editor/Staging/`, 186 lines) and `MultiClientMapStagingTests.cs`
  (`tests/WowViewer.Core.Tests/`) both exist and are real, not stubs.
  **Phase 5 (checked tasks)** — `MopAdtChunkParser.cs`, `AdtTileFamilyResolver.cs`,
  `AdtRawChunkBlobCollector.cs` (`src/core/WowViewer.Core.IO/Maps/`) all exist; `MapConversionFormat.cs`
  defines `MopSplitAdt` as a target enum value that returns `false`/is explicitly guarded off in its
  writer-capability switch, matching T121/T123's "native MoP split output stays disabled" claim.
- Partial: T117/T117a/T117b (Ghidra native-semantics extraction for `MHID`/`MDID`/`MCXH`, WMO seam
  blending) are honestly marked in-progress/open in the task text itself — matches code state (parser
  recognizes these chunks per spec text but "evidence-gated," not asserted as native-parity-verified).
- Not implemented: **Phase 1 (T101–T105, workspace mode switcher)** — no `ViewerWorkspaceMode` type
  found anywhere under `src/`; correctly unchecked. **Phase 2 (T106–T108, MK Dataset purge)** — NOT
  done: `MkDatasetHarvester.cs` (`src/viewer/WoWViewer/Terrain/Vlm/`) still exists as a live file, and
  `MkDataset` references remain in `ViewerApp.cs` and `ViewerApp_CaptureAutomation.cs`; correctly
  unchecked. T122/T123 (compact-MCIN consumer audit, slot-aware canonical document + native MoP split
  writer) genuinely not implemented — correctly unchecked.
- Checkbox accuracy: 6 unchecked-but-present (T109–T115, PM4 selection + multi-client staging — both
  entire phases implemented with zero boxes checked) in addition to the tracked checked items being
  accurate. This is a significant under-reporting of real progress, not over-reporting.
- Operator gates owed: T105 (mode-switch usability test — moot, feature absent), T112 (PM4 mouse-pick
  test), T115 already has unit tests but no operator confirmation recorded, real 5.0.1 client render
  parity for height/shader/WMO-seam blending (T117b/T119's own stated caveat).
- Open residue (spec-stated only): T101–T108 (workspace mode + MK Dataset purge, wholly unstarted);
  T117/T117a/T117b (native Ghidra semantics incomplete); T122/T123 (sparse-merger coverage, native MoP
  split writer — writer intentionally kept disabled pending this).
- Superseded by / overlaps: none named; this is a live, multi-pillar spec not claimed by any other
  spec in this batch. The MoP-pipeline pillar (Phase 5) is the only piece in this batch's theme.
- Disposition: KEEP-ACTIVE (too large/multi-pillar to fold cleanly; Phase 5 alone could fold into a
  split-ADT/MoP epic but Phases 1–4 are unrelated UI/editor scope that would need their own epic)
- Proposed epic theme: split-adt-mop-pipeline (Phase 5 only); Phases 1–4 belong to a UI/editor epic,
  out of this batch's theme.
- Confidence: medium — full-spec breadth, verified the highest-signal claims per phase but did not
  trace every task.

---

### 205 MH2O LiquidObject vertex-format resolution
- Stated status: "Draft — ready to implement." Tasks: 0/many checked (T101–T503) — but implementation
  is substantially real.
- Scope: fix MH2O's `liquid_object_or_lvf` field being cast straight to a vertex-format enum with no
  `default` case, discarding real per-vertex river/stream heightmaps (ids ≥42 are `LiquidObject.dbc`
  ids, not raw vertex formats) in both `Mh2oChunk.Parse` (render path) and `AdtLiquidReader` (harvest
  path).
- Verified implemented: `DbcLiquidObjectTable.cs`, `DbcLiquidMaterialTable.cs`, and
  `LiquidVertexFormatChain.cs` (`src/core/WowViewer.Core.IO/Dbc/`) all exist; `LiquidVertexFormatChain`
  is consumed by **both** `Mh2oChunk.cs` and `AdtLiquidReader.cs` (grep-confirmed), satisfying Phase 4's
  "one decoder" consolidation goal (T401) despite it being unchecked. A dedicated unit test file
  `tests/WowViewer.Core.Tests/Dbc/LiquidVertexFormatChainTests.cs` exists. `evidence/phase1-dbc-chain-verified.md`
  is present, matching Phase 1's gate.
- Partial: the resolver's `LiquidObjectIdThreshold` constant is still `42` in code. The user's own
  memory-bank note ("MH2O LiquidObject ids — FIXED 2026-09-01... the '≥42' wiki threshold is WRONG
  here (ids start at 57)") suggests the corpus-observed IDs start at 57, not that 42-as-cutoff is wrong
  per se (no id in 42–56 was ever observed in the measured corpus) — this report could not fully
  reconcile the memory note against the literal constant in the time available; flagged for the
  operator rather than asserted either way.
- Not implemented: operator-gated verification tasks T206 (re-run `inspect adt liquid-formats`, expect
  zero unresolved), T304 (river height-spread match), T501/T502 (fly-through seam check, cross-era
  no-regression check) — no evidence these real-client runs were performed under this spec's own
  `evidence/` (only Phase 1 has a receipt).
- Checkbox accuracy: essentially all of Phase 1/2/3/4's code tasks (T101–T103, T201–T204, T301–T303,
  T401) are unchecked-but-present — this spec undercounts far more than it overcounts.
- Operator gates owed: T206, T304, T501, T502 (all real-client/real-data confirmation, per the spec's
  own `**(operator)**` tags) — no receipts found for these specifically.
- Open residue (spec-stated only): the operator-tagged verification tasks above; Phase 5's T503
  (retain float-plausibility probe as a reporting-only cross-check) not separately confirmed.
- Superseded by / overlaps: none — not named by any other spec in this batch or by 235.
- Disposition: ARCHIVE-COMPLETE (core implementation + consolidation verified in code; residue is
  operator-run confirmation only, which should be handed to the operator as a bounded follow-up rather
  than kept as an open spec)
- Proposed epic theme: liquid-format-resolution
- Confidence: medium-high — code-verified; did not independently re-run the `inspect adt
  liquid-formats` command myself (no client root available to this read-only audit).

---

### 209 WLW / MCLQ liquid convergence
- Stated status: "Draft — diagnosis not yet done" (stale — diagnosis is in fact done). Tasks: 3/6
  checked (209-T1–T3, Phase 1 diagnosis); Phase 2 remediation (T4–T6) unchecked and genuinely not done.
- Scope: diagnose then fix why WL*/MCLQ/MH2O liquid layers fail to converge at coastlines (operator
  report: "gaps in MCLQ data where WLW's overlap"), via two named candidate mechanisms (A: MCLQ
  partial-presence-quad interpolation poisoning height; B: `KeepOnlyAboveTerrain` culling WL* at
  shorelines with no MCLQ replacement).
- Verified implemented: `LiquidConvergenceAnalyzer.cs` (`src/core/WowViewer.Core/Maps/`) and
  `AdtLiquidConvergenceSupport.cs` (`tools/inspect/WowViewer.Tool.Inspect/`) both exist, backing the
  checked `inspect adt liquid-convergence` CLI verb (T1). `evidence/phase1-convergence-measured.md`
  exists and its task-file summary states Mechanism B is the confirmed root cause (585,108 WL* cells
  culled by terrain across Azeroth; 459,374 of those had no MCLQ replacement — the union invariant
  itself verified intact, ruling out a merge-drops-cells bug).
- Partial: Mechanism A ("quad edge interpolation validated") is noted as measured but the task text
  does not state whether it was confirmed as a real contributing cause or ruled out — ambiguous from
  the task file alone.
- Not implemented: T4 (soften `KeepOnlyAboveTerrain` shoreline culling in
  `src/core/WowViewer.Core.IO/Maps/WlLiquidRasterizer.cs`, confirmed present and unmodified per this
  spec), T5 (shoreline-connectivity unit tests), T6 (re-run convergence report to verify the fix) — the
  actual remediation is wholly unbuilt; only the diagnostic instrument exists.
- Checkbox accuracy: accurate (Phase 1 checked and verified real; Phase 2 unchecked and verified
  absent).
- Operator gates owed: T6's real-client re-measurement, and any visual fly-through confirmation once
  T4/T5 land.
- Open residue (spec-stated only): T4–T6 in full — this is a real, unstarted fix with a precise root
  cause already in hand.
- Superseded by / overlaps: none named.
- Disposition: KEEP-ACTIVE (diagnosis complete, fix genuinely pending — small enough to fold into a
  liquid epic as a defined next step, not abandon)
- Proposed epic theme: liquid-format-resolution
- Confidence: high

---

### 221 Converter regression harness — corpus gates & real-client validation
- Stated status: no explicit Status line (spec opens directly with "Overview & User Intent"). Tasks:
  0/22 checked across all 5 phases — but Phase 0 is substantially done in practice.
- Scope: a corpus-gate command (`converter validate-corpus`) plus an object-converter round-trip
  validator, a gillijimproject oracle cross-check, and an operator-executed 0.5.3 real-client smoke +
  collision harness — all currently absent, replacing "we think it still works" with a measured gate.
- Verified implemented: `evidence/phase0-baseline.md` exists and documents real work: T001's inventory
  table (25 converter commands audited, confirming every object-converter unit test — WMO V14↔V17,
  M2↔MDX — uses synthetic bytes only, zero real-client data, exactly as the spec's own "Measured
  baseline" predicted); T002's terrain round-trip baseline run on real 0.5.3.3368 Azeroth MPQ data,
  with **three genuine defects found and fixed during that measurement pass** (MCLY flag-contract bug
  in `AlphaToLkConverter.cs`, an 8-byte MCAL/MCSH header-stripping bug in `LkAdtReader.cs`, and a
  validator resolution-mismatch bug in `ValidateRoundTripCommand.cs`) with a pinned-red regression test
  left in place for the one remaining open defect.
- Partial: none beyond what's listed — Phase 0 is a clean, real, evidenced stopping point.
- Not implemented: Phase 1 (`converter validate-object-roundtrip` — grepped, no such CLI verb exists
  under `tools/converter/`), Phase 2 (`converter validate-corpus` aggregation — does not exist), Phase 3
  (`converter oracle-crosscheck` against gillijimproject — does not exist), Phase 4 (0.5.3 real-client
  smoke/collision harness doc + checklist — no `docs/real-client-harness-053.md` found).
- Checkbox accuracy: Phase 0 (T001–T004) is unchecked-but-substantially-present; Phases 1–4 (T101–T405)
  are unchecked and accurately absent.
- Operator gates owed: the entirety of Phase 4 (T401–T404) is explicitly operator-executed and was
  never started; T206-equivalent real-corpus runs for the not-yet-built object validator likewise
  pending.
- Open residue (spec-stated only): Phases 1–4 in full (US-2 object round-trip validator, US-1 corpus
  gate aggregation, US-3 oracle cross-check, US-4 real-client harness) — none of this exists yet, and
  it is exactly the kind of "prove it, don't assume it" work the reconciliation effort should preserve
  given the fixes already found during Phase 0 alone.
- Superseded by / overlaps: none named; complementary to 235's FR-015 (reconcile the two drifted
  M2↔MDX converter implementations — confirmed still unreconciled: both
  `src/core/WowViewer.Core.IO/M2/M2ToMdxConverter.cs` and
  `src/viewer/WoWViewer/Terrain/Transfer/M2ToMdxConverter.cs` still exist as of this audit).
- Disposition: KEEP-ACTIVE (Phase 0 findings are real and valuable; Phases 1–4 are substantial
  unstarted scope, not abandonable residue)
- Proposed epic theme: converter-validation-harness
- Confidence: high

---

### 235 Legacy MDX & M2 rendering correctness (1.0.0–3.0.1) & fuckported-asset compatibility
- Stated status: "Draft — authored from operator directive plus reconciliation of prior specs; not
  planned" (stale — extensive implementation has since landed). Tasks: all of Phase 0–4 checked
  (T001–T041, ~24 items) with 8 evidence receipts in `evidence/`.
- Scope: supersedes 104 and 154's unimplemented residue; survey-first build-by-build M2/MDX correctness
  across 1.0.0–3.0.1, embedded-skin reading, bone-layout fix, bounding-box fallback, plus two new
  operator-directive pillars: "fuckported" (non-standard third-party re-chunked) asset rendering
  parity with Warcraft.NET/Benilla (US4/FR-008/FR-009), and MDX torch/light-emitter visual effects
  (US5/FR-010).
- Verified implemented: `M2ModelReaderDispatcher.cs` routes all `MD20` versions `0x100`–`0x107`
  through `M2Era100ModelReader.ValidateLayout` (T010, confirmed by direct code read — the
  `0x102`–`0x107` `NotSupportedException` wall is gone). Eight evidence receipts in
  `specs/235-legacy-mdx-m2-rendering/evidence/` document real, dated fixes: bone-stride/quaternion
  handling, texture-wrap inversion, skybox/transparent blending, creature-variant + replaceable-texture
  resolution, and 2.0.0 sequence-offset/geoset fixes — these are concrete, itemized root-cause-and-fix
  writeups, not restated task text. `MdxLightSummary.cs`/`MdxLightType.cs` (`src/core/WowViewer.Core/Mdx/`)
  are consumed in `src/viewer/WoWViewer/Rendering/ModelRenderer.cs` for Omni/Ambient local-light
  uniform wiring — real light-type-aware rendering exists, though no evidence receipt or task
  explicitly claims this satisfies US5/FR-010 (torch/light-glow effect).
- Partial: FR-014 (reconcile `FormatProfileRegistry` vs `M2ModelReaderDispatcher` as the single M2
  resolution authority) — plan.md's own research names `M2ModelReaderDispatcher` as authoritative, but
  the inert `M2Profile` half of `FormatProfileRegistry` was **not deleted** (verified: still present,
  identical strides, per the 105 finding above) — FR-014 says "either extended... or confirmed
  superseded/unused **and updated accordingly**"; the "updated accordingly" (removal) half is not done.
- Not implemented: **FR-015** (reconcile the two drifted M2↔MDX converter implementations into one
  owned implementation) — both `WowViewer.Core.IO/M2/{M2ToMdxConverter,MdxToM2Converter}.cs` and the
  separate `src/viewer/WoWViewer/Terrain/Transfer/M2ToMdxConverter.cs` still exist; no task in
  tasks.md addresses this at all despite it being a named FR. **US4 "fuckported" asset compatibility**
  (FR-008/FR-009) — no task, no evidence file, no code found referencing non-standard/re-chunked-asset
  detection or a Warcraft.NET-parity fallback path; this entire pillar from the operator directive is
  untouched. **US5 torch/light-glow effect** (FR-010) — local-light uniform plumbing exists (see above)
  but nothing in tasks.md or evidence claims the specific "torch shows its defined light/glow effect"
  acceptance scenario was verified; likely a partial-credit gap, not a clean miss, but unconfirmed
  either way.
- Checkbox accuracy: all tracked tasks (T001–T041) show real, evidenced work — accurate as far as it
  goes — but **two full user stories (US4, US5) and one full FR (FR-015) from the spec's own
  Requirements section have no corresponding tasks in tasks.md at all**, meaning tasks.md itself
  under-scopes the spec, independent of checkbox truth.
- Operator gates owed: SC-002/SC-003 (visible-geometry + bounding-box-fallback user confirmation across
  the surveyed 1.0.0–3.0.1 range), SC-008 (real-client no-regression spot check) — receipts show code
  proof and build-gate passes; no artifact records an operator visually confirming rendering in a live
  client session.
- Open residue (spec-stated only): **FR-015** (single M2↔MDX converter), **US4/FR-008/FR-009**
  (fuckported-asset parity), **US5/FR-010** (light-emitter visual effects), and the "delete-not-migrate"
  half of **FR-014** (inert `M2Profile` removal) — all four are spec-stated, unimplemented, and
  untracked by any task.
- Superseded by / overlaps: absorbs 104 and 154 (confirmed, see above); narrower relationship to 105
  (absorbs the symptom, not the architecture) and 193 (cites, does not execute).
- Disposition: KEEP-ACTIVE (large, live epic-in-fact; substantial real progress but material named
  scope — FR-015, US4, US5 — genuinely still open and not safe to archive as complete)
- Proposed epic theme: legacy-mdx-m2-rendering
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 104 | ARCHIVE-SUPERSEDED | 0 (fully absorbed by 235) | legacy-mdx-m2-rendering |
| 105 | FOLD | 2 (profile unification FR-008–014; animation addressing model FR-001–007) | legacy-mdx-m2-rendering |
| 154 | ARCHIVE-SUPERSEDED | 1 (US4 cross-era rig comparison, deliberately deferred) | legacy-mdx-m2-rendering |
| 193 | FOLD | 11 (T101–T303, unused Benilla-comparison methodology) | legacy-mdx-m2-rendering |
| 197 | KEEP-ACTIVE | 5 (T101–108 UI/purge; T117/117a/117b native Ghidra semantics; T122/T123 sparse-merger + native MoP writer) | split-adt-mop-pipeline (Phase 5 only) / UI epic (Phases 1–4) |
| 205 | ARCHIVE-COMPLETE | 4 (operator-run verification only: T206, T304, T501, T502) | liquid-format-resolution |
| 209 | KEEP-ACTIVE | 3 (T4–T6, shoreline-culling remediation, root cause already known) | liquid-format-resolution |
| 221 | KEEP-ACTIVE | 4 phases (object round-trip validator, corpus-gate aggregation, oracle cross-check, real-client harness) | converter-validation-harness |
| 235 | KEEP-ACTIVE | 4 (FR-015 converter reconciliation; US4 fuckported-asset parity; US5 light-emitter effects; FR-014 inert-profile deletion) | legacy-mdx-m2-rendering |
