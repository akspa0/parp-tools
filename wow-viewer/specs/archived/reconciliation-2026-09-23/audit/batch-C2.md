# Batch C2 — Renderer correctness & draw-call/decode performance

Specs audited: 160, 198, 199, 200, 201, 202, 204, 206, 207, 242.
Method: read spec.md/tasks.md for each, then verified claims against source (Grep/Read) in
`src/viewer/WoWViewer/`, `src/core/WowViewer.Core.Runtime/`, `src/core/WowViewer.Core.IO/`,
`tests/`. No `evidence/` directory exists for any spec in this batch — receipts, where present,
are informal notes inline in `tasks.md`.

---

### 160 Skybox Rendering
- Stated status: Draft | Tasks: 0/72 checked
- Scope: Fix 5 confirmed sky defects (D1-D5): authored colours discarded every frame, skybox model
  gated to night-only, 2-of-5 sky bands used, WMO `MOSB` skybox name dropped, filename-heuristic
  classification. 8-phase plan with automated capture-diff proof infrastructure as a blocking
  Phase 1.
- Verified implemented: none.
- Partial: none.
- Not implemented: everything. Confirmed in code: `WorldScene.cs:12883` still gates the skybox
  model draw on `_skyDome.NightVisibility > 0.001f` (D2, FR-010, unchanged). No
  `SkyProvenance`/`SkyGradientSource`/`SkySourceResolver`/`SkySourceSelection` classes exist
  anywhere under `src/` — Phase 2's foundational scaffold (T009-T017) was never started, so no
  later phase (US1-US5) can have landed either.
- Checkbox accuracy: accurate (0 checked, 0 present).
- Operator gates owed: all — capture/pixel-diff automation (T001-T004) doesn't exist yet either.
- Open residue (spec-stated only): the entire spec — FR-001 through FR-024, all 5 user stories.
- Superseded by / overlaps: none directly; shares "sky/fog colour" boundary note with terrain
  lighting work but is self-contained.
- Disposition: FOLD
- Proposed epic theme: renderer-correctness (sky/material)
- Confidence: high

---

### 198 M2 and WMO Shader Permutation System
- Stated status: Draft | Tasks: no tasks.md (spec/plan/research/contracts/data-model only — the
  Spec Kit workflow never reached `tasks.md`)
- Scope: Resolve the client's real vertex (`Diffuse_*`, 16) x pixel (`Combiners_*`, 31) shader-pair
  table per M2 batch, plus WMO's 6 map-object programs, replacing the single hardcoded program.
  Explicitly not a performance fix.
- Verified implemented: none of the spec's actual ask.
- Partial: `src/core/WowViewer.Core.Runtime/M2/M2EffectRecipe.cs` (`M2DiffuseEffectFamily`,
  `M2CombinerEffectFamily`, `M2EffectRegistry.Resolve`) is a **pre-existing, much coarser**
  CPU-side classification — only 6 combiner families (`Opaque/AlphaKey/Decal/Add/Mod/Mod2X/Fade`)
  and 5 diffuse families, driven from `M2BlendMode`, not from the client's named permutation
  table. It produces a `RecipeKey`/state bucket for CPU blend-state decisions; it does not compile
  or select distinct vertex/pixel GLSL programs per permutation, so none of environment-mapping,
  edge-fade, dual-crossfade, or two-layer blending (the spec's own motivating examples) are
  implemented. No `PermutationRegistry`, no per-permutation shader compilation, and a repo-wide
  search for the client's actual names (`Combiners_`, `Diffuse_T1`) matches only this one
  unrelated file's `*FamilyName` string-formatting helpers, not a permutation table.
- Not implemented: FR-001 through FR-010 (permutation resolution, provenance to native names,
  fallback reporting, WMO's 6 programs) in full.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: all (side-by-side real-client capture comparison, SC-002).
- Open residue (spec-stated only): the whole spec, US1 (P1) through US3 (P3).
- Superseded by / overlaps: none in this batch, but overlaps future lighting-input work the spec
  explicitly defers ("lighting inputs... sequenced after this one").
- Disposition: FOLD
- Proposed epic theme: renderer-correctness (material/shader)
- Confidence: high

---

### 199 MCAL Alpha Map Decode Correctness
- Stated status: Draft | Tasks: 0/~25 checked (T101 marked `[-]` in-progress; everything else `[ ]`)
- Scope: Consolidate 4 independent MCAL decoders into one canonical owner, establish the real
  decode rule by measurement, delete the alpha-fabrication in `StandardTerrainAdapter`, and make
  harvest/render agree.
- Verified implemented: none.
- Partial: none — the "instrument first" Phase 1 scaffolding (`AdtAlphaEncoding`,
  `AdtAlphaDecodeReport`, etc.) does not exist in the repo (`AdtAlphaDecodeReport` /
  `AdtLayerDecodeOutcome` / `AdtAlphaDecodeRule` — zero matches under `src/`).
- Not implemented: everything. Confirmed all **four** decoders described in the spec's own table
  still exist independently and unconsolidated: `Mcal.GetAlphaMapForLayer`/`…Relaxed` in
  `src/core/WowViewer.Core.IO/Lk/Mcal.cs`, `AdtMcalDecoder.ReadCompressedAlpha` in
  `src/core/WowViewer.Core.IO/Maps/AdtMcalDecoder.cs`, `DecodeLayerBySpan` in
  `src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs`, and `AlphaMapService.ReadBigAlpha` in
  `src/viewer/WoWViewer/Terrain/Vlm/AlphaMapService.cs`. The fabrication
  `SynthesizeCataclysm400ResidualAlpha` is still present and called from
  `StandardTerrainAdapter.cs`. `VlmDatasetExporter.cs:1908` still hardcodes
  `GetAlphaMapForLayer(layer, false)` — exact line the spec cites, unchanged, so harvest/render
  disagreement (FR-006) is unresolved.
- Checkbox accuracy: accurate (0 checked, 0 present; T101's `[-]` matches "in progress" honestly —
  no evidence it concluded).
- Operator gates owed: all (Ghidra client-side MCAL-consumer identification T101, corpus sweeps).
- Open residue (spec-stated only): the entire spec, all 4 user stories, FR-001 through FR-009.
- Superseded by / overlaps: none in this batch; flagged by the spec itself as a Constitution II
  violation (one canonical owner per format surface) that is still live.
- Disposition: FOLD
- Proposed epic theme: decode-correctness / data-harvest-parity
- Confidence: high

---

### 200 WMO Portal Admission — Eliminating the Conservative Fallback
- Stated status: Draft | Tasks: 0/~15 checked
- Scope: Diagnose why 0 of 806 groups / 0 of 145 placements were ever rejected on a flown route
  (portal fallback firing 41% of the time); split the "portal data absent" reason into "no portal
  chunks" vs "chunks present but no usable graph"; make portal culling actually reject something
  where data supports it.
- Verified implemented: none.
- Partial: the underlying reason enum already exists
  (`WmoPortalVisibilityDecision.FallbackReason`, `WorldScenePortalVisibilityEvaluator`,
  `WorldScenePortalGraph` — all in `src/core/WowViewer.Core.Runtime/World/...`) with the named
  reason `"portal_data_absent"`, matching the spec's Context section. But
  `WmoAdmissionStats.FirstPortalFallbackReason` (`src/core/WowViewer.Core.Runtime/World/Visibility/WmoAdmissionStats.cs:105`)
  records only the **first** reason per the stats' own field name — the spec's T001 ask
  (aggregate counts per reason across a flight) is not done, and there is no split of
  `portal_data_absent` into "no chunks" vs "chunks present, no graph" (T003) anywhere in the repo.
- Not implemented: FR-001 (aggregate counts), FR-002 (WMO paths per reason), FR-003 (the
  chunks-absent vs graph-not-built split — "the distinction the whole spec turns on"), FR-004
  (portal culling actually rejecting).
- Checkbox accuracy: accurate.
- Operator gates owed: all (re-fly Wandering Isle with the diagnostic in place, before/after
  admitted-group comparison).
- Open residue (spec-stated only): the whole spec, US1-US3.
- Superseded by / overlaps: spec 151 (which built the instrumentation this spec extends); spec 207
  US2 explicitly takes "the group-bounds slice" of this same problem independent of portals.
- Disposition: FOLD
- Proposed epic theme: draw-call reduction (WMO admission/culling)
- Confidence: high

---

### 201 Separating M2 and MDX Render-Path Metrics
- Stated status: Draft, Phase 1 marked landed 2026-09-01 in tasks.md | Tasks: 4/6 checked in
  Phase 1 (T001-T004); Phase 2 (T101-T104) unchecked
- Scope: Stop conflating M2-routed and MdxDirect-routed models under one `MDX` counter; distinguish
  unbatched-by-toggle from structurally-unbatchable-by-route; then (P2, gated) give native-route M2
  a real batch path.
- Verified implemented: `WorldModelRenderPath` decomposition and per-path
  batched/unbatched/unbatchable counters exist in
  `src/core/WowViewer.Core.Runtime/World/Passes/ModelSubmissionAccounting.cs`; consumed from
  `WorldScene.cs`, `ModelRenderer.cs`, `M2Renderer.cs`; surfaced in
  `ViewerApp_Sidebars.cs`; the FR-005 sum-check test exists at
  `tests/WowViewer.Core.Tests/ModelSubmissionAccountingTests.cs` (test name
  `PerPathCounts_SumExactlyToAggregateTotals` matches the tasks.md description exactly). A
  dedicated `ModelDrawCallCounter.cs` also exists (landed jointly with spec 202 Phase 0 per the
  tasks.md note, and independently confirmed in code).
- Partial: Phase 1's two operator-only steps (T005 re-fly with batching off to size the
  native-M2 share, T006 confirm no frame-time regression) have no recorded evidence in the repo —
  not verifiable from code.
- Not implemented: Phase 2 (US3, P2) — native-route M2 batch key and toggle. This is explicitly
  gated on T005's measurement and the tasks.md itself says it may legitimately not be worth doing.
- Checkbox accuracy: accurate for Phase 1 (T001-T004 genuinely present in code); Phase 2 correctly
  unchecked.
- Operator gates owed: T005/T006 flight comparison (Phase 1 close-out); all of Phase 2 if pursued.
- Open residue (spec-stated only): T005/T006 operator verification; Phase 2 (US3) native-M2 batch
  key, contingent on T005's result.
- Superseded by / overlaps: spec 202 explicitly consumes this spec's Phase 1 output and states
  "Phase 3 T301 absorbs spec 201 Phase 2" — the native-M2 batch-key work is the same work as 202's
  T301, not a second effort.
- Disposition: FOLD
- Proposed epic theme: draw-call reduction (metric attribution — feeds 202)
- Confidence: high

---

### 202 Unified Model Batching Across Render Paths and Eras
- Stated status: Draft, phased with informal "landed" notes through Phase 3 | Tasks: ~9/~35
  checked (Phase 0 T001-T004 done, T005 answered-from-source; Phase 1 T101-T102 done, T103 `[-]`;
  Phase 3 T303 done, T302 half-done `[-]`; Phases 2, 4, 5, 6 unchecked)
- Scope: The operator-mandated "overhaul" — one component decides batchability from render state
  for every model instance regardless of loader/route/era, replacing today's per-renderer opt-in.
- Verified implemented: Phase 0's measurement decomposition
  (`ModelSubmissionAccounting.cs`/`ModelDrawCallCounter.cs`, shared with 201) is real, per above.
  Phase 1's `DistinctModelCount` claim is consistent with symbols found alongside
  `ModelDrawCallCounter` in `ModelRenderer.cs`/`WorldScene.cs`. Phase 3's fade-in-instance-payload
  work (T302, "half done") matches spec 207's confirmed `aInstanceFade`/opaque-vs-faded split in
  `ModelRenderer.cs` (`QueueGpuInstance`, `EndGpuInstanceBatch`, `DrawGpuInstanceSet`) — the two
  specs converged on the same code.
- Partial: T302 itself says the `>= 0.999` opaque/faded split gate is "deliberately retained" (not
  closed) pending a real blended-batch split — matches what's in code today (two draw calls: one
  opaque, one faded, not a unified per-instance blend decision).
- Not implemented: Phase 2 (`ModelBatchKey`, `ModelBatchDecision`, `ModelBatchPlanner` — zero
  matches anywhere under `src/`, confirming the "one decision point" central abstraction was never
  built) and Phase 3's actual blocker, T301 (native-route M2 real batch path — the thing tasks.md
  calls "the whole remaining blocker"). Phases 4 (era coverage), 5 (transparent), 6
  (cross-model, optional) are all untouched.
- Checkbox accuracy: accurate — the informal "landed" annotations in tasks.md correspond to real
  code; the unchecked phases correspond to absent code.
- Operator gates owed: T104 (Phase 1 flight), T205 (Phase 2 flight), T305/T306 (Phase 3 flight +
  capture), T401-T404 (era coverage), T501-T503 (transparent).
- Open residue (spec-stated only): Phase 2 (the actual planner abstraction) through Phase 6 —
  the majority of the spec's stated value (the "overhaul" itself, native-M2 batching, era coverage,
  transparent batching) is not yet built.
- Superseded by / overlaps: absorbs spec 201 Phase 2 (T301) and spec 153 US3's mechanism (T304);
  spec 207 Phase 1 independently implemented the fade-split half of what 202's T302 describes on
  the same files — these need reconciling into one owner, not two overlapping partial
  implementations.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: draw-call reduction (this is effectively the epic anchor already)
- Confidence: high

---

### 204 Off-Thread Asset Decode
- Stated status: Draft | Tasks: 1/~30 checked as "done" (T003, and that was discovering pre-existing
  unread counters, not new code); Phase 1 (thread-safety audit, a hard gate) unchecked
- Scope: Move model parse/adapt/BLP-decode/mesh-construction off the render thread; only GPU
  upload stays on it. Currently `WorldAssetManager` has "no threading whatsoever."
- Verified implemented: none of the actual concurrency work.
- Partial: Phase 0's instrumentation ask (split `DeferredAssetLoads` into fetch/decode/upload,
  T001/T002) has no corresponding code found; T003's "done" note in tasks.md is honestly scoped
  as just discovering that `OversizedAdmissionCount` etc. already existed, not new work.
- Not implemented: the actual seam — `DecodedAssetPayload`, `AssetLoadPipeline`,
  `AssetLoadRequest` (Phase 2/3's core types) do not exist anywhere under `src/`. Phase 1's
  mandatory thread-safety audit (T101-T104, a hard gate per the spec's own text: "nothing in
  Phase 3 starts until this is written down") has not been written. Phases 4-7 (retire oversized
  admission, textures out of draw pass, prefetch follow-parse, era coverage) all depend on the
  above and are untouched.
- Checkbox accuracy: accurate.
- Operator gates owed: all (baseline flight, cold-cache streaming-rate measurement, era coverage).
- Open residue (spec-stated only): essentially the entire spec — this is the largest single gap in
  the batch relative to its measured urgency (26.4-68.1 ms hitches, 2047/2048 frames over budget
  on the cited MoP flight).
- Superseded by / overlaps: spec 206 explicitly calls itself complementary, not a substitute
  ("206 removes work from the pipeline... 204 moves work off the render thread... neither
  substitutes for the other").
- Disposition: FOLD
- Proposed epic theme: decode/streaming performance
- Confidence: high

---

### 206 Zarr-First Asset Residency
- Stated status: Draft, member of `epic-client-datastore` (explicitly not a standalone spec — "read
  the epic first... this spec is a member of that epic, not a new one") | Tasks: no tasks.md
  (checklists/requirements.md only)
- Scope: Retire the `output/cache/` byte-copy extraction (US1, ~1.75 GB, independent of everything
  else); build a C# Zarr array reader (US2, currently missing entirely per the spec's own
  admission); make render-ready decoded content resident in the store (US3); ingest on first
  contact (US4); keep the store a universal interchange format (US5). Explicit architecture
  constraint: Python owns the datastore, C# must not implement Zarr/TensorStore I/O (operator
  directive, cost 2 months to undo previously).
- Verified implemented: none.
- Partial: none — this spec's own Context section already states the current gaps accurately and
  code confirms them unchanged: `ViewerApp.cs` still performs the exact byte-copy the spec names
  (`Directory.CreateDirectory(CacheDir); ... File.WriteAllBytes(cachePath, data);` around line
  12483-12493, now per-client-root-segmented per spec 222 but still a full extraction — US1 not
  done). `ZarrTileDatasetLoader.LoadTile` still `throw new NotImplementedException(... "the
  remaining work is the Blosc+Zstd+bitshuffle chunk decoder" ...)` — US2, the gating capability
  for US3/US4, is unbuilt exactly as the spec describes.
- Not implemented: FR-001 through FR-021, all 5 user stories.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: all.
- Open residue (spec-stated only): the whole spec, but framed correctly as residue of
  `epic-client-datastore` (specs 179-183, 190) rather than a new epic — the spec is explicit that
  it does not overturn that epic's design and should not be treated as freestanding.
- Superseded by / overlaps: epic-client-datastore (parent, pre-existing); spec 204 (complementary,
  explicitly not overlapping — 204 reschedules decode, 206 eliminates it); spec 201 (FR-011 depends
  on 201's attribution pattern, already partly landed).
- Disposition: FOLD
- Proposed epic theme: NOT this batch's renderer-performance epic — folds into the existing
  `epic-client-datastore`, cross-referenced from the draw-call/decode epic for its US3 overlap with
  spec 204.
- Confidence: high

---

### 207 Object Draw-Call Reduction
- Stated status: Draft (plan.md exists) | Tasks: ~12/~25 checked — Phase 0 T001/T002 done, T003
  unchecked; Phase 1 (US1, doodad instancing) T101-T106 all done; Phase 1b (asset load ordering,
  added mid-spec) T151/T152 done; Phase 2 (US2, WMO group admission) fully unchecked; Phase 3 (US3,
  skin-profile LOD) fully unchecked
- Scope: Operator ablation showed objects are ~97% of frame cost (WMOs ~55.5 ms, doodads ~41.2 ms
  of a ~100 ms frame), traced to three defects: faded doodads still costing a draw call, the
  instancing gate excluding the entire distance-fade population (~36% of visible doodads by area),
  and WMO groups admitted wholesale (0 of 80/806 ever rejected).
- Verified implemented: Phase 1 (US1) is real and matches tasks.md's claims precisely.
  `src/viewer/WoWViewer/Rendering/ModelRenderer.cs`: `QueueGpuInstance` no longer silently
  `return`s for sub-threshold fade (T102/FR-004) — it now routes to `_gpuInstanceData` (opaque,
  `fadeAlpha >= 0.999`) or `_gpuInstanceFadedData` (faded) instead of discarding (T101);
  `EndGpuInstanceBatch`/`DrawGpuInstanceSet` issue two separate `DrawGpuInstanceSet` calls — opaque
  first, faded second with correct depth/blend comment (T103). `FullyFadedMdxCount` exists in
  `WorldVisibilityFrame.cs`/`WorldObjectVisibilityCollector.cs` (T001, pre-spec landing confirmed).
  Phase 1b's `MaxPriorityLoadBacklog` (16 streaming / 8 WMO-only per T152) is present and wired at
  4 call sites in `WorldScene.cs` (lines ~926-9503).
- Partial: Phase 0's T003 (distinct-visible-model-count, the SC-001 floor) is unchecked and not
  independently confirmed beyond what 202's `DistinctModelCount` may already supply — worth
  reconciling rather than re-building.
- Not implemented: Phase 2 (US2 — WMO group-level admission independent of portals, FR-006/007/008)
  and Phase 3 (US3 — skin-profile LOD, FR-009) are both fully unbuilt; no code found for per-group
  frustum/projected-size rejection independent of the placement-level admission, and no skin-profile
  selection code.
- Checkbox accuracy: accurate.
- Operator gates owed: T004 (Phase 0 baseline), T107/T108 (Phase 1 capture proof), T153 (Phase 1b
  confirmation), T207 (Phase 2), T305 (Phase 3) — all unchecked/operator-owned as tasks.md states.
- Open residue (spec-stated only): Phase 2 (US2, WMO group admission — directly overlaps specs
  200/151) and Phase 3 (US3, skin-profile LOD, depends on spec 193).
- Superseded by / overlaps: spec 202 (same fade/instancing mechanism, converged on the same
  `ModelRenderer.cs` code — needs one owner, not two specs both claiming it); spec 200/151 (US2's
  WMO slice is explicitly scoped to avoid portal correctness, which 200 owns); spec 193 (skin
  profiles for US3).
- Disposition: FOLD
- Proposed epic theme: draw-call reduction (US1 delivered; US2/US3 residue)
- Confidence: high

---

### 242 Per-Placement WMO Shell Instancing Under Scene Lights
- Stated status: Draft, operator-directed 2026-09-18, opened from a regression report against
  v0.6.0-alpha, no per-spec branch (rides `v0.5.4-dev`) | Tasks: no tasks.md (checklists/
  requirements.md only)
- Scope: Spec 236 Phase 2 added scene-emitted lights and disabled WMO shell GPU instancing
  whole-scene (`_sceneLightManager.Count == 0`) whenever any light is active anywhere, rather than
  per-placement. On modern 1.60.1 CASC data this gate is effectively always false, so every WMO
  opaque placement falls back to the unbatched per-placement path: measured 16,431 WMO draw calls,
  ~5.5 FPS, WMO pass ≈5,493 ms of a 7,716 ms frame. Fix: move the decision to whether any light's
  attenuation actually reaches that placement's bounds (the per-placement query already exists via
  `SceneLightManager.QueryAffecting`, just not used for the batching decision).
- Verified implemented: none.
- Partial: none — the exact defect line is still present, unchanged. Confirmed at
  `src/viewer/WoWViewer/Terrain/WorldScene.cs:11415`:
  `bool canBatch = _sceneLightManager.Count == 0 && renderer is IGpuInstancedWmoRenderer ...` — a
  whole-scene gate, not the per-placement `QueryAffecting` test the spec's own fix describes (that
  method is already used per-placement for the *lighting upload* in `ModelRenderer.cs:1174`,
  `WmoRenderer.cs:1984`, and `TerrainRenderer.cs:1483`, but not for the *batching decision* named
  in this spec).
- Not implemented: FR-001 through FR-007 in full.
- Checkbox accuracy: n/a (no tasks.md).
- Operator gates owed: all — before/after draw-call and FPS receipt on `wow_classic_beta`
  `Azeroth` (FR-007, SC-001/002/003).
- Open residue (spec-stated only): the entire spec — a small, precisely bounded, well-evidenced
  regression fix (move one boolean from scene-scope to per-placement using an already-existing
  query method).
- Superseded by / overlaps: same defect family as 202/207 (batching gates excluding populations
  that should instance); FR-006 explicitly requires following AGENTS.md §10 (no new members in
  `WorldScene`/`ViewerApp` god classes — the fix site is currently inline in `WorldScene.cs` and
  must land in an owned service, not as another `WorldScene` edit).
- Disposition: FOLD
- Proposed epic theme: draw-call reduction (small, high-value, unimplemented regression fix)
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 160 | FOLD | 5 (D1-D5 / US1-US5, full spec) | renderer-correctness (sky/material) |
| 198 | FOLD | 3 (US1-US3, full spec) | renderer-correctness (material/shader) |
| 199 | FOLD | 4 (US1-US4, full spec) | decode-correctness / data-harvest-parity |
| 200 | FOLD | 3 (US1-US3, full spec) | draw-call reduction (WMO admission) |
| 201 | FOLD | 2 (T005/T006 close-out + Phase 2, absorbed by 202) | draw-call reduction (metric attribution) |
| 202 | KEEP-ACTIVE | 5 (Phases 2, 3-T301, 4, 5, 6) | draw-call reduction (epic anchor) |
| 204 | FOLD | 7 (Phases 0-7, essentially full spec) | decode/streaming performance |
| 206 | FOLD | 5 (US1-US5, full spec) | epic-client-datastore (not this epic) |
| 207 | FOLD | 2 (Phase 2 US2, Phase 3 US3) | draw-call reduction (US1 delivered) |
| 242 | FOLD | 1 (whole spec, small + bounded) | draw-call reduction (regression fix) |

**Cross-cutting note for the reconciliation pass**: 201, 202, 207, and 242 all touch the same
`ModelRenderer.cs`/`WorldScene.cs` batching surface and have partially overlapping/duplicated
implementations already in flight (e.g., 202's T302 and 207's Phase 1 both landed the same
opaque/faded instance split). Any new epic covering "draw-call reduction" should treat 202 as the
architectural anchor (`ModelBatchKey`/`ModelBatchPlanner`, not yet built) and fold 200/201/207/242's
residue into it explicitly, rather than letting four specs keep independent claims on the same
code. 206 is the one spec in this batch that should NOT fold into that new epic — it belongs to
the pre-existing `epic-client-datastore` and should be cross-referenced, not absorbed.
