# Spec Reconciliation — 2026-09-23

Operator-directed pass (branch `v0.6.0-dev`): audit every open Spec Kit spec against the **code**,
reconcile the open plans into a handful of epics, archive (never delete) the old specs, and correct
the memory bank and `STATUS.md`. This file is the pass's receipt (AGENTS.md §9.2 / §9.5).

## Method

1. 12 read-only audit batches (137 spec directories) checked each spec's claims against source,
   tests, CLI verbs and evidence — checkboxes were **not** treated as evidence. Brief:
   [audit/AUDIT-BRIEF.md](audit/AUDIT-BRIEF.md). Reports: `audit/batch-*.md`. Memory-bank fact check:
   [audit/memory-bank-factcheck.md](audit/memory-bank-factcheck.md).
2. Contested claims re-verified by hand (PM4 placement restore exists as `inspect pm4 restore-placements`;
   Spec 118 instance masks exist end to end; zone music is disabled by policy, not shipped).
3. Every spec directory plus the three old `epic-*` folders moved with `git mv` to `specs/archived/`
   (history preserved). Each archived record carries a banner naming its disposition and successor
   ([stamp_banners.py](stamp_banners.py)).
4. Documentation links re-resolved from each file's old location ([relink.py](relink.py), docs only —
   `.md`, `.json`, `.gitignore`). **Code comments were deliberately not edited** (AGENTS.md §4 reader
   freeze); spec paths quoted in source comments now resolve under `specs/archived/`.
5. Residue carried into 7 epics (248–254). No new scope was introduced; every backlog item cites its
   source spec id. Implementation approach is deferred until the operator triages
   [TRIAGE.md](../../TRIAGE.md) (operator directive, 2026-09-23).

## Result

| Disposition | Count | Meaning |
|---|---|---|
| COMPLETE | 16 | Code verified; operator gates (if any) carried into the successor epic's verification sweep |
| FOLDED | 104 | Open, spec-stated residue carried into an epic backlog item |
| SUPERSEDED | 13 | Replaced by later work; any narrow residue carried |
| COLD | 4 | Historical reference; no live residue |

| Epic | Primary successor of |
|---|---|
| [248 Formats, Readers, Writers & Conversion](../../248-epic-formats-and-conversion/spec.md) | 21 |
| [249 Renderer Performance, Lighting & Correctness](../../249-epic-renderer-performance-and-correctness/spec.md) | 24 |
| [250 Map Reconstruction, Composition & Editor Platform](../../250-epic-reconstruction-and-editor-platform/spec.md) | 27 + old editor-platform epic |
| [251 Viewer UX, Shell & Code Health](../../251-epic-viewer-ux-and-code-health/spec.md) | 15 |
| [252 World Simulation, Audio & Interaction](../../252-epic-world-simulation-and-audio/spec.md) | 15 |
| [253 PM4/PD4 Navmesh](../../253-epic-pm4-navmesh-research/spec.md) | 10 + old PM4 epic |
| [254 Datasets, Client Datastore & Terrain ML](../../254-epic-datasets-and-terrain-ml/spec.md) | 25 + old datastore epic |

## Checkbox accuracy findings (not corrected in the archived records; they are history)

- Under-reported (implemented, unchecked): 197 Phases 3–5, 205 Phase 4, 219 T007/T008, 222 T108,
  223 T103, 176 T001, 104 T006/T007, 237 (0/52 with the feature shipped), 238 (0/41, local CASC shipped).
- Over-reported (checked, not effective): 072 T005 toolbar width scoping.
- Stale docs inside archived specs: 115 `quickstart.md` still says US2 is not implemented.

## Duplicate threads merged

146+148 (audio) · 147+148 (residency/batching) · 185+188+189 (PM4 field semantics) · 132+140
(archaeology) · 134+139 (clean-signal lane) · 239 FR-008 + 240 US1 (survey) · 219 Ph7 + 232 T030 +
234 US1/US2 + 236 Ph5 (map save) · 136/138/153/202/207/236 (batching; the opaque/faded split landed
twice) · 223→227→231 (UI consolidation) · 213+178 (MCP automation).

## Per-spec ledger
| Spec | Disposition | Successor | Note |
|---|---|---|---|
| [009-full-project-reimplementation-spec](../009-full-project-reimplementation-spec/) | COLD | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Historical reimplementation reference; not an execution contract. |
| [046-pm4-asset-matching](../046-pm4-asset-matching/) | SUPERSEDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Superseded by 065 surface-triangle correlation. |
| [056-viewerapp-gpu-lod-modernization](../056-viewerapp-gpu-lod-modernization/) | COLD | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | 81-task refactor never executed; its perf intent landed piecemeal via 136/153/207/236. Evergreen constraints only. |
| [065-pm4-correlation-to-world-assets](../065-pm4-correlation-to-world-assets/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Correlation + fingerprint matching shipped; placement restore shipped outside the spec (inspect pm4 restore-placements). Residue: T034-T036, rotation. |
| [069-viewer-ui-overhaul](../069-viewer-ui-overhaul/) | SUPERSEDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Tab-bar design never built; replaced by the Workbench navigator (231). |
| [072-sidebar-resize-cleanup](../072-sidebar-resize-cleanup/) | COMPLETE | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Hotfix complete (T005 width scoping inaccurate, not reopened). |
| [073-ui-surface-revamp](../073-ui-surface-revamp/) | SUPERSEDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Converters piece owned by 231; rest never planned. |
| [104-legacy-m2-rendering](../104-legacy-m2-rendering/) | SUPERSEDED | [248](../../248-epic-formats-and-conversion/spec.md) | Absorbed by 235 (own supersession notice). |
| [105-format-version-profiles](../105-format-version-profiles/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Residue: FR-008..FR-014 profile unification, FR-001..FR-007 animation addressing. |
| [106-native-daynight-lighting](../106-native-daynight-lighting/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Direction constants shipped; residue FR-004/FR-012 calibration, FR-008..FR-010 provenance (verify vs 109/122). |
| [107-lighting-quick-inspection](../107-lighting-quick-inspection/) | COMPLETE | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Quick lighting + far-plane + single-ray hover shipped. |
| [108-image-wdl-prior](../108-image-wdl-prior/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Code shipped; residue T012/T013/T015 user-run training. |
| [109-v50-clean-room-audit](../109-v50-clean-room-audit/) | COMPLETE | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | v50 clean-room dataset built; T050/T051 doc-only residue. |
| [110-viewer-stabilization](../110-viewer-stabilization/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | US1/US5/US7 shipped; US2 to 248 (via 235), US3/US4 residue. |
| [111-minimap-lighting-calibration](../111-minimap-lighting-calibration/) | COMPLETE | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Calibration shipped; T009/T019 operator runs owed. |
| [112-v50-height-model](../112-v50-height-model/) | SUPERSEDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | US3 training happened under 114-T017. |
| [114-direct-terrain-reconstruction](../114-direct-terrain-reconstruction/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Geometry chain complete; residue Phase 6/7 texture-family + alpha-stack. |
| [115-terrain-feature-classifier](../115-terrain-feature-classifier/) | COMPLETE | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Classifier + retrain shipped (its quickstart.md is stale). |
| [117-wdl-lattice-prior](../117-wdl-lattice-prior/) | SUPERSEDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Premise closed by 123. |
| [118-object-occlusion-masks](../118-object-occlusion-masks/) | COMPLETE | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Instance/occlusion masks shipped; real paired training comparisons owed. |
| [123-real-wdl-detailer](../123-real-wdl-detailer/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open (next-model design, no code). |
| [124-legacy-detangle-runpod](../124-legacy-detangle-runpod/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open (legacy python detangle + C# RunPod tooling). |
| [125-minimap-dxt1-inversion](../125-minimap-dxt1-inversion/) | SUPERSEDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Superseded by 126/133/139; parked residue FR-021 super-res, FR-010/FR-012. |
| [126-minimap-terrain-reconstruction](../126-minimap-terrain-reconstruction/) | SUPERSEDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Core ambition moved to 139; parked residue US5 seams, US6/US8 texture tier. |
| [127-weak-tile-explorer](../127-weak-tile-explorer/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Residue US1/US3 + US2 UI wiring. |
| [128-pm4-negative-bsp-matching](../128-pm4-negative-bsp-matching/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Entire spec open, gated on 130. |
| [129-pm4-zarr-dataset](../129-pm4-zarr-dataset/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Entire spec open, gated on 130. |
| [130-pm4-remaining-decode](../130-pm4-remaining-decode/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Decode phases open; partially advanced by 188/189 research. |
| [132-terrain-brush-signature-classification](../132-terrain-brush-signature-classification/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | US1 shipped; US2..US6 residue (merged with 140). |
| [133-unbaked-minimap-decomposition](../133-unbaked-minimap-decomposition/) | COMPLETE | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | terrain_shadow_256 delivered end-to-end. |
| [134-v60-unified-dataset-model](../134-v60-unified-dataset-model/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Shares clean_signal modules with 139; residue US6 later-client adapters. |
| [135-phased-terrain-dual-map-overlay](../135-phased-terrain-dual-map-overlay/) | SUPERSEDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Generalized by PhaseLayers (222/232). |
| [136-m2-doodad-rendering-performance-optimization](../136-m2-doodad-rendering-performance-optimization/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Code landed; T008/T011 operator proofs owed. |
| [137-phased-minimap-overlay-and-consistent-teleport](../137-phased-minimap-overlay-and-consistent-teleport/) | SUPERSEDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Minimap phase tiles via PhaseLayers; Armed teleport via 147 US1. |
| [138-cataclysm-renderer-evolution](../138-cataclysm-renderer-evolution/) | COLD | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Broad 4.x evidence epic cold; residue: WmoRenderer.DrawBatch access violation. |
| [139-v7-clean-signal-reconstruction](../139-v7-clean-signal-reconstruction/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Live clean-signal lane; residue cross-tile family regression, promotion. |
| [140-terrain-paste-motif-archaeology](../140-terrain-paste-motif-archaeology/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open (merged with 132). |
| [141-terrain-method-translation](../141-terrain-method-translation/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Residue US4 research-lead ledger. |
| [142-world-scene-graph](../142-world-scene-graph/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Graph + bounded admission shipped (graph selector default-off); residue US4/US5, T048, T054-T057, promotion gates. |
| [143-world-context-lighting](../143-world-context-lighting/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | US1 area names + Alpha world clock shipped; US2/US3/US5 to 252, US4 lighting to 249. |
| [144-camera-capture-paths](../144-camera-capture-paths/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Code complete; 5 user-run capture proofs owed. |
| [146-audio-camera-playback](../146-audio-camera-playback/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | WorldAudioRuntime shipped; residue US1/US3/US4/US5, T017a (merged with 148). |
| [147-minimap-fog-instancing](../147-minimap-fog-instancing/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | US1 minimap gesture shipped; US2 fog residency, US3 doodad batching, US4 diagnostics open. |
| [148-world-simulator](../148-world-simulator/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phase 1 audio shipped; Phases 2-4 to 249 (merged with 147), Phase 5 + T006/T006a to 252. |
| [149-pm4-region-audio-controls](../149-pm4-region-audio-controls/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | US4 area overlay shipped; residue US1/US2 + audio enablement. |
| [150-alpha-renderer-performance](../150-alpha-renderer-performance/) | SUPERSEDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Superseded by 152/153. |
| [151-portal-game-mode-surface](../151-portal-game-mode-surface/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | US1 portal slice shipped; T030 + US5 to 249; US2 to 251; US3/US4 to 252. |
| [152-renderer-frame-stability](../152-renderer-frame-stability/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phase 0/1 shipped; Phase 6 era lighting to 249, Phase 7 UI ownership to 251. |
| [153-renderer-hitch-and-batching](../153-renderer-hitch-and-batching/) | COMPLETE | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phases 0-3 landed; residue Phase 5 step 2 async decode, WMO admission handoff. |
| [154-m2-era-reader-parity](../154-m2-era-reader-parity/) | SUPERSEDED | [248](../../248-epic-formats-and-conversion/spec.md) | Absorbed by 235; US4 cross-era rig comparison parked. |
| [155-wmo-doodad-chronology](../155-wmo-doodad-chronology/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | US1 inventory shipped; US2..US6 open. |
| [156-precise-object-selection](../156-precise-object-selection/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Cursor covered by 210/211; residue triangle-precise picking (252), PM4 match library (253). |
| [157-lit-documentation-update](../157-lit-documentation-update/) | COMPLETE | [248](../../248-epic-formats-and-conversion/spec.md) | LIT documentation drafted; wiki submission is an operator action. |
| [158-alpha-demo-restoration](../158-alpha-demo-restoration/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | US1 via 159; residue US2..US6. |
| [159-wtf-command-inspection](../159-wtf-command-inspection/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Sweep tool shipped; residue full build sweep, --listfile rerun. |
| [160-skybox-rendering](../160-skybox-rendering/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [166-editor-plugin-host](../166-editor-plugin-host/) | COMPLETE | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Plugin host shipped and tested; FR-006/SC-003 minor residue. |
| [167-editor-runtime-bridge](../167-editor-runtime-bridge/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Bridge partial; residue FR-007 staging retirement, applier completion. |
| [168-editor-session-undo](../168-editor-session-undo/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Session partial; residue exit warning, all-ops undo/save. |
| [169-chunk-clipboard-plugin](../169-chunk-clipboard-plugin/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Migration not started. |
| [170-dbc-table-browser](../170-dbc-table-browser/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open. |
| [171-dbc-table-editing](../171-dbc-table-editing/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open. |
| [172-editor-edit-journal](../172-editor-edit-journal/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open. |
| [173-asset-integrity-gate](../173-asset-integrity-gate/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Gate shell only; residue validators, census. |
| [174-asset-repair-patterns](../174-asset-repair-patterns/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open (after 173). |
| [175-placement-authoring](../175-placement-authoring/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Residue FR-002 add placement, session routing. |
| [176-object-transfer](../176-object-transfer/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | 8/13 tasks; residue T002-T008. |
| [177-adt-tile-creation](../177-adt-tile-creation/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open. |
| [178-mcp-automation-surface](../178-mcp-automation-surface/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Entire spec open (merged with 213). |
| [179-patch-chain-resolver](../179-patch-chain-resolver/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open (datastore gate). |
| [180-multi-build-datastore](../180-multi-build-datastore/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open. |
| [181-incremental-processing](../181-incremental-processing/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open. |
| [182-adaptive-encoding](../182-adaptive-encoding/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open. |
| [183-datastore-viewer-load](../183-datastore-viewer-load/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open. |
| [184-pm4-generation-from-geometry](../184-pm4-generation-from-geometry/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Entire spec open. |
| [185-pm4-pd4-format-documentation](../185-pm4-pd4-format-documentation/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Merged with the 188/189 field-semantics thread. |
| [186-server-data-transformer](../186-server-data-transformer/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open. |
| [187-museum-world-simulation](../187-museum-world-simulation/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open (after 186). |
| [188-pm4-field-semantics](../188-pm4-field-semantics/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Residue US1..US3 (merged). |
| [189-pm4-complete-field-map](../189-pm4-complete-field-map/) | FOLDED | [253](../../253-epic-pm4-navmesh-research/spec.md) | Residue MSLK._0x04, AdtPm4MaskBuilder space. |
| [190-rosetta-calibration-corpus](../190-rosetta-calibration-corpus/) | COMPLETE | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Rosetta corpus shipped. |
| [191-procedural-garden-museum-generator](../191-procedural-garden-museum-generator/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Residue T022-T024 (overlaps 236 US4). |
| [192-terrain-template-brush-generator](../192-terrain-template-brush-generator/) | COMPLETE | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Template brush library + generator shipped. |
| [193-benilla-112-client-reference](../193-benilla-112-client-reference/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Benilla oracle methodology unused; T101-T303 parked. |
| [194-temporal-stratigraphy-weak-signals](../194-temporal-stratigraphy-weak-signals/) | COMPLETE | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Stratigraphy shipped; runtime NFR proofs owed. |
| [196-wdl-lattice-magnetization-stratigraphy](../196-wdl-lattice-magnetization-stratigraphy/) | COMPLETE | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | WDL magnetization shipped; AC-002/AC-005 runtime proofs owed. |
| [197-workspace-profiles-editor-and-mop-adt-pipeline](../197-workspace-profiles-editor-and-mop-adt-pipeline/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Phase 5 MoP to 248; Phases 1-4 workspace/editor mode to 251. |
| [198-m2-shader-permutations](../198-m2-shader-permutations/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [199-mcal-decode-correctness](../199-mcal-decode-correctness/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Entire spec open. |
| [200-wmo-portal-admission-fallback](../200-wmo-portal-admission-fallback/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [201-render-path-metric-attribution](../201-render-path-metric-attribution/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phase 1 landed; T005/T006 + Phase 2 open. |
| [202-unified-model-batching](../202-unified-model-batching/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phase 0/1 landed; Phases 2-6 open (batching anchor). |
| [203-multi-phase-map-composition](../203-multi-phase-map-composition/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Composition shipped; residue FR-002/FR-003 uniqueId reconciliation. |
| [204-off-thread-asset-decode](../204-off-thread-asset-decode/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [205-mh2o-liquid-object-vertex-format](../205-mh2o-liquid-object-vertex-format/) | COMPLETE | [248](../../248-epic-formats-and-conversion/spec.md) | MH2O LiquidObject resolution shipped; operator verifications owed. |
| [206-zarr-first-residency](../206-zarr-first-residency/) | FOLDED | [254](../../254-epic-datasets-and-terrain-ml/spec.md) | Entire spec open (datastore residue). |
| [207-object-draw-call-reduction](../207-object-draw-call-reduction/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phase 1 landed; Phase 2/3 open. |
| [208-cross-map-tile-transplant](../208-cross-map-tile-transplant/) | SUPERSEDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Built inside 219/222/232; narrow residue FR-007, picker, FR-008. |
| [209-wlw-mclq-convergence](../209-wlw-mclq-convergence/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Diagnosis complete; T4-T6 fix open. |
| [210-3d-scene-cursor-selection](../210-3d-scene-cursor-selection/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Shipped; T401-T404 owed. |
| [211-wmo-interior-picking-ghost-wireframe](../211-wmo-interior-picking-ghost-wireframe/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Shipped; T407/T505 owed. |
| [212-spatial-ui-shell](../212-spatial-ui-shell/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | US6 selection silhouettes carried; rest cold. |
| [213-mcp-tooling-harness](../213-mcp-tooling-harness/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Entire spec open (merged with 178). |
| [214-mop-physics-domino](../214-mop-physics-domino/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | .phys reader + policy shipped; T002, T021-T031 open. |
| [215-mop-weather-system](../215-mop-weather-system/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open. |
| [216-model-cursor-light-source](../216-model-cursor-light-source/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open. |
| [217-audio-lifecycle](../217-audio-lifecycle/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open. |
| [218-creature-staging](../218-creature-staging/) | FOLDED | [252](../../252-epic-world-simulation-and-audio/spec.md) | Entire spec open. |
| [219-phase-layer-rotation](../219-phase-layer-rotation/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Phases 1/2/5 largely live; Phases 3/4/6/7/8 open. |
| [220-wmo-doodad-editing](../220-wmo-doodad-editing/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Entire spec open. |
| [221-converter-validation-harness](../221-converter-validation-harness/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Phase 0 findings landed; Phases 1-4 open. |
| [222-map-composition-workbench](../222-map-composition-workbench/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Residue T109-T114. |
| [223-ui-consolidation-audit](../223-ui-consolidation-audit/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Residue T101-T107, T201-T203, T301/T302, T401, T501/T502, T609/T610. |
| [224-speckit-governance](../224-speckit-governance/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Rules codified in AGENTS.md section 9; Gate 1 + ledger sync open. |
| [225-overhead-ortho-view](../225-overhead-ortho-view/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Phase 1 open. |
| [226-renderer-polish](../226-renderer-polish/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | US1/US2 open. |
| [227-ui-reaudit](../227-ui-reaudit/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | T001/T002 done; US2..US4 open. |
| [228-source-decomposition](../228-source-decomposition/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Entire spec open (blocked on 227-T004). |
| [229-wow-shell-keybind-profiles](../229-wow-shell-keybind-profiles/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Entire spec open. |
| [230-reconstruction-editor](../230-reconstruction-editor/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | US1 Rosetta placement open; US2/US3 went to 234. |
| [231-editor-archaeology-ui-overhaul](../231-editor-archaeology-ui-overhaul/) | FOLDED | [251](../../251-epic-viewer-ux-and-code-health/spec.md) | Residue T041, T050-T052, T061-T064, T080. |
| [232-cartography-composition-project](../232-cartography-composition-project/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Residue T020-T022, T030-T032, T055, T060-T071 + witnesses. |
| [233-marketing-capture-automation](../233-marketing-capture-automation/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | US1 source landed; residue T015-T029. |
| [234-map-save-new-map](../234-map-save-new-map/) | FOLDED | [250](../../250-epic-reconstruction-and-editor-platform/spec.md) | Save pipeline FR-001..FR-007 open; New Map creator partly exists. |
| [235-legacy-mdx-m2-rendering](../235-legacy-mdx-m2-rendering/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Phases 0-4 landed; residue FR-014, FR-015, US4, US5 + visual gates. |
| [236-scene-lighting-doodad-performance](../236-scene-lighting-doodad-performance/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Phases 1/2 landed; T013 + Phase 3 to 249, Phase 4/5 to 250. |
| [237-adt-v26-terrain](../237-adt-v26-terrain/) | COMPLETE | [248](../../248-epic-formats-and-conversion/spec.md) | DAT v22/v23/v26 read + render shipped; v22 AMAP carried via 247. |
| [238-casc-data-source](../238-casc-data-source/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Local CASC shipped; residue US2 CDN streaming, US3 builds by id. |
| [239-modern-client-assets](../239-modern-client-assets/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | FDID readers shipped; residue FR-001 resolver, FR-008 survey, tier A/B. |
| [240-format-conformance](../240-format-conformance/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | T000a-f shipped; survey + Phases 3-7 open. |
| [241-dat-v26-interchange](../241-dat-v26-interchange/) | COLD | [248](../../248-epic-formats-and-conversion/spec.md) | Operator-deferred; one-way export lives in 247. |
| [242-wmo-instancing-performance](../242-wmo-instancing-performance/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [243-modern-to-legacy-map-conversion](../243-modern-to-legacy-map-conversion/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Plan authored; entire implementation open. |
| [244-modern-liquid-flow](../244-modern-liquid-flow/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Entire spec open. |
| [245-modern-chunk-completeness-survey](../245-modern-chunk-completeness-survey/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | Entire survey open (one scoping note exists). |
| [246-modern-m2-camera-paths-and-benchmarking](../246-modern-m2-camera-paths-and-benchmarking/) | FOLDED | [249](../../249-epic-renderer-performance-and-correctness/spec.md) | Entire spec open. |
| [247-dat-capture-and-adt-export](../247-dat-capture-and-adt-export/) | FOLDED | [248](../../248-epic-formats-and-conversion/spec.md) | US3/US5 shipped; US1 AMAP, US2 capture, witnesses open. |
