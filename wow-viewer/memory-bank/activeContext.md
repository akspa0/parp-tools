# Active Context — wow-viewer

Last updated: 2026-08-10

**This file is a dashboard, not a log.** It says what is live, what changed last, and where the
detail lives. Findings belong in the workstream file, not here — see "Memory bank layout" in
`coding_standards.md`. If a section here grows past a screen, it belongs somewhere else.

## Workstreams

| Workstream | State | Detail |
|---|---|---|
| PM4 decode | **active** — versioning formatted; placement solved; scene graph tree view restored | [workstream-pm4-decode.md](workstream-pm4-decode.md) |
| Terrain / viewer runtime | **active** — phased dual-map overlay (135), CPU and GPU M2 doodad batching (136), phased minimap & teleport (137) landed; 4.x renderer evolution (138) is an evidence-gated epic note | [activeContext.md](activeContext.md) |
| World scene graph / renderer performance | **active** — Spec 142 graph, conservative traversal, default-on `WorldScene` object adapter/selector, metadata-only graph rebuilds, ADT M2 doodads partitioned beneath terrain chunk buckets, nested WMO groups, WMO read-model portal adapter, bounded portal view volumes, graph-side runtime portal traversal, safe opaque WMO doodad batching, and production headless CPU-stage reporting implemented; WMO submission parity, pass/query reuse, terrain-mesh ownership, and GPU-stage evidence remain pending | [Spec 142](../specs/142-world-scene-graph/spec.md) |
| Terrain / minimap ML | **active** — Spec 139 clean-signal reconstruction, Spec 140 paste/fractal/tileset evidence, and Spec 141 external-method translation; object identity remains parked | [workstream-terrain-ml.md](workstream-terrain-ml.md) |
| Tile archaeology | **active** — old v50 synthetic minimap output needs fresh regeneration after renderer fixes; spec 132 phase 1 landed | [weak-signal-tile-archaeology.md](weak-signal-tile-archaeology.md) |

## Now — Spec 142 renderer grounding

- Spec 142 now distinguishes `synthetic_world_scene`, `synthetic_minimap_asset`,
  `real_client_scene`, and `mixed_validation_scene` as separate evidence classes.
- The synthetic renderer lane must use the same runtime graph, visibility, pass, and query
  contracts as real content; a generated minimap or 2-D preview cannot satisfy 3-D renderer proof.
- The required order is baseline current path, deterministic synthetic identity, synthetic
  scaling, real-client parity, then promotion. Reports must separate CPU traversal, submission,
  GPU/driver wait, upload, and query stages with median and p95 samples.
- Phase 1 now builds `WorldSceneGraph` plus nested WMO/M2/PM4 synthetic fixtures in
  `WowViewer.Core.Runtime`; focused proof is 8 passing tests. This does not yet change `WorldScene`.
- Phase 2 now provides `WorldSceneTraversal`: one injected visibility test can reject an entire
  region, with explicit skipped-descendant attribution and fail-open handling for unknown bounds.
  Focused graph/workload/traversal proof is 11 tests.
- The current bounded integration adapts existing `WorldObjectInstance` lists into stable
  `map -> tile/external bucket -> placement` nodes. Client-backed `WmoMeshSummary.GroupSummaries`
  now mount as nested `WmoGroup` children. `WorldScene.UseHierarchicalSceneTraversal` is now
  default-on and feeds one conservative graph traversal into the existing WMO/MDX collectors; the
  legacy path remains available as a runtime fallback. `WorldScenePortalGraph` now provides graph-only adjacency and bounded traversal
  diagnostics for malformed, cyclic, missing-entry, absent-data, and depth-limited cases.
  `WorldScenePortalAdapter` consumes existing `WmoRenderDocument` portal read models without
  changing readers, preserving valid geometry and failing open for malformed geometry or unknown
  groups. `WorldScenePortalViewVolumeBuilder` now preserves parent planes and creates bounded
  child portal volumes with explicit fallback reasons. Focused proof is 26 tests and the viewer
  project compiles. Loaded `WmoRenderer` portal read models now populate placement-keyed
  `SceneGraphPortalAdapters` during opt-in graph rebuilds; the current WMO visibility path is
  unchanged. `WorldScenePortalVisibilityEvaluator` now walks reachable groups through nested
  volumes and applies only to opt-in `WmoGroup` graph traversal, failing open to all groups when
  uncertain. Focused proof is 33 tests.
- Traversal diagnostics now expose individually tested nodes, rejected subtree roots, and skipped
  descendants by node kind, so ADT Chunk rejection can be separated from individual M2 culling.
- Opt-in graph traversal now defers exact visibility of M2 leaves under Chunk nodes to the existing
  M2 collector, while retaining graph-owned chunk rejection; external, skybox, WMO, and WMO doodad-set
  paths are unchanged.
- The opt-in graph set now gives each resident ADT its own `Tile`-rooted graph and keeps external
  content separate; WorldScene traverses those graphs independently. This is an ownership and
  traversal-boundary change, not yet a measured performance result.
- A severe residency regression was traced to graph rebuilds calling `TryGetWmoMeshSummary`, which
  synchronously read and parsed each resident WMO just to discover optional group children.
  Rebuilds now use `TryGetCachedWmoMeshSummary`; missing metadata remains fail-open and does not
  block tile residency on WMO I/O.
- The first safe submission-batching slice is now wired: static legacy-backed M2s can use the
  existing shared batch path, while native-runtime M2s remain isolated behind their distinct state
  path. Opaque WMO doodads group by `IModelRenderer`; transparent and particle/ribbon cases retain
  ordered or unbatched fallbacks.
- The first true GPU slice is now wired for compatible opaque M2/MDX batches: one dynamic instance
  VBO carries model matrices/fade values, and each compatible geoset uses `glDrawElementsInstanced`.
  Native-runtime state, transparent/effect-heavy paths, and unsupported fades remain fallback-safe.
  Runtime visual parity and frame-time proof are still user-run.
- A second runtime-loading multiplier was then found: deferred WMO doodad loads were being advanced
  from every visible `WmoRenderer.RenderWithTransform` call. `WorldAssetManager` now owns one
  scene-wide deferred-doodad budget per frame, and `MinimapRenderer` uses one background reader
  against the shared client data source instead of four. This is an I/O fan-out containment fix;
  user-run stage diagnostics still own the proof of whether parsing, terrain upload, or GPU wait is
  the remaining bottleneck.
- `WowViewer.Tool.ValidationCapture profile-render` now creates a hidden OpenGL surface and calls
  the real `WorldScene.Render` loop rather than a terrain-only or render-plan stand-in. Its
  `world-render-diagnostic-v1` report preserves all current CPU stages, frame/submission counts,
  streaming queues, initialization time, and MPQ read-cache counters, then emits actionable
  findings. Per-stage GPU/driver timer queries remain explicitly unproven and are the next report
  expansion rather than an assumption about the bottleneck.
- The adapter does not invent missing group bounds. Resident non-skybox ADT M2 placements now mount
  beneath deterministic terrain chunk buckets in the opt-in graph; unresolved bounds keep the chunk
  fail-open. This is object-population partitioning, not terrain mesh ownership. The existing
  `WmoRenderer` still owns WMO group and WMO-internal doodad-set submission. Heavy real-scene
  captures and GPU measurements remain user-run. Runtime stats now expose graph roots and rejection
  counts plus AOI camera/retention and last WMO-bearing tile-unload evidence. Next gaps are runtime
  WMO submission/doorway parity, pass/query ownership, terrain-mesh mounting, GPU visual parity,
  and stage-level performance evidence.

## Now — Viewer Runtime & Terrain Improvements (Specs 135, 136, 137)

## Next — Cross-era terrain foundation (Spec 138)

- The new reference dossier is preserved at `.reference_data/4.0.0.11792/` and indexes 19 Ghidra
  audit modules for Build 11792. It is reference input, not yet production proof.
- The priority is now a reusable, profile-gated terrain core spanning basic terrain support from
  0.5.3 through 11.x. Build 11792 is the first modern evidence anchor; 0.5.3 parity follows this
  foundation rather than becoming a separate first renderer lane.
- Existing 4.0.0 support is partial but usable: terrain/world content and basic WMO/M2 rendering
  work, while shaders, visual effects, some lava-effect models, fog, proper lighting/point lights,
  batching, and CPU-bound submission remain the main renderer gaps.
- Spec 138 owns the cross-era profile, terrain-signal, M2/WMO, synthesis-lineage, and performance
  roadmap. File ownership, chunk availability, vertex layout, and optional signals must be profile
  capabilities; no broad renderer rewrite, harvest, training, or long benchmark is authorized by
  the note.
- Live Ghidra evidence for Build 11792 now confirms active `MCLV`/`MCCV`/`MCNR`/`MCSH`
  consumers and terrain shader axes for vertex color, shadows, PCF, layer count, point lights,
  environment mapping, and tessellation. `MCTV`/`MCMT` were not observed in this build. The
  CPU vertex builder preserves raw MCNR order; trace the Terrain shader before changing the
  normal transform. Detailed addresses and implications are recorded in Spec 138.
- Priority findings to verify against real clients: 8+ observed `MCLY` capacity, `MCLV`, `MCTV`,
  `MCMT`, `MCLY` `0x100`/`0x200`, `MD21`, 4.x WMO materials, instancing, and monolithic versus
  split ADT ownership.
- wow.export is an allowed comparative reference for modern 12.x-era behavior, but not a runtime
  dependency or a reason to fork the repo-independent viewer contract.
- DBCD, WoWDBDefs, and wowdev listfiles are already integrated project authorities. Spec 138 must
  reuse them; only the missing MPQ/CASC source-adapter and build-profile capability seams are new.

## Now — Spec 139 v7 clean-signal reconstruction

**Current route pivot (2026-08-10):** the one-channel `terrain_shadow_256` → `height_257` experiment
is retained as negative evidence, not the active model contract. Spec 139 now owns the next lane:
recover v7's coarse/detail and structural-loss bias while replacing its WDL trestle, height hints,
normals, liquid, object, and other target-derived channels with a four-channel albedo-normalized
observation package: luma, x/y gradients, and albedo confidence.

The first v60 architecture bakeoff rejected all four candidates against the `0.191047` tile-mean
baseline; `pyramid_cnn` was best at `0.236665`, while cross-tile lightning/burn drove the failure.
That result does not close the v7 hypothesis because it did not test v7's coarse/detail structural
loss stack or a deployment-safe albedo-normalized input.

- The earlier v50/multi-client harvest direction is not the active v60 deliverable. No authored
  real-data corpus has passed the albedo gate; a separate `real_terrain_synthetic` bridge now
  exists for diagnostic testing on real-client terrain geometry.
- The current first experiment is a project-owned, deterministic control corpus: 27 terrain families
  × 4 variants = 108 rows, with complete-family holdouts and `easy`/`medium`/`hard`/`pathological`
  buckets. It also emits a sibling `object-sieve-v1` derivative with 540 rows.
- The control taxonomy now includes mountainous relief, arbitrary-angle sheer drop-offs, zone-style
  blends, fBm, ridged fractal, dendritic lightning-burn terrain proxies, and two global 2×2
  cross-tile families. Non-grid families carry deterministic sub-cell offsets; only `chunk_grid` is
  exactly cell-aligned. Cross-tile metadata and stitched visual atlases are required before any
  model run.
- The C# generator, Python validators, visual reviews, and object-sieve model/loss variants are
  implemented and focused checks are being finalized.
  The user still runs the actual corpus generation, any 0.x/1.x client transfer sample, and all
  training/GPU work.
- The object-sieve control lane is now emitted alongside the terrain corpus: synthetic tree/rock/
  building/bridge overlays, clean terrain-shadow targets, and a separate screen-space contamination
  mask across none/sparse/dense/overlap/boundary-crossing regimes. The mask is loss-supervised and
  optionally predicted-mask-guided, never supplied as a ground-truth inference channel.
- The promoted object lane is now `v60-object-library-sieve-v1`: it reads the real
  `object_mask_library_0_5_3_3368.zarr` (5,349 captured 0.5.3 top-down objects with RGB + masks),
  composites exact silhouettes over clean controls, and emits union masks plus per-instance IDs and
  library provenance. The old v50 curriculum `object_mask`/`object_precise_mask` experiment
  produced tile-level dots and is diagnostic/rejected, not model evidence.
- New builder, validator, visual atlas, trainer, and focused tests are under `data-harvester/`.
  The user still runs corpus generation and all CUDA training; no real-library corpus or GPU run has
  been launched by Codex. The user's first two builds stopped at 251 NPZs before their manifests
  because a thin library silhouette was erased by nearest-neighbour downsampling; both partial
  outputs are invalid and untouched. The corrected user run `object-library-sieve-v3` now passes:
  540 rows, 304 train / 236 validation, five complete regimes, 1,033 sampled objects, and 115
  library families.
- First user-run `library-guided-v1` training proved the mask head learns (`nonempty_mask_iou`
  peaked at 0.4183, versus 0.0 for the non-empty zero-mask baseline), but the absolute clean head
  lost to the identity baseline (best clean MAE 0.0372 versus contaminated-input MAE 0.0066).
  The clean head is now an identity-preserving residual with a required clean-vs-identity gate;
  `library-guided-residual-v1` is the next user-run experiment.
- The object lane is parked. The user-run marker experiment did not learn identity: held-out
  retrieval top-1 ended at 0, negatives were frequently accepted as known, and the input corpus did
  not preserve the real object RGB signal required for identity. Its checkpoint and reports remain
  diagnostic only and must not enter the terrain model.
- Spec 139 contract, model, synthetic packaging, visual-review, and loss slices are implemented in
  `data-harvester/src/harvester/v60/`, with fail-closed validator/builder/visual CLIs in `scripts/`.
  The shared model contract covers `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2`; the loss
  contract now exposes versioned parity and v7 structural profiles with independent differentiable
  point/gradient/frequency/curvature/edge/transition/border/LF/HF metrics. The shared trainer and
  PowerShell-ready dry-run CLI now
  fixes deterministic split identities, four-channel lazy loading, independent final/coarse/detail
  reports, family/bucket metrics, and best/last checkpoint binding. Focused loss/trainer/CLI proof is
  9 tests and the full v60 folder is 76 passing after this slice. The clean corpus builder publishes
  a hash-bound synthetic manifest atomically and visual review emits family/variant/cross-tile
  atlases with provenance JSON. The user completed the six-cell CUDA within-family matrix and the
  full-profile `pyramid_cnn/v7_structural_v1` complete-family gate. The full-profile checkpoint's
  best epoch 37 reached final-height MAE `0.173904` versus the `0.191047` tile-mean baseline
  (`8.97%` overall improvement) across 76 train and 32 held-out rows. `cross_tile_burn` regressed
  `15.52%`, `cross_tile_lightning` regressed `229.79%`, and the pathological bucket regressed
  `2.81%`; the explicit cross-tile acceptance scenario therefore holds promotion. The checkpoint
  is diagnostic evidence only and real transfer remains blocked. A prediction-only checkpoint
  diagnostic CLI now exports held-out per-row predictions/errors and full/cross-tile atlases; the
  user-run diagnostic exposed a constant-field failure: `flat-v00` and
  `cross_tile_lightning-v01` have nearly identical inputs, but the legacy zero-padding checkpoint
  emits the same non-flat ramp for both near-zero targets. New model identities now use
  `reflect-3x3-v1` padding; old identities remain loadable for comparison. The user completed the
  `v2-reflect-padding` full-profile run at best epoch 80 with MAE `0.137891` versus the `0.191047`
  baseline (`27.82%` improvement). The flat-input ramp is gone, but cross-tile lightning and burn
  still regress `61.17%` and `30.15%` against their baselines. The next user-owned gate is a
  full-profile within-family run using all 81 training rows to distinguish missing family coverage
  from missing clean-signal information. The existing 16-row Alpha/Azeroth real-terrain bridge was
  also materialized and evaluated: MAE `0.323879` versus `0.157124` tile-mean baseline
  (`-106.13%`), with zero forbidden reads. The user then trained on 15 bridge rows; best epoch 4
  reached `0.313952` versus `0.109902` baseline, and all-16 evaluation was `0.293371` versus
  `0.157124` (`-86.71%`). That 16-row directory was an old diagnostic subset, not the intended
  real-data scale. The complete v50.1 mixed curriculum Zarr store has 1,330 synthetic rows
  (688 Kalimdor, 642 Azeroth), but it is pre-Spec-133 and lacks `terrain_shadow_256`; it exposes
  raw `shadow_mask` instead. Raw MCSH is rejected as an inference route because minimaps do not
  carry it. The active real-observable baseline now reads `minimap_rgb`: the store has 1,325
  authored rows (688 Kalimdor, 637 Azeroth), and the builder derives raw luma/gradients with
  explicit absent confidence and no albedo-gate claim. Two old-subset rows are effectively flat
  and source dynamics vary widely, so authored raw-pixel learnability is the next user-owned gate.
  Authored RGB remains unaccepted until albedo normalization is implemented and measured. Codex
  launched no training.
- Detail and commands: [Spec 139 quickstart](../specs/139-v7-clean-signal-reconstruction/quickstart.md).

## Now — Spec 140 terrain paste and fractal motif archaeology

Spec 140 is the parallel evidence lane for the multi-stage reconstruction workflow. It treats
observation normalization, tileset identity, alpha/texturing, fractal descriptors, recurring
cross-tile pastes, geometry reconstruction, and object placement as separate signals with explicit
availability, provenance, confidence, and ablation ownership. The leaked 10.2 workflow map is
corroborating workflow evidence only; 0.x/1.x client data remains authoritative.

The first gate is a deterministic synthetic/real visual atlas and transformed-motif retrieval
benchmark. No neural motif model, broad harvest, or GPU run is authorized until recurrence and
cross-family leakage checks pass. Validated guidance may later feed Spec 139; unconfirmed matches
must remain soft evidence or be omitted. Exact object identity is deferred behind normalized
object-slot evidence.

The current authoring-order hypothesis is now explicit: opaque layer-0 base/“brain” texture,
recurring layer-1 rocky paste, later alpha-painted additions, terrain sculpting, then
surface/object refinement. Source-side alpha is therefore a candidate upstream paint/sculpt
scaffold, not merely a texture correlate. The implementation must preserve MCLY order and MCAL
offsets, treat layer 0 as opaque base rather than paste evidence, inspect layer 1 as the first
paste/paint candidate, and classify paint/relief links as intact, retextured, resculpted, unknown,
or insufficient-data.

The early Python connected-component extractor and later C# full-map segmentation are now treated
as complementary scales: `atomic_brush` components, `paste_block` groupings, and
`macro_prefab_context` parent regions. The C# macro/block result is not a bug or a replacement for
the Python atomic evidence. Spec 140 must preserve parent/child links, separate per-scale metrics,
and allow one-off or boundary-truncated records to remain unconfirmed. A frozen synthetic reference
model may score per-signal and seam difficulty for curriculum sampling (`easy`, `learnable_hard`,
`pathological`), but that score is explicitly not a staleness indicator or pseudo-target.

Alpha is the primary evidence substrate for this lane. Preserve every available layer and its
MCLY/MCAL/tile/map/build provenance before deriving any interpretation. Raw occupancy,
transition/stroke, atomic, paste-block, macro-context, ordered-layer, and cross-tile views are all
kept open; an unhelpful view must not erase the others, and unavailable/opaque data must not become
an invented empty mask.

- Spec: [140 terrain paste and fractal motif archaeology](../specs/140-terrain-paste-motif-archaeology/spec.md)
- Plan: [Spec 140 plan](../specs/140-terrain-paste-motif-archaeology/plan.md)

### What was accomplished this session (2026-08-08)

1. **PM4/PD4 Version Header Formatting** — `Pm4VersionFormatter.cs` parses version headers (`0x10` Cataclysm = v16, `0x30` WoD = v48). Integrated into status bar (`WorldScene.cs`) and CLI inspect tool (`WowViewer.Tool.Inspect`).
2. **Phased Terrain Dual-Map Overlay (Spec 135)** — `ITerrainAdapter`, `StandardTerrainAdapter`, `TerrainManager`, and `WorldScene` support `SecondaryOverlayMap` / `OverlayMapName`. Resolves split ADT payloads (`root`, `_tex0`, `_obj0`) from secondary map folders (`World\Maps\<OverlayMapName>\`) when tiles exist, evicting and re-streaming affected tiles in real time without unloading resident world tiles. Added a searchable map dropdown selector built from `_discoveredMaps` in `ViewerApp_Investigation.cs`.
3. **M2 Doodad Rendering Performance Optimization (Spec 136)** — Fixed massive framerate drops (<1 FPS) on dense object maps. Removed `_isM2AdapterModel` from `ModelRenderer.RequiresUnbatchedWorldRender` so static M2 doodads use high-throughput batched instancing (`BeginBatch()` once per pass + `RenderInstance()`). Deduplicated `UpdateAnimation()` in `WorldScene.cs` so shared models update at most once per frame.
4. **Phased Minimap Overlay & Consistent Minimap Teleport (Spec 137)** — `MinimapRenderer` & `MinimapHelpers` query active secondary overlay tile BLPs first, rendering phased minimap tiles on the minimap surface. Unified fullscreen minimap to use 3-click armed teleport (`MinimapTeleportMode.Armed`), matching the small dockable minimap panel.

### What was accomplished this session

1. **PM4 Scene Graph** — full scene outliner restored (Blender-style tree view with tile/CK24/Part hierarchy, MSLK linking summary, search filter, click-to-select). See [workstream-pm4-decode.md](workstream-pm4-decode.md).

2. **Single-command archaeology pipeline** — [`run-archaeology.ps1`](../scripts/run-archaeology.ps1) does harvest MPQ → V50 Zarr store → tile inventory → synthesis → composites. Proven working on TBC 2.0.0.5610 (Expansion01, 741 tiles, 34 weak signal, 186 white plate).

3. **Batch archaeology** — [`run-batch-archaeology.ps1`](../scripts/run-batch-archaeology.ps1) discovers all 15 1.x Windows clients in H:\CLIENTS, finds terrain maps via discover-maps, and runs the pipeline on each.

4. **Spec 132 drafted** — 6 user stories for terrain brush signature classification, including the Nov 2001 rescale boundary detection (33.33% horizontal roll).

### What was accomplished this session (2026-08-05)

**Spec 132 Phase 1 — three-tier brush-signature classification, implemented.**

- [`classify.py`](../data-harvester/src/harvester/v50/classify.py) — `compute_signal_tier()` with published criteria: weak (range < 5), normal (5-50 range OR 8-64 surviving levels OR low alpha<->height correlation), strong otherwise; `na` for zero-relief tiles. Deterministic (FR-006), never fabricates a score when alpha data is absent (FR-007).
- [`v50_tile_classify.py`](../data-harvester/scripts/v50_tile_classify.py) — CLI over V50 Zarr store or NPZ shard dir -> classify.csv/json + summary.json.
- `tile_inventory.py` gains `signal_class` / `signal_class_evidence` per row + `by_signal_class` summary; `tile_composite.py` gains green normal-tier outline; both archaeology orchestrators (`v50_archaeology.py`, `build_v50_store_from_npz.py`) run the classifier.
- 13 new unit tests pass; 22 existing inventory/composite tests still pass.
- Committed as `f19fc774` on branch `132-terrain-brush-signature-classification`. Spec/plan/tasks committed in the same change; tasks.md covers all 6 phases.

Next: Phase 2 (nested weak signal detection) per `tasks.md`.

### Harvested data already on disk

- `output/archaeology/2_0_0_5610/npz/Expansion01/` — 741 NPZ shards
- `output/archaeology/2_0_0_5610/store/Expansion01.zarr/` — V50 Zarr store
- `output/archaeology/2_0_0_5610/archaeo/` — tile inventory + synthesis sheets

### Open

- **3.x terrain darkness** — procedural fallback in `TerrainLighting.Update()` produces very dark night values. DBC lighting may not load for 3.x pre-release builds.
- **Composite images** — need to filter out non-weak tiles and add minimap overlay. The composite script renders all tiles; the `textured` mode needs minimap_rgb_256 present.

PM4 placement is **fixed and visually confirmed** (2026-08-04): tiles aligned, tents correctly
identified, previously-rotated walls and buildings correct. That unblocked the scene-graph work.

### Scene graph tree view restored (2026-08-04)

The **PM4 Scene Graph** panel now shows a **full hierarchical scene outliner** (like Blender's
outliner) with two modes:

- **Full Scene** mode — all PM4 objects organized by tile → CK24 → Part, with MSLK Group and
  linked MPRL ref counts shown at each level. Click any item to select it and frame the camera.
  Includes a search filter and right-click context menu (Select All Parts, Frame All Parts).
- **Selected Object** mode — existing detailed graph decomposition (TypeBucket → LinkGroup →
  MscnRef → Part), now with improved MSLK linking info display.

### MSLK Linking Summary (new)

The Full Scene panel now includes an **MSLK Linking Summary** section that shows corpus-wide
statistics computed from all loaded PM4 research contexts:

- Anchor-only vs path-window link counts (MspiFirstIndex < 0 vs >= 0)
- Component coverage (CK24 groups with and without MSLK links)
- RefIndex mismatch counts
- Research leads section pointing to the next open questions

### API additions

- `WorldScene.GetPm4TileObjectSummaries()` — returns flat tuple-based summaries for the outliner
- `WorldScene.SelectPm4ObjectByKey(int tileX, int tileY, uint ck24, int objectPart)` — direct
  selection without region lookup
- `WorldScene.GetPm4MslkLinkingStats()` — computes MSLK linking statistics across all loaded files
- `Pm4MslkLinkingStats` — public readonly struct for the stats

### Open

- **Component coverage.** 34.4% of components have no MSLK link at all, so GroupObjectId names
  only a minority. The anchor-only MSLK entries (`MspiFirstIndex < 0`, 53% of 1.27M links) are the
  next place to look.
- **MPRR.** The length-3 and length-7 record shapes are undecoded.
- **MSCN** as a co-equal connective-geometry candidate is still untested.

## Test state

`WowViewer.Core.PM4.Tests`: **102 passed, 1 failed** —
`Pm4RegionObjectGrouperTests.AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects`,
**pre-existing**, confirmed failing at baseline.

## Durable constraints

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- The user runs training, capture, client-backed proof, and all heavy/GPU work. Hand over the exact
  command; never launch it.
- No DepthAnything / multi-head / shared-weight model paths.
- Constitution IV: per-signal evidence — a strong signal must never mask a dead one.

## Incidental

`pm4 inspect` and `pm4 audit` accept `--output` and silently ignore it; the other `pm4` report
commands honour it.

## 2026-08-10 — v60 dataset viewer consumption boundary

Spec 134 now records the verified dataset boundary: v50.1 Zarr stores contain liquid mask/height
signals, per-build liquid type coverage, object placement/mask evidence, MCLY layer/tileset
metadata, and minimap/texture path evidence, but the current `output/datasets/v60` tree contains
control NPZ and experiment artifacts rather than a built unified v60 map Zarr.

Implemented the first viewer slice in `wow-viewer`: shared `DatasetVersionCatalog` discovery,
catalog tests, current liquid/object/tileset/texture/placement reporting in
`ZarrTileDatasetLoader`, and a persistent Settings selector with explicit VLM activation. Control
NPZ/run directories are excluded. VLM switching preserves camera state; current Zarr entries are
summary-only and fail closed because `LoadTile`/tensor rehydration remains unimplemented. Client
source/build and secondary client-map overlay remain separate authorities.

## 2026-08-10 — real tile observation boundary

Real reconstruction inputs are not limited to client-harvested tiles. Spec 134 now distinguishes
client-backed tiles, authored minimaps, and low-resolution media/reference imagery. A real image may
provide only RGB; unknown height, normal, liquid, alpha, object, tileset, and texture targets must
remain unknown. Native resolution, source bytes/hash, provenance, optional map/tile hints, and every
crop/alignment/de-albedo/upscale operation must be preserved separately from derived artifacts.

Added `RealTileObservation` and `RealTileObservationKind` to the shared dataset contract. The viewer
catalog recognizes explicitly named real-observation folders as reference-only, input-eligible but
non-renderable/non-target entries. The next bounded slice is an observation manifest/materializer
and original-versus-derived inspection surface; it is independent of direct Zarr tile decoding.

## 2026-08-10 — Spec 141 terrain-method translation Phase 0/1 complete

Created branch `141-terrain-method-translation` and the Speckit package under
`specs/141-terrain-method-translation/`. The lane translates external DSM/DTM, LiDAR ground-filter,
aerial object-mask, and geospatial-encoder research into explicit method records and modality gates.
DSM2DTM/ResDepth/SMRF/CSF remain offline/reference methods; the first executable WoW branch is
RGB-only object-aware terrain completion with no-mask, predicted-mask, and withheld-mask conditions.
The package also adds bounded research-lead records so novel signal observations retain provenance,
falsification tests, confidence, and next actions. The v60 implementation now exposes six method
records, four input-contract branches, canonical signal aliases, translation decisions, and fail-closed
input-read audits. The Phase 0/1 dry-run CLI and regression proof passed (`17` focused, `106` v60). The
manifest-only RGB planner now produces no-mask/predicted-mask/withheld-mask plans and correctly marks
the real `object-library-sieve-v3` as synthetic luma control-only: 540 rows, 304 train / 236 validation,
zero runtime-eligible RGB rows. The authored RGB planner remains blocked only by the missing user-built
corpus. Focused Spec 141 proof is now `22` tests total; full v60 is `111` passing. No external weights,
client data, corpus build, or training was run. Next gate: user materializes
the authored RGB corpus, then reviews the combined benchmark dry plan.
