# Active Context — wow-viewer

Last updated: 2026-08-08

**This file is a dashboard, not a log.** It says what is live, what changed last, and where the
detail lives. Findings belong in the workstream file, not here — see "Memory bank layout" in
`coding_standards.md`. If a section here grows past a screen, it belongs somewhere else.

## Workstreams

| Workstream | State | Detail |
|---|---|---|
| PM4 decode | **active** — versioning formatted; placement solved; scene graph tree view restored | [workstream-pm4-decode.md](workstream-pm4-decode.md) |
| Terrain / viewer runtime | **active** — phased dual-map overlay (135), M2 doodad batching (136), phased minimap & teleport (137) landed | [activeContext.md](activeContext.md) |
| Terrain / minimap ML | **active** — Spec 134 synthetic v60 control corpus is being finalized; no corpus generation or training has run | [workstream-terrain-ml.md](workstream-terrain-ml.md) |
| Tile archaeology | **active** — harvest pipeline working on 1.x clients; spec 132 phase 1 landed | [weak-signal-tile-archaeology.md](weak-signal-tile-archaeology.md) |

## Now — Viewer Runtime & Terrain Improvements (Specs 135, 136, 137)

## Now — Spec 134 v60 control corpus

- The earlier v50/multi-client harvest direction is not the active v60 deliverable. No working v60
  real-data corpus has been accepted or generated in this lane.
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
- Detail and commands: [Spec 134 quickstart](../specs/134-v60-unified-dataset-model/quickstart.md).

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
