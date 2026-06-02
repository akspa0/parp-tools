# ACTIVE CONTEXT — wow-viewer

## Branch
- `036-renderer-improvements`

## Renderer Planning Lane
- Renderer modernization planning now has an explicit convergence owner:
  - `wow-viewer/specs/036-renderer-improvements/spec.md`
  - `wow-viewer/specs/036-renderer-improvements/plan.md`
  - `wow-viewer/specs/036-renderer-improvements/tasks.md`
  - `wow-viewer/specs/036-renderer-improvements/research.md`
- Purpose:
  - converge source plans `030-wmo-render-pass-architecture`
  - `031-terrain-cell-awareness`
  - `032-native-renderer-parity`
  - into one library-first owner plan for terrain, WMO, lighting, sky/fog, liquid, and thin viewer-host integration
- Source specs 030-032 remain reference slices and now point readers to 036 as the active owner plan
- Spec `035-m2-render-parity-recovery` remains a separate adjacent lane and is not absorbed into 036
- 2026-06-01 RE-grounded planning refresh landed for `036` using staged `3.3.5.12340` Ghidra evidence:
  - added runtime-controls inventory and phase gates for:
    - terrain/light/fog/liquid controls
    - projected-texture and water LOD toggles
    - bounded M2 runtime optimization flags as dependency diagnostics (not ownership transfer)
  - added telemetry-first validation requirement so control snapshots/logs are required before screenshot-only signoff
- 2026-06-01 first implementation slice from the new LOD/cell route is now landed as telemetry scaffolding (before topology changes):
  - `WorldRenderFrameStats` now carries terrain and far-terrain visibility counters for runtime diagnostics:
    - `TerrainChunksCulled`
    - `WdlHiddenTileCount`
  - `WoWViewer` runtime frame assembly now populates those counters from live renderer state:
    - terrain from `TerrainRenderer.ChunksCulled`
    - WDL from `WdlTerrainRenderer.HiddenTiles`
  - sidebar diagnostics now surface rendered/culled terrain chunks plus visible/hidden WDL tiles in one line
- 2026-06-01 second implementation slice from the new LOD/cell route is now landed as core topology/hole correction:
  - fixed `WowViewer.Core.Renderer` terrain vertex topology mapping to the canonical interleaved 145-vertex layout:
    - replaced flat `row=i/17,col=i%17` decode with 9/8 row-walk decode in `TerrainMeshBuilder.GetVertexPosition(...)`
  - fixed terrain index generation to the canonical 8x8 cell fan topology:
    - each cell now emits 4 triangles around the matching inner vertex (12 indices/cell)
  - fixed hole-mask semantics to 4x4 groups over 2x2 cells:
    - replaced incorrect `cell/4` bit usage with `holeBit = 1 << ((cellY/2)*4 + (cellX/2))`
  - added focused regression tests in `wow-viewer/tests/WowViewer.Core.Tests/TerrainMeshBuilderTopologyTests.cs` covering:
    - no-hole index count
    - single-group hole removal count
    - full-hole empty geometry
    - interleaved vertex decode anchors
  - focused validation command passed (`18/18`):
    - `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TerrainMeshBuilderTopologyTests|FullyQualifiedName~WorldRenderCompositionBuilderTests|FullyQualifiedName~WorldRenderOptimizationAdvisorTests"`
- 2026-06-01 wireframe control parity follow-up landed for world object rendering:
  - world-scene wireframe toggle now applies to object renderers in addition to terrain:
    - `WorldScene.ToggleWireframe()` now forwards state to `WorldAssetManager.SetObjectWireframeEnabled(...)`
  - `WorldAssetManager` now tracks object wireframe preference and applies it to:
    - all currently loaded MDX/WMO renderers when toggled
    - newly lazy-loaded MDX/WMO renderers as they enter cache (immediate and deferred load paths)
  - this closes the "wireframe toggle for objects" gap in world mode while preserving existing standalone model/wmo wireframe behavior.
  - Alpha `0.5.3` layout guardrail remains explicit in active viewer path:
    - `AlphaTerrainAdapter` still reorders non-interleaved MCVT/MCNR (81 outer + 64 inner) into canonical interleaved 145 runtime layout before mesh generation.
- 2026-06-01 3.3.5 WMO interior bleed follow-up landed as bounded blend-route correction:
  - corrected `WmoRenderer.ResolveWmoBlendMode(...)` mapping for WMO MOMT blend ids:
    - `1` now maps to `AlphaKey` (cutout) instead of `Blend`
    - `2` now maps to `Blend`
    - `3` now maps to `Add`
  - rationale: mode `1` misrouted to transparent pass caused shell cutout surfaces to skip opaque depth ownership and made interior faces appear through exterior walls in some 3.3.5 placements.
  - proof: focused `WowViewer.Core.Tests` filter rerun passed (`18/18`).
- 2026-06-01 renderer convergence scope update (spec-only, no implementation yet):
  - spec `036-renderer-improvements` now explicitly tracks a `3.3.5.12340` liquid-family misclassification risk where MCNK-flag-only routing can render river/ocean as magma.
  - convergence requirements now require build-aware liquid classification evidence (per-build table semantics such as LiquidType/DBC-resolved meaning) instead of one hard-coded mapping across builds.
  - this is currently an outstanding implementation lane, added to spec/quality checklist and continuity as a bounded next liquid slice.
- 2026-06-01 liquid-family classification lane (spec 036 FR-012/FR-013) is now partially implemented in active viewer terrain path:
  - strengthened `StandardTerrainAdapter` MH2O liquid-family resolution with a build-aware trust gate for `LiquidType.dbc` class-column mappings.
  - `LoadMh2oLiquidTypeLookup(...)` still attempts class-column decoding (`Type`/`LiquidType`/`TypeID`), but now validates anchor IDs before accepting that mapping:
    - expected anchors: id `2` -> ocean, `3` -> water, `4` -> slime, `5` -> magma.
  - if anchor checks fail, class-column mapping is rejected and the adapter falls back to ID-family mapping (`MapMh2oLiquidTypeFallback`) for every loaded row, preserving build-specific ID interpretation without one global class hard-code.
  - fallback route is explicitly logged with build identifier for validation evidence.
  - bounded compile proof passed (isolated output paths; 0 errors).
- 2026-06-01 liquid-family follow-up for staged `3.3.5.12340` narrowed the remaining gap further:
  - verified the active viewer does load `LiquidType.dbc` for standard WDT terrain sessions; `ViewerApp` passes `_dbcBuild`/`_dbcProvider`/`_dbdDir` into [`new StandardTerrainAdapter(...)`](wow-viewer/src/viewer/WoWViewer/ViewerApp.cs:11800), and the adapter calls [`LoadMh2oLiquidTypeLookup(...)`](wow-viewer/src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:1715) when those are present.
  - verified `LiquidType.dbd` for build `3.3.5.12340` does **not** expose a simple family/class column like `Type`; it exposes fields such as `Name`, `Flags`, `MaterialID`, `Texture[6]`, `Color[2]`, `Float[18]`, `Int[4]` instead.
  - bounded fix landed in [`StandardTerrainAdapter`](wow-viewer/src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs:22): when class-column decoding is absent or fails trust checks, the adapter now classifies loaded rows by DBC `Name` heuristics (`ocean/sea`, `magma/lava`, `slime/ooze`, `river/water/lake/fast/slow`) and only then falls back to per-ID family defaults.
  - added `_mh2oLiquidTypeLookupAttempted` so unknown MH2O IDs still log evidence even when the loaded table resolves via heuristic/fallback rather than a dense class-column projection.
  - bounded compile proof passed again with isolated outputs (`wow-viewer/output/validation/liquid-fix-build`).
- 2026-06-02 terrain-cell runtime-knowledge slice landed for the new terrain/world performance lane:
  - implemented [`WorldTerrainHoleMask`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainHoleMask.cs:1) and [`WorldTerrainCellGrid`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainCellGrid.cs:1) in `WowViewer.Core.Runtime` as the first bounded port of spec `031`/`036` terrain-cell knowledge.
  - extended [`WorldTerrainChunkData`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs:7) so each runtime chunk now carries a typed hole-mask view plus a default 8x8 cell grid with stable vertex-index mapping for the 145-vertex interleaved layout.
  - this slice does **not** implement final distance LOD selection yet; it establishes the runtime-owned cell/hole semantics needed before chunk-level or cell-aware terrain LOD decisions can be layered in safely.
  - focused proof passed via [`WorldTerrainCellGridTests`](wow-viewer/tests/WowViewer.Core.Tests/WorldTerrainCellGridTests.cs:1), [`WorldTerrainTileBuilderTests`](wow-viewer/tests/WowViewer.Core.Tests/WorldTerrainTileBuilderTests.cs:9), and [`TerrainMeshBuilderTopologyTests`](wow-viewer/tests/WowViewer.Core.Tests/TerrainMeshBuilderTopologyTests.cs:6) with `12/12` passing under isolated outputs in `wow-viewer/output/validation/terrain-cell-lod-tests`.
- 2026-06-02 first terrain distance-LOD selector slice landed immediately after the runtime cell/hole foundation:
  - added [`WorldTerrainLodSelector`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainLodSelector.cs:1) with four bounded outcomes: `FullDetail`, `FadeToBaseLayer`, `BaseLayerOnly`, and `LowDetail`.
  - the selector now uses runtime chunk knowledge already exposed by [`WorldTerrainChunkData`](wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs:7), including layer count plus cell-grid-derived renderable cell counts from native-style hole masks.
  - current scope is decision logic only: no viewer-host shader wiring or low-detail terrain mesh submission changes yet. This is the policy seam needed before `TerrainRenderer`/runtime pass integration.
  - focused proof passed via [`WorldTerrainLodSelectorTests`](wow-viewer/tests/WowViewer.Core.Tests/WorldTerrainLodSelectorTests.cs:1) together with the earlier terrain-cell tests (`17/17` pass under isolated outputs in `wow-viewer/output/validation/terrain-lod-selector-tests`).
- 2026-06-02 WMO dark-lighting follow-up for staged `3.3.5.12340` landed as a bounded shader/input rebalance:
  - inspection showed the current WMO shader multiplies dynamic lighting by baked vertex/lightmap lighting uniformly with a hard-coded `0.6` mix, regardless of whether the group is interior or exterior.
  - spec/runtime docs still say interior 3.3.5 groups should rely much more heavily on baked illumination while exterior groups keep more dynamic light contribution; the old fixed-weight blend was collapsing many WMOs into overly dark masses.
  - fix landed in [`WmoRenderer`](wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs:1241): per-group baked-lighting weight is now computed from group flags plus sampled vertex-light contribution, uploaded as an extra vertex attribute, and used by the fragment shader instead of the previous constant `0.6` bake mix.
  - interior groups now bias heavily toward baked lighting; exterior groups still use baked lighting but keep a lower weight range.
  - bounded compile proof passed with isolated outputs in `wow-viewer/output/validation/wmo-lighting-fix-build`.
- 2026-06-01 3.3.5 WMO dark-surface follow-up landed as a bounded lighting-input correction:
  - `WmoRenderer` now prefers parsed WMO `MONR` normals when they are present and count-aligned with vertices, instead of always regenerating normals from triangle geometry.
  - parsed normals are normalized per-vertex with finite/length guards; invalid entries fall back to `Vector3.UnitY`; if the parsed set is unusable the renderer falls back to generated normals.
  - rationale: 3.3.5 WMOs can carry authored normal data that better matches client shading; always-regenerated normals made many surfaces appear darker than expected.
  - bounded build proof was captured with isolated output paths to avoid active viewer binary locks:
    - `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/build/wowviewer/bin/ -p:IntermediateOutputPath=i:/parp/parp-tools/output/tmp/build/wowviewer/obj-wowviewer/`
- 2026-06-01 taxi route video unexpected app-exit hotfix landed as capture-queue close-guard hardening:
  - symptom: recording a taxi route video and finishing ffmpeg write could close the entire viewer unexpectedly.
  - bounded fix in capture queue semantics:
    - `PendingCaptureRequest` now carries explicit `AllowWindowCloseOnCapture`.
    - queue completion now closes window only when both flags are true: `ExitAfterCapture && AllowWindowCloseOnCapture`.
    - startup automation capture paths explicitly set `AllowWindowCloseOnCapture` only for startup-intended `--exit-after-capture` behavior.
    - interactive UI capture/route-video paths keep `AllowWindowCloseOnCapture=false`, so normal recording completion no longer exits the app.
  - touched files:
    - `wow-viewer/src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs`
    - `wow-viewer/src/viewer/WoWViewer/ViewerApp_StartupAutomation.cs`
  - bounded build proof (isolated output paths) passed with 0 errors.
- New M2 note:
  - staged `3.0.1.8303` Northrend currently shows some `.mdx` placements failing both `.skin` resolution and M2-to-MDX fallback
  - treat these as a likely prototype `MD20` / `Model2` build-profile boundary
  - future proof owner is a Ghidra pass over staged `3.0.1.8303` `wow.exe`, not more blind renderer guessing
  - staged `3.3.5.12340` wrong-axis M2 animation regression is now localized to runtime compressed-quaternion payload interpretation drift:
    - runtime sampler had been reading `M2CompQuaternion` as a swizzled `(y, -x, z, w)` payload
    - converter/compatibility path was already reading direct little-endian `(x, y, z, w)`
    - current fix centralizes the direct read path in `WowViewer.Core.M2.M2CompQuaternion.FromRawLittleEndian(...)`
    - focused proof currently comes from `WowViewer.Core.Tests.M2RuntimeTests`; live viewer signoff still requires restarting the running viewer binary
  - follow-up `3.3.5.12340` motion-coupling seam is now localized to skin vertex bone remap:
    - `.skin` `BoneEntries` were parsed but ignored by `M2StaticRenderModelBuilder`
    - runtime now prefers `skin.BoneEntries[localSkinVertexIndex]` over raw `M2Vertex.bone_indices` when constructing runtime render vertices
    - this is the current best explanation for cases where separate visual sections move together after axis decode is fixed

## Primary Live Lane
- V16 terrain dataset + training is the current execution path.
- New V18 direction is now explicitly spec'd as a dataset-refinement-first lane:
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/spec.md`
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/plan.md`
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/tasks.md`
  - key shift: detect/mined pastes on stitched map canvases (multi-tile), then cross-build dedupe into canonical paste families, then build refined manifests for model training
  - rationale: tile-local mining/training overcounts copy-pasted motifs and fragments large authored regions
  - latest refinement captured in spec pack:
    - paste library is now a first-class metadata contract (stable IDs, canonical names, aliases, role/shape tags)
    - alpha-layer-aware signatures are required for dedupe/variant separation
    - MCNK AreaID overlap/distribution is required for macro-zone composition grouping
    - auto-naming + confidence + review/lock workflow is now planned as a dedicated phase
    - family-balanced manifest generation is now explicit (balance by paste family, not raw row frequency)
- **Spec 009 landed (2026-05-22):** 2,650-line comprehensive design specification at
  `wow-viewer/specs/009-full-project-reimplementation-spec/spec.md`.
  Covers all 28 sections: binary format specs, rendering pipeline, ML pipeline,
  CLI tools, PM4 research, WMO portals, terrain edge cases, M2 animation, GLSL
  shaders, converters, liquid system, object masking, world rendering, legacy
  renderer. Sufficient for full from-scratch reimplementation. This is now the
  master design reference for the project.
- `wow-viewer/specs/012-real-validation-batch-extraction/` now has bounded wow-viewer GPU proof on both staged anchors: `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`. `ValidationWorldSceneAdapter` now owns the hidden-window OpenGL render/readback path behind `IValidationWorldSceneAdapter`, and `WowViewer.Tool.ValidationCapture capture --gpu-viewer-style` completes `4/4` variants on both anchors. The same bounded runs now also emit compatible `images/<tile>_object_visibility_mask.png` and `images/<tile>_no_objects.png` outputs under the dataset root, so the first Phase 5 artifact handoff is landed for the proof surface. This still bypasses `WowViewerWorldScenePlanner` preview framing, but it continues to reuse `WorldGpuPreviewRenderer` as a temporary backend; the next open step is broader automation cutover plus longer-range renderer extraction.
- V16.1.1 is now the named next bounded implementation slice on top of the
  landed V16.1 base:
  - normal-first again, not a fresh family reset
  - smarter curated manifests, not blind full-pool replay
  - difficulty buckets: `easy`, `medium`, `hard`, `pathological`
  - bucket-aware epoch sampling for short scouting runs
  - stronger hard-region weighting inside the normal loss
  - optional uncertainty-guided normal training if the simpler weighting lane
    proves insufficient
  - target fresh-chat spec pack:
    - `wow-viewer/specs/007-v16-1-1-curated-normal-acceleration/`
  - the first implementation slice is now landed:
    - curation profile:
      - `normal_terrain_v16_1_1`
    - manifest fields:
      - usefulness scoring
      - difficulty buckets
      - scouting recipe metadata
    - trainer seam:
      - `--bucket-sampling-profile v16_1_1_normal`
      - `evidence/train_epoch_bucket_usage.jsonl`
    - hard-region weighting now includes:
      - height gradients
      - local normal variation
      - painted alpha transitions
      - MCLY transitions
      - terrain-valid masking remains authoritative
  - focused proof now exists at:
    - `wow-viewer/output/datasets/v16/curation/smoke_v16_1_1_curation_335/`
    - `wow-viewer/models/v16_1/normal/runs/smoke_v16_1_1_bucket_cpu/`
    - `wow-viewer/models/v16_1/normal/runs/smoke_v16_1_1_bucket_rotation_cpu/`
  - first longer GPU normal run now exists at:
    - `wow-viewer/models/v16_1/normal/runs/v16_1_1_normal_pool800_epoch256_autotune12_compile/`
  - current operator truth:
    - the first longer V16.1.1 normal run finished, but it still needs more epochs
    - V16.1.x normal training resumes with `--resume-checkpoint`, not `--resume-from auto`
    - `--epochs` remains the total run ceiling, so the continuation command must raise that ceiling above the completed checkpoint epoch
    - the shared V16.1 trainer now extends the cosine scheduler when resuming to a higher total epoch ceiling, so resume is no longer pinned to the original scheduler `T_max`
- V16.1.3 height-channel normal model is now the active normal-lane iteration:
  - adds `height_norm` as a 4th input channel to the normal model
  - single model, clean gradient flow — no separate refiner, no distillation loops
  - `V161NormalHeightModel` in `v16_1_models.py`: 3,549,955 params, Conv2d(4,64) first layer
  - CLI: `--height-channel` flag on `train_v16_1_normal.py`
  - smoke proof: 10 epochs, val_loss dropped from 2.01 to 0.54
  - 1000-epoch long run now training at `v16_1_3_height_normal_pool4000` with 12GB autotune
  - autotune selected batch-size=48, torch.compile enabled, ~172s/epoch
  - V16.1.2 refiner approach was attempted but failed due to detached computation graph; V16.1.3 replaces it
  - spec: `wow-viewer/specs/016-v16-1-3-height-channel-normal-model/`
- V16.1 is now the named next architecture lane for terrain models:
  - one independent model per target family
  - `minimap -> height`
  - `minimap -> normal`
  - `minimap -> holes`
  - `minimap -> liquid footprint + liquid type`
  - `minimap -> MCLY/MCAL decomposition + recomposition`
  - shared object-mask loss gating stays available across appropriate trainers
  - linked together into resulting terrain outputs after per-family prediction
- V16 stays as the baseline/reference trainer until V16.1 lands smoke proof.
- The first V16.1 implementation slice is now landed in code:
  - `v16_curation.py`
  - `v16_1_dataset.py`
  - `v16_1_models.py`
  - `build_v16_curation_manifest.py`
  - `train_v16_1_common.py`
  - `train_v16_1_height.py`
  - `train_v16_1_normal.py`
  - `train_v16_1_holes.py`
  - `train_v16_1_liquid.py`
  - `train_v16_1_texcomp.py`
  - `infer_v16_1.py`
- Focused proof that is already real:
  - normal-oriented curation manifest:
    - `wow-viewer/output/datasets/v16/curation/smoke_normal_curation_335/`
  - normal-only curated CPU smoke run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_curated_cpu/`
  - normal-only CPU smoke run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_cpu/`
  - height-only CPU smoke run:
    - `wow-viewer/models/v16_1/height/runs/smoke_height_cpu/`
  - stitched inference smoke from the normal checkpoint:
    - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_normal/3_3_5_12340.pred.zarr`
  - stitched inference smoke from the height checkpoint:
    - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_height/3_3_5_12340.pred.zarr`
- Current V16.1 liquid typing is a coarse `16x16` five-class grid derived from
  `mcnk_flags_16`: `none`, `water`, `ocean`, `magma`, `slime`.
- Current V16.1 normal-loss focus now explicitly combines:
  - `normal_mask`
  - object-filter-derived terrain weighting
  - `mddf_mask` / `modf_mask`
  - `liquid_mask`
  - deformation-aware detail steering from target height gradients plus local
    normal variation
  - operator knob: `--normal-detail-boost`
  - raw supervision guidance channels:
    - `terrain_valid_mask_257`
    - `object_presence_257`
    - `alpha_painted_256`
    - `mcly_any_16`
    - `what_plate_flag`
- Current object-mask correction state:
  - archive-backed ADT harvest now attempts geometry-derived WMO footprints in
    `AdtTensorPackBuilder` before falling back to projected MODF bounds
  - the new path is wired through `WowViewer.Tool.Harvest` via archive asset
    reads, so LK/Cata/V16 rebuilds can stop depending only on coarse WMO AABBs
  - MDDF loss gating is no longer just filename regex plus a fake scale-based
    height guess; archive-backed harvest now resolves doodad model bounds and
    filters tree or clutter assets plus tiny/tall doodads out of
    `object_filtered_mask_257`
  - `inspect_v16_harvest_samples.py` can now run raw-only without any finalized
    `.zarr` store, so object-mask QA can happen before dataset mutation
  - `inspect_v16_harvest_samples.py` now renders raw `mddf`, raw `modf`, and
    `object_filtered_mask_257` explicitly, so preview QA shows the real terrain
    loss gate instead of only the merged raw object channel
  - MdxViewer validation capture now also emits renderer-truth
    `object_visibility_mask` / `no_object_minimap` artifacts from the live
    primary/noobjects/objectsonly families instead of depending only on the
    harvester-side approximate object masks
  - capture overrides now keep doodads visible with world objects, so MDX/M2
    silhouettes flow into the same renderer-truth object-mask path as WMOs
  - startup automation can now queue a bounded validation batch directly with:
    - `--validation-dataset-root`
    - `--validation-output`
    - `--validation-resolution`
    - `--force-validation-regeneration`
    - `--exit-after-validation`
  - object-mask artifact generation is now build-aware:
    - `0.x` builds prefer the direct `objectsonly` silhouette so early
      underground-object bleed-through is preserved
    - later builds prefer `primary` vs `noobjects` diffs so terrain occlusion
      wins over terrain-hidden silhouettes
  - **Capture batch tuning hotfix (2026-05-28):**
    - hardcoded settle throttles replaced with plan-level configurable fields:
      - `RequiredSettledFrames` default `12` (was `48`)
      - `MaxFramesBeforeCapture` default `480` (was `2400`)
      - `BatchSettledFrames` default `2` (new)
      - `FastSettleAfterBatchReady` default `true` (new)
    - batch-fast-settle: after first tile in a batch settles successfully,
      subsequent tiles use `BatchSettledFrames` instead of
      `RequiredSettledFrames`, dramatically reducing per-tile settle wait in
      multi-tile batched sessions
    - per-tile capture metadata JSON now emitted alongside each PNG:
      `{baseName}_capture_metadata.json` with build, map, tile, variant,
      settledFrames, totalFrames, timedOut status
    - wow-viewer `ValidationCaptureCommand` CLI now exposes:
      `--settled-frames`, `--max-frames`, `--batch-settled-frames`
    - stub-scene test: 4/4 variants passed with new fast-settle defaults
    - unit tests: 33/33 pass including 3 new batch-fast-settle tests
  - Zarr-mutating `build_v16_dataset.py` commands now require
    `--allow-zarr-write`; preview-first is the enforced operator path
  - current proof level:
    - `dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug`
    - raw harvest preview rerun for `3_3_5_12340 / Azeroth` succeeded after the
      MDDF filter and inspector changes
    - `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug`
      succeeded after the renderer-truth doodad/mask policy hotfix
    - bounded renderer-truth validation runs now exist at:
      - `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48`
      - `output/tmp/mdxviewer_validation_smoke/3_3_5_12340_Azeroth_30_48`
        - `output/tmp/mdxviewer_validation_smoke_fix_wmo/3_3_5_12340_Azeroth_30_48`
        - `output/tmp/mdxviewer_validation_smoke_heightfilter/3_3_5_12340_Azeroth_30_48`
      - fresh bounded rerun after the policy hotfix succeeded on staged
        `0_5_3_3368 / Azeroth / 30_48` with proof owner still `MdxViewer`
        validation capture:
        - primary proof image:
          `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48/viewer_validation_minimaps/Azeroth_30_48_viewer_validation.png`
        - regenerated object-mask proof artifact:
          `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48/images/Azeroth_30_48_object_visibility_mask.png`
    - the `0.5.3.3368` exported mask matched the direct `objectsonly`
      silhouette exactly on `Azeroth_30_48`
    - the `3.3.5.12340` exported mask diverged from the direct
      `objectsonly` silhouette on the same tile, proving the later-build
      occluded diff path was the one used at runtime
      - `WmoRenderer` now carries a bounded near-camera visibility hotfix so
        large nearby WMO roots no longer collapse to a single visible group in
        the validation captures
      - validation capture batches now wait longer before capture and can hide
        very tall MDX clutter through a bounds-height threshold during the batch
      - current real renderer-truth proof still covers only:
        - `0_5_3_3368`
        - `3_3_5_12340`
      - current throughput risk is still open:
        - the bounded single-tile `MdxViewer` validation rerun took a little
          over 3 minutes for one tile, which is too slow for broad proof or
          patch workflows
        - next operational pressure point is batching multiple tiles per loaded
          world session instead of paying this cost one tile at a time
      - remaining build proof is still open for:
        - `0_5_5_3494`
        - `0_7_0_3694`
        - `3_0_1_8303`
        - `4_0_0_11927`
      - active `V16.2` direction is now sidecar-first:
        - keep finalized base V16 stores intact
        - stage renderer-truth and richer precise-mask signals into separate
          sidecar stores first
        - only consider merge-back after broader cross-build validation exists
      - renderer-truth capture pipeline is now integrated into the V16 build:
        - `build_v16_dataset.py generate-viewer-stubs` reads index.parquet and
          writes per-tile JSON stubs for MdxViewer tile discovery
        - `generate_all_renderer_truth_captures.bat` runs MdxViewer batches per build
        - `build_v16_dataset.py patch-renderer-truth` patches captured PNGs into Zarr
        - all 20,627 stubs generated across 6 builds as of 2026-05-22
      - MdxViewer CLI requires: --game-path, --build, --listfile, --world,
        --validation-dataset-root, --validation-output, --validation-resolution,
        --force-validation-regeneration, --exit-after-validation
      - harvest stream produces all core terrain signals (height, normals, alpha,
        holes, liquid, object masks, minimap, shadow, MCLY/MCAL, placements)
      - viewer produces only renderer-truth overlay: object_visibility_mask and
        no_object_minimap (diff of primary vs noobjects capture families)
      - future direction: wow-viewer reads terrain geometry from Zarr stores
        directly instead of game client archives (Zarr is compact and fast
        compared to MPQ)
  - remaining open issue: the shared harvester-side raw WMO/MODF preview is
    still approximate; the precise renderer-truth path currently lives in
    MdxViewer validation capture artifacts instead
- The shared V16.1 trainer now has real gradient accumulation through
  `--grad-accum-steps`; this is the intended path for the 4070 Ti SUPER instead
  of pretending large micro-batches fit in VRAM.
- The shared V16.1 trainer now also carries forward the useful V16 runtime
  seams:
  - `torch.compile`
  - `--num-workers -1` auto resolution
  - `--persistent-workers`
  - `--prefetch-factor`
- Focused proof exists at:
  - `wow-viewer/models/v16_1/normal/runs/smoke_normal_compile_gpu/`
  - GPU smoke completed with `torch.compile: enabled`
- Current V16.1 direction is to treat normals as the first terrain-signal proof
  lane and let that inform later height-lane shaping.
- V16.1 now has a separate reusable curation layer between Zarr and trainers.
  - trainer consumption path: `--curation-manifest`
  - first profile: `normal_terrain_v1`
  - curation builder now supports multi-process tile auditing:
    - `--workers`
    - `--chunk-size`
  - rule direction: all future model families should train from curated
    manifests, not raw tile rows
  - current blank-genesis rule:
    - reject `blank_what_plate_tile` before normal training
- Canonical short docs were rewritten and should now be the first read for this lane:
  - `wow-viewer/README.md`
  - `wow-viewer/data-harvester/README.md`
  - `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md`
- The README surfaces now explicitly document the curation-first V16.1 normal
  workflow:
  - build `normal_terrain_v16_1_1` manifest
  - train via `--curation-manifest`
  - preferred current operator launch:
    - curation: `--workers -1 --chunk-size 128`
    - train: `--batch-size 16 --grad-accum-steps 1`
    - small scouting pool: `--train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48`
  - current VRAM truth on the 16 GB card:
    - if `8 x 1` sits under about `5 GB`, it is too conservative
    - prefer raising micro-batch before adding accumulation
- The shared V16.1 trainer now also preserves the useful V16 small-run seam:
  - bounded persistent train/val pools:
    - `--train-max-tiles`
    - `--val-max-tiles`
  - rotating per-epoch train subsets:
    - `--train-epoch-tiles`
  - proof root:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_curated_epoch_rotation_cpu/`
  - evidence files:
    - `evidence/train_pool_summary.json`
    - `evidence/val_pool_summary.json`
    - `evidence/train_epoch_orders.jsonl`
 - The shared V16.1 trainer now explicitly tracks:
   - `best_val`
   - `best_epoch`
- V16.1 validation preview writing is best-gated again:
  - previews write only on new best checkpoints
  - output path:
    - `validation/best_epoch_XXXX.png`
- Validation previews are no longer single-sample lies:
  - the shared V16.1 preview path now renders up to `8` samples from the
     selected validation batch per artifact instead of only batch item `0`
- Validation previews now draw labels again:
  - panel labels such as `input`, `normal_gt`, `normal_pred`, `train_mask`
  - per-row sample headers with build/map/tile metadata
- Full-map debug-signal quilts are now scriptable from the same normal-loss math:
  - script: `wow-viewer/data-harvester/scripts/quilt_v16_1_debug_signals.py`
  - outputs stitched `hard_region` / `transition` (local+global) and `train_mask` PNG quilts per `build/map`
  - supports optional `--curation-manifest`, map/build filters, and tile caps for bounded dataset slices
- Paste/prefab mining now has a first script surface in `wow-viewer`:
  - script: `wow-viewer/data-harvester/scripts/mine_v17_pastes.py`
  - current scope: candidate extraction from normal guidance signals + optional grid-cell library seeds
  - supports cross-build perceptual-hash dedupe (`--dedupe`) and emits brush/cell library seed manifests
- Spec 024 Phase 1 canvas mining is now landed:
  - script: `wow-viewer/data-harvester/scripts/mine_v18_pastes_canvas.py`
  - stitched canvas candidates now emit `canvas_bbox` and `tile_coverage`
  - evidence artifacts now include `summary.json`, `candidates.jsonl`, `canvas_summary.json`, and `config.snapshot.json`
  - bounded proof run:
    - command: `uv run python -u scripts/mine_v18_pastes_canvas.py --builds 3_3_5_12340 --maps Azeroth --max-tiles 1024 --seed 42 --component-threshold 0.28 --out-dir ../output/tmp/v18_canvas_smoke_dense`
    - result: `candidates=24`, `multi_tile_candidates=6`, `multi_tile_ratio=0.25`
- Spec 024 Phase 2 dedupe is now landed in the same script:
  - deterministic candidate fingerprinting: `rgb_fingerprint`
  - tile coverage now includes `tile_id` per covered tile for trainer-manifest projection
  - alpha-layer-aware metadata: `layer_means`, `layer_coverage`, `dominant_layers`, `alpha_layer_signature`
  - cluster lineage metadata per candidate: `cluster_id`, `canonical_id`, `variant_rank`, `cluster_size`, `is_canonical`
  - dedupe outputs: `candidates_deduped.jsonl`, `cluster_summary.jsonl`, `dedupe_stats.json`
  - cluster QA outputs: `cluster_atlas/*.png` and `clusters_canonical_top_atlas.png`
  - deterministic rerun proof (same command/seed twice) produced identical:
    - `selection_hash=999c2e6880225c24fd979b70538f1353d60f8187b51ba2abfe5c43b40cefabe0`
    - `cluster_hash=4dc8ffa09cac92cd3c07ede9f8ad88aec91167c99837ffc418ec438622db14a8`
    - byte-identical `cluster_summary.jsonl`
- Spec 024 Phase 3 refined manifest generation is now landed:
  - script: `wow-viewer/data-harvester/scripts/build_v18_refined_manifest.py`
  - consumes deduped canvas candidates and emits trainer-compatible manifest rows:
    - `kept_tiles.parquet`
    - `tiles.parquet`
    - `tiles.jsonl`
  - normal-aware gating + cluster-balancing controls:
    - `--min-score-mean`
    - `--min-transition-mean`
    - `--min-hard-mean`
    - `--min-train-mask-mean`
    - `--max-clusters`
    - `--max-variants-per-cluster`
    - `--max-tiles`
  - evidence outputs include:
    - `summary.json` with `selection_hash`, cluster distribution, duplicate-ratio metrics, bucket/build counts
    - `selected_candidates.jsonl`
    - `config.snapshot.json`
  - trainer load proof:
    - command: `uv run python -u scripts/train_v16_1_normal.py --builds 0_5_3_3368 3_3_5_12340 --curation-manifest ../output/tmp/v18_refined_manifest_phase3_smoke --device cpu --epochs 1 --batch-size 2 --train-max-tiles 24 --train-epoch-tiles 8 --val-max-tiles 8 --rotate-val-tiles --val-epoch-tiles 4 --num-workers 0 --no-compile --run-name v18_refined_manifest_load_smoke`
    - result: run completed (`train=24`, `val=3`), confirming manifest/trainer seam compatibility
- Spec 024 Phase 4 composition graph layer is now landed:
  - script: `wow-viewer/data-harvester/scripts/build_v18_composition_graph.py`
  - composition outputs:
    - `composition_candidates.jsonl` (candidate-level `area_id_coverage`, `dominant_area_ids`, `composition_family_id`)
    - `composition_nodes.jsonl` (cluster nodes + area distribution)
    - `composition_edges.jsonl` (adjacency/co-occurrence edges with stable counts/distances)
    - `composition_families.jsonl` (stable `composition_family_id` macro groups)
  - AreaID behavior:
    - supports optional external `--area-id-map` for tile-level labels
    - soft-label fallback to `unknown` when AreaID data is absent
  - deterministic graph proof:
    - run1: `../output/tmp/v18_composition_phase4_run1`
    - run2: `../output/tmp/v18_composition_phase4_run2`
    - identical hashes:
      - `graph_hash=eccd70abda63f7e5dcbabe2528f2809b62aed7455780c2cef125291a7333c09a`
      - `family_hash=800a2029757e7988f736d70dc787087d8b4f72e692c3007e32fa4bdbdd8c2771`
    - byte-identical `composition_edges.jsonl`
- Phase 4 metadata integration into refined manifests is now landed:
  - `build_v18_refined_manifest.py` accepts `--composition-graph`
  - refined rows now carry:
    - `source_composition_family_ids`
    - `source_composition_family_count`
    - `composition_balance_weight_mean`
  - summary now includes composition-family distribution evidence
- Spec 024 Phase 5 auto-naming + paste-library catalog is now landed:
  - script: `wow-viewer/data-harvester/scripts/build_v18_paste_library_catalog.py`
  - deterministic naming from role/shape/layer/family descriptors with stable `paste_id`
  - metadata fields include:
    - `canonical_name`
    - `aliases`
    - `name_confidence`
    - `review_state` (auto)
    - `review_required`
  - catalog outputs:
    - `paste_library_catalog.json`
    - `paste_library_catalog.jsonl`
  - deterministic stability proof:
    - rerun output at `../output/tmp/v18_paste_library_phase5_run2`
    - byte-identical `paste_library_catalog.jsonl` vs run1
    - stable hash: `1ae9a2d2900a24aba4f7b34c260f747bd683527317278abdb2a22a783f372a2f`
- Spec 024 Phase 6 baseline launch contract is now landed:
  - script: `wow-viewer/data-harvester/scripts/run_v18_baseline_contract.py`
  - defines profile contract (`small` / `medium` / `large`) in `baseline_profiles.json`
  - executes bounded refined baseline run + non-ref comparison run
  - writes comparison artifacts:
    - `comparison_report.json`
    - `comparison_report.md`
  - bounded proof root:
    - `../output/tmp/v18_baseline_contract_phase6`
- Spec 025 bounded continuation now spans Phases 1-4 proof surfaces:
  - Phase 1 roof-library build+validation remains green on staged `3_3_5_12340`:
    - `output/datasets/object_roof_library/smoke_spec025_phase1_335/`
    - `validate_v18_object_roof_library.py` status: `pass`
  - Phase 2 learned fallback + mask validation landed:
    - `scripts/infer_v18_object_roof_masks.py`
    - `scripts/validate_v18_object_roof_masks.py`
    - bounded anchor proof: `output/tmp/v18_object_roof_infer_smoke_335_30_53/`
      - `tiles_non_empty=1`, validator status: `pass`
  - Phase 2 patch/report lane now writes side artifacts outside `.zarr` by default:
    - report root: `output/tmp/object_roof_patch_reports/`
    - bounded report: `output/tmp/object_roof_patch_reports/smoke_spec025_patch_335/3_3_5_12340/object_roof_patch_report.json`
    - label contract artifact: `.../object_roof_label_contract.json`
  - Phase 3 training integration landed for normal lane:
    - variant: `v18_object_roof_aux`
    - dataset/model/trainer consume `object_roof_mask_256` and `object_roof_weight_257`
    - bounded CUDA aux run: `models/v18/normal/runs/v18_oroof_smoke_spec025_v18_oroof_aux_cuda/`
    - bounded same-pool baseline run: `models/v18/normal/runs/smoke_spec025_v18_baseline_samepool_cuda/`
  - task closure state in spec checklist:
    - `wow-viewer/specs/025-object-roof-mask-library-and-minimap-sieve/tasks.md`
    - complete: `T001`, `T003`-`T022`
    - open by design: `T002` (MdxViewer one-at-a-time asset capture seam)
- Spec 025 T002 object-capture audit + first wow-viewer slice is now landed:
  - architecture note:
    - `wow-viewer/docs/architecture/spec025-t002-object-capture-audit-2026-05-26.md`
  - implemented seam closure (bounded):
    - `ValidationCaptureScenePolicy` now carries explicit capture culling-override flags
    - `ValidationWorldScenePolicyApplier` propagates those flags into policy state
    - `ValidationWorldSceneAdapter.BuildFrameRequest(...)` forwards fog/object-streaming/MDX-height/culling knobs into runtime request
    - `WowViewerWorldRuntimeFrameRequest` and `WowViewerWorldRuntimeBridge` now route those knobs into `WorldObjectVisibilityContext`
    - `WorldObjectVisibilityCollector` now honors capture override toggles and MDX max-bounds-height suppression via shared runtime context
  - bounded proof:
    - focused tests passed (`ValidationCaptureScenePolicyTests`, `ValidationWorldScenePolicyApplierTests`, `ValidationWorldSceneAdapterTests`, `WorldObjectVisibilityCollectorTests`)
    - staged real-scene dry-run passed for `3_3_5_12340 / Azeroth_30_48` at staged client root `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`
  - T002 status after this slice:
    - policy-propagation and culling-hook gap is closed in wow-viewer
    - full dedicated object-render backend parity and one-at-a-time asset-pose capture orchestration remain open follow-up work
- Spec 025 T002 second bounded automation slice is now landed in `wow-viewer`:
  - `WowViewer.Tool.ValidationCapture` now supports `capture-batch` with ledger input (`--ledger-path`) for manifest-driven multi-tile batch execution
  - batch path reads `manifest_capture_ledger.json`, skips `captured_complete`, expands each pending tile into 4 variant requests, and reuses the same default scene/variant policy composition as single-tile `capture`
  - `build_v16_dataset.py generate-viewer-stubs` guidance now points to `capture-batch` as primary capture lane (legacy MdxViewer scripts retained only for compatibility comparison)
  - focused command test proof passed (`ValidationCaptureCommandTests`, `5/5`)
  - T002 status after automation cutover slice:
    - manifest/ledger automation is now wow-viewer-owned
    - dedicated object-render backend parity + per-asset pose-capture orchestration still remain for full T002 closure
- Spec 025 T002 third bounded automation bridge slice is now landed in `wow-viewer/data-harvester`:
  - `build_v16_dataset.py` now includes `capture-renderer-truth`
  - the command discovers `WowViewer.Tool.ValidationCapture.exe` and invokes `capture-batch` per build/map ledger group
  - it reads `manifest_capture_ledger.json`, skips `captured_complete`, groups pending tiles by map, emits temp per-map ledgers, and forwards mode flags/resolution/build label into wow-viewer capture
  - focused proof:
    - `uv run python scripts/build_v16_dataset.py --help` now lists `capture-renderer-truth`
    - bounded dry-run for `3_3_5_12340` resolved tool + root and cleanly skipped missing ledger with exit `0`
  - T002 status after this bridge slice:
    - end-to-end batch orchestration from data-harvester into wow-viewer capture is now wired
    - dedicated object-render backend parity + one-at-a-time per-asset pose-capture seam still remain for full T002 completion
- Spec 025 T002 fourth bounded pose-metadata slice is now landed:
  - `capture-batch` now accepts optional pose metadata fields from ledger rows (`asset_path`, `instance_type`, `unique_id`, `rot_x`, `rot_y`, `rot_z`, `scale`)
  - render-mode `capture-batch` runs now emit per-tile pose artifacts at:
    - `<dataset-root>/pose-metadata/<tile_name>_pose.json`
  - `build_v16_dataset.py generate-viewer-stubs` now enriches ledger rows from real `<build>.zarr/placements.parquet` data (prefers `modf`, then `mddf`) so pose metadata is no longer dependent on bogus stub JSON
  - focused proof:
    - `ValidationCaptureCommandTests` now include pose-artifact assertion and pass (`6/6`)
    - bounded `capture-renderer-truth --stub-scene` run produced non-null pose metadata in:
      - `output/tmp/mdxviewer_validation_smoke/3_3_5_12340/pose-metadata/AhnQiraj_46_27_pose.json`
- Spec 025 T002 fifth bounded per-tile placement-resolution slice is now landed:
  - `build_v16_dataset.py generate-viewer-stubs` now materializes all placement rows per tile into ledger payload:
    - `object_instance_count`
    - `object_instances[]`
  - `WowViewer.Tool.ValidationCapture capture-batch` pose artifact outputs now preserve those full fields per tile.
  - focused parity proof against dataset truth (`3_3_5_12340`):
    - regenerated ledger rows: `5134`
    - `placements.parquet` rows: `1,015,470`
    - mismatches between ledger per-tile counts and `placements.parquet` per-tile counts: `0`
    - multi-instance samples confirmed (e.g. `Northrend_21_23=3580`, `Northrend_22_22=3437`, `Azeroth_32_39=3015`).
  - this closes the first-instance-only gap for ledger pose/placement coverage.
- V18 raw-blob datastore sketch is now documented for undecoded payload preservation:
  - `wow-viewer/docs/architecture/v18-undecoded-blob-datastore-sketch-2026-05-27.md`
  - defines sidecar `raw_blobs` manifest + content-addressed payload layout, phased migration path, and validation contract without rewriting current readers.
- New canonical dataset-build requirement from latest user direction is now reflected in implementation:
  - one script (`build_v16_dataset.py`) must produce complete dataset signals and decoded metadata coverage without depending on patch/fixup scripts.
  - build flow now writes `decoded_metadata.parquet` with one decoded metadata row per harvested tile and validates parity against `index.parquet`.
  - merge flow now carries `decoded_metadata.parquet` forward and validates merged parity.
  - validation flow now checks both signal coverage and decoded metadata table integrity.
- V17.1 normal trainer behavior adjustments landed during this session:
  - `v17_1_normals` no longer enables refiner/distillation by default
  - `height_supervision_weight` default for `v17_1_normals` restored to `1.0`
  - explicit startup guard now fails if `v17_1_normals` runs with height supervision disabled
  - invalid-region neutral normal loss added (`--invalid-neutral-weight`) to suppress object-region leakage in predicted normals
  - batch autotune probing now uses warmup+measured steps with configurable safety/probe controls, and `v17_1_normals` loader defaults are more conservative on host RAM
- The shared V16.1 trainer now supports startup VRAM autotune:
  - `--target-vram-gb`
  - `--autotune-batch-size`
  - `--autotune-batch-candidates`
  - `--autotune-keep-epoch-steps`
  - evidence path:
    - `evidence/batch_autotune.json`
  - per-epoch CUDA memory guidance prints are back too
- Canonical flow:
  - `WowViewer.Tool.Harvest harvest-stream --stream-profile v16`
  - `inspect_v16_harvest_samples.py` raw preview before any store write
  - `build_v16_dataset.py build --allow-zarr-write`
  - `validate_v16_training_ready.py`
  - `train_v16.py`
- Current real-run trainer shape:
  - train pool: `--train-max-tiles 4000`
  - epoch budget: `--train-epoch-tiles 1350`
  - val budget: `--val-max-tiles 150`
  - batch size: `72`
  - GPU throttle: `--gpu-duty-cycle 100`
- `wow-viewer` is the implementation owner. `gillijimproject_refactor` is reference/continuity/validation only.

## Current V16 Corpus Truth
- Finalized stores exist for:
  - `0_5_3_3368`
  - `0_5_5_3494`
  - `0_7_0_3694`
  - `3_0_1_8303`
  - `3_3_5_12340`
  - `4_0_0_11927`
- All six current `signal_validation.json` files pass.
- Human-eye QA artifacts exist for all six under:
  - `wow-viewer/output/datasets/v16/inspection/`
- Only standing allowed warning in the current corpus:
  - `0_7_0_3694` has `has_holes_16 = 0`

## Current Trainer Contract
- Dataset loader: `wow-viewer/data-harvester/src/harvester/v16_dataset.py`
- Current terrain model host: `wow-viewer/data-harvester/src/harvester/v15_model.py`
- Current supervised terrain heads:
  - height
  - normals
  - alpha
  - holes
  - liquid mask
  - MCLY logits
- `liquid_height` stays in the dataset contract but is deferred from the current terrain trainer/inference path.
- Terrain loss weighting uses `object_filtered_mask`.
- `object_instance_mask` is readable but not yet used by the terrain trainer.
- Validation snapshot alpha QA now uses a painted-layer composite (`max(ch1..3)` with fallback) instead of raw `alpha[...,0]`, because channel `0` is commonly the implicit base layer and was producing false-black GT panels.
- `train-max-tiles` is now the persistent run-level train pool, while `train-epoch-tiles` can rotate a fresh per-epoch subset from that pool.
- CUDA-oriented loader defaults are less conservative now: `--num-workers=-1` auto-resolves a worker count and `persistent_workers` defaults on when workers are active.
- Trainer curation now has a basic quality gate by default: it drops obviously low-signal flat tiles and writes `train_quality_audit.json` / `val_quality_audit.json`.
- Every new best `val_h` epoch now writes a fresh random validation snapshot set under `validation/best_epoch_XXXX/`, separate from the normal interval snapshots.
- Current conclusion from the long V16 run: the shared-head trainer is not the
  long-range architecture owner. Future model work should target the V16.1
  dense-correlation family instead of adding more complexity to the V16 monolith.
- Liquids are no longer treated as "mask only" in the next architecture lane;
  V16.1 should carry liquid type as a first-class prediction surface.
- Alpha is no longer treated as a standalone generic mask head in the next
  architecture lane; V16.1 should handle it as a dedicated MCLY/MCAL
  decomposition + recomposition family.
- That decomposition family is not greenfield: existing `train_d1.py` /
  `D1UNet` / `D1Dataset` work should be migrated onto the V16 Zarr-quality
  signals and current loss-gating contract.

## Harvest / Dataset Truth
- Stream format is lean `ARRY`, not legacy `NPZB`.
- Archive-backed ADT families now route through the in-memory byte path.
- Default dataset compression is Blosc `lz4` / level `1` / `shuffle`.
- `repair-index` is the fast fix for coordinate-only damage.
- `patch-liquids` can rewrite only liquid arrays + liquid provenance flags in-place.
- `inspect_v16_dataset.py` is the human-eye QA surface.
- Operator routing is intentionally simpler now:
  - root README = repo + workflow orientation
  - data-harvester README = commands + outputs
  - V16 spec = contract and boundaries

## Critical Recent Fixes
- Mixed Cataclysm archive tiles can carry inline root `MCLY` / `MCAL` without `_tex0`.
  - `AdtTensorPackBuilder.ReadTextureDataFromBytes(...)` now falls back to inline root texture parsing when `_tex0` bytes are absent.
  - Focused proof on staged `4_0_0_11927 / AhnQiraj / (27,46)` restored `mcly_texture_ids`, `mcly_layer_mask`, and `mcal_alpha_pack_256`.
- Alpha placeholder `map=memory` metadata was fixed at the harvest / repair-index seam.
- Liquid derivation now prefers explicit `mh2o_presence_mask` / `mclq_presence_mask`; WL* remains last-resort fallback.

## Known Nuance
- WL* liquid coverage still does not always fill the whole chunk footprint that the raw data spans.
- This is currently treated as a downstream loader / trainer semantic issue, not a harvest-corruption issue.
- The corpus is now considered consistent enough for training work.

## Inference Direction
- Keep the paired contract:
  - input: `wow-viewer/output/datasets/v16/<build>.zarr`
  - output: `wow-viewer/output/datasets/v16_inference/<run>/<build>.pred.zarr`
- Current `infer_v16.py` emits:
  - `<build>.pred.zarr`
  - per-tile `inference_summary.json`
  - `predicted_height_257.npy`
  - `predicted_liquid_mask_256.npy`
- Downstream patch/export path remains:
  - `terrain-patch-adt`
  - `convert-lk-to-alpha`
  - `convert-alpha-to-lk`

## Focused Proof Pointers
- Trainer-readiness proof:
  - `wow-viewer/output/datasets/v16/validation/3_3_5_12340.training_readiness.json`
  - `wow-viewer/output/datasets/v16/validation/all-builds.training_readiness.json`
- Visual QA root:
  - `wow-viewer/output/datasets/v16/inspection/`
- Current per-build summaries:
  - `<build>.summary.json`
  - `<build>.samples.json`
  - `<build>.validation_audit_overview.png`
- Multi-build training smoke run:
  - `wow-viewer/models/v16/runs/smoke_v16_full_corpus_post_fix/`
  - 1 epoch on CPU completed cleanly against curated tiles from the finalized six-build corpus
- Alpha-validation snapshot fix proof:
  - `wow-viewer/models/v16/runs/smoke_alpha_validation_fix/validation/epoch_0001/tile_00/alpha_gt_painted_max.png`
  - `alpha_gt_painted_max.png` now carries nonzero GT intensity; the prior false-black symptom was a channel-selection issue, not a corpus-alpha loss issue
- Epoch-rotation proof:
  - `wow-viewer/models/v16/runs/smoke_epoch_rotation/evidence/train_epoch_orders.jsonl`
  - epoch `1` selected positions `[7,4,2,0]`; epoch `2` selected `[5,6,4,2]`, proving fresh epoch subsets from a larger curated pool
- Current production-oriented launch contract:
  - run name: `v16_full_corpus_epoch_rotation`
  - command uses `train-max-tiles 4000`, `train-epoch-tiles 1350`, `val-max-tiles 150`, `batch-size 72`, `gpu-duty-cycle 100`
- Immediate next planning truth:
  - do not widen into a giant new foundation-model rewrite
  - use V16.1.1 to harden the normal lane first through curation intelligence
    and training-efficiency improvements
  - treat a mixed `400`-tile scouting pool as the first proof surface before
    any longer rerun
  - the current longer-run continuation path is a resume, not a fresh launch
    from scratch
- Alpha/minimap alignment audit:
  - `wow-viewer/output/datasets/v16/validation/alpha_minimap_alignment/alpha_minimap_alignment.summary.json`
  - sampled corpus result: `edge_f1_mean≈0.54`, `median≈0.64`, but `p10=0.0`, confirming a real zero-match bad tail
- Quality-curation proof:
  - `wow-viewer/models/v16/runs/smoke_quality_curation/evidence/train_quality_audit.json`
  - basic gate dropped `196` obviously low-signal flat train tiles from the `3_3_5_12340` smoke candidate pool (`4621 -> 4425`)

## Next Likely Slice
- Run smoke proof for V16.1 liquid, texcomp, and holes trainers using the
  current V16 corpus as the dataset contract.
- Extend the curation layer with additional target-aware profiles after the
  normal lane:
  - height
  - liquid
  - texture decomposition
- Re-launch the first real normal run with `batch-size 1` or `2` plus
  accumulation instead of high micro-batch counts that trigger WDDM offload.
- Preferred current launch contract after the runaway-process cleanup:
  - start normal training at `8 x 1`
  - fall back to `4 x 2`, `2 x 4`, then `1 x 8` only if VRAM forces it
- Write the short note on what the normal lane teaches the height lane, then
  tighten the height loss around that terrain-only framing.
- Tighten the stitched-output contract so the final V16.1 `.pred.zarr` bundle
  consistently carries all per-family signals plus provenance.
- Reuse the existing D1 tileset/decomposition lane as the starting point for
  V16.1 texture decomposition instead of redesigning it from scratch.
- Treat `v16_full_corpus_epoch_rotation*` as baseline evidence, not as the main
  future architecture investment surface.
- If WL* chunk-fill behavior matters to loss semantics, handle it in the loader/trainer, not by reopening harvest.
- PM4 follow-up now has a library-owned `MSHD.Field04` region-id seam feeding `MdxViewer` overlay coloring/debug/export, selected-region peer summaries, and LLM-oriented visible-overlay evidence bundles; broader PM4 object-mapping work can build on that without reintroducing viewer-owned decode logic.

## Ghidra RE Session (2026-05-30)

### WMO Render Pass Architecture (Build 3368)
- Full WMO group render pipeline decompiled and documented
- Key: `RenderGroup` dispatches interior/exterior via function ptrs at `DAT_00ec1b98`/`DAT_00ec1ca0`
- 11 render pass functions confirmed (Int/Ext ColorTex, LightTex, Lightmap, LightmapTex, Tex, BSP, Normals, Portals)
- Per-batch MOMT flags: bit0=lighting, bit1=fog, bit2=culling, bit0x10=emissive, bit0x20=window-lit
- Interior fog: applied when `intFog != 0` and WMO == camMapObj
- Liquid dispatch: types 0/4/8 → water (int/ext), types 2/3/6/7 → magma
- Lightmap split: Int = lighting off + lightmap on tex1; Ext = lighting on + no lightmap on tex1
- Group flags: `0x88` = skip, `0x48` = exterior, `0x1000` = liquid, `0x10000` = always-render
- Portal walk: recursive visibility traversal with screen-rect clipping and depth limiting
- Architecture doc: `wow-viewer/docs/architecture/wmo-render-pass-architecture-2026-05-30.md`
- Spec: `wow-viewer/specs/030-wmo-render-pass-architecture/`

### Terrain Cell System (Build 3368)
- Vertex layout: 9x9 outer (81) + 8x8 inner (64) = 145 vertices per MCNK
- Inner vertices at cell centers enable diagonal splits per cell
- Face planes: 256 per chunk (8x8 cells × 4 triangles per cell)
- Hole mask: 16-bit `holes` field, 4x4 grouping, each bit covers 2x2 cell block
- Cell addressing: 13-bit packed coords (3 sub-chunk + 4 chunk + 6 ADT bits per axis)
- Normals: packed 3 signed bytes, Y/Z/X order, scale ~0.251388
- Rendering: 26 distance buckets, texture LOD drops layers at distance, low-detail 17x17 far area
- Spec: `wow-viewer/specs/031-terrain-cell-awareness/`

### MdxViewer Migration (033) + WoWViewer Rename (034) — COMPLETED 2026-05-30/31
- MdxViewer moved from `gillijimproject_refactor/src/MdxViewer/` → `wow-viewer/src/viewer/WoWViewer/`
- Renamed to WoWViewer, version bumped to v0.5.0 in csproj (AssemblyName: ParpToolsWoWViewer)
- MDX-L_Tool vendored into `wow-viewer/libs/WoW-Tools/MDX-L_Tool/`
- gillijimproject-csharp.csproj vendored into `wow-viewer/libs/WoW-Tools/GillijimProject/`
- WowViewer.App moved to `wow-viewer/src/viewer/WowViewer.App.Defunct/` (defunct)
- WoWViewer added to `wow-viewer/WowViewer.slnx`
- Remaining gap: Many source files still reference `MdxLTool.Formats.Mdx` namespace (not yet ported to WowViewer.Core.IO)
- **MDX/M2 rendering is BROKEN** — namespace mismatch after migration; `MdxLTool.Formats.Mdx` types not fully ported
- Spec: `wow-viewer/specs/033-mdxviewer-migration/` (spec+plan+tasks complete)
- Spec: `wow-viewer/specs/034-wowviewer-rename/` (spec+plan+tasks complete)

### WMO Minimap Naming (Build 3368) — previously confirmed
- Pattern: `<WMOName>_<groupIdx>_<quadY>_<quadX>.blp` under `Textures\Minimap\`
- Resolved through MINIMAPMD5NAME hash table in `SetupQuad`
- Spec: `wow-viewer/specs/029-wmo-minimap-signal/` (spec+plan+tasks written, not yet implemented)
