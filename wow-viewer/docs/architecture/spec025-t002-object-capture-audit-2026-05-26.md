# Spec 025 T002 — Object Capture Audit and First Slice (2026-05-26)

## Scope

This note captures the focused T002 audit for object capture parity and the first bounded implementation slice that removes major capture-policy propagation gaps in `wow-viewer`.

## Current vs Desired (Gap Analysis)

### 1) Policy expression existed, but runtime consumption was incomplete

- Current state before this slice:
  - capture policy/state had fields for fog/object-streaming/height and related toggles in `ValidationCaptureScenePolicy` and `ValidationWorldScenePolicyState`
  - runtime request path into world visibility mostly only carried pass booleans (terrain/objects/sky/liquids)
- Gap:
  - capture-specific culling intent (distance/projection/cone/frustum/max-distance relaxation) was not explicitly represented at world-runtime visibility call sites

### 2) Visibility collector hard gates reduced object-capture recall

- Current state before this slice:
  - `WorldObjectVisibilityCollector` always enforced:
    - distance gates
    - projected-size gates
    - rear-cone/frustum interplay gates
    - max-view-distance gates
- Gap:
  - for dataset object-capture variants, those gates can suppress objects we intentionally need in object-only and roof-library flows

### 3) MDX height suppression needed a shared-runtime control

- Current state before this slice:
  - legacy `MdxViewer` capture lane used max MDX bounds-height suppression in capture mode
  - wow-viewer visibility path did not carry this as first-class runtime visibility context
- Gap:
  - no direct shared-runtime parity hook for deterministic tall-clutter suppression in capture policy-driven runs

### 4) Proof-level needed staged-client confirmation using the new path

- Current state before this slice:
  - tests existed for pass-option and policy-state behavior, but no focused assertion that new capture-policy visibility tuning actually survives runtime-request mapping
- Gap:
  - lacked direct bounded proof that the policy hook lane is wired and operational on staged client roots

## Target Architecture (T002 Direction)

For capture mode, routing must be:

1. `ValidationCaptureScenePolicy` owns deterministic capture-visibility tuning intent.
2. `ValidationWorldScenePolicyApplier` maps that intent into policy state.
3. `ValidationWorldSceneAdapter.BuildFrameRequest(...)` forwards state into runtime frame request.
4. `WowViewerWorldRuntimeBridge.Build(...)` builds `WorldObjectVisibilityContext` from request-level capture tuning.
5. `WorldObjectVisibilityCollector` honors these tuning knobs while keeping default behavior unchanged for non-capture callers.

This creates a clean shared-runtime contract without re-routing through legacy shell UI concerns.

## First Bounded Slice Implemented

### A) Runtime visibility context extension

- Added capture-tuning fields to `WorldObjectVisibilityContext`:
  - `MaxVisibleMdxBoundsHeight`
  - `IgnoreDistanceCulling`
  - `IgnoreProjectedSizeCulling`
  - `IgnoreVisionConeCulling`
  - `IgnoreFrustumCulling`
  - `IgnoreMaxViewDistanceCulling`

### B) Visibility collector gating hooks

- Updated `WorldObjectVisibilityCollector` to conditionally bypass corresponding culling branches when capture overrides are enabled.
- Added MDX height suppression directly in the collector using the context value.

### C) Runtime request propagation

- Extended `WowViewerWorldRuntimeFrameRequest` with capture-visibility tuning parameters.
- Updated `WowViewerWorldRuntimeBridge` to build `WorldObjectVisibilityContext` from request-level values (instead of hardcoded fog/range-only defaults).

### D) Scene-policy contract extension

- Extended `ValidationCaptureScenePolicy` with explicit culling-override booleans.
- Extended `ValidationWorldScenePolicyState` + applier mapping to carry these values.
- Updated `ValidationWorldSceneAdapter.BuildFrameRequest(...)` to forward all tuning fields.

### E) Tool default policy for capture lane

- Updated `WowViewer.Tool.ValidationCapture` bounded capture policy setup to enable culling overrides for capture runs.

## Validation Evidence

### Focused test proof

- Focused test filter run passed (`18` tests):
  - `ValidationWorldSceneAdapterTests`
  - `ValidationWorldScenePolicyApplierTests`
  - `ValidationCaptureScenePolicyTests`
  - `WorldObjectVisibilityCollectorTests`

Added tests include:

- `CollectVisibleMdx_RespectsMaxVisibleMdxBoundsHeight`
- `CollectVisibleMdx_CaptureOverridesCanBypassDistanceCulling`

### Bounded staged-client dry-run proof

- Command root: `WowViewer.Tool.ValidationCapture capture ... --real-scene-dry-run`
- Staged client used:
  - `I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft`
- Tile:
  - `Azeroth_30_48`
- Result:
  - all 4 variants reported `sceneContent=True`, `tileLoaded=True`, `pendingObjects=0`

## Remaining T002 Work (Not Claimed Done)

This slice closes policy-propagation and visibility-hook wiring, but T002 is still open for:

## Second Bounded Slice Implemented (Automation Cutover)

### A) `capture-batch` command surface in `WowViewer.Tool.ValidationCapture`

- Added `capture-batch` command to the validation-capture host shell.
- New required args:
  - `--client-root`
  - `--map-input`
  - `--dataset-root`
  - `--output-root`
  - `--ledger-path`
- The command now reads `manifest_capture_ledger.json`, filters out `captured_complete` tiles, and builds one `ValidationCaptureBatchPlan` spanning all remaining tiles (all four capture variants per tile).

### B) Shared plan/policy composition reuse

- Refactored command internals so both `capture` and `capture-batch` reuse:
  - default scene policy creation
  - default variant policy map
  - batch-plan construction across tile inputs
- Keeps policy parity between single-tile and ledger-driven batch automation paths.

### C) Dataset builder guidance cutover

- Updated `build_v16_dataset.py generate-viewer-stubs` command text/help to position ledger-driven `WowViewer.Tool.ValidationCapture capture-batch` as the primary next step for renderer-truth capture.
- Legacy MdxViewer batch scripts remain available for compatibility comparison only.

### D) Focused automation tests

- Added command tests:
  - `Execute_CaptureBatchMissingLedger_ReturnsOne`
  - `Execute_CaptureBatchDryRun_ReturnsZeroAndPrintsSummary`
- Dry-run proof verifies tile filtering semantics from ledger status and variant expansion (`2` pending tiles -> `8` variant requests).

### E) Focused execution proof

- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ValidationCaptureCommandTests"`
- Result: `5/5` passed.

## Fourth Bounded Slice Implemented (Pose Metadata Carry-Through)

### A) Ledger pose metadata contract in `capture-batch`

- Extended `capture-batch` ledger parsing to accept optional per-tile pose fields:
  - `asset_path`
  - `instance_type`
  - `unique_id`
  - `rot_x`
  - `rot_y`
  - `rot_z`
  - `scale`
- These fields are now carried from ledger rows into runtime tile planning inputs.

### B) Per-tile pose artifact emission

- Added bounded artifact emission after render-mode runs (`--stub-scene`, `--gpu-viewer-style`, `--native-renderer`):
  - output root: `<dataset-root>/pose-metadata/`
  - file name: `<tile_name>_pose.json`
- Each artifact records tile identity + captured pose metadata values used for one-at-a-time object-capture continuity.

### C) Ledger generation now hydrates pose metadata from real dataset placements

- Updated `build_v16_dataset.py generate-viewer-stubs` to enrich ledger rows from `<build>.zarr/placements.parquet` (not stub JSON).
- Per tile, it now carries:
  - full `object_instance_count`
  - full `object_instances[]` list (all placement rows for that tile)
  - a representative top-level pose entry for backward-compatible consumers (prefers first `modf`, then first `mddf`, then first row).
- Enriched ledger fields include the same pose metadata keys consumed by `capture-batch`, plus full per-tile instance payload.

### D) Focused test + functional proof

- Added command test:
  - `Execute_CaptureBatchStubScene_WritesPoseMetadataArtifacts`
- Focused test command:
  - `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ValidationCaptureCommandTests"`
  - Result: `6/6` passed.
- Functional proof path:
  1. regenerate ledger with `generate-viewer-stubs` for `3_3_5_12340`,
  2. run `capture-renderer-truth --stub-scene` (single-tile trimmed ledger),
  3. verify emitted pose file at `output/tmp/mdxviewer_validation_smoke/3_3_5_12340/pose-metadata/AhnQiraj_46_27_pose.json` carries non-null dataset placement metadata (`asset_path`, `instance_type`, `unique_id`, `rot*`, `scale`).

## Fifth Bounded Slice Implemented (Full Per-Tile Placement Resolution)

### A) All placement rows now flow into ledger rows

- `build_v16_dataset.py` now materializes all placement rows per tile into ledger payload via:
  - `object_instance_count`
  - `object_instances[]`
- This resolves the prior limitation where only a representative pose was effectively exposed as the actionable capture context.

### B) `capture-batch` pose artifact output preserves full tile placement payload

- `WowViewer.Tool.ValidationCapture` ledger tile contract now includes:
  - `object_instance_count`
  - `object_instances`
- Emitted pose artifacts under `<dataset-root>/pose-metadata/` persist those full fields for downstream one-at-a-time orchestration logic.

### C) Focused parity proof against dataset truth (`placements.parquet`)

- Regenerated ledger proof build:
  - `uv run python scripts/build_v16_dataset.py generate-viewer-stubs --build 3_3_5_12340 --capture-root ../../output/tmp/mdxviewer_validation_smoke`
- Count parity check:
  - ledger rows: `5134`
  - placement rows: `1,015,470`
  - mismatches (`ledger.object_instance_count` vs `placements.parquet` row count per `tile_id`): `0`
- Multi-instance sample proof (all `>=3` instances per tile):
  - `Northrend_21_23` (`tile_id=3704`): `3580`
  - `Northrend_22_22` (`tile_id=3676`): `3437`
  - `Azeroth_32_39` (`tile_id=600`): `3015`

This confirms the ledger now resolves every object placement on each tile from Zarr placement truth, not first-instance-only summaries.

## Third Bounded Slice Implemented (Python Automation Bridge)

### A) End-to-end data-harvester command

- Added `capture-renderer-truth` to `build_v16_dataset.py`.
- This command drives wow-viewer capture directly by invoking:
  - `WowViewer.Tool.ValidationCapture.exe capture-batch ...`
- Per build, it:
  1. reads `manifest_capture_ledger.json`
  2. filters `captured_complete`
  3. groups pending tiles by `map`
  4. emits temporary per-map ledgers
  5. runs `capture-batch` with canonical staged client roots

### B) Tool discovery and routing

- Added validation-capture executable discovery (`_find_validation_capture_tool`) alongside existing harvest-tool discovery.
- Keeps this lane wow-viewer-owned and avoids new MdxViewer-only batch dependence for primary automation routing.

### C) Mode forwarding

- `capture-renderer-truth` forwards exactly one bounded run mode to validation-capture:
  - `--dry-run` (default)
  - `--real-scene-dry-run`
  - `--gpu-viewer-style`
  - `--native-renderer`
  - `--stub-scene`
- Also forwards build label and requested resolution.

### D) Focused command proof

- Help surface proof:
  - `uv run python scripts/build_v16_dataset.py --help`
  - includes new `capture-renderer-truth` command.
- Bounded dry-run execution proof:
  - `uv run python scripts/build_v16_dataset.py capture-renderer-truth --build 3_3_5_12340 --dry-run`
  - Result: command resolves validation-capture tool and capture root, then safely skips when ledger is absent (expected in this environment), exits cleanly.

## Remaining T002 Work (Not Claimed Done)

After this automation cutover slice, T002 remains open for:

1. full object-render parity backend (move beyond temporary preview-surface reuse)
2. dedicated one-at-a-time asset capture orchestration with explicit per-asset pose metadata contract
3. richer segmentation-comparator artifact suite and large-surface performance characterization
4. broader staged-client build matrix reruns beyond the bounded 3.3.5 anchor
