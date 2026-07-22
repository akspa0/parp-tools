# CLI Contract: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature**: 117-wdl-lattice-prior | **Date**: 2026-07-21

All commands are thin wrappers under `wow-viewer/data-harvester/scripts/` importing library logic
from `harvester.spec117`. Every command is **dry-run / print-plan by default** and refuses to write
or train without an explicit flag. Every training command is **user-run**; the assistant never
launches them. Run from `wow-viewer/data-harvester/` via `uv run`.

No new JSON schema is defined by this feature (research.md D-01, data-model.md Run-Record Schema):
run records validate against the existing `v50-model-stage-run-v1` schema, and the generated
lattice store validates against the existing `v115-feature-map-v1` schema — both already enforced
by code that predates this feature.

---

## US1 — signal export (C# harvest side, no new CLI)

**Implementation finding (2026-07-21)**: `TerrainWdlLattice` was already computed in
`AdtTensorPackBuilder` and already streamed by `RawArraySerializer.WriteTerrainVertexArrays` under
the real names `wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/`wdl_inner_present` in every
stream profile (Full/V16/V22) — no C# change was needed. The only real gap was the frozen v50
signal catalog (`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`), which didn't
yet declare these four arrays, so the existing 1:1 name-matched store builder
(`scripts/v50_build_dataset.py::_cmd_build`) never selected them. Fixed by adding four catalog rows
and regenerating the derived config (no hand-editing, no new CLI):

```
uv run python scripts/v50_generate_manifest_template.py \
  --catalog-doc ../docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md \
  --build-id 0_5_3_3368 --release v50.1 \
  --output v50_configs/v50-manifest-template-0_5_3_3368.json \
  --signals-output v50_configs/v50-signals-0_5_3_3368.json
```

Verified by re-running the existing store build/finalize path against a build already on disk and
confirming the four new arrays (`wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/
`wdl_inner_present`) are present with the expected shapes (see quickstart.md §1).

## US2 — standalone lattice predictor (USER-RUN, dry-run first)

```
uv run python scripts/spec117_train_lattice.py \
  --store <v50 curriculum store with wdl_outer_17/wdl_inner_16/*_present arrays> \
  --held-out-split <Spec 116 held-out split dir> \
  --output <run dir> --run-id <run id> --source <authored|all|synthetic> \
  [--epochs 100] [--batch 16] [--lr 2e-4] [--lr-schedule constant|onecycle] [--base 24] \
  [--confirm-run]
```

- `--run-id`, `--source`, `--output` are required, matching every other v50 trainer's convention.
- `--held-out-split` is **required** (no `--val-key`/`--val-value` fallback exists on this trainer)
  — FR-004 says the standalone predictor "MUST refuse to run against a leaky or unspecified split,"
  not merely default away from one.
- **Default (no `--confirm-run`)**: validates inputs — including that the store actually carries
  the four `wdl_*` arrays (fails closed with a clear message otherwise) — prints the full plan,
  exits without training.
- **With `--confirm-run`**: trains `LatticeNet`; writes `checkpoint_best.pt` +
  `model_stage_run.json` (schema `v50-model-stage-run-v1`, `stage="lattice_prior"`).
- Rows where every lattice sample is absent are excluded from training/evaluation and counted
  (`excluded_no_present_lattice` in the printed plan), never scored as zero/max error.
- **Gate**: held-out lattice-point MAE vs the per-tile-mean lattice baseline (D-02), reported as
  `beats_tile_mean_baseline` in `model_stage_run.json`'s metrics; `promotion_verdict` starts
  `pending`.
- Both the dry-run plan and the missing-array refusal were exercised for real against a fixture
  store during implementation, not just unit-tested in isolation.

## US3(i) — bridge the frozen predictor's output into the existing feature-store shape

```
uv run python scripts/spec117_lattice_to_feature_map.py \
  --store <v50 curriculum store> \
  --checkpoint <lattice predictor checkpoint_best.pt> \
  --output <derived feature-map store> \
  [--device cpu|cuda] [--write]
```
- **Default**: prints the materialization plan only.
- **With `--write`**: runs the frozen checkpoint over every tile in the source store. The 545-sample
  lattice is two REGULAR grids at the same 16-world-unit stride (17×17 outer spanning the full
  tile, 16×16 inner offset by 8) — the bridge independently bilinear-upsamples each to 256×256
  (`align_corners=True`) and averages them (a documented approximation, not a precision
  reconstruction of the true quincunx-offset sample set). Writes a `feature_map` array of shape
  `(N, 1, 256, 256)` under `schema="v115-feature-map-v1"`, `class_count=1` (D-01) — the exact
  shape/schema the existing coarse and detailer trainers' `--feature-store` already validates.
  **No changes to those trainers** — proven for real during implementation by dry-running both
  `v50_train_direct_geometry.py --feature-store <bridge output>` and
  `v50_train_geometry_detailer.py --feature-store <bridge output>` against fixture stores; both
  accepted the shape/schema with zero code edits (`input_channels: 4`, `deployment_inputs` includes
  `generated_terrain_feature_map`).
- Source store is never mutated; the derived store is immutable once written, checkpoint-hash bound
  (mirrors `structure_materialize.py`/`spec116_structure_to_feature_map.py`).

## US3(ii) — paired chain-integration comparison (USER-RUN, existing trainers, unmodified)

Run the **existing, unmodified** coarse and/or detailer trainers twice each — once without the
bridged lattice store, once with — on the identical held-out split, exactly as the structure-
augmented detailer comparison ran this session:

```
uv run python scripts/v50_train_direct_geometry.py \
  --store <store> --architecture mit_b0_regression --source <authored|all> \
  --held-out-split <split dir> \
  --feature-store <bridged lattice feature-map store> \
  --output <run dir> --run-id <run id> \
  --confirm-run
```

```
uv run python scripts/v50_train_geometry_detailer.py \
  --store <store> --coarse-store <materialized coarse store> --source <authored|all> \
  --held-out-split <split dir> \
  --feature-store <bridged lattice feature-map store> \
  --output <run dir> --run-id <run id> \
  --confirm-run
```
- Both `--feature-store` flags already exist and are already validated (the coarse trainer since
  Spec 115; the detailer trainer since this session's structure-augmented extension). This feature
  adds no flags to either script.
- The report names which feed point (coarse, detailer, both) measurably reduced relief-region MAE
  against the pre-existing real baseline, per spec US3 acceptance scenario 3.
