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

Widens the existing harvest signal selection; exported through whatever command already writes the
v50 store for a build (no new command surface). Verified by re-running the existing store build/
finalize path against a build already on disk and confirming the four new arrays
(`wdl_lattice_outer17`/`inner16`/`outer_present`/`inner_present`) are present with the expected
shapes.

## US2 — standalone lattice predictor (USER-RUN, dry-run first)

```
uv run python scripts/spec117_train_lattice.py \
  --store <v50 curriculum store with wdl_lattice_* arrays> \
  --held-out-split <Spec 116 held-out split dir> \
  --output <run dir> \
  [--epochs 100] [--batch 16] [--lr 2e-4] [--lr-schedule onecycle] \
  [--confirm-run]
```
- **Default (no `--confirm-run`)**: validates inputs, prints the full plan + time/memory estimate,
  exits without training.
- **With `--confirm-run`**: trains the standalone predictor; writes `checkpoint_best.pt` +
  `model_stage_run.json` (schema `v50-model-stage-run-v1`, `stage="lattice_prior"`).
- Refuses a leaky or unspecified held-out split, matching the existing coarse/detailer trainers'
  `--held-out-split` behavior exactly (reused code path, not reimplemented).
- **Gate**: held-out lattice-point MAE vs the per-tile-mean lattice baseline (D-02);
  `promotion_verdict` starts `pending`.

## US3(i) — bridge the frozen predictor's output into the existing feature-store shape

```
uv run python scripts/spec117_lattice_to_feature_map.py \
  --store <v50 curriculum store> \
  --checkpoint <lattice predictor checkpoint_best.pt> \
  --output <derived feature-map store> \
  [--write]
```
- **Default**: prints the materialization plan only.
- **With `--write`**: runs the frozen checkpoint over every tile in the source store, writes a
  `feature_map` array of shape `(N, 1, 256, 256)` under `schema="v115-feature-map-v1"`,
  `class_count=1` (D-01) — the exact shape/schema the existing coarse and detailer trainers'
  `--feature-store` already validates. **No changes to those trainers.**
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
