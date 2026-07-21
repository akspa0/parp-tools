# CLI Contract: Relational Terrain Layer Reconstruction

**Feature**: 116-relational-terrain-layers | **Date**: 2026-07-21

All commands are thin wrappers under `wow-viewer/data-harvester/scripts/` that import library
logic from `harvester.spec116`. Every command is **dry-run / print-plan by default** and refuses to
write or train without an explicit flag (FR-015). Every training/heavy command is **user-run**
(FR-018); the assistant never launches them. Run with `uv run python -m scripts.<name>` from
`wow-viewer/data-harvester/`.

---

## US1 — family→slot consistency (analysis, no model)

```
uv run python scripts/spec116_family_slot_consistency.py \
  --store <v50 curriculum store> \
  --dumps <texture-name dump json>... \
  --threshold 0.70 \
  [--output <report.json>] [--write]
```
- **Reads**: v50 store (`mcly_texture_ids`, `index.parquet`), texture-name dumps.
- **Writes** (only with `--write`): `family-slot-consistency-<build>-v1.json` (see
  `analysis-report.schema.json`).
- **Default**: prints the per-family slot distribution, summary consistency score, and the
  `slot_keyed`/`family_keyed` recommendation; writes nothing.
- **Time**: <1 min CPU over ~1.4k rows.

## US2 — shape→coverage coupling (analysis, no model)

```
uv run python scripts/spec116_shape_coverage_coupling.py \
  --store <v50 curriculum store> \
  [--output <report.json>] [--write]
```
- **Reads**: `height_257`, `mcly_layer_mask`.
- **Writes** (only with `--write`): `shape-coverage-coupling-<build>-v1.json`.
- **Default**: prints per-(tile,layer) explained variance summary, dip-test p-value, mixture BIC,
  high-coupling tile share, and the linear-vs-nonlinear note.
- **Time**: a few min CPU (per-tile GradientBoosting fits).

## US4 — spatially-isolated held-out split

```
uv run python scripts/spec116_build_held_out_split.py \
  --store <v50 curriculum store> \
  --buffer-rings 1 \
  --seed 116 \
  --output <split dir> \
  [--write]
```
- **Reads**: `index.parquet` (tile coords).
- **Writes** (only with `--write`): `split.parquet` + `split.json` (see
  `held-out-split.schema.json`).
- **Default**: prints the train/held_out counts and the **verified_violation_count** (must be 0);
  writes nothing.
- **Invariant**: exits non-zero if any held-out tile is 8-neighbour-adjacent to a training tile.

## US4 — relief-stratified re-score (re-scores an existing model)

```
uv run python scripts/spec116_train_structure.py \
  --store <v50 curriculum store> \
  --split <split dir> \
  --rescore-checkpoint <existing Spec 114/115 geometry checkpoint.pt> \
  [--relief-threshold <std>] [--rescore-in-channels <int, default 3>] \
  [--rescore-output <report.json>] [--print-only]
```
- Reports flat vs relief-bearing MAE + trivial baseline per stratum, and whether the checkpoint
  beats the trivial baseline on relief-bearing regions (FR-011). No training; read-only.
- `--rescore-checkpoint` switches the CLI into evaluation mode: `--dumps`/`--vocabulary`/
  `--output`/`--slot` are not required in this mode.
- `--rescore-in-channels`: `3` for the RGB-only baseline; a Spec 115 `--feature-store` checkpoint
  needs its trained value.
- Always prints the report; `--rescore-output` additionally persists it unless `--print-only`.

## US3 — structure prediction training (USER-RUN, dry-run first)

```
uv run python scripts/spec116_train_structure.py \
  --store <v50 curriculum store> \
  --split <split dir> \
  --dumps <texture-name dump json>... \
  --slot 1 \
  --vocabulary <US1 family_slot_consistency report JSON> \
  --epochs 100 --batch 16 --lr 1e-3 --max-class-weight 15.0 \
  --output <run dir> \
  [--device cuda] [--confirm-run]
```
- **Default (no `--confirm-run`)**: validates inputs, prints the full plan + time/memory estimate,
  and exits **without training** (FR-015).
- **With `--confirm-run`**: trains one independent per-slot family classifier; writes
  `checkpoint_best.pt` + `structure-run.json` (see `structure-run.schema.json`).
- **Gate**: per-class IoU/recall; `promotion_verdict` starts `pending`; aggregate accuracy is
  reported but never gates (D-08).
- **Time/Mem (estimate)**: ~1.5M params, ~1.4k rows → roughly 1–2 min/epoch on a single GPU;
  <2 GB GPU memory. Printed exactly by the dry run.

## US3 — structure inference + legality audit

Two mutually exclusive input modes:

```
uv run python scripts/spec116_infer_structure.py \
  --checkpoint <structure checkpoint_best.pt> \
  --inputs <tile png | dir>... \
  [--tile-table <MTEX table json>] \
  --slot 1 \
  --output <audit.json> [--write]
```
or, batch mode over an existing v50 store:
```
uv run python scripts/spec116_infer_structure.py \
  --checkpoint <structure checkpoint_best.pt> \
  --store <v50 curriculum store> \
  --dumps <texture-name dump json>... \
  --slot 1 \
  --output <audit.json> [--write]
```
- Predicts family probabilities per chunk/slot; resolves legal local ids when a tile table
  (`--inputs` mode) or texture-name dump (`--store` mode) is supplied; emits `v50-structure-
  infer-v1` (see `structure-run.schema.json` infer variant).
- OOD images (`--inputs` with no `--tile-table`) set `legal_table_available=false` and never
  fabricate references (D-05). `--inputs` runs unchanged on a hand-painted image with no store
  backing at all.

## US5 — materialize predicted structure + geometry comparison

```
uv run python scripts/spec116_materialize_structure.py \
  --store <v50 curriculum store> \
  --checkpoint <structure checkpoint_best.pt> \
  --output <derived structure store> \
  --dumps <texture-name dump json>... \
  --slot 1 \
  [--write]
```
Bridge the derived store into the geometry trainer's feature-store shape (its `--feature-store`
was built for Spec 115's per-pixel `v115-feature-map-v1`, not this per-chunk store):
```
uv run python scripts/spec116_structure_to_feature_map.py \
  --structure-store <derived structure store> \
  --output <feature-map store> \
  [--write]
```
Then the user runs `scripts/v50_train_direct_geometry.py` (`--architecture mit_b0_regression`)
**with** and **without** `--feature-store <feature-map store>` on the same split -- passing
`--held-out-split <split dir>` both times to consume the Spec 116 split directly instead of the
trainer's own `--val-key`/`--val-value` column -- and the comparison is recorded in
`v50-structure-geometry-comparison-v1` (data-model.md).