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
  --rescore-checkpoint <existing geometry checkpoint.pt> \
  [--relief-threshold <std>] [--print-only]
```
- Reports flat vs relief-bearing error + trivial baseline per stratum (FR-011). No training.

## US3 — structure prediction training (USER-RUN, dry-run first)

```
uv run python scripts/spec116_train_structure.py \
  --store <v50 curriculum store> \
  --split <split dir> \
  --dumps <texture-name dump json>... \
  --slot 1 \
  --vocabulary <US1 recommendation: family_keyed|slot_keyed> \
  --epochs 100 --batch-size 16 --lr 1e-3 --max-class-weight 15.0 \
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

```
uv run python scripts/spec116_infer_structure.py \
  --checkpoint <structure checkpoint_best.pt> \
  --inputs <tile png | dir>... \
  [--tile-table <MTEX table json>] \
  --output <infer dir>
```
- Predicts family probabilities per chunk/slot; resolves legal local ids when a tile table is
  supplied; emits `structure-infer.json` (see `structure-run.schema.json` infer variant).
- OOD images (no `--tile-table`) set `legal_table_available=false` and never fabricate references
  (D-05).

## US5 — materialize predicted structure + geometry comparison

```
uv run python scripts/spec116_materialize_structure.py \
  --store <v50 curriculum store> \
  --split <split dir> \
  --checkpoint <structure checkpoint_best.pt> \
  --output <derived structure store> \
  [--write]
```
Then the user runs the existing Spec 114 geometry trainer **with** and **without**
`--feature-store <derived structure store>` on the same split, and the comparison is recorded in
`v50-structure-geometry-comparison-v1` (data-model.md).