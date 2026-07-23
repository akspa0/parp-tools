# CLI Contract: Per-Object Occlusion-Aware Masks

**Spec**: [../spec.md](../spec.md) | **Date**: 2026-07-22

All commands run from `wow-viewer/data-harvester/` via `uv run python scripts/<name>.py`.
All training/write CLIs are **dry-run-first**: without the explicit write/confirm flag they print a
JSON plan and write nothing. The user runs every heavy step (FR-012). `&&` chains are PowerShell-7
compatible (Rule 0A).

## 1. Config regeneration (US1, after catalog rows land)

```powershell
uv run python scripts/v50_generate_manifest_template.py --catalog-doc ../docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md --build-id 0_5_3_3368 --release v50.1 --output v50_configs/v50-manifest-template-0_5_3_3368.json --signals-output v50_configs/v50-signals-0_5_3_3368.json
```

Exit 0 on success; the committed drift-guard test must pass unmodified afterwards.

## 2. Mask audit (US1 verification, read-only)

```powershell
uv run python scripts/spec118_audit_object_masks.py --store <store.zarr> [--map Kalimdor] [--output <audit.json>] [--write]
```

Dry-run prints the `v118-object-mask-audit-v1` document to stdout. Fails closed when the store
lacks `object_geometry_visible_mask_257` (expected until a US1 rebuild lands).

## 3. Object-masked height training (US2 — user-run)

Existing trainers, one new flag each:

```powershell
uv run python scripts/v50_train_direct_geometry.py --store <store.zarr> --held-out-split <split.json> --object-mask-weight 1.0 --output <run-dir> [--confirm-run]
uv run python scripts/v50_train_geometry_detailer.py --store <store.zarr> --coarse-store <coarse.zarr> --held-out-split <split.json> --object-mask-weight 1.0 --output <run-dir> [--confirm-run]
```

- `--object-mask-weight` float in [0,1], default `0.0` (bit-parity with current behavior).
- Missing `object_geometry_visible_mask_257` in the store → warning + behaves as 0.0 (mirrors
  `--liquid-mask-weight`), recorded in the dry-run plan.
- Run record reports aggregate, relief-stratified, and object-touched-subset metrics (FR-008).

## 4. Segmenter training (US3 — user-run)

```powershell
uv run python scripts/spec118_train_objects.py --store <store.zarr> --held-out-split <split.json> --output <run-dir> [--base 24] [--epochs 100] [--confirm-run]
```

- `--held-out-split` is REQUIRED (no `--val-key` fallback; refuses an unspecified or leaky split —
  same contract as `spec117_train_lattice.py`).
- Refuses closed when the store lacks `object_geometry_visible_source_257`.
- Writes `training_plan.json`, `run_identity.json`, `checkpoint_best.pt`,
  `model_stage_run.json` (`stage = "object_segmentation"`, `promotion_verdict = "pending"`).

## 5. Segmenter inference (US3)

```powershell
uv run python scripts/spec118_infer_objects.py --checkpoint <checkpoint_best.pt> --inputs <tile.png | <dir>> --output <out-dir> [--write]
uv run python scripts/spec118_infer_objects.py --checkpoint <checkpoint_best.pt> --store <store.zarr> --dumps <out-dir> [--write]
```

The two modes are mutually exclusive. `--inputs` mode needs no store and no ground truth (FR-009;
runs unchanged on a hand-painted OOD tile). Both modes emit per-tile class PNGs and a
`v118-object-infer-v1` audit record; store mode additionally scores per-class IoU/recall where
ground truth exists.

## 6. Feature-store bridge (US3)

```powershell
uv run python scripts/spec118_objects_to_feature_map.py --store <store.zarr> --checkpoint <checkpoint_best.pt> --output <feature-store.zarr> [--write]
```

Writes `schema = "v115-feature-map-v1"`, `class_count = 2` (doodad, building softmax channels),
`feature_map` (N,2,256,256) float16, row-aligned `index.parquet`, checkpoint path+sha256 in attrs.
Refuses a non-empty output; never mutates the source store (FR-013). The output is consumed by the
existing `--feature-store` flag on both geometry trainers with zero trainer changes (FR-011).
