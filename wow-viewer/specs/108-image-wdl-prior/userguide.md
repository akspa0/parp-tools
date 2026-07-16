# Superseded: Analytic-Only WDL Prior Runbook

> Do not start this sequence for the universal model. It remains useful as a smoke reference only.
> Spec 108 now requires a capped mixed real-plus-synthetic curriculum before another training run;
> see `plan.md` Phase 3 and `tasks.md` T009–T015.

The active, complete post-build runbook is [mixed-curriculum-userguide.md](mixed-curriculum-userguide.md).

## Active next command

Run from `I:\parp\parp-tools\wow-viewer\data-harvester`. This is the user-owned CPU/I/O build;
it writes the new <256-row store and does not start CUDA training.

```powershell
uv run python scripts/spec108_build_mixed_curriculum.py --real-store "I:\parp\parp-tools\wow-viewer\output\datasets\v18\0_5_3_3368.zarr" --synthetic-store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --real-rows 144 --synthetic-rows 96 --max-rows 240 --output "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr"
```

It writes `summary.json` and `index.parquet`. Confirm the summary reports 240 total rows, 144 real,
96 synthetic, and all four 0.5.3 maps before starting a training run.

Run every command from `I:\parp\parp-tools\wow-viewer\data-harvester`.

This lane has one source of training truth: the synthesized authored-lighting store below. Do not
replace it with V18, curation, a real minimap, macro/blocky output, or chunk-motif output.

```text
Store:      I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr
Holdout:    pattern=crater (96 rows)
Train:      864 rows
Checkpoint: checkpoint_best.pt
```

## 1. Train the RGB-to-WDL prior

The existing completed run is at
`I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater`.
Its best checkpoint is usable now. Run this only when deliberately training a fresh replacement:

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --val-key pattern --val-value crater --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2" --epochs 80 --batch 32 --workers 4 --patience 10
```

Expected output: `checkpoint_best.pt` and `training_summary.json`. Early stopping is success: it
means the held-out crater score stopped improving.

## 2. Inspect the held-out synthetic family

Use the new checkpoint path from step 1; if skipping retraining, use the existing checkpoint path
shown above.

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\checkpoint_best.pt" --row 192 --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\heldout_crater_row192"
uv run python scripts/visualize_spec103_wdl_prior.py --evaluation "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\heldout_crater_row192"
```

Open `visual_review\predicted_wdl_reconstruction.obj` and
`visual_review\truth_wdl_reconstruction.obj` side by side. They should agree on the crater’s coarse
shape; the full-resolution mesh shows detail the WDL prior deliberately cannot encode.

## 3. Generate WDL priors for every synthetic row

This is the bridge to V8. It runs the trained prior on RGB only and writes one predicted 17x17 outer
lattice plus one 16x16 inner lattice per row. It does not read WDL or height as model input.

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\checkpoint_best.pt" --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\generated_wdl_all_rows.npz" --device cuda
```

Expected output: `generated_wdl_all_rows.npz`. The next step refuses an archive produced from a
different store or missing a selected row.

## 4. Train V8 with the generated WDL prior

This is the deployment-shaped synthetic terrain model: ch6 and the WDL-derived height hints come
from the generated archive, never from ground-truth WDL.

```powershell
uv run python scripts/train_spec103_v7.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --generated-wdl-priors "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\generated_wdl_all_rows.npz" --height-hints wdl --val-key pattern --val-value crater --output "I:\parp\parp-tools\wow-viewer\output\spec108_v8_synthetic_generated_wdl_crater_v1" --arch v8 --epochs 80 --batch 8 --workers 4 --patience 12 --wdl-prior-dropout 0.15
```

Expected output: `checkpoint_best.pt`, validation previews, and `history.json`. Confirm
`history.json` contains `generated_wdl_priors.path` and a SHA-256; otherwise stop—the run was not
deployment-shaped.

## 5. Run the synthetic holdout through V8 and validate it

```powershell
uv run python scripts/infer_spec103_v7.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_v8_synthetic_generated_wdl_crater_v1\checkpoint_best.pt" --generated-wdl-priors "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater_v2\generated_wdl_all_rows.npz" --val-key pattern --val-value crater --output "I:\parp\parp-tools\wow-viewer\output\spec108_v8_synthetic_generated_wdl_crater_v1\predictions" --device cuda
uv run python scripts/validate_spec103_labelfree.py --predictions "I:\parp\parp-tools\wow-viewer\output\spec108_v8_synthetic_generated_wdl_crater_v1\predictions" --report "I:\parp\parp-tools\wow-viewer\output\spec108_v8_synthetic_generated_wdl_crater_v1\labelfree_report.json" --gt-store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr"
```

`[PASS]` means the reconstruction is structurally self-consistent. The `--gt-store` value adds
development-only diagnostics; it does not decide the pass/fail result.

## Stop conditions

- If step 2’s WDL visual review is visibly wrong, stop before steps 3–5 and keep the checkpoint/output.
- If step 4 says the generated archive has a wrong store or missing rows, regenerate it with step 3.
- If step 5 fails, preserve `history.json`, `labelfree_report.json`, and the prediction directory; do
  not switch the training store to real data.
