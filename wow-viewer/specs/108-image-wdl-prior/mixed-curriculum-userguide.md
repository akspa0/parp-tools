# Mixed WDL Curriculum: Runbook

Run every command from `I:\parp\parp-tools\wow-viewer\data-harvester`.

This is the universal-model lane. It uses the 240-row mixed store, never the old analytic-only
store. The builder assigns `split=train|val` by whole source group, so use `split=val` everywhere
below; do not pick one arbitrary tile as validation.

```text
Mixed store: I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr
Training cap: 240 rows total (144 real 0.5.3 + 96 synthetic)
Holdout:     all rows whose index.parquet split is val
```

## 1. Confirm the mixed store

```powershell
Get-Content "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr\summary.json"
```

Continue only when it says 240 total rows, 144 real rows, 96 synthetic rows, and contains all four
0.5.3 maps.

## 2. Train the WDL prior on the mixed store

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr" --val-key split --val-value val --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1" --epochs 100 --batch 16 --workers 4 --patience 15 --max-object-coverage 0.0
```

Expected output: `checkpoint_best.pt` and `training_summary.json`. Early stopping is normal; it is
not a failure.

## 3. Generate predicted WDL lattices for the whole mixed store

This is RGB-only inference. The archive is the only WDL input permitted for the following V8 run.

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\checkpoint_best.pt" --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\generated_wdl_all_rows.npz" --device cuda
```

## 4. Train V8 on generated—not ground-truth—WDL

```powershell
uv run python scripts/train_spec103_v7.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr" --generated-wdl-priors "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\generated_wdl_all_rows.npz" --height-hints wdl --val-key split --val-value val --output "I:\parp\parp-tools\wow-viewer\output\spec108_v8_mixed_generated_wdl_v1" --arch v8 --epochs 100 --batch 8 --workers 4 --patience 15 --wdl-prior-dropout 0.15 --max-object-coverage 0.0
```

Open `history.json` after the run. It must contain `generated_wdl_priors.path` and
`generated_wdl_priors.sha256`. If either is absent, stop: that run did not use generated WDL.

## 5. Infer every held-out mixed row and validate

```powershell
uv run python scripts/infer_spec103_v7.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_v8_mixed_generated_wdl_v1\checkpoint_best.pt" --generated-wdl-priors "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\generated_wdl_all_rows.npz" --val-key split --val-value val --output "I:\parp\parp-tools\wow-viewer\output\spec108_v8_mixed_generated_wdl_v1\predictions" --device cuda
uv run python scripts/validate_spec103_labelfree.py --predictions "I:\parp\parp-tools\wow-viewer\output\spec108_v8_mixed_generated_wdl_v1\predictions" --report "I:\parp\parp-tools\wow-viewer\output\spec108_v8_mixed_generated_wdl_v1\labelfree_report.json" --gt-store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr"
```

## 6. Inspect several held-out WDL prior examples

This prints the first six held-out row numbers. Run the evaluator and OBJ visualizer once for a real
row and once for a synthetic row from this list; do not use a training row.

```powershell
uv run python -c "import pyarrow.parquet as pq; p=r'I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr\index.parquet'; rows=pq.read_table(p).to_pylist(); print([(i,r['source_kind'],r['map']) for i,r in enumerate(rows) if r['split']=='val'][:6])"
```

For each displayed row number `<ROW>`, run:

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\mixed_053_synthetic_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\checkpoint_best.pt" --row <ROW> --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\review_row_<ROW>"
uv run python scripts/visualize_spec103_wdl_prior.py --evaluation "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_mixed_v1\review_row_<ROW>"
```

## Stop conditions

- Do not continue if the builder summary is not 144 real + 96 synthetic across four maps.
- Do not switch to the analytic-only corpus if a result is poor; retain the mixed run artifacts.
- Do not trust a V8 run unless `history.json` records the generated archive identity.
