# Quickstart

Run from `wow-viewer/data-harvester`. The user owns these CUDA runs.

1. Build a compact representative store using the existing Spec 103 curation output; retain source-group metadata and reserve one complete group for validation.
2. Train the independent WDL prior:

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store <representative-store.zarr> --output ../output/spec108_wdl_prior_v1 --val-key pattern --val-value <held-out-pattern> --epochs 80 --batch 32
```

3. Produce generated priors and use them in V8:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --store <representative-store.zarr> --checkpoint ../output/spec108_wdl_prior_v1/checkpoint_best.pt --output ../output/spec108_wdl_prior_v1/generated_priors.npz
uv run python scripts/infer_spec103_v7.py --store <representative-store.zarr> --checkpoint <v8-checkpoint.pt> --generated-wdl-priors ../output/spec108_wdl_prior_v1/generated_priors.npz --output ../output/spec108_v8_generated_prior
```

The current V8 checkpoint may still depend on its other auxiliary channels. This slice proves the WDL handoff; replacing those auxiliaries is a later, separate residual-model slice.
