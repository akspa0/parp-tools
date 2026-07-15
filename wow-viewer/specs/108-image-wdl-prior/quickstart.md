# Quickstart

Run from `wow-viewer/data-harvester`. The user owns these CUDA runs.

1. Use the real paired V18 store with the existing Spec 103 representative-pattern curation manifest.
   The trainer reads only the manifest's selected rows; it does not train on the whole store.
2. Train the independent WDL prior:

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store <real-v18-store.zarr> --curation-manifest <spec103-curation-manifest-or-directory> --output ../output/spec108_wdl_prior_v1 --val-key map --val-value ignored-when-manifest-partitioned --epochs 80 --batch 32
```

3. Produce generated priors and use them in V8:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --store <real-v18-store.zarr> --checkpoint ../output/spec108_wdl_prior_v1/checkpoint_best.pt --output ../output/spec108_wdl_prior_v1/generated_priors.npz
uv run python scripts/infer_spec103_v7.py --store <real-v18-store.zarr> --checkpoint <v8-checkpoint.pt> --generated-wdl-priors ../output/spec108_wdl_prior_v1/generated_priors.npz --output ../output/spec108_v8_generated_prior
```

4. Before trusting V8, evaluate a held-out real tile. This model call consumes its minimap RGB;
   `height_257` is opened afterwards only to score the prediction:

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store <real-representative-store.zarr> --checkpoint ../output/spec108_wdl_prior_v1/checkpoint_best.pt --row <held-out-row> --output ../output/spec108_wdl_prior_v1/real_tile_row<held-out-row>
```

5. Prove the standalone image route using the evaluator's exported minimap. This has no WDL/store
   input at all:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --image ../output/spec108_wdl_prior_v1/real_tile_row<held-out-row>/input_minimap.png --checkpoint ../output/spec108_wdl_prior_v1/checkpoint_best.pt --output ../output/spec108_wdl_prior_v1/real_tile_row<held-out-row>/standalone_wdl_lattice.npz
```

The current V8 checkpoint may still depend on its other auxiliary channels. This slice proves the WDL handoff; replacing those auxiliaries is a later, separate residual-model slice.
