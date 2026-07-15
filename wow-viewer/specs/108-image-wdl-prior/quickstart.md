# Quickstart

Run from `wow-viewer/data-harvester`. The user owns these CUDA runs.

1. Use the real paired V18 store with the existing Spec 103 representative-pattern curation manifest.
   The trainer reads only the manifest's selected rows; it does not train on the whole store.
2. Train the independent WDL prior:

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\v18\3_3_5_12340.zarr" --curation-manifest "I:\parp\parp-tools\wow-viewer\output\datasets\v18\curation\v18_focus_tiny_800ish\kept_tiles.parquet" --val-key map --val-value ChamberOfAspectsBlack --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber" --epochs 80 --batch 16 --patience 10
```

3. Produce generated priors and use them in V8:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\v18\3_3_5_12340.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_expansion01\checkpoint_best.pt" --val-key map --val-value Expansion01 --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_expansion01\generated_Expansion01_priors.npz"
```

4. Before trusting V8, evaluate a held-out real tile. This model call consumes its minimap RGB;
   `height_257` is opened afterwards only to score the prediction:

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\v18\3_3_5_12340.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber\checkpoint_best.pt" --row 906 --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber\real_ChamberOfAspectsBlack_29_27"
```

Its `report.json` records truth error and a `standalone_png_vs_store_rgb` round-trip metric after
reloading the exported PNG. Both should be inspected before using the generated prior in V8.

5. Prove the standalone image route using the evaluator's exported minimap. This has no WDL/store
   input at all:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --image "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber\real_ChamberOfAspectsBlack_29_27\input_minimap.png" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber\checkpoint_best.pt" --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_tiny800_335_chamber\real_ChamberOfAspectsBlack_29_27\standalone_wdl_lattice.npz"
```

The current V8 checkpoint may still depend on its other auxiliary channels. This slice proves the WDL handoff; replacing those auxiliaries is a later, separate residual-model slice.
