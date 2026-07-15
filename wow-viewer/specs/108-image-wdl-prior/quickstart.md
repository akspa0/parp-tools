# Quickstart

Run from `wow-viewer/data-harvester`. The user owns these CUDA runs.

1. Generate the varied controlled corpus (320 known-height tiles: ten terrain families, two
   amplitudes, sixteen parameterized variants each). This is CPU preparation; the printed ADT/capture
   commands remain optional because the authored-lighting store path below renders from known height.

```powershell
uv run python scripts/spec103_make_synthetic_adts.py --output "I:\parp\parp-tools\wow-viewer\output\spec108\synthetic_varied_v1" --map-name synth108 --variants-per-pattern 16 --seed 103
uv run python scripts/spec103_build_synthetic_store.py --manifest "I:\parp\parp-tools\wow-viewer\output\spec108\synthetic_varied_v1\synthetic_manifest.json" --lighting-time 0.25 --lighting-time 0.35 --lighting-time 0.50 --synthesize-mcsh --output "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr"
```

2. Train the independent WDL prior, holding out the entire crater family (96 rows):

```powershell
uv run python scripts/train_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --val-key pattern --val-value crater --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater" --epochs 80 --batch 32 --patience 10
```

3. Evaluate the first held-out synthetic crater row (row 192 = first crater variant at time 0.25),
then measure the synthetic-to-real gap with the same checkpoint.

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\spec108\synthetic_varied_lighting_v1.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater\checkpoint_best.pt" --row 192 --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater\heldout_crater_row192"
```

4. Then measure the synthetic-to-real gap on a bright real minimap. The model consumes only RGB;
   `height_257` is opened afterwards solely to score it:

```powershell
uv run python scripts/evaluate_spec103_wdl_prior.py --store "I:\parp\parp-tools\wow-viewer\output\datasets\v18\3_3_5_12340.zarr" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater\checkpoint_best.pt" --row 906 --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater\real_ChamberOfAspectsBlack_29_27"
```

Its `report.json` records truth error and a `standalone_png_vs_store_rgb` round-trip metric after
reloading the exported PNG. Both should be inspected before using the generated prior in V8.

5. Write visible review artifacts for either evaluation directory. This creates textured OBJ meshes
for the predicted paired-WDL reconstruction, the truth WDL reconstruction, and the actual 257×257
truth terrain, plus a signed-error PNG heatmap:

```powershell
uv run python scripts/visualize_spec103_wdl_prior.py --evaluation "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_varied_crater\real_ChamberOfAspectsBlack_29_27"
```

Open `visual_review\predicted_wdl_reconstruction.obj` and
`visual_review\truth_wdl_reconstruction.obj` side by side in an OBJ viewer. The error PNG is red
where predicted terrain is above truth and blue where it is below.

6. Prove the standalone image route using the evaluator's exported minimap. This has no WDL/store
   input at all:

```powershell
uv run python scripts/infer_spec103_wdl_prior.py --image "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_lighting_plateau\real_ChamberOfAspectsBlack_29_27\input_minimap.png" --checkpoint "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_lighting_plateau\checkpoint_best.pt" --output "I:\parp\parp-tools\wow-viewer\output\spec108_wdl_prior_synthetic_lighting_plateau\real_ChamberOfAspectsBlack_29_27\standalone_wdl_lattice.npz"
```

The current V8 checkpoint may still depend on its other auxiliary channels. This slice proves the WDL handoff; replacing those auxiliaries is a later, separate residual-model slice.
