# Quickstart: V60 Controlled Reconstruction Experiment

This is a staged workflow. The first commands produce and validate only the small project-owned
control corpus. The user runs synthesis, client-backed processing, and GPU training.

## 1. Build the control tool

```powershell
dotnet build "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj" -c Debug --no-restore
```

## 2. Generate the tiny control corpus

```powershell
dotnet "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll" control-corpus --output-dir "I:/parp/parp-tools/wow-viewer/output/datasets/v60/control-v1" --variants 4 --holdout-families chunk_grid,island_sea,sheer_dropoff,zone_style_blend,cross_tile_lightning,cross_tile_burn,noise,pathological
```

This writes 108 deterministic terrain NPZ rows across the full default taxonomy, a control manifest,
and a sibling `object-sieve-v1` corpus with 540 object-overlay rows (five placement regimes per
terrain row). No client harvest is involved. Reduce the family list or variant count only as an
explicit experiment decision; the default run is designed to expose coverage gaps before training.

## 3. Validate before any model run

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run python scripts/v60_validate_control_corpus.py --corpus "../output/datasets/v60/control-v1" --write-report
uv run python scripts/v60_validate_object_sieve.py --corpus "../output/datasets/v60/control-v1/object-sieve-v1" --write-report
```

The validator must report `"valid": true`. Do not train if it reports missing arrays, hash errors,
family leakage, missing complexity buckets, non-finite values, or out-of-range textureless inputs.

## 4. Render visual coverage

```powershell
uv run python scripts/v60_visualize_control_corpus.py --corpus "../output/datasets/v60/control-v1" --output-dir "../output/datasets/v60/control-v1/visual-review" --variants-per-family 4
uv run python scripts/v60_visualize_object_sieve.py --corpus "../output/datasets/v60/control-v1/object-sieve-v1" --output-dir "../output/datasets/v60/control-v1/object-sieve-v1/visual-review"
```

Inspect `control-family-atlas.png` for one representative of every family and
`control-variant-atlas.png` for within-family variation. Inspect
`control-cross-tile-atlas.png` to verify that the lightning/burn motifs continue across seams.
Inspect `object-sieve-input-atlas.png` and `object-sieve-mask-atlas.png` for flat, mountainous,
sheer-dropoff, style-blend, dense-object, and boundary-crossing coverage. The JSON reports must
show all four complexity buckets, `cross_tile_complete: true`, and
`coverage_complete: true` before model work.

## 5. Run the object-sieve control experiment

The object decomposition data, validator, visual review, and model/loss variants are now present.
The bounded experiment must use the emitted `object-sieve-v1` manifest and compare `clean_only`,
`auxiliary_mask_loss`, and `predicted_mask_guided` variants. The guided model receives its predicted
mask, never the ground-truth mask. The training/evaluation command remains withheld until its
loader and report writer land.

## 6. Run the limited height control experiment

The training command is intentionally not declared complete by this plan until the control-v1
loader/evaluator task lands. The eventual command must accept the manifest, keep the family holdout
fixed, run the limited row sizes, and write an `experiment-report.schema.json` report. GPU training
is user-run.

## 7. Normalize a tiny 0.x/1.x real sample

Only after the control result is recorded, process an explicit small source manifest from approved
0.x/1.x roots through the versioned albedo-normalization operation. The operation must write an
`albedo-gate-report.schema.json` report. There is no valid shortcut that treats the synthetic
textureless control image as the result for a real authored tile.

The first real-data run accepts only rows whose gate decision is `accepted`. Rejected and
quarantined rows remain visible in the report and do not enter the model input directory.

## 8. Transfer gate

Run the tiny accepted real sample through the same evaluator and compare its distribution, failure
cases, and baseline-relative metrics with the control result. Broader real-data processing is
allowed only when the transfer report says `expand`; otherwise the next task is diagnosis of
albedo normalization or domain shift.
