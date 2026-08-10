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

## 5. Build the real-library object-sieve corpus

The old `real-object-masks-v1` run is rejected for precision-object work: its v50 curriculum targets
are tile-level placement projections, not the per-object silhouettes in the v50 library. Use the
read-only Spec 118 library directly and derive a separate corpus over the validated control terrain:

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run python scripts/v60_build_object_library_sieve.py --control-corpus "../output/datasets/v60/control-v1" --object-library "output/spec118/object_mask_library_0_5_3_3368.zarr" --output "../output/datasets/v60/object-library-sieve-v1" --samples-per-terrain 1 --seed 6001
uv run python scripts/v60_validate_object_library_sieve.py --corpus "../output/datasets/v60/object-library-sieve-v1" --write-report
uv run python scripts/v60_visualize_object_library_sieve.py --corpus "../output/datasets/v60/object-library-sieve-v1" --output-dir "../output/datasets/v60/object-library-sieve-v1/visual-review" --rows-per-regime 3
```

Review `object-library-sieve-atlas.png`. Its panels are objectified input, clean terrain target,
exact union mask, and per-instance ID map. The manifest must report the v50 library provenance and
non-empty silhouettes across sparse, dense, overlap, and boundary-crossing rows before training.

## 5b. Train the corrected object-sieve variants

First print the dry-run plans for all three ablations:

```powershell
uv run python scripts/v60_train_object_sieve.py --corpus "../output/datasets/v60/object-library-sieve-v1" --output "../output/datasets/v60/object-sieve-runs/library-clean-only" --variant clean_only --epochs 40 --batch 8
uv run python scripts/v60_train_object_sieve.py --corpus "../output/datasets/v60/object-library-sieve-v1" --output "../output/datasets/v60/object-sieve-runs/library-auxiliary-mask" --variant auxiliary_mask_loss --epochs 40 --batch 8
uv run python scripts/v60_train_object_sieve.py --corpus "../output/datasets/v60/object-library-sieve-v1" --output "../output/datasets/v60/object-sieve-runs/library-predicted-mask-guided" --variant predicted_mask_guided --epochs 40 --batch 8
```

After the visual and validation reports pass, the user launches each GPU run by adding
`--confirm-run` to the corresponding command. The trainer never supplies the ground-truth mask as
an input; it reports clean-terrain and contamination-mask signals independently by regime.

The old `real-object-masks-v1/experiment_report.json` must not be used as a v60 model result.

## 5c. Build and train the footprint-guided object marker

The marker specialist is separate from the sieve. It consumes an image plus one candidate
footprint, learns knownness and an embedding, and resolves exact identity through the read-only
v50 object library gallery. The first command builds candidates from the validated library-sieve
corpus; it does not use the old v50 dot projections.

```powershell
uv run python scripts/v60_build_object_marker.py --sieve-corpus "../output/datasets/v60/object-library-sieve-v3" --object-library "output/spec118/object_mask_library_0_5_3_3368.zarr" --output "../output/datasets/v60/object-marker-v1" --seed 6001
uv run python scripts/v60_validate_object_marker.py --corpus "../output/datasets/v60/object-marker-v1" --write-report
uv run python scripts/v60_train_object_marker.py --corpus "../output/datasets/v60/object-marker-v1" --object-library "output/spec118/object_mask_library_0_5_3_3368.zarr" --output "../output/datasets/v60/object-marker-runs/library-marker-v1" --epochs 40 --batch 16
```

Fully occluded overlap instances are recorded in `skipped_instances` and excluded from candidate
training because the visible-winner instance map contains no footprint for them. If a build stops,
use a fresh output name; an incomplete output is never reused.

The last command is a dry run. After the corpus/visual report is accepted, the user adds
`--confirm-run` to launch CUDA training. The eventual marking command takes an input minimap and
explicit candidate footprints and writes `known_object_marker_256` plus an identity table:

```powershell
uv run python scripts/v60_mark_known_objects.py --minimap-npz "../output/datasets/v60/marker-input/example.npz" --checkpoint "../output/datasets/v60/object-marker-runs/library-marker-v1/checkpoint_best.pt" --object-library "output/spec118/object_mask_library_0_5_3_3368.zarr" --output "../output/datasets/v60/marked/example"
```

This first marker slice does not discover footprints. It measures identity/knownness conditional on
the supplied footprint, leaving proposal recall as a separate future specialist.

## 5d. Review paired real/flat validation evidence

The existing v50.1 mixed curriculum contains same-tile authored and legacy flat synthetic rows for
most source groups. This command selects a deterministic 16-tile Azeroth holdout, compares authored
RGB to the flat fake maptexture, and writes both a JSON absolute-difference report and a visual
atlas. It does not train and it never uses the real masks as inputs.

```powershell
uv run python scripts/v60_validate_real_synthetic_pairs.py --store "../output/datasets/v50/v50.1/curriculum-0_5_3_3368-obj_v1.zarr" --output "../output/datasets/v60/real-synthetic-pair-validation-v1" --split map_holdout --val-map Azeroth --validation-rows 16
```

Review `real-synthetic-pair-atlas.png` and `real-synthetic-pair-report.json`. The report must show
complete pair identity, no split leakage, `labels_used_as_inputs: false`, and
`legacy_synthetic_is_terrain_shadow_target: false`. The observed initial 16-tile sample had mean
normalized RGB MAE `0.1812`; this is flat-vs-authored absolute-difference evidence, not a shadow
target.

To compare the flat-row absolute difference with the fixed terrain-shadow renderer, first rebuild
the harvest tool and have the user harvest only these same 16 tiles from the approved 0.5.3 client:

```powershell
dotnet build "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj" -c Debug --no-restore
dotnet "I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/bin/Debug/net10.0/WowViewer.Tool.Harvest.dll" harvest-map-mpq --client-root "H:/CLIENTS/Vanilla/0.x/0_5_3_3368/World of Warcraft" --map "Azeroth" --output-dir "I:/parp/parp-tools/wow-viewer/output/datasets/v60/real-shadow-npz-v1" --tile-list "0,0;24,53;27,52;27,53;27,54;27,55;27,56;27,57;27,58;24,54;28,22;28,23;28,24;28,25;28,26;28,27" --force
```

The harvest must emit `terrain_shadow_256` in every requested NPZ. Then rerun the pair report with
`--shadow-npz-dir`:

```powershell
uv run python scripts/v60_validate_real_synthetic_pairs.py --store "../output/datasets/v50/v50.1/curriculum-0_5_3_3368-obj_v1.zarr" --output "../output/datasets/v60/real-synthetic-pair-validation-v1" --split map_holdout --val-map Azeroth --validation-rows 16 --shadow-npz-dir "../output/datasets/v60/real-shadow-npz-v1"
```

The fixed-shadow correlations are calibration diagnostics, not a substitute target or a claim that
flat-vs-authored absolute difference is pure terrain shadow. This pair report is separate from the
library-derived object-sieve training in section 5b and does not turn the old dot projections into
precision labels.

## 6. Run the terrain-only architecture bakeoff

The active lane uses only the validated control NPZs. It does not read the object-sieve,
object-library, marker, real-client, or albedo artifacts. The first bakeoff runs one common
32-row training subset through each architecture. First print the fixed-family split, tile-mean
baseline, parameter counts, and model contracts:

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run --no-cache python scripts/v60_run_experiment.py --corpus "../output/datasets/v60/control-v1" --output "../output/datasets/v60/terrain-architecture-runs/control-v1" --architectures "unet_lite_v2,pyramid_cnn,dpt_small,segformer_b0" --train-sizes 32 --epochs 40 --batch-size 8 --lr 1e-3 --seed 6001
```

After reviewing the dry-run plan, the user may launch CUDA training by adding `--confirm-run`. It
writes `unet_lite_v2/train-032/`, `pyramid_cnn/train-032/`, `dpt_small/train-032/`, and
`segformer_b0/train-032/`, plus `experiment-report.json` with per-architecture, per-family
baseline-relative metrics. Use a fresh output directory for every confirmed run. To add the
learning curve after this first one-of-each comparison, use `--train-sizes 8,16,32`; the trainer
uses one shared seeded nested schedule so the larger sets extend the smaller sets.

The object-sieve, object-library, and footprint-guided marker commands above are parked for later;
they are not part of this terrain gate.

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
