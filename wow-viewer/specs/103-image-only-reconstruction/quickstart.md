# Quickstart: Spec 103 v7 revival

Agent-prepared commands. **The USER runs every capture/training/GPU step** (AGENTS RULE 0).
Python runs from `wow-viewer/data-harvester/`; dotnet runs from `wow-viewer/`.

## 0. Sanity (CPU, safe anytime)

```powershell
uv run pytest tests/spec103 -q          # 7 tests: channel order, trestle, dropout, round trip
```

## 1. Synthetic PoC (MVP — US1)

```powershell
# 1a. author known-height tiles (fast, CPU). Prints ALL commands below with resolved paths.
uv run python scripts/spec103_make_synthetic_adts.py --output ../output/spec103/synthetic
```

Then, from `wow-viewer/` (USER runs; commands are printed by 1a with exact paths):

```powershell
# 1b. blank ADT+WDT+WDL per tile (frozen writers used as-is) — one command per tile
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Release -- map generate-blank ...

# 1c. patch the known heights into the blank ADTs (tiles are non-adjacent by construction,
#     so the patcher's seam stitching never mutates a pattern)
dotnet run --project tools/converter/WowViewer.Tool.Converter -c Release -- terrain-patch-adt ...

# 1d. capture per-tile renders (GPU). Perspective-camera caveat: research-v7-contract.md §7.
dotnet run --project tools/capture/WowViewer.Tool.Capture -c Release -- render ...
```

Back in `data-harvester/`:

```powershell
# 1e. assemble the 13-channel synthetic store (CPU). Use --synthesize-minimaps to proceed
#     with the labeled hillshade fallback before a capture run exists.
uv run python scripts/spec103_build_synthetic_store.py `
    --manifest ../output/spec103/synthetic/synthetic_manifest.json `
    --minimap-dir ../output/spec103/synthetic/captures `
    --output ../output/datasets/spec103/synthetic_v1.zarr

# 1f. TRAIN (GPU — USER runs). Holds out the complete crater pattern.
uv run python scripts/train_spec103_v7.py `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --output ../output/spec103_v7_synth_v1 `
    --val-key pattern --val-value crater `
    --epochs 60 --batch 4 --wdl-prior-dropout 0.25
```

Watch `val_previews/` (minimap | prior | prediction | GT) and the `noprior_l1_g` column —
that is the prior-dropout robustness. T011: record every caveat in
`research-v7-contract.md` §8 after this run.

## 2. Review + label-free validation (US2)

```powershell
# 2a. inference on the held-out pattern (GPU — USER runs)
uv run python scripts/infer_spec103_v7.py `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --checkpoint ../output/spec103_v7_synth_v1/checkpoint_best.pt `
    --val-key pattern --val-value crater `
    --output ../output/spec103_v7_synth_v1/predictions

# deployment-shaped variant: add --drop-prior (image + flat prior only)

# 2b. OBJ meshes for eyeball review (CPU)
uv run python scripts/spec103_export_mesh.py `
    --predictions ../output/spec103_v7_synth_v1/predictions `
    --store ../output/datasets/spec103/synthetic_v1.zarr `
    --output ../output/spec103_v7_synth_v1/meshes

# 2c. label-free acceptance (CPU). --gt-store adds DEV-ONLY diagnostics; never gates.
uv run python scripts/validate_spec103_labelfree.py `
    --predictions ../output/spec103_v7_synth_v1/predictions `
    --report ../output/spec103_v7_synth_v1/labelfree_report.json `
    --gt-store ../output/datasets/spec103/synthetic_v1.zarr
```

Predictions are `terrain-patch-adt`-compatible: patch them into blank ADTs (as in 1c) to view
the reconstructed terrain in WoWViewer itself.

## 3. Real clean data — curate first, then train

Spec Principle #5: height under an object is occluded in the minimap, so object tiles are
**impossible targets** and must be dropped, not learned. Curation is mandatory, not optional.

```powershell
# 3a. verify + pin the real store (CPU, read-only; V18 already pairs minimap + height)
uv run python scripts/spec103_build_real_store.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --output ../output/spec103/real_store_contract.json

# 3b. CURATE + BUCKET (CPU, read-only, ~a few min). Drops object-contaminated, blank, and
#     height/normal-mismatch tiles; writes an auditable manifest + per-map/regime buckets.
#     V18 result: 5134 -> 3131 kept (410 blank, 1593 object-contaminated dropped).
#     Purist zero-object set: add --max-object-coverage 0.0 (2650 tiles).
uv run python scripts/spec103_curate_dataset.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --output ../output/spec103/curation_v18_v1

# 3c. TRAIN on the curated set (GPU — USER runs). Complete-map holdout inside the kept tiles
#     (Azeroth: 2666 train / 465 val, all clean terrain).
uv run python scripts/train_spec103_v7.py `
    --store ../output/datasets/v18/3_3_5_12340.zarr `
    --curation-manifest ../output/spec103/curation_v18_v1 `
    --output ../output/spec103_v7_real_v1 `
    --val-key map --val-value Azeroth `
    --epochs 80 --batch 8 --wdl-prior-dropout 0.25
```

Without `--curation-manifest` the trainer still drops object tiles by default
(`--max-object-coverage 0.02`); pass `1.0` only for the v7-faithful keep-all ablation.

## 4. Shadow capture lane (T018, exploratory — USER runs)

The deterministic fixed-light capture contract is Spec 102 N011-N013
(`WowViewer.Tool.ValidationCapture capture --variants ...`, objects/textures/liquids off).
Run it against the patched synthetic map from 1c and correlate shadow luminance with the
known height gradients; record findings in `research-v7-contract.md` §8.
