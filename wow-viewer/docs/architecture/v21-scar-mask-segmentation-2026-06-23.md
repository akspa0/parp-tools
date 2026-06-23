# V21 Scar Mask Segmentation

> Deprecated as primary route (2026-06-23): This model is a coarse diagnostic baseline only. It predicts whole-tile alpha-scar presence and does not recover reusable brush/fractal/paste units. Active decomposition work is now specified in `wow-viewer/specs/076-full-map-fractal-brush-library/` and `full-map-fractal-brush-library-2026-06-23.md`.

## Purpose

V21 scar-mask segmentation is the first trainable model that consumes the spec 074 alpha-brush/scar signals. It predicts one signal only:

```text
minimap_rgb_256 -> alpha_scar_mask_256
```

The target is the binary union of MCAL alpha layers L1-L3 after thresholding at `0.05`. This detects where authored alpha brushwork exists. It does not classify exact scar IDs and does not predict multi-tile prefabs.

## Why This Is First

Spec 074 found `320,368` component instances and `263,188` exact binary scar patterns. A direct exact-scar classifier would be too sparse and would mostly learn local hand edits. A binary scar-presence model is the smallest useful segmentation baseline and keeps the output in the same 256x256 coordinate space as the scar catalog.

The on-disk dataset remains the patched V18 Zarr corpus. The model lane is V21-era because V18 is no longer the active model generation.

## Model Contract

- Input: `(B, 3, 256, 256)` float minimap RGB in `[0,1]`.
- Output: `(B, 1, 256, 256)` logits.
- Loss: BCE-with-logits plus soft Dice loss.
- Checkpoint: one model, one signal, no auxiliary heads.

## Validation

```powershell
cd wow-viewer/data-harvester
uv run pytest src/harvester/test_v21_scar_mask.py
uv run python scripts/train_v21_scar_mask.py --builds 0_5_3_3368 3_3_5_12340 --max-steps 2 --val-max-steps 1 --batch-size 2 --run-name smoke
```

Expected smoke artifacts:

```text
../models/v21/scar-mask/runs/smoke/latest.pt
../models/v21/scar-mask/runs/smoke/best.pt
../models/v21/scar-mask/runs/smoke/metrics.json
../models/v21/scar-mask/runs/smoke/run_config.json
../models/v21/scar-mask/runs/smoke/preview_latest.png
../models/v21/scar-mask/runs/smoke/preview_best.png
../models/v21/scar-mask/runs/smoke/best_epoch_0001.png
../models/v21/scar-mask/runs/smoke/previews/epoch_0001.png
```

## Resume

Checkpoints include model weights, optimizer state, completed epoch, best validation loss, and metric history. Older pre-resume checkpoints can still resume model weights; when optimizer state is absent, the trainer starts a fresh optimizer and recovers best/history from the sibling `metrics.json` when present.

Resume by pointing `--resume` at `latest.pt` or `best.pt` and setting `--epochs` to the new final epoch number:

```powershell
uv run python scripts/train_v21_scar_mask.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --epochs 40 `
  --batch-size 8 `
  --base-channels 32 `
  --lr 0.001 `
  --threshold 0.05 `
  --layers 1,2,3 `
  --device auto `
  --run-name v21_scar_mask_two_build_v2 `
  --resume ../models/v21/scar-mask/runs/v21_scar_mask_two_build_v2/latest.pt
```

## Next Step After V75

Once the V21 scar-mask model can segment scar regions, a later spec can assign predicted components to scar families using the 074 exact/near dedupe outputs. That should be a separate single-output retrieval or embedding model, not a multi-head classifier bolted onto this model.
