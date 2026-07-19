# Quickstart: Direct Minimap-to-Terrain Reconstruction

## Status

This is a planning handoff, not a runnable trainer handoff. The commands under **Planned commands**
name the intended CLI contract; those scripts do not exist until their tasks are implemented and
fixture-proven. Do not substitute the old WDL trainer.

## Gate 0 — finish the corrected source evidence

Before Spec 114 can build a real corpus:

1. Complete Spec 113 T010b with the fixed-noon-white synthetic renderer.
2. Visually compare the real authored tile, synthetic 256 tile, and synthetic 1024 detail tile.
3. Refresh only the stale synthetic RGB arrays; numeric height, normals, liquid, material, alpha,
   authored RGB, and other unaffected v50 signals remain valid.
4. Record the new store/manifest hashes and `NoonWhiteGlobal` provenance.

Spec 112 T021 may still be run as the current lean direct-height baseline. Its result becomes the
mandatory `direct_cnn_v112` comparison rather than a discarded lane.

## Architecture comparison to implement

| Stage | Required baseline | Candidate | Output | Explicitly excluded |
|---|---|---|---|---|
| Direct geometry | Spec 112 lean CNN | MiT-B0/SegFormer-style continuous decoder | `relative_height_257` | WDL prior, DA-V2, GAN/diffusion height |
| Object visibility | Empty/all-object baselines | Compact SegFormer semantic mask | one object mask | authored-minus-synthetic RGB labels |
| Terrain features | Majority family | Compact SegFormer semantic classifier | one feature map | shared geometry weights |
| Texture families | Per-map majority | Small ordered family selector | ordered family IDs | alpha prediction head |
| Alpha stack | Base-only/uniform blend | Lean U-Net/FPN regressor | one ordered alpha stack | texture identity head |
| Visual detail | Spec 113 RRDB floor | Spec 113 RealPLKSR | detailed RGB | numeric terrain truth |

Pretrained weights are optional ablations. Any Hub artifact must be license-recorded and pinned to
an immutable revision/hash; every stage must retain a from-scratch/local baseline on the same split.

## Planned commands — available only after the named tasks land

From `wow-viewer/data-harvester`:

```powershell
# After T008: build the leak-safe authored + corrected-synthetic geometry curriculum.
uv run python scripts/v50_build_reconstruction_curriculum.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr `
  --output ../output/datasets/v50/v50.1/reconstruction-direct-v1.zarr

# After T015: user-run baseline. Expected first calibration: roughly 1-3 hours, <=16 GB VRAM.
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/reconstruction-direct-v1.zarr `
  --architecture direct_cnn_v112 `
  --output ../output/v50/v50.1/direct_geometry/direct_cnn_v112 `
  --confirm-run

# After T015: user-run candidate on the identical split/target.
uv run python scripts/v50_train_direct_geometry.py `
  --store ../output/datasets/v50/v50.1/reconstruction-direct-v1.zarr `
  --architecture mit_b0_regression `
  --output ../output/v50/v50.1/direct_geometry/mit_b0_regression `
  --confirm-run
```

The runtime estimate is a planning bound for the local RTX 4070 Ti SUPER and must be recalibrated
from the trainer's non-training dry-run report before the user launches either run. Later object,
feature, texture, and alpha commands are added only when their preceding phase passes.

## Geometry promotion gate

The first Spec 114 checkpoint is promotable only when all of these hold:

- No WDL, ground-truth normal, height, alpha, material, or mask enters deployment inference.
- Authored and synthetic views share one group/split; leakage count is zero.
- Validation MAE beats flat/tile-mean and the strongest recorded Spec 112 result by at least 5%.
- Best epoch is later than epoch 1.
- Adjacent-border error passes SC-002 and the user accepts the held-out visual sheet.
- The run summary validates against `contracts/model-stage-and-curriculum.schema.json`.

Stop after this gate. Object-mask implementation is the next phase, not concurrent work.
