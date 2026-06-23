# Research: Alpha Brush Library Phase 0

**Date**: 2026-06-23

**Scope**: Validate connected-component extraction from V18 `alpha_256` and confirm DINOv2 embeddings can be generated for component patches.

## Command

Run from `wow-viewer/data-harvester/`:

```powershell
uv run python scripts/_research_alpha_components.py --map Azeroth --tile-limit 12 --max-components 96 --examples-per-layer 8 --batch-size 16 --device auto
```

Input dataset:

```text
wow-viewer/output/datasets/v18/0_5_3_3368.zarr
```

Output directory:

```text
wow-viewer/output/analysis/alpha-brush-library/research/
```

Generated artifacts:

```text
summary.json
components_sample.jsonl
patches_sample.npy
embeddings_sample.npz
projection.png
projection_cls.png
projection_mean.png
patches/layer*_example*.png
```

## Sample

The script selected the first 12 alpha-bearing `Azeroth` tiles from build `0_5_3_3368`:

```text
tile_ids = 0, 1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 26
```

Default component settings:

```text
min_area = 16
reject_edge = false
connectivity = 8-connected
```

## Threshold Sweep

| Alpha threshold | L0 | L1 | L2 | L3 | Total |
|-----------------|----|----|----|----|-------|
| 0.03 | 0 | 69 | 68 | 78 | 215 |
| 0.05 | 0 | 78 | 80 | 89 | 247 |
| 0.10 | 0 | 98 | 129 | 106 | 333 |

Decision: keep `0.05` as the default for Phase 1. It produces more separation than `0.03` without fragmenting as aggressively as `0.10`.

Layer note: L0 was empty in this sample, which matches the expectation that base fill often has no alpha mask. L1-L3 carried the brushwork.

## DINOv2 Embeddings

Model: `facebook/dinov2-small`

Sample size: 96 largest components at threshold `0.05`.

Both `[CLS]` token and mean-pooled patch-token embeddings were saved. The mean-pooled patch-token projection is also copied to `projection.png` because it grouped the sample more by visible component scale and shape than the `[CLS]` projection in this first pass.

Decision: use mean-pooled patch-token embeddings as the default token strategy for Phase 1, while keeping `[CLS]` available for comparison.

## Findings

- Connected components are numerous even in a 12-tile sample: 247 components at threshold `0.05`.
- The threshold sweep behaves as expected: raising the threshold splits blended strokes into more components.
- DINOv2 loads and runs through `transformers.Dinov2Model` in the data-harvester environment.
- The script produced reusable evidence artifacts for visual review: patch examples per layer, embeddings, and PCA projections colored by layer.

## Open Follow-Up For Phase 1

- Add the shared `alpha_brush.py` library around the proven extraction, patch rendering, DINOv2 embedding, and clustering seams.
- Decide whether edge-touching components should be rejected by default after reviewing larger map samples. Phase 0 kept them to avoid prematurely dropping large brush strokes.
