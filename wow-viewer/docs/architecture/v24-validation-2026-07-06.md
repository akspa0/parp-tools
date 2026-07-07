# V24 Validation Report — 2026-07-06 (Spec 094)

Two bounded 50-tile Northrend (`3_3_5_12340`) validation runs. Both exercise
the full pipeline (C# WDL shim → merged prior → Stage A → Stage B →
`validate_v24.py`) on real data. `all_pass` is `false` on both, for two
different and instructive reasons documented below — this is the honest
result, not a forced green checkmark.

| Run | Selection | Real coverage | `output/v24_validation/<run_id>/` |
|---|---|---|---|
| `v24_northrend50_20260706` | first 50 Northrend tiles (unfiltered) | 100% real | ✅ |
| `v24_northrend_rough50_20260706` | first 50 Northrend tiles with `height_std ≥ 15` | 78% real / 22% synthetic | ✅ |

## SC-001 — merged coverage

Both runs: `real_plus_synthetic_ratio_of_non_empty = 1.0` (PASS — no
learned-fill cells in either 50-tile sample).

**Confidence bound is terrain-dependent.** SC-001 also requires
`wdl_prior_confidence ≥ 0.9` on `≥ 80%` of real-WDL-available cells:

- Flat run: **100%** of real cells hit confidence 1.0 (PASS). Flat terrain has
  no local height variation, so the client's int16 WDL and our synthetic
  point-sample always land within the 1.0 world-unit `disagree_threshold`.
- Rough run (`height_std ≥ 15`): **78.3%** (FAIL, just under 80%). On steep
  terrain, the exact sub-pixel location the client sampled at WDL-build time
  and the location our synthetic extractor samples at can disagree by more
  than 1.0 world unit even though both are "correct" — the disagreement is
  sampling-phase noise on a steep gradient, not a wrong algorithm. This is an
  honest empirical finding: the user's "match 99% of tiles" claim holds at the
  *tile* level (every tile gets real+synthetic coverage) but the *cell-level*
  confidence bound softens on rough terrain at `disagree_threshold=1.0`.
  Raising the threshold would trivially fix this at the cost of accepting
  larger disagreements as "confident" — not done here since the spec sets
  the threshold and this finding is more useful undisturbed.

## SC-002 — Stage A beats `block_reduce`

| Run | Stage A cheat L1 | `block_reduce` baseline L1 | real-cell L1 | synth-cell L1 |
|---|---|---|---|---|
| Flat | **0.087** | 0.906 | 0.087 | n/a (no synthetic cells) |
| Rough | **0.506** | 0.603 | 0.479 | 0.736 |

PASS on both runs. Real-cell L1 < synth-cell L1 on the rough run (0.479 <
0.736), confirming Stage A learned the real-WDL correlation rather than just
memorizing the synthetic pattern. Params: 337,485 (≤ 1M cap).

Minimap-only regime (no synthetic-WDL cheat channel; `--synth-dropout 0.5`
during training) is dramatically worse (172–601 world units) — expected,
since Stage A's residual architecture anchors on the synthetic quincunx and
the model was not trained long enough / with enough tiles to close that gap
from minimap alone. This regime exists for future work, not this bound.

## SC-003 — Stage B beats the upsampled prior and `block_reduce + bilinear`

| Run | Final L1 | Upsampled-prior L1 | `block_reduce+bilinear` L1 |
|---|---|---|---|
| Flat | 0.031 | 0.868 | **0.0000052** |
| Rough | **1.783** | 3.563 | 3.247 |

Rough run: PASS on both bounds — Stage B's final height beats both baselines.
Flat run: PASS vs upsampled-prior, **FAIL vs `block_reduce+bilinear`** — on
dead-flat terrain the trivial "sample height at the WDL lattice points and
bilinear-upsample" baseline is already a near-exact reconstruction of the
ground truth (5×10⁻⁶ world units of error), so no model can beat it. This is
exactly Risk 6 from the spec ("the baseline isn't a perfect WDL-grid-shaped
baseline... any model that beats it is doing real work") playing out in the
degenerate direction: on trivial terrain there is no work to do. The rough-50
selection (`--min-height-std`) exists specifically to produce a non-vacuous
SC-003 comparison, and it does. Params: 827,681 (≤ 2M cap; combined with
Stage A, 1,165,166 ≤ 3M total cap).

## SC-004 — determinism

Two `infer_v24_stage_b.py` runs with different seeds (11, 22), same
checkpoints, same input: bit-identical (`np.array_equal`). PASS on both runs.

## SC-005 — hardware envelope

Peak VRAM 0.187 GB (well under the 4 GB bound); mean wall-time 0.019–0.021 s
and max 0.034–0.049 s per tile (well under the 3 s bound) — both runs, on an
RTX 4070 Ti SUPER. PASS on both. The ≤ 3M-param design leaves enormous
headroom under the "6 GB consumer GPU" target hardware.

## What "V24 works" means given these results

- The full pipeline — C# WDL shim, merged-prior builder, Stage A, Stage B,
  determinism, hardware envelope — is proven end-to-end on real client data
  across two WoW client eras (shim validated on `0_5_3_3368` and
  `3_3_5_12340` in the Phase 0 audit; models trained on `3_3_5_12340`).
- On terrain with actual relief, both models measurably beat the trivial
  no-learning baselines (SC-002, SC-003 both pass on the rough-50 run).
- The one substantive open finding is the cell-level confidence bound on
  rough terrain (78.3% vs the 80% target) — a data-quality signal about
  sampling-phase disagreement between the client WDL and our synthetic
  extractor at the default 1.0-unit threshold, not a pipeline defect.

## Curated open-world run (2,011 tiles, 4 maps) — 2026-07-06/07

The 50-tile Northrend-only runs above are a *pipeline* proof. For a
*reliable, terrain-generalizable* model, V24 now consumes the existing V18
curation manifest (`kept_tiles.parquet` from `build_v18_curation_manifest`),
which already omits mismatched-signal tiles (blank minimap/normals,
normal/minimap edge mismatch, WMO loss wipeout, insufficient trainable
terrain) and buckets survivors by difficulty. `build_wdl_prior.py` gained
`--curation-manifest` and `--difficulty-bucket` (join on `(build, tile_id)`,
keep only `keep == True`).

This run trains on the curated open-world corpus for `3_3_5_12340`:
Azeroth (488), Kalimdor (741), Northrend (423), Expansion01 (359) = 2,011
kept tiles, all `hard`/`pathological` buckets, 76% real / 24% synthetic WDL
coverage. 1,609 train / 402 val, 30 epochs each stage.

| Check | Result |
|---|---|
| SC-001 coverage | PASS (100% real+synthetic of non-empty) |
| SC-001 confidence bound | FAIL (75.9% vs 80% — same documented rough-terrain sampling-phase disagreement; data-quality signal, not a pipeline defect) |
| SC-002 Stage A beats `block_reduce` | PASS — cheat L1 **1.652** < baseline 1.760; real-cell L1 **0.412** < synth-cell L1 6.540 |
| SC-003 Stage B beats prior + `block_reduce+bilinear` | PASS — final L1 **0.649** < upsampled-prior 4.307 < block_reduce+bilinear 4.199 (~6.5× better than both) |
| SC-004 determinism | PASS (bit-identical across seeds) |
| SC-005 hardware envelope | PASS — peak VRAM 0.187 GB, max wall 0.111 s/tile |

This is a substantially stronger and more generalizable result than the
50-tile bounded run: Stage B beats both no-learning baselines by a wide
margin across four terrain-distinct continents, and Stage A's real-cell L1
(0.41) is far below its synth-cell L1 (6.54), confirming it learned the
real-WDL correlation rather than memorizing the synthetic pattern. The one
remaining `all_pass=false` is the same SC-001 cell-level confidence bound
that is terrain-dependent by construction at the default 1.0-unit
`disagree_threshold`.

### Reproduce (curated open-world)

```bash
cd wow-viewer/data-harvester
uv run python scripts/build_wdl_prior.py build \
  --v18-store ../output/datasets/v18/3_3_5_12340.zarr \
  --staged-client ../../output/tmp/wowarchive-clients/3_3_5_12340 \
  --output ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet \
  --maps Azeroth Kalimdor Northrend Expansion01

uv run python scripts/train_v24_stage_a.py \
  --v24-store ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
  --output ../output/v24_validation/v24_openworld_curated_20260706 --epochs 30

uv run python scripts/train_v24_stage_b.py \
  --v24-store ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
  --stage-a-checkpoint ../output/v24_validation/v24_openworld_curated_20260706/stage_a.pt \
  --output ../output/v24_validation/v24_openworld_curated_20260706 --epochs 30

uv run python scripts/validate_v24.py \
  --v24-store ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
  --stage-a-checkpoint ../output/v24_validation/v24_openworld_curated_20260706/stage_a.pt \
  --stage-b-checkpoint ../output/v24_validation/v24_openworld_curated_20260706/stage_b.pt \
  --run-id v24_openworld_curated_20260706
```

## Reproduce (bounded 50-tile Northrend rough-50)

```bash
cd wow-viewer/data-harvester
uv run python scripts/build_wdl_prior.py build \
  --v18-store ../output/datasets/v18/3_3_5_12340.zarr \
  --staged-client ../../output/tmp/wowarchive-clients/3_3_5_12340 \
  --output ../output/datasets/v24/3_3_5_12340_northrend_rough50.zarr \
  --maps Northrend --min-height-std 15 --limit 50

uv run python scripts/train_v24_stage_a.py --v24-store ../output/datasets/v24/3_3_5_12340_northrend_rough50.zarr \
  --output ../output/v24_validation/v24_northrend_rough50_20260706 --epochs 50

uv run python scripts/train_v24_stage_b.py --v24-store ../output/datasets/v24/3_3_5_12340_northrend_rough50.zarr \
  --stage-a-checkpoint ../output/v24_validation/v24_northrend_rough50_20260706/stage_a.pt \
  --output ../output/v24_validation/v24_northrend_rough50_20260706 --epochs 50

uv run python scripts/validate_v24.py \
  --v24-store ../output/datasets/v24/3_3_5_12340_northrend_rough50.zarr \
  --stage-a-checkpoint ../output/v24_validation/v24_northrend_rough50_20260706/stage_a.pt \
  --stage-b-checkpoint ../output/v24_validation/v24_northrend_rough50_20260706/stage_b.pt \
  --run-id v24_northrend_rough50_20260706
```

## Related documents

- [`wdl-reader-shape-audit-2026-07-06.md`](wdl-reader-shape-audit-2026-07-06.md) — Phase 0 C# shape audit + synth-vs-real convergence check.
- [`v22-dataset-audit-2026-07-06.md`](v22-dataset-audit-2026-07-06.md) — C#-grounded V22/V18 signal audit (amendment A8), including the `holes_16` polarity defect this spec worked around.
- [`../../specs/094-wdl-prior-v24/spec.md`](../../specs/094-wdl-prior-v24/spec.md) — spec + Implementation Amendments (A1–A8).
