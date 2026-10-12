# Reconstruction Target Contract (v126.1)

**Feature**: 126-minimap-terrain-reconstruction | **Date**: 2026-08-02

Versioned so a checkpoint can be bound to the target definition it was trained against. Changing any
rule here requires a new version; silently changing one invalidates every prior comparison.

## Accuracy bands

A hobby restoration project, not metrology and not an exact-reproduction effort. Read as Pearson
correlation of recovered relief against real MCVT on held-out tiles unless stated otherwise.

| Band | Meaning |
|------|---------|
| **≥ 0.75** | **Project target met — ship it and move on** |
| 0.60–0.75 | Partial — useful as a prior, not a final answer |
| < 0.60 | Not working — diagnose before scaling |

**0.75 is the cutoff, not a floor to build on.** A run that reaches it is finished. Past that the
returns do not justify the training time, and this project explicitly declines to spend it.

**100% is not required by any gate and is not reachable.** Three effects bound it from above: the
DXT1 quantisation floor, Lambert saturation destroying back-facing slope information, and genuinely
flat terrain containing no relief evidence at all.

## Input normalization

Per-tile brightness/contrast normalization is applied to the input. Measured 2026-08-02 over 83
random Azeroth tiles: on land our synthetic renders sit roughly 8–11 lighter than authored across all
three channels with about 15% less contrast, of which DXT1 accounts for −0.56. The offset is small
and near-uniform, so normalizing the input removes it without a lighting recalibration or a re-render.

Water is excluded from terrain relief scoring. Authored 0.5.3 water is cyan-teal (≈52/133/147) and
our synthetic water is close to our synthetic land, but water carries no terrain relief so this does
not affect the height target.

## Height target

Per-tile min-max normalization with a denominator floor, matching the established relative-height
contract:

```text
normalized = clip((h - tile_min) / max(tile_max - tile_min, RANGE_FLOOR), 0, 1)
```

- **Altitude-offset invariance is structural**: adding a constant to a tile's heights leaves the
  target unchanged. Cross-tile altitude therefore cannot leak into supervision.
- The same property means **absolute elevation is unrecoverable from the target by construction**. It
  is composed from the WDL lattice prior at export, never predicted.
- Tiles without a WDL prior export relative-only and state that absolute elevation is absent. A
  fabricated datum is a contract violation.
- float64 intermediates for the encode; float32 storage.

## Input contract

- **Default input is the DXT1-degraded minimap.** Authored tiles are DXT1; training on the pristine
  render leaves a codec domain gap the loss never sees.
- The pristine variant stays in the store for ablation. Both remain queryable.
- Objects remain in the input at inference time. Masking applies to the **loss**, not the input.

## Loss masking

- Height loss is masked by `object_geometry_visible_mask_257` — terrain actually hidden in the
  rendered view.
- `object_precise_mask` is the full ground footprint and **must not be substituted**; it over-masks
  heavily.
- Excluded fraction is recorded per tile. Tiles above the occlusion threshold are bucketed, not
  silently trained on.
- Liquid regions are excluded from terrain relief scoring.

## Per-signal evidence (constitution v2.0.0, FR-023)

Every head reports its own metric against its own baseline:

| Head | Baseline it must beat |
|------|----------------------|
| Residual | Per-tile mean residual |
| Height | Tile-mean height |
| Albedo | Per-tile mean colour |
| Layer alpha | Dominant-layer-everywhere |
| Texture identity | Most-frequent-texture prior |

**A run is not successful while any head sits at its baseline**, regardless of the aggregate. An
aggregate win with a dead head is a partial failure and must be reported as one. This is the
obligation that replaced the retired single-signal prohibition, and it is a stop, not a note.

Every head must be independently ablatable — droppable or freezable without retraining the others.

## Evaluation

- Held-out tiles must be spatially disjoint from training tiles.
- **Kalimdor and Azeroth only.** PVPZone02 and Kalidar are never validation targets.
- Accuracy on authored input is reported separately from synthetic input, and the gap is stated.
- Every reported result names its evaluation set. Results from different held-out sets are never
  compared as equivalent.
- Best-epoch-1 is recorded as a structural failure, not a success.
- Per-tile results are reported alongside aggregates, with the failing fraction stated explicitly.

## Provenance

Every trained artifact records: dataset identity, evaluation set identity, release, target contract
version, per-head loss weights, and whether the input was codec-degraded.
