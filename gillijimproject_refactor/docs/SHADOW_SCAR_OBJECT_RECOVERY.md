# Shadow-Scar Object Recovery

This note defines the actual thing we are trying to learn from `MCSH`.

The goal is not "predict all objects from minimaps." The goal is narrower and
much more defensible:

- use `MCSH` as persistent evidence that an object or mixture of objects once
  affected the terrain shading
- compare that evidence against the current surviving MDDF/MODF placement set
- isolate the unexplained residual as a `shadow scar`
- use that residual to recover missing-object family, footprint, and placement
  hypotheses

This is the third model family alongside terrain-height recovery and
texture/alpha decomposition.

## Why This Matters

On many maps, especially `0.5.3`, we have most of the terrain shadow state even
when the corresponding placements are incomplete, moved, or deleted.

That happens because the world was heavily copy-pasted and iterated:

- object placements may have been moved or removed late
- shadow state may still preserve the older object footprint
- the same object or object mixture may still exist elsewhere on the same map,
  on another map, or in another build

So `MCSH` is not just a generic terrain-shadow channel. It is often historical
object evidence.

## What MCSH Actually Gives Us

`MCSH` is a chunk-local static shadow bitmap, not an object list.

What it provides directly:

- where the terrain is shadowed at chunk scope
- connected dark regions that look like object footprints or casts
- rough spatial extent and silhouette of those regions

What it does not provide directly:

- object ids
- asset paths
- rotations/scales
- whether a region comes from one object, several overlapping objects, terrain,
  or other baked/static causes

So the recovery problem must be posed as inference over residual evidence, not
as direct decoding.

## Current Foothold In The Exporter

The active exporter already carries more than raw shadow PNGs.

`terrain_data` currently exposes:

- `shadow_maps`: stitched tile-level shadow images
- `shadow_bits`: raw per-chunk `MCSH` bit payloads
- `shadow_analysis`: derived connected regions and nearby current-object
  candidate summaries
- `objects`: current MDDF/MODF placements with optional bounds

The current `shadow_analysis` foothold already computes:

- connected shadow regions per chunk
- region bounding boxes and centroids
- chunk/world coordinates for those regions
- nearby current object candidates projected into chunk-shadow space

That means the missing seam is not "start shadow analysis from nothing." The
missing seam is to decide which regions are already explained and which ones are
true shadow scars.

## Core Concept: Explained vs Unexplained Shadow

The right mental model is:

1. current placements explain some of the observed shadow
2. the remaining shadow residual may indicate missing objects

So every shadow region should eventually be classified into one of these
buckets:

- `explained-current`: adequately matched by one or more current placements
- `ambiguous-mixed`: partly explained, but likely a mixture or overlap
- `unexplained-scar`: no adequate current placement explanation
- `non-object-shadow`: likely terrain/self-shadow/noise/other static artifact

The useful training signal is mainly the `unexplained-scar` bucket.

## What We Are Actually Trying To Learn

The third model should learn this mapping:

`minimap appearance + MCSH evidence + surviving placement context -> missing object hypotheses`

That breaks down into three subproblems.

### 1. Scar Detection

Input:

- minimap tile
- chunk-local `MCSH` bits / stitched shadow map
- current object footprints and bounds

Output:

- mask or scored regions of unexplained shadow

This is the first necessary step. If we cannot distinguish explained shadow from
unexplained shadow, the rest of the pipeline will just rediscover already
present objects.

### 2. Scar Attribution

Input:

- unexplained shadow region
- minimap crop/context
- current nearby object context
- retrieval candidates from other maps/builds where similar shadow+object pairs
  still co-exist

Output:

- candidate object family or mixture of families
- confidence that the scar is one object vs multiple objects

This is where the copy-paste nature of the world helps us: if the missing object
is gone here but still present somewhere else, we can use those other places as
teacher examples.

### 3. Placement Reconstruction

Input:

- scar region geometry
- chosen object family candidates
- map/local context

Output:

- reconstructed placement hypotheses: position, approximate yaw, scale, and
  optional mixture weights if the scar is composite

This should be treated as candidate generation, not perfect one-shot truth.

## Recommended Data Regimes

Think about the data in three regimes.

### Regime A: Paired Object + MCSH

We have both current placements and shadow data.

Use this regime to learn:

- how objects of a given family project into `MCSH`
- how much shadow extent survives for different footprint sizes/scales
- how overlapping objects create mixed scars

This is the primary teacher regime.

### Regime B: MCSH With Weak/Partial Placement Coverage

We have shadow data and some objects, but the match is incomplete.

Use this regime to learn:

- residual detection
- partial explanation scoring
- cases where one current object explains only part of a larger scar

This is the bridge regime between paired truth and orphan recovery.

### Regime C: MCSH-Only / Orphan Scar

We have shadow evidence but no convincing current placement explanation.

Use this regime for:

- candidate mining
- retrieval from other tiles/maps/builds
- pseudo-label generation once strong cross-map matches are found

This is the actual recovery target regime.

## Best Working Hypothesis For 0.5.3

For `0.5.3`, the most plausible hypothesis is not that orphan `MCSH` regions are
random missing objects. The stronger hypothesis is:

- many orphan scars come from reused object kits or repeated map assembly
- a matching object or object mixture often still exists somewhere else
- therefore cross-map/cross-build retrieval is likely more useful than trying to
  infer object family from scratch using only one scar

That means we should treat the third model as retrieval-assisted, not purely
end-to-end generative.

## Practical Labeling Pipeline

The next extractor/label pass should do this explicitly.

### Stage 1. Rasterize Current Explanations

- rasterize current object footprints into chunk-shadow space
- for bounded objects, project approximate footprint extents
- produce an `explained_shadow_mask` from current placements

### Stage 2. Compute Residual Shadow

- compare `explained_shadow_mask` against real `MCSH`
- derive `residual_shadow_mask`
- split residual into connected regions

### Stage 3. Score Region Explainability

Per region, compute:

- overlap with projected current objects
- distance to nearest current object candidate
- region compactness / elongation
- whether region looks single-object vs multi-object

### Stage 4. Retrieval Candidate Search

For unexplained regions, search a library built from paired regions where object
families are known.

Candidate search keys should include:

- region shape/silhouette
- area / bbox / aspect ratio
- minimap crop context
- nearby roads/buildings/terrain texture cues
- local object neighborhood pattern

### Stage 5. Pseudo-Label Promotion

Only promote a scar to a pseudo-label when:

- it has strong retrieval agreement
- the retrieved candidate is stable across multiple examples
- the predicted placement produces a plausible re-explanation of the observed
  shadow

## What Should Be Added To The Dataset Next

The current exporter already has the right base ingredients. The next schema
extension should make the residual explicit instead of forcing downstream code to
recompute it every time.

Current active status:

- the exporter now emits first-pass residual fields inside `shadow_analysis`
  for each chunk/region:
  - chunk-level explained/residual shadow pixel counts and ratios
  - region-level `explained_by_current_objects`
  - region-level `explained_overlap_ratio`
  - region-level `scar_candidate_score`
  - region-level `scar_type`
- this is still heuristic pseudo-labeling based on projected current-object
  footprints, not proof that the dataset is correct or that the ML path works

Recommended additions inside `shadow_analysis` or an adjacent object-recovery
surface:

- `retrieval_candidate_object_ids` or family labels beyond the current nearby
  object ids
- richer `scar_type` resolution such as `single` vs `mixture` once retrieval is
  available
- `recovered_placement_hypotheses`

These fields should be treated as durable ML/export surfaces, not notebook-only
ephemera.

## What The Third Model Is Not

To avoid future drift, this model is not:

- a generic object detector over the minimap
- a replacement for current placement parsing
- a promise that every shadow region maps cleanly to one object
- a guarantee that `MCSH` always encodes missing objects rather than terrain or
  baked/static shadowing

It is a residual recovery model over historical shadow evidence.

## Short Memory Version

If this needs to be remembered quickly, the core idea is:

> `MCSH` is sometimes the last surviving evidence of deleted or moved objects.
> Learn the residual between current placements and observed shadows, then use
> repeated object patterns elsewhere in the copied/pasted world to recover what
> likely used to be there.