# V7.4 Canvas-Aware Reconstruction Plan

## Apr 12, 2026 - Direction Reset

- The next model direction should stop treating the world as a bag of independent tiles and start treating it as a hierarchical authored canvas.
- The current `V7.3` run family is useful proof that the geometry path can learn from the trusted corpus, but it does not yet prove end-to-end terrain restoration quality or produce a trustworthy terrain mesh output.
- The next real gain is not another round of generic hyperparameter churn. The next gain is to fix dataset truth ownership, duplicate-concept density, alpha-brush archaeology, and liquid/object supervision sanity before hardening the next model family.
- The immediate user correction that must be preserved here is that liquid masks are still not semantically trustworthy when liquids exist below terrain and should not be treated as clean positive surface supervision until that is fixed.

## Problem Statement

The current training path has four structural problems that can keep the model from converging on the right concepts even when loss appears to improve:

1. The active corpus still behaves too much like a tile-level sample pile rather than a world-scale authored canvas.
2. Near-duplicate concepts across builds, lighting variants, and repeated copied world motifs can overweight some ideas and cause the model to hedge or oscillate.
3. Alpha-mask layers are being treated too much like raw target images and not enough like reused authored brush strokes or grouped prefab motifs laid onto a chunked terrain canvas.
4. Some conditioning/supervision channels are still not semantically clean enough, especially liquids that may be present in data but not visible or walkable at the terrain surface.

## Core Observation

The world data is authored like a quilted infinite canvas:

- map-scale authored surface
- `ADT` tile partitions
- chunk-scale sub-canvases
- layer `0` as chunk base assignment
- layers `1..3` as local alpha-authored edits
- terrain deformation, minimap appearance, object placement, `WMO`, `PM4`, and later `PD4` all reinforcing one authored scene

That means the next model should be built around scene-structure recovery, not only per-pixel regression over disconnected tile crops.

## Recovered Historical Authoring Pipeline

The planning direction also needs to preserve the likely original production workflow, because that workflow explains why the exported data looks the way it does.

Working reconstruction of the upstream art process:

1. rough world or zone ideas began as notebook pencil drawings
2. those drawings were scanned and cleaned in Photoshop
3. bespoke internal tooling converted those drawings into terrain/height foundations
4. artists then worked over the resulting terrain using map-scale and zone-scale `blockout` surfaces
5. the terrain canvas was subdivided across `ADT` tiles, chunks, and texture layers
6. artists painted broad forms with layer `0`, then refined or repainted locally with higher alpha layers
7. chunks or regions could be erased and repainted iteratively on a per-layer basis

The practical implication is that some of the signals in the current corpus are traces of upstream authored media and tools, not only final runtime terrain state. For example, unusual paper-like or early-brush texture in specific minimap tiles should be treated as evidence about the authoring process rather than dismissed as random noise.

## Why This Historical Pipeline Matters

This history changes how the reverse-engineered tooling should be designed:

- the data is not only a runtime render target; it is the residue of an authoring system
- alpha masks are not generic segmentation labels; they are authored paint operations over a structured terrain canvas
- repeated patterns may represent reused brushes, reused Photoshop-era motif fragments, or later prefab-like authoring conventions
- terrain deformation, texture painting, object placement, and later `PM4`/`PD4` precision structure all belong to one scalable world-authoring system
- the clean-room implementation should aim to recover the semantics of that authoring system, not merely decode file bytes into isolated images

## Viewer-To-Editor Architectural Implication

The long-term product direction should be written down explicitly:

- today the active app is still primarily a viewer
- the long-term destination is an editor-capable world-authoring environment that can reason about terrain, layer painting, prefabs/brushes, object placement, and scene-scale reconstruction coherently
- that means every planning slice should prefer recovering authoring concepts that will still matter when the viewer becomes an editor

This is one more reason `V7.4` should not be framed as only a better predictor. It should be framed as one part of a larger clean-room recovery of the original scalable authoring model.

## Additional Non-Negotiable Rule

- When a recovered behavior likely reflects the original world-building pipeline, prefer preserving that behavior as an explicit authoring concept in the plan instead of flattening it into a generic ML label.

## Non-Negotiable Rules

- Do not describe a lower validation loss alone as terrain-restoration closure.
- Do not keep expanding the training corpus without dedupe/capping rules for same-concept variants.
- Do not add arbitrary rotation augmentation to the current terrain trainer first; orientation and lighting cues still matter.
- Do not trust liquid masks until below-terrain liquid cases are filtered or reclassified.
- Do not collapse alpha-mask layers into generic segmentation targets without preserving their authored-layer semantics.
- Do not split the long-term ML dataset contract across `WoWMapConverter`, ad hoc scripts, and viewer-local logic once `wow-viewer` ownership can absorb the seam.

## What V7.4 Should Be

`V7.4` should be one shared reconstruction system with staged responsibilities:

1. a canonical dataset contract with provenance, completeness, and curation metadata
2. a corpus-capture and clustering layer that controls duplicate concept density
3. a brush/prefab harvesting layer that models alpha-mask authoring as reusable motifs
4. one shared encoder backbone for terrain understanding
5. multiple task heads or follow-on consumers instead of one immediate giant monolithic output surface

This preserves the user's desired `train once, use many` direction without pretending one first-pass network should own every output family at full fidelity on day one.

## Model Tier Strategy

### Tier 1 - Base world-structure model

Own the big concepts first:

- local/global terrain shape
- terrain bounds/global context
- low-frequency scene interpretation from minimap + normal + WDL + known-loss masks
- concept embedding for later brush/texture retrieval

This is the model that should learn the large coherent structure of the world.

### Tier 2 - Detail/refiner model

Own the detail pass later:

- ridge sharpening
- cliff detail cleanup
- localized alpha/deformation refinement
- brush-aligned terrain or texture detail recovery

This should remain an explicit refinement stage rather than forcing the base model to learn every fine detail from the start.

### Tier 3 - Texture or alpha authoring model

Own authored surface decisions later:

- base-layer classification or palette planning
- upper alpha-layer brush selection
- transform/placement of reusable brush motifs
- residual local edits where no known brush fits well

This tier should consume the same shared latent scene understanding instead of relearning terrain semantics from scratch.

## Dataset Architecture Changes

### 1. Canonical provenance-first dataset contract

Extend the current dataset/manifest surface so that every tile or harvested sub-region carries:

- source build/profile
- map/tile/chunk/layer provenance
- source-presence flags for root ADT, `_tex0`, `_obj0`, WDL, liquid source, and object source
- observed vs stitched vs generated vs missing status per modality
- concept-cluster membership
- dedupe signatures
- brush-candidate membership where applicable

The dataset contract should continue moving toward canonical ownership under `wow-viewer`, not deepen legacy split ownership.

### 2. Concept clustering and duplicate-density control

Add a headless curation layer over the manifest that produces:

- exact duplicate groups
- near-duplicate groups within map/build families
- cross-build concept groups for the same terrain idea under different minimap styles or lighting
- retained representative set plus capped allowed variants

Selection rule:

- keep one canonical representative per concept cluster
- keep a small bounded set of materially useful variants across builds
- discard excess redundant repeats

This should happen before training, not only through in-trainer complexity trimming.

### 3. Liquid supervision sanity repair

Current concern to preserve:

- liquid masks can be positive even when the liquid is effectively below the terrain surface
- that makes the model learn false liquid occupancy or shape cues

Required dataset fix:

- classify liquids into at least:
  - visible surface liquid
  - submerged or below-terrain liquid
  - uncertain
- train the main terrain model only on the semantically valid liquid cues for the intended task
- keep the raw liquid source for archaeology/debugging, but stop treating all positive liquid-mask pixels as equally valid supervision

### 4. Alpha-mask brush archaeology

Alpha masks must be treated as authored reusable patterns.

The harvested unit should include both:

- 2D alpha-mask crop/pattern
- normalized 3D terrain deformation patch beneath it

Each harvested entry should also include:

- provenance
- layer index/indices
- transform metadata (rotation, mirror, offset, optional scale)
- dedupe signature
- acceptance state (candidate / approved brush / prefab member)

### 5. Chunk-border and tile-border-aware extraction

Brush candidates must not be naively clipped to one chunk or one tile.

The extraction path should:

- track connected authored regions across chunk boundaries
- preserve motifs that span multiple chunks or multiple tiles
- avoid turning one large authored pattern into many junk fragments

## Training-Contract Changes

### Keep

- minimap
- terrain normals
- WDL prior
- height bounds/global context
- trusted object/loss conditioning where semantically valid

### Change

- move from random or loosely curated tile emphasis to cluster-aware sample weighting
- introduce explicit same-concept cap rules
- add optional concept/brush embedding auxiliary supervision
- preserve the current fast AMP/TF32/Dataloader path unless resources force temporary fallback

### Do not add first

- arbitrary 90-degree rotations
- many brand-new fragile input channels
- one giant everything-output head

## V7.4 Model Shape Recommendation

Recommended architecture:

- one shared encoder backbone for scene understanding
- one geometry head for local/global height outputs
- one bounds/global-context head
- one concept/brush embedding head
- later optional texture/alpha planning head or a separate refiner consumer over the shared latent

This is a unified system, but not a single undifferentiated loss soup.

## Phased Plan

### Phase 1 - Dataset closure

Land first:

1. manifest/provenance extensions
2. concept clustering and duplicate report generation
3. liquid visibility/below-terrain classification rules

Exit condition:

- the corpus can say exactly what is present, trustworthy, duplicated, retained, or rejected

### Phase 2 - Brush-harvest foundation

Land next:

1. restore robust per-layer alpha inspection
2. add first candidate extraction over real alpha masks + terrain patches
3. add transform-aware dedupe and candidate review surface

Exit condition:

- real alpha motifs can be harvested, clustered, and inspected across repeated/rotated/mirrored usage

### Phase 3 - V7.4 training revision

Land after the data layer is stable:

1. cluster-aware train/val/test selection
2. shared-backbone plus auxiliary concept/brush head
3. dataset weighting by concept density and provenance confidence

Exit condition:

- V7.4 trains against curated concept-balanced data, not only raw tile count

### Phase 4 - Detail/refiner path

Land only after the base model is trustworthy:

1. explicit detail/refiner stage for high-frequency terrain cleanup
2. optional texture/alpha refinement over shared latent or retrieved brush priors

Exit condition:

- the system has a clean split between world-structure prediction and detail restoration

## Validation Standard

Every future slice should state proof at four levels:

1. build/syntax validation
2. real-data export/corpus validation
3. curation/manifest truth validation
4. model-quality validation against real reconstructed terrain outputs

Do not describe any future model as successful until it produces at least one terrain mesh or terrain output that is visibly and structurally credible under the current corrected heightmap pipeline.

## What This Plan Is Explicitly Not Claiming

- This is not claiming the current `V7.3` run has already solved terrain reconstruction.
- This is not claiming the liquid channel is already trustworthy.
- This is not claiming alpha-brush harvesting exists yet in active code.
- This is not claiming the first `V7.4` implementation should be the final all-in-one production model.

## Recommended Next Implementation Slice

Do this next in code mode:

1. extend the manifest/corpus-curation contract with dedupe/cluster metadata
2. add liquid visibility vs below-terrain classification to dataset audits
3. add a first alpha-brush candidate extraction service that emits paired 2D/3D candidates with provenance
4. only then revise the next training model design around those new dataset truths

## Continuity Files To Keep In Sync

- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/memory-bank/progress.md`
- `gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md`
- `.github/prompts/vlm-dataset-reconstruction-plan.prompt.md`

