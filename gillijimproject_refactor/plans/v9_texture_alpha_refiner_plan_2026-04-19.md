# V9 Texture And Alpha Refiner Plan

## Intent

- keep `v9.1` as the active base terrain-shape lane
- add a separate refiner lane for terrain texturing and alpha reconstruction instead of trying to force those responsibilities into the height model
- treat the refiner as an inverse-rendering problem over real exporter outputs, not as a free-form image hallucination task
- allow a bounded GAN term later, but only after the deterministic decomposition and forward-render loop is working

## Why This Exists

- the current `v9.1` lane is finally showing promising terrain-shape behavior on native supervision and higher curated coverage
- terrain shape and terrain texturing are related, but they are not the same target and they fail in different ways
- minimaps alone are ambiguous for exact texture reconstruction; a pure image-to-image GAN can produce plausible-looking but structurally wrong layer assignments
- the exporter already emits enough real supervision to attempt texture-layer and alpha reconstruction directly:
  - stitched alpha masks
  - alpha atlases
  - exported tileset textures
  - terrain-only and no-object and no-liquid and no-MCCV minimap variants
  - object/liquid/PM4/hole/semantic masks

## Core Recommendation

- do not make the first refiner a pure minimap-to-minimap GAN
- do not ask the refiner to directly hallucinate one final pretty output and then reverse-engineer metadata from it
- instead, build a chunk-first model that predicts texture-layer metadata and alpha masks, then renders a synthetic terrain-only minimap from those predictions and trains against the real terrain-only minimap

## Problem Framing

### What The Refiner Should Consume

- use inputs that exist at inference time:
  - terrain-only minimap when available, otherwise the best cleaned minimap variant available to the pipeline
  - `v9.1` terrain outputs or native-shape priors derived from them
  - liquid mask
  - object-visibility mask
  - PM4 mask when available
  - holes mask
  - optional area-id or chunk-flag or liquid-type semantic maps when they prove useful

- do not rely on ground-truth alpha masks as an inference-time input
- ground-truth alpha masks are training targets, not a required runtime dependency

### What The Refiner Should Predict

- chunk-local texture-slot metadata
- chunk-local alpha masks for overlay layers
- chunk or tile confidence/ambiguity scores
- optionally a synthetic terrain-only minimap reconstruction for debugging and validation
- final tile-level metadata bundle that can be stitched beside the terrain outputs

### What Success Looks Like

- the model can recover a chunk-local decomposition that re-renders the terrain-only minimap closely enough to be useful
- alpha masks are structurally plausible and spatially aligned with real exported alpha masks
- texture identity becomes at least a ranked or family-level prediction even when exact BLP-path recovery remains ambiguous
- inference outputs can be stitched into one tile package together with the `v9.1` terrain outputs

## Existing Anchors In The Repo

- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs`
  - already exports tileset textures
  - already exports stitched alpha masks and alpha atlas paths
  - already exports terrain-only and related minimap variants
  - already exports object/liquid/PM4/hole/semantic helper surfaces

- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`
  - already proves that a bounded PatchGAN discriminator path existed in the older image-heavy training lane
  - should be mined for scheduling and stabilization ideas only, not treated as the source-of-truth contract for this new lane

- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9.py`
  - remains the active terrain-shape lane and should stay separate from the first texturing refiner implementation

## Non-Negotiable Rules

- keep the first refiner chunk-first, not tile-first
- train against real exported alpha masks and terrain-only minimaps from real maps
- do not make a GAN the first or only teacher
- do not require unavailable runtime signals at inference time
- do not force exact texture-file classification as the first success criterion; family-level or ranked-candidate outputs are acceptable first milestones
- do not merge this lane into `v9.1` until the supervision contract and outputs are stable

## Proposed Data Contract

### Chunk-Level Inputs

- terrain-only minimap patch or fallback cleaned minimap patch
- coarse or mid or full terrain shape signals from `v9.1`
- liquid mask patch
- object visibility patch
- PM4 mask patch
- holes mask patch
- optional semantic maps:
  - area id
  - chunk flags
  - liquid type
  - dominant effect id

### Chunk-Level Targets

- base texture slot id
- overlay texture slot ids for active layers
- per-layer alpha masks
- optional alpha-atlas patch target
- synthetic terrain-only minimap rendered from ground-truth texture layers and alpha masks

### Tile-Level Outputs

- stitched texture-slot metadata for all chunks
- stitched alpha-mask outputs or atlas outputs
- debug reconstructed minimap
- provenance bundle linking each predicted slot to an exported tileset texture file path or ranked candidate list

## Representation Recommendation

### Texture Identity

- do not start with direct unrestricted BLP-path classification over the full texture universe
- first collapse textures into a smaller training representation:
  - tileset-family clusters
  - or top-k texture candidate shortlist per map/profile
  - or map/profile-scoped texture vocabulary

- later, once the decomposition works, add exact texture-path ranking inside the reduced candidate set

### Alpha Layout

- predict explicit overlay alpha masks per chunk layer
- keep the base layer implicit where possible and predict overlay masks for layers `1..n`
- align the first target contract with the real ADT chunk semantics rather than an arbitrary image segmentation contract

## Model Direction

### Phase 1 - Deterministic Decomposition Baseline

- build a chunk-first multi-head model with:
  - texture-slot classification head
  - alpha-mask regression head
  - optional confidence head

- no GAN in the first runnable slice
- primary losses:
  - texture classification loss
  - alpha-mask reconstruction loss
  - synthetic minimap reconstruction loss after forward rendering

### Phase 2 - Forward Terrain Renderer In The Loop

- add a renderer or approximate compositor that can synthesize a terrain-only minimap patch from:
  - predicted texture slots
  - predicted alpha masks
  - exported tileset textures

- use this as the central consistency bridge between hidden metadata and visible image supervision

### Phase 3 - Bounded PatchGAN Refiner

- once the deterministic decomposition is stable, add a lightweight PatchGAN discriminator over the synthetic terrain-only minimap patches
- adversarial pressure should act on rendered realism, not directly on latent alpha-mask tensors
- keep GAN scheduling delayed and bounded, similar in spirit to the older `v7` lane:
  - do not start adversarial training on epoch `1`
  - keep reconstruction and metadata losses dominant

## Implementation Slices

### Slice 1 - Dataset Audit And Contract Freeze

1. add a small audit script that measures real coverage for:
   - alpha masks per chunk/tile
   - tileset texture references per map/profile
   - terrain-only minimap availability
   - chunk-layer counts and active-layer distributions
2. define the first reduced texture vocabulary or clustering strategy
3. document the chunk-level training schema

### Slice 2 - Refiner Cache Builder

1. build a cache builder that emits chunk-first tensor shards for the refiner lane
2. include:
   - minimap patch
   - shape priors
   - semantic masks
   - texture-slot targets
   - alpha-mask targets
3. keep this separate from the active `v9` terrain cache unless the schemas later converge naturally

### Slice 3 - Non-GAN Baseline Trainer

1. add a first trainer script for chunk-first texture and alpha prediction
2. train without adversarial loss initially
3. prove that alpha masks and coarse texture families can be recovered from real exported data

### Slice 4 - Synthetic Terrain-Only Renderer

1. implement the forward compositor for predicted chunk outputs
2. add minimap reconstruction loss against real terrain-only minimap patches
3. expose debug outputs for:
   - predicted alpha masks
   - predicted texture slots
   - reconstructed minimap
   - target terrain-only minimap

### Slice 5 - PatchGAN Refinement

1. add a lightweight discriminator over rendered terrain-only minimap patches
2. delay GAN start until the baseline converges to a sane decomposition
3. keep adversarial loss low-weight and explicitly monitored so it cannot dominate metadata correctness

### Slice 6 - Tile Stitch And Inference Bundle

1. stitch chunk predictions back into tile-level outputs
2. emit a bundle that can live beside `v9.1` outputs:
   - terrain output
   - texture metadata output
   - alpha-mask output
   - reconstructed minimap preview
3. define the first consumer path for those outputs in downstream inference/viewer tooling

## Validation Gates

### Gate 1 - Data Reality

- confirm real exported datasets contain enough alpha and texture supervision to support the task at useful coverage

### Gate 2 - Deterministic Baseline

- prove a non-GAN model can recover chunk-local alpha masks and coarse texture families with measurable fidelity

### Gate 3 - Forward-Render Consistency

- prove that predicted metadata can synthesize a terrain-only minimap close enough to the real target to act as a strong supervision bridge

### Gate 4 - GAN Value Add

- add GAN only if it improves rendered realism without materially degrading metadata correctness or alpha alignment

### Gate 5 - Unified Inference Package

- prove the stitched outputs can be emitted together and consumed as one coherent terrain package

## Risks

- exact texture identity may be underdetermined from minimap evidence alone
- some tiles may only support ranked candidates or family-level labels, not exact file-path recovery
- chunk-local supervision may still contain exporter-side ambiguity in split-ADT texture ordering or alpha decode semantics
- if the forward renderer is weak, the model can learn the renderer’s errors instead of the terrain truth
- if the GAN is introduced too early, it can reward plausible-looking but wrong decompositions

## Recommended Immediate Next Steps

1. freeze `v9.1` as the active terrain-shape lane while the current higher-coverage run matures
2. build the refiner data-audit script first
3. define the first chunk-first tensor cache contract for texture slots plus alpha masks plus semantic context
4. implement a non-GAN baseline trainer before any adversarial work
5. add the forward minimap compositor before deciding whether the GAN term is actually necessary

## Recommendation

- yes, a GAN-based refiner is possible here
- no, the first implementation should not be a pure GAN
- the right first implementation is a chunk-first inverse-rendering lane that predicts texture metadata and alpha masks, re-renders a terrain-only minimap, and only later adds a bounded PatchGAN for local realism