# Research: Terrain Paste and Fractal Motif Archaeology

## Evidence already in the repository

- Spec 132 already frames height, alpha, texture-layer identity, weak signals, and cross-map copied fragments as related evidence. Spec 140 should reuse that question while narrowing the first proof to recurrence and retrieval.
- Spec 133 established that the baked minimap is a mixture of albedo and lighting. The albedo-normalized observation is therefore an input preparation step, not a new ground-truth terrain signal.
- Spec 134 showed that a one-channel terrain-shadow model can improve during training while still losing to the tile-mean baseline. Cross-tile lightning and burn controls were the dominant failure families. This is evidence for adding context and structural guidance, not for adding more random architecture candidates.
- Spec 139 preserves the useful v7 idea of coarse relief plus detail residual and structural losses while removing WDL and target-derived inference inputs. Spec 140 must feed it optional, confidence-bearing evidence rather than replacing its deployment contract.

## Decisions

### Decision: promote paint order to an intermediate signal

The strongest current hypothesis is that terrain texturing is upstream of final relief: a base
tileset/brain texture is established, rocky paste families are painted into alpha layers, and the
terrain is sculpted around those painted regions. Therefore the first learned latent should be a
paint/sculpt-intent scaffold that includes ordered layer evidence and paste membership. It must be
evaluated as an intermediate, not confused with a height target.

The source contract must preserve `MCLY` identity and `MCAL` offsets. Layer 0 is the opaque base
tileset and is not treated as a paste signal. Layer 1 is the first alpha-bearing layer and the
first paste/paint candidate. The analyzer must never fabricate an `alpha_0` tensor.

### Decision: classify broken relationships explicitly

For each window, compare paint additions and relief features but emit one of `intact`,
`retextured`, `resculpted`, `unknown`, or `insufficient_data`. Re-textured zones are valuable
training/evaluation evidence precisely because they break the original relationship; they are not
valid reasons to force current alpha to explain current height.

### Decision: use a staged evidence pipeline

The initial topology is a workflow of small specialists:

1. observation normalizer and albedo-confidence estimator;
2. tileset/biome profile encoder;
3. ordered alpha and texture-layout descriptor;
4. multiscale motif detector and retrieval index;
5. paint/sculpt-intent scaffold estimator;
6. clean-signal geometry model;
7. optional object-footprint sieve;
8. renderer/referee loop for consistency and refinement.

Each stage emits a signal, confidence, provenance, and an explicit unavailable state. The stages can be ablated independently.

### Decision: retain complementary extraction scales

The April 12 Python brush-imprint path is valuable as an atomic extractor: it thresholds one
alpha layer, finds localized connected components, and preserves centered alpha/mask patches for
shape descriptors. The later C# full-map path is valuable as a parent-context extractor: it groups
nearby alpha evidence into macro paste/scar regions and middle-scale blocky children while retaining
map-wide and cross-tile relationships. The C# result is not a bug; it answers a different question.

Spec 140 therefore keeps `atomic_brush`, `paste_block`, and `macro_prefab_context` as distinct
ontologies. The join is through spatial overlap, layer/MCLY provenance, map coordinates, and
height/normal or texture response. An atomic component can be a child of a block or macro region,
but neither is automatically the historical identity of the component. Every scale remains
independently reviewable and ablatable.

### Decision: make alpha preservation the fan-out boundary

Alpha is the strongest shared substrate available across the competing interpretations. Preserve
all source layers and their provenance first, then derive multiple views: raw occupancy,
transition/stroke evidence, atomic components, paste blocks, macro context, ordered additions, and
cross-tile relationships. These views may disagree or be useful for different terrain families;
that disagreement is evidence to analyze, not a reason to discard one implementation. An absent or
opaque layer is an availability state, never a fabricated empty mask.

### Decision: use validation error only for curriculum difficulty

A frozen reference model trained on the synthetic control corpus may score candidate controls and
produce a reproducible difficulty guide. Per-signal error, seam/boundary error, confidence, and
coverage can place a sample in `easy`, `learnable_hard`, or `pathological` bands and adjust future
synthetic sampling weights. This score is not a staleness check, a pseudo-target, a replacement for
ground truth, or evidence that a corpus changed. The reference checkpoint, corpus hash, and scoring
configuration must travel with the guidance report.

### Decision: start with classical recurrence proof

The first implementation should compare descriptors and transformed synthetic controls before training a large motif model. If a nearest-neighbor or correlation baseline cannot retrieve known transformed motifs, a neural model will only make the failure less legible.

### Decision: treat fractals as a control family and a descriptor family

FBM, ridged, dendritic, lightning-like, and burn-like patterns are required synthetic regimes. They are not labeled as historical Blizzard brushes unless recurrence and cross-channel evidence support that claim.

### Decision: retain cross-tile context as a first-class boundary

Windows are sampled independently of chunk and tile boundaries. A motif family owns its split, so overlapping windows cannot leak the same pattern into validation.

### Decision: keep tileset evidence separate

Texture IDs, alpha masks, albedo, normals, specular-like channels, and height are stored as related but distinct signals. Auxiliary channels can guide a later stage only after their correlation with geometry is measured for the relevant build.

### Hypothesis: Warcraft III lineage and recursive pattern reuse

The early WoW mapping workflow may have inherited more than engine code from Warcraft III: base
tileset textures, Photoshop-like pattern overlays, alpha-layer brushes, and macro terrain motifs may
share a reusable visual vocabulary across scales. A green brain-like base texture is therefore a
valid observation of an early authoring state, not evidence of completed terrain relief. Similar
patterns appearing in tiled textures, alpha masks, and macro terrain should be tested through
transformed recurrence and spatial/height correlation, not assumed to be the same brush. Client
evidence, real media references, and synthetic controls must remain separate provenance classes.

### Decision: defer exact object identity

Object placements are represented first as normalized structural slots or footprints. Exact library identity remains an optional later head because the earlier dot-like mask result did not establish useful identity learning.

## Alternatives rejected for the first gate

- **One model that regenerates every signal**: cannot distinguish an unobservable target from a dead head and makes ablations opaque.
- **Training directly on the leaked map screenshot or assumed modern workflow**: it is not a 0.x/1.x client contract.
- **Using tileset alpha as a height target**: alpha may carry correlated brush evidence, but correlation must be measured and can be broken by repainting.
- **Using WDL or target-derived height hints at deployment**: violates the clean-signal input contract and prevents arbitrary minimap inference.
- **A large real-data harvest before controls**: would reproduce the previous debugging failure without proving which signal is responsible.
