# Pattern-Aware Corpus Curation Research

**Status**: planned Spec 103 Phase 3B | **Date**: 2026-07-15

## Decision

Use the existing Spec 076 full-map fractal brush library as the source of alpha/paste/fractal
evidence. Do not resurrect tile-local connected components as brush truth and do not make a second
alpha-mining pipeline. Spec 103 consumes Spec 076 evidence only to decide which otherwise-clean
terrain tiles are worth training on.

This is a curation signal, not a runtime feature: V8's eventual deployment interface remains one
minimap image. Alpha, MCLY, mesh, object, and liquid data may explain a training selection, but
cannot be appended to the deployed input or read at inference.

## Reconstruction purpose

This work is reverse engineering of an art pipeline, not generic image classification. The intended
workflow is: recover evidence-backed terrain-art prefabs from surviving game data and historical
image breadcrumbs; use models to propose compatible terrain/prefab structure; then let the operator
make deliberate hand edits and inspect the result in the viewer or export it through the existing
ADT/alphaWDT-compatible paths. A model output is therefore a reconstruction aid, never a claim of
historical truth and never a replacement for artist review.

The user's working historical model is that early game worlds were hand-authored with reusable
prefab-like terrain/texture/object pairings, while later editor tooling could re-decorate their
object relationships for further reuse. This is a useful hypothesis for evidence design: preserve
both the canonical terrain-art prefab and its placement/decorative context, rather than treating
the observed map as a flat collection of final pixels.

## What counts as a pattern

A pattern family is a stable upstream full-map region/family identifier plus the contextual state
that makes it meaningfully distinct. In the map-modding/exploration vocabulary, call this a
**terrain-art prefab**: a reusable pairing of terrain/texture, terrain/object, or mixed placement
data which may be copied, translated, mirrored, rotated, and retextured for a different zone.

The user's initial cross-era paste analysis found approximately **140 distinct prefab families** over
the 0.5.3–3.3.5 map corpus. That is the working evidence baseline—not a quota to force the next
run to reproduce. It is plausible precisely because a large zone can be assembled from transformed
and retextured placements of a small authored prefab vocabulary. In particular, a placement's
tileset must be modeled as a variant/provenance clue, not as proof that it belongs to a different
family.

The ledger must preserve, not collapse:

- atomic/repeated alpha/fractal motifs;
- `blocky_paste` and rectangle-page authored regions;
- composite canvases and one-off/non-brush patterned regions;
- terrain height/normal relief response;
- MCLY texture/layer context; and
- object/liquid placement relationships where present.

The selection question is therefore not “is this a brush?” but “does this tile contribute a
previously unrepresented authored pattern-and-context relationship?”

Canonical prefab identity is separate from placement identity:

- `prefab_family_id` groups transform-equivalent underlying terrain-art structure;
- `placement_id` records its exact build/map/canvas location, orientation/reflection, extent, and
  local neighbours; and
- `tileset_variant_id` records the active MCLY texture/path assignment without breaking the
  canonical family merely because a copied placement was retextured.

## Map-wide composition, not isolated regions

An ADT is a storage page, not an artist's unit of work. Analysis begins with the complete map
canvas for each alpha layer and examines it at several scales. A component, crop, or blocky-paste
region is only a coordinate anchor into that canvas; it is never the final observation by itself.

The ledger must therefore retain map-global composition features for every region/family:

- full-map and multi-tile spatial extent, including crossings of ADT and MCNK boundaries;
- recurrence and relative placement vectors between family members (grids, chains, rings, scars,
  branching, repeated neighbours, and other cellular/game-of-life-style local arrangements);
- multi-scale alpha occupancy/transition descriptors, so a large fractal, a repeated substructure,
  and a local stroke can be related without asserting they are identical;
- parent/child and neighbouring-region links, including gaps or alternate layer relationships; and
- the map-local tileset baseline plus retained, anomalous MCLY texture/layer choices.

“Game-of-life-style” here is an observation rule, not a claim about the authoring algorithm: record
the local-neighbour configuration and its repeated transformations across the map. We are looking
for an artist's reusable composition grammar, not merely exact bitmap duplicates.

## Tileset provenance as a paste signal

Tileset identity is a first-class discriminator. For each alpha-family placement, record active
MCLY texture IDs/paths by chunk/cell, their coverage, and a map-local frequency baseline. A texture
that is rare in its surrounding area but recurs with the same family or neighbour arrangement is a
`tileset_provenance_anomaly` candidate: evidence of a copied/pasted authored fragment, not an
automatic error. Preserve the original texture identity/path and era/build scope; never flatten it
to a generic texture class.

## Prefab-aware selection and validation

The reduced corpus selects placements that explain each canonical prefab under meaningful
transforms, terrain response, and tileset variants—not hundreds of near-identical copies. Conversely,
a transformed/retextured placement remains valuable when it adds a previously unrepresented
variant/context. For validation, `prefab_family_id` is the leakage group: transformed placements of
the same prefab cannot be split across train and validation, even when they occur on different maps.

The corpus should additionally distinguish what the data supports:

- **recovered evidence** — source map/canvas/ADT/MCLY/object provenance and directly observed
  placements;
- **model proposal** — a predicted terrain/prefab relationship with its generating model/checkpoint
  and confidence/diagnostic record; and
- **operator revision** — manual viewer edits or export preparation, retaining the immutable
  recovered evidence and the reason for each deliberate change.

This preserves the chain needed to recreate partially documented historical builds without
pretending an inferred placement is an original asset.

## Evidence chain

Each ledger membership carries source store identity and upstream library hash, then:

`build → map → tile_id/tile_x/tile_y → ADT chunk/cell coverage → alpha layer → full-map region → family → context → selection decision`.

An absent alpha layer, missing object mask, or unjoined upstream region is a recorded state. It
cannot be silently turned into a zero or inferred family.

## Selection and split policy

1. Apply the existing impossibility/quality filter first (object contamination, blank minimap,
   missing signal, height-normal mismatch).
2. Build ledger memberships for every remaining tile and all available alpha layers/maps.
3. Aggregate family/context coverage per tile, retaining per-membership rows for audit.
4. Derive map-global composition and tileset-anomaly features before selection. A tile's value may
   be its relationship to neighbouring placements or an out-of-place retained tileset, even if its
   local alpha crop is unremarkable.
5. Select deterministic prefab representatives across map/build, transform, relief, MCLY variant,
   alpha-family/state, composition grammar, and placement context. Every removal points to a
   representative or an explicit non-duplicate reason.
6. Group by canonical prefab family before assigning partitions. A prefab cannot cross train/validation; complete-map
   holdout still wins if it is stricter.

No global “top N most repeated masks” cutoff is valid: it would erase rare but meaningful terrain
contexts and bias the set toward empty/repeated canvas pages. The diversity budget must be explicit
and reported, never tuned by hidden ordering.

## Evidence required before use

The first proof is CPU-only and bounded: schema tests, deterministic rerun hashes, no split-family
leakage, and a map report showing retained/excluded representatives and their exact ADT locations.
Only after that proof may a reduced manifest be handed to the user for a training run.
