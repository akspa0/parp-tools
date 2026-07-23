# Feature Specification: Per-Object Occlusion-Aware Masks for Object-Deconfounded Terrain Height

**Feature Branch**: `118-object-occlusion-masks`

**Created**: 2026-07-22

**Status**: Draft

**Input**: User description: "We need to build a proper image segmentation/classifier that understands the object data well enough to identify the class and potentially the entire object, just from the minimap input, and use that data to drive loss for the objects placed on the maps, so they don't contribute to land knowledge. Precise object masks were to be used for LOSS signals — per-object, each object its own mask, class identified via the roof/object data we can harvest from the viewer. The old precise masks over-masked because WMOs and M2/MDX can be placed underground and only a small portion pokes through the terrain; using the full footprint lost 80–90% of the tile. We need a fragmented, visible-portion-only mask so underground/occluded objects don't contribute too much loss."

## Overview *(context, not a template section — kept brief)*

Authored WoW minimaps have world objects (WMO buildings, M2/MDX doodads) painted onto them. Those objects occlude the ground and carry no terrain-height information, so a terrain-height model regressing height from minimap RGB is fed a confounded input on every object-touched tile (~52% of Azeroth, ~54% of Kalimdor). The Spec 117 RGB→WDL-lattice result confirmed the height model does not beat the tile-mean baseline on relief tiles; the unhandled object confound is a leading suspect. This feature reintroduces an object signal — dropped from the v50 clean-room dataset — **correctly**: a per-object, occlusion-aware (visible-portion-only) mask with an object class, used to keep object pixels from corrupting land-height learning. It follows the proven Spec 115 deconfounding pattern (a generated terrain-feature map cut road-region height MAE 21%), applied to the object confound.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Occlusion-aware per-object mask + class signal in the dataset (Priority: P1)

Harvest, into the v50 dataset, a signal that marks — for every terrain tile — which minimap pixels are covered by which *visible* object, plus each object's class. A pixel is marked only where an object actually pokes through the terrain and appears in the minimap; an object buried underground (or fully occluded by terrain) contributes little or no marked area. Each placed object is distinguishable (its own identity), and each carries a class label.

**Why this priority**: This is the missing foundation the user asked for repeatedly and that was dropped from v50. Every downstream use (loss deconfounding, the segmentation model) depends on a *correct* mask. It is the one thing whose absence — and whose previous broken form (full-footprint over-masking) — has blocked all object-aware work. It delivers standalone value: with a correct signal on disk, object pixels can immediately be excluded from any existing loss using ground truth.

**Independent Test**: Harvest a known city tile (dense WMO/M2 placement) and a known underground-heavy tile. Verify: (a) marked pixels coincide with objects visible in the authored minimap, not their full ground footprint; (b) an object known to be underground contributes ≈0 marked pixels; (c) each object has a distinct identity and a class label; (d) the fraction of the tile marked as object is consistent with what is visibly object-covered (single-digit-to-tens of percent on typical tiles), NOT the 80–90% the full-footprint mask produced.

**Acceptance Scenarios**:

1. **Given** a tile where a large WMO is placed mostly below the terrain surface with only its roof exposed, **When** the object mask is harvested, **Then** only the exposed-roof pixels are marked for that object and the buried remainder is not.
2. **Given** a tile with several distinct doodads, **When** the per-object signal is harvested, **Then** each doodad occupies a separately identifiable region and carries a class label, rather than being merged into one undifferentiated object blob.
3. **Given** a tile with no objects, **When** harvested, **Then** the object mask is empty and the tile is unaffected (no false marking).
4. **Given** the same tile harvested twice, **When** compared, **Then** the object mask is byte-identical (deterministic), and provenance records the render/visibility method used.

---

### User Story 2 - Object-masked terrain-height loss that removes the confound without over-masking (Priority: P2)

Use the visible-object mask to keep object pixels from contributing to the terrain-height training loss, weighted by visibility so a barely-visible or invisible object adds little or no loss exclusion. Re-train (or re-score) a terrain-height model with this object-masked loss and show it improves height accuracy on object-touched tiles versus the identical model trained without it, measured on the honest spatially-isolated, relief-stratified held-out split.

**Why this priority**: This is the payoff the user is after — objects no longer corrupt "land knowledge." It can be proven with **ground-truth** masks alone (loss-side use is admissible), so it needs only US1, not the trained model. It is the cheap proof that the whole direction is worth pursuing: if excluding perfectly-known object pixels does not help, the model in US3 is not worth building.

**Independent Test**: Train two otherwise-identical terrain-height runs on the same store/split — one with the object-masked loss, one without — and compare relief-stratified MAE on the subset of held-out tiles that contain visible objects. Verify the object-masked run is at least as good overall and better on object-touched relief tiles, and that the mask excludes only a small, visible fraction of each such tile (no 80–90% wipeout).

**Acceptance Scenarios**:

1. **Given** two identical height runs differing only by the object-masked loss, **When** scored on object-touched held-out relief tiles, **Then** the object-masked run's MAE is lower (or a null result is reported honestly if it is not).
2. **Given** a tile with an underground object, **When** the object-masked loss is applied, **Then** the effective loss coverage of that tile is nearly unchanged (the invisible object barely reduces the trainable land area), unlike the full-footprint mask which removed most of the tile.
3. **Given** the object-masked loss is enabled, **When** aggregate and relief-stratified metrics are recorded, **Then** both are reported so a flat-tile-dominated aggregate cannot mask a relief-region change.

---

### User Story 3 - From-scratch object segmentation + classifier from any minimap input (Priority: P3)

Train a small, from-scratch model that takes a single minimap tile — including out-of-distribution or hand-painted tiles with no ground truth — and predicts, per pixel, whether it is a visible object and of which class (with per-object separation as a stretch goal). Its generated output can then serve object-aware terrain reconstruction at inference time (as an extra input channel, the Spec 115 pattern) and can be validated on minimaps that have no harvested mask.

**Why this priority**: This is the eventual deployable capability the user described ("identify what an object is, in contrast to everything else on any random minimap tile used as input or validation"). It depends on the US1 signal for supervision and is validated by the US2 proof that object-awareness helps. It is prioritized after the signal and the value proof because a model is only worth training once the signal is correct and the lever is shown to work.

**Independent Test**: Train on the US1 masks; evaluate per-class segmentation quality (e.g., IoU/recall at the authoring unit) on the spatially-isolated held-out split and on at least one hand-painted OOD minimap with a human-verified object region; confirm the model runs on a loose minimap image with no store or ground truth present.

**Acceptance Scenarios**:

1. **Given** a held-out minimap tile with visible objects, **When** the model predicts, **Then** predicted visible-object regions overlap the ground-truth visible mask above a defined threshold and object classes are correct above a defined rate.
2. **Given** a hand-painted minimap tile with an obvious building, **When** the model predicts, **Then** it marks the building region as a visible object of the appropriate class without any ground-truth input.
3. **Given** the predicted object map is fed to a terrain-height model as an extra input channel, **When** the height model is trained/scored, **Then** it is accepted through the existing generated-feature contract with no bespoke wiring.

---

### Edge Cases

- **Object exactly at terrain grade / flush with ground**: define a small visibility tolerance so a flush object is treated consistently (marked or unmarked by a documented threshold), not flickering per-pixel.
- **Object visible but its color matches terrain** (e.g., a dirt-textured hut): the harvested mask is renderer-truth (not color-based), so it is still marked; the *model* may struggle and that is a measured limitation, not a mask error.
- **Tile that is almost entirely a giant WMO** (a city building footprint): the mask may legitimately be a large fraction of the tile; US2's loss weighting must down-weight, not hard-drop, so such a tile still contributes some land signal where terrain is visible.
- **Roof/model class unavailable for an object** (missing asset): the object is still marked as a generic visible object with an "unknown" class rather than dropped.
- **Object partially occluded by another object** (WMO behind WMO): the visible mask reflects what the renderer shows; per-object separation may be ambiguous where two objects overlap in screen space — resolve to the front-most object's identity.
- **Synthetic (terrain-only) minimap rows**: they contain no objects by construction; the object mask is empty and they are valid negatives.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The dataset MUST carry a per-tile object mask that marks a pixel only where an object is **actually visible in the minimap** (pokes through / is not occluded by terrain), never the object's full ground-projected geometry footprint.
- **FR-002**: The object signal MUST be **per-object**: distinct placed objects are separately identifiable within a tile, not merged into a single object region.
- **FR-003**: Each identified object MUST carry a **class label** derived from the object/roof/model identity (at minimum distinguishing building-type from doodad-type objects; finer classes where readily available), with an explicit "unknown" class when identity is unavailable.
- **FR-004**: Objects that are underground or fully terrain-occluded MUST contribute approximately **zero** marked area, such that the total object-marked fraction of a typical tile reflects visible coverage (single-digit-to-tens of percent) rather than the ~80–90% the full-footprint mask produced.
- **FR-005**: The object signal MUST be **deterministic and provenance-tracked**: identical inputs yield an identical mask, and the record states how visibility was determined and against which build/fingerprint.
- **FR-006**: The terrain-height training loss MUST be able to **exclude or down-weight** object-marked pixels so those pixels do not contribute to land-height learning, **weighted by visibility** so a barely-visible object reduces trainable land area only slightly.
- **FR-007**: The object-masked loss MUST NOT remove more than the visibly-object-covered portion of a tile; a tile with a mostly-underground object MUST retain nearly all of its trainable land area.
- **FR-008**: Height-model results using the object-masked loss MUST be reported **both aggregate and relief-stratified** on the spatially-isolated held-out split, and compared against the identical model without the object-masked loss; a null result is a valid, reportable outcome.
- **FR-009**: The segmentation/classification model MUST accept a **single minimap tile with no accompanying store or ground truth** (including hand-painted OOD tiles) and produce a per-pixel visible-object + class prediction.
- **FR-010**: The segmentation model MUST be **small and trained from scratch** (no DepthAnything-family or mandatory pretrained-backbone dependency), and be an independently checkpointed specialist whose output feeds downstream stages by output, not shared weights.
- **FR-011**: The model's generated object map MUST be consumable by the existing terrain-height trainers through the **already-validated generated-feature contract**, requiring no bespoke per-feature trainer wiring.
- **FR-012**: All harvest, dataset rebuild, and training steps MUST be **user-run**: the tooling prepares and prints exact commands and never launches heavy/billed work itself.
- **FR-013**: The feature MUST NOT mutate existing v50 source stores in place; the object signal is added via the dataset's normal signal-catalog path, and any derived stores are separate and provenance-bound.
- **FR-014**: Ground-truth object masks are admissible **loss-side only**; the deployed/inference input to the terrain-height model remains the minimap plus the **model-predicted** object map, never a ground-truth mask.

### Key Entities *(include if feature involves data)*

- **Visible object mask (per tile)**: which minimap pixels show an object, restricted to terrain-visible portions. The core corrected signal.
- **Object instance identity (per tile)**: an assignment of visible object pixels to distinct placed objects, so per-object regions can be counted, separated, and analyzed.
- **Object class label (per object)**: a category for each object (building/doodad, finer where available, or "unknown"), sourced from object/roof/model identity — not from minimap color.
- **Object-masked land-loss weight (per tile)**: a per-pixel weight the terrain-height loss uses to discount object-covered land, derived from the visible mask and its visibility strength.
- **Object segmentation model**: a small from-scratch predictor mapping a minimap tile to a visible-object + class map; an independent checkpoint in the terrain reconstruction chain.
- **Held-out evaluation split**: the existing spatially-isolated, relief-stratified split reused to judge both the height improvement (US2) and the segmentation quality (US3) honestly.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a representative object-heavy tile, the harvested object mask covers only the visibly object-covered area; the object-marked fraction is within a small tolerance of the human-verified visible coverage and is dramatically smaller than the full-footprint mask (target: at least a 3× reduction in over-marked area on tiles with underground objects).
- **SC-002**: On tiles containing a mostly-underground object, enabling the object-masked loss changes the tile's trainable land area by no more than a small margin (target: ≤10% reduction), versus the full-footprint approach that removed 80–90%.
- **SC-003**: A terrain-height model trained with the object-masked loss achieves lower relief-stratified height error on object-touched held-out tiles than the identical model without it (target: a measurable reduction on the object-touched relief subset), with the result reported honestly whether positive or null.
- **SC-004**: The segmentation model, given held-out minimap tiles, identifies visible-object regions and their class above defined thresholds (targets to be fixed in planning, e.g., visible-object region overlap and per-class accuracy at the authoring unit), and runs successfully on at least one hand-painted OOD minimap with a human-verified object.
- **SC-005**: The segmentation model is small and from-scratch (target: single-digit-millions of parameters, no pretrained-backbone requirement), consistent with the project's tiny-modular-specialist constraint.
- **SC-006**: The end-to-end path (harvest object signal → object-masked height training → object segmentation model → predicted map into the height chain) runs entirely from user-issued commands with no assistant-launched heavy jobs, and no existing source store is mutated.

## Assumptions

- **Visibility source**: the occlusion-correct visible mask is obtained from renderer truth (a with-objects vs without-objects render difference) rather than any color heuristic, reusing the viewer's existing object-visibility capability; the full-geometry footprint projection is explicitly NOT the mask.
- **Per-object identity source**: per-object separation reuses the existing per-object instance identity, intersected with the visible mask, so "per-object" is achieved without new instance-segmentation infrastructure at harvest time.
- **Class taxonomy**: the initial class set is coarse (building/WMO vs doodad/M2, plus "unknown"), extensible to finer model-family classes where object identity readily provides them; a fully fine-grained taxonomy is not required for the first cut.
- **Model output granularity**: the segmentation model's first target is per-pixel visible-object **class** segmentation (semantic), which directly serves the loss and OOD identification; predicting fully separated per-object **instances** from the minimap is a stretch goal, not required for US2 or the core of US3.
- **Loss vs input use**: the primary use is loss-side exclusion during height training (ground-truth masks admissible); the input-channel deconfounding use (predicted map as extra channel) reuses the Spec 115 generated-feature contract.
- **Evaluation**: judgments use the existing spatially-isolated, relief-stratified held-out split; aggregate-only metrics are treated as untrustworthy on this corpus.
- **Execution ownership**: the user runs all harvest, dataset rebuild, and training; the assistant prepares exact commands and never launches heavy work.
- **Scope boundary**: this feature delivers the object signal, the object-masked loss proof, and the from-scratch object segmenter. It does NOT attempt to reconstruct true ground height *under* objects (an occluded, separate problem), nor does it re-open the dropped inpainted ground-intent-height signal.
- **Prior art reused (not reinvented)**: the viewer's object-visibility rendering, the existing per-object instance identity, and the roof-capture object renders for class; the v50 signal-catalog path for adding the new signal; the existing generated-feature trainer contract for the input-channel use.
