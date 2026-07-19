# Feature Specification: Minimap Super-Resolution (Real-ESRGAN) from Authored LR to Detail-Rendered HR

**Feature Branch**: `113-minimap-superres`

**Created**: 2026-07-18

**Status**: Draft

**Input**: User description: "train a Real-ESRGAN model off our synthesized HR's and the real LR's.
We probably have a unique case where we can generate giant versions of every tile, preserving
extreme details that are otherwise lost."

## Context and prior findings this spec encodes

- **The "unique case" is real, but our current HR is not detail-rich.** A model whose source assets
  we hold can render minimaps at giant resolution with *genuine* texture detail — a thing no
  ordinary super-resolution pipeline has, because it normally must hallucinate HR detail from a
  learned prior. But the existing `minimap_rgb_1024` does NOT contain that detail: the compositor
  deliberately uses each terrain texture's flat *average color*
  (`TerrainMinimapCompositor.cs` `CalculateAverageColor`, chosen to kill moire at 256px minimap
  scale), so the 1024 render is ~256px of chunk/alpha structure upsampled — one flat color per
  texture, no real texture pixels. Delivering "extreme details" therefore requires a NEW
  detail-preserving render path.
- **Why detail rendering is viable now (the giant insight).** The material-average hack existed
  because sampling real diffuse texels while downsampling hard to 256px produced moire. At giant
  resolution (1024+/tile) the downsample is gentle, so real texel sampling becomes viable again —
  the very reason a "giant version" can preserve detail a 256px render cannot.
- **The training pair crosses render styles, and alignment is unverified.** The LR input is the
  *real authored client minimap* (the actual deployment input); the HR target is *our* detail
  render. These are two different renders of the same tile, so the model learns a combined
  style+detail transfer, not pure denoise-upscale. For the pair to be valid at all, the authored
  image and our render must be spatially registered (same tile bounds, same orientation). This
  codebase has a documented history of orientation subtleties (the north/south solar-direction
  saga, the open GLB Y-axis texture-mirror bug), so authored↔render alignment is a hard,
  must-prove prerequisite, not an assumption.
- **Depends on Spec 112's store.** Post-Spec-112, each per-build store already carries
  `minimap_rgb_authored` (real client minimap, 256, honest partial coverage) and `minimap_rgb_1024`
  (the 1024 render). This spec turns the latter into a detail-preserving render and pairs it with
  the former. Corpus scope inherits Spec 112: Kalimdor and Azeroth only; PVPZone02/Kalidar excluded
  (too small to gauge anything).
- **User scope decisions (2026-07-18)**: HR detail from a NEW detail-preserving render; LR = real
  authored minimap; first target scale ×4 (256→1024), pipeline built to go bigger later.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Detail-Rendered HR, Spatially Aligned to the Authored LR (Priority: P1) — MVP

The dataset operator produces a detail-preserving 1024 render for each tile that samples real
terrain texture pixels, and proves it is spatially registered to the authored client minimap for
the same tile — so an (authored 256, detail 1024) pair is a valid super-resolution pair, not two
misaligned pictures.

**Why this priority**: Every downstream step (pair assembly, training, evaluation) is worthless if
the HR is either detail-free or misaligned with the LR. This story is the make-or-break foundation:
it establishes that the "unique case" actually exists in our data.

**Independent Test**: For a sample of tiles that have both an authored minimap and a detail render,
(1) the detail render measurably contains real high-frequency texture detail absent from a bicubic
upscale of the same tile's 256 material-average render, and (2) the authored image and the detail
render register within a small pixel tolerance under the identity transform (no flip, rotation, or
transpose needed) — or the exact misregistration is characterized so it can be corrected.

**Acceptance Scenarios**:

1. **Given** a tile with decodable terrain textures, **When** the detail-preserving render runs,
   **Then** its output contains real texture-pixel detail (high-frequency content materially above
   a bicubic upscale of the material-average render), not merely upsampled flat colors.
2. **Given** a tile with both an authored client minimap and a detail render, **When** the two are
   registered, **Then** their alignment error under the identity transform is below the stated
   tolerance, OR a fixed corrective transform (flip/rotate/transpose/offset) is identified that
   brings them into alignment for all sampled tiles consistently.
3. **Given** a tile whose textures cannot be decoded, **When** the detail render is attempted,
   **Then** it is honestly skipped and recorded — never emitted as a flat or fabricated HR.
4. **Given** the detail render at 1024, **When** compared to the moire failure that motivated the
   material-average hack, **Then** it shows no minimap-scale moire (the gentle-downsample premise
   holds), validated visually on a sample.

---

### User Story 2 - Aligned Super-Resolution Pair Dataset (Priority: P2)

The dataset operator assembles a training set of aligned (authored LR, detail HR) pairs from
Kalimdor and Azeroth, including only tiles that have BOTH a real authored minimap and a successful
detail render, with an honest held-out evaluation split and no tile crossing it.

**Why this priority**: Real-ESRGAN training consumes paired LR/HR data; the pairing, coverage
honesty, and leak-safe split are what make the trained model's evaluation trustworthy.

**Independent Test**: The assembled pair set contains only tiles with both sources present; each
pair is the registered (authored, detail) pair for one tile; the eval split is disjoint by tile
from the train split; and the pair count and per-map coverage are reported against the store's
actual `minimap_rgb_authored`/detail coverage.

**Acceptance Scenarios**:

1. **Given** the rebuilt Kalimdor/Azeroth stores, **When** the pair set is assembled, **Then** it
   includes exactly the tiles with both a populated authored minimap and a successful detail render
   — tiles missing either are excluded and counted, never zero-filled into a pair.
2. **Given** the pair set, **When** the train/eval split is assigned, **Then** no tile's pair
   appears in both splits, and the split is deterministic across rebuilds.
3. **Given** any request to include PVPZone02 or Kalidar tiles, **When** the pair set is built,
   **Then** those maps are excluded from this lane.

---

### User Story 3 - Trained Super-Resolution Model and Evaluation (Priority: P3)

The model operator trains a Real-ESRGAN super-resolution model on the aligned pairs and evaluates
it: given a real authored low-res client minimap, the model produces a detailed high-resolution
minimap that is measurably and visibly superior to a naive upscale, on held-out tiles.

**Why this priority**: This is the deliverable, but it is only trustworthy after US1/US2 establish
that the HR is real detail and the pairs are valid.

**Independent Test**: On held-out tiles, the model's HR output beats a bicubic (and the
material-average 1024) baseline on the chosen SR metrics, and a user side-by-side judges the outputs
to preserve genuine detail rather than plausible-but-invented texture.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint and a held-out authored LR minimap, **When** the model upscales
   it, **Then** the result scores better than a bicubic upscale of the same LR on the chosen
   full/no-reference SR metrics, recorded in the run summary.
2. **Given** the trained model, **When** evaluation is requested for a map outside Kalimdor and
   Azeroth, **Then** the tooling refuses or clearly labels the result out-of-scope.
3. **Given** a training run, **When** its summary is reviewed, **Then** it records the pair-set
   identity, split, model/loss configuration, per-metric baseline comparison, and enough provenance
   to reproduce or reject the run.
4. **Given** the model's HR output, **When** a user reviews held-out tiles side by side against the
   authored LR and the baseline, **Then** the output is judged to add real, plausible terrain
   detail without introducing obviously fabricated or hallucinated structure.

---

### Edge Cases

- **Misalignment beyond correction**: if no single fixed transform registers authored↔detail across
  sampled tiles (e.g. the authored minimap uses a different projection/crop than our render), the
  real-LR→synthetic-HR pairing is invalid as-is. The spec must stop at the US1 gate and surface the
  finding rather than train on misaligned pairs; a documented fallback is pure-synthetic pairs
  (detail HR degraded down to LR), which changes the deployment story and requires a separate
  decision.
- **Detail render reintroduces moire** at 1024 on some texture families: treated as a US1 failure
  for those tiles, characterized, and either resolved or those tiles excluded — never shipped as HR.
- **Authored minimap present but detail render fails** (or vice versa): the tile has no valid pair
  and is excluded from US2, counted in coverage.
- **Style gap dominates**: if the model learns to repaint authored minimaps into our render's style
  rather than add detail, that is an evaluation failure (US3 scenario 4), caught by the visual gate.
- **Giant-scale follow-on**: 2048/4096 renders and training are explicitly out of scope for the
  first pass (×4 only); the render and data pipeline should not hardcode 1024 in a way that blocks
  a later scale bump, but building/validating those scales is future work.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a detail-preserving minimap render that samples real terrain
  texture pixels (with MCAL blending and MCNR lighting) at 1024/tile, distinct from the existing
  material-average render, and MUST NOT emit a flat/fabricated image where textures cannot be
  decoded (honest skip instead).
- **FR-002**: The detail render MUST contain measurably more real high-frequency texture detail
  than a bicubic upscale of the material-average render, and MUST NOT exhibit minimap-scale moire
  at 1024 (the condition that justified the material-average hack must be shown not to recur).
- **FR-003**: The system MUST verify spatial registration between the authored client minimap and
  the detail render for the same tile, quantify the alignment error, and either confirm identity
  alignment within tolerance or identify a single fixed corrective transform that aligns all
  sampled tiles; it MUST NOT assume alignment.
- **FR-004**: The super-resolution pair set MUST include only tiles that have both a populated
  authored minimap and a successful detail render, from Kalimdor and Azeroth only; excluded tiles
  MUST be counted, never zero-filled into pairs.
- **FR-005**: The pair set MUST carry a deterministic train/eval split in which no tile appears on
  both sides; PVPZone02 and Kalidar MUST NOT appear in this lane.
- **FR-006**: The model MUST be a single-purpose image super-resolution model (Real-ESRGAN family);
  it MUST NOT be multi-task or share weights with any terrain-signal model.
- **FR-007**: Training MUST be user-executed with explicit per-run go-ahead; tooling prepares and
  prints exact commands but never launches training itself.
- **FR-008**: Every training run MUST record pair-set identity, split, model/loss configuration,
  per-metric baseline comparison (at minimum vs bicubic), and reproducible provenance.
- **FR-009**: Evaluation MUST restrict reported metrics to Kalimdor and Azeroth held-out tiles; any
  out-of-scope map request MUST fail closed or be clearly labeled.
- **FR-010**: All persisted dataset artifacts MUST remain per-build Zarr with the established
  identity/lineage discipline (the detail HR is the store's `minimap_rgb_1024` signal, its render
  semantics upgraded; pairs reference store rows, no NPZ side-channels).
- **FR-011**: The render and pair pipeline MUST NOT hardcode 1024 in a way that blocks a later
  scale increase; the first delivered scale is ×4 (256→1024).

### Key Entities

- **Detail-Preserving Render**: the new HR minimap render that samples real texture pixels; the
  content upgrade to the store's `minimap_rgb_1024` signal.
- **Authored LR Minimap**: `minimap_rgb_authored` from Spec 112 — the real client minimap, the
  model's low-res input and eventual deployment input.
- **Alignment Report**: the registration finding between authored and detail renders (error
  statistics and any corrective transform) — the US1 gate artifact.
- **SR Pair Set**: the trainer-facing collection of aligned (authored LR, detail HR) pairs with a
  leak-safe split and honest coverage.
- **SR Training Run Summary**: the record binding a checkpoint to its pair-set identity, split,
  configuration, metric-vs-baseline comparison, and provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a sampled set of tiles, the detail render's high-frequency content exceeds a
  bicubic upscale of the material-average render by a stated margin, with no visible moire at 1024.
- **SC-002**: Authored↔detail registration error is characterized on a sample and is either within
  tolerance under identity, or a single corrective transform brings all sampled tiles within
  tolerance — the US1 gate passes explicitly or the spec halts with the finding.
- **SC-003**: The SR pair set contains only tiles with both sources present from the two big maps,
  with a deterministic leak-free split, and its coverage is reported against the store's actual
  authored/detail coverage.
- **SC-004**: The trained model beats bicubic (and the material-average 1024) on the chosen SR
  metrics on held-out Kalimdor and Azeroth tiles, recorded in the run summary.
- **SC-005**: A user side-by-side review of held-out tiles judges the model's HR to add genuine,
  plausible detail (not fabricated structure and not mere restyling), using the established visual
  proof discipline.
- **SC-006**: No training or heavy render/data run in this spec is executed by the assistant; every
  such run's command appears in documentation with an estimate and is user-launched.

## Assumptions

- Spec 112 is complete: the rebuilt Kalimdor/Azeroth stores carry `minimap_rgb_authored` and
  `minimap_rgb_1024` with honest coverage, and the authored-signal capture works.
- The authored client minimap and our render share the same per-tile world bounds (both are
  per-(tileX,tileY) tile renders); only pixel-level orientation/offset is in question, which US1
  resolves.
- Real-ESRGAN is the chosen model family (user-specified); the exact generator/discriminator/loss
  and whether to fine-tune from public pretrained SR weights are plan-level decisions, subject to
  the constitution (repo-independence: any dependency is a package, not a path reference).
- The material-average render (`minimap_rgb`, 256) remains the height model's input (Spec 112) and
  is unchanged by this spec; only the 1024 render becomes detail-preserving.
- ×4 (256→1024) is the first and only delivered scale; 2048/4096 are future work.
- Training compute is the user's local CUDA GPU unless a cloud run is separately authorized; the
  Real-ESRGAN training is heavier than the Spec 112 height model and may need patch-based training,
  which is a plan-level concern.
- Constitution principle IV (terrain residual chain) governs terrain-signal models; the minimap SR
  model is a distinct single-purpose image model and is evaluated under that lens in the plan's
  constitution check rather than forced into the terrain-residual framing.
