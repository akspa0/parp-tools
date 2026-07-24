# Feature Specification: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Feature Branch**: `121-v7-wdl-height`

**Created**: 2026-07-24

**Status**: Draft

**Input**: User description: "Go back to training a v7-style model with current signals. Building a minimap-to-WDL prior model is important. Object masking and segmenting on minimaps does not work — Specs 119 and 120 do not work at all. Use the precise object masks as a loss signal for the v7-style model instead. Use a smaller 3–30M parameter model, not the 100–130M model v7 used. A HuggingFace backbone such as SegFormer is acceptable."

## Problem Statement

The minimap object-identity line is dead. Spec 119's own retrieval proof-of-concept measured real
minimap object instances at **p50 = 10px, max = 29px** and showed a trained library embedding
cannot discriminate blobs at that scale (every crop matched unrelated round textures at ~0.99
cosine). Spec 120's DINOv2 retrieval pivot inherits the same scale problem. Segmenting and
classifying objects **on minimaps** does not work and is abandoned as a task.

Meanwhile the expensive, correct object signal already exists and is unused: the v50 Full-profile
store streams precise, occlusion-aware object visibility masks
(`object_geometry_visible_mask_257`, `object_geometry_visible_source_257`,
`object_geometry_visible_instance_257`, Spec 118). Those masks are ground truth rendered from real
geometry. Their correct use is **loss-side**: down-weight or exclude object-contaminated pixels
when training terrain models. They must never again be framed as something a minimap model should
predict.

The height line also regressed away from what worked. The v7 idea — a low-resolution WDL prior
plus a small residual detailer — was the last architecture the user judged as working. Spec 117
proved a from-scratch ~675K-param RGB→WDL model plateaus above the tile-mean baseline; the
conclusion was not "the idea is wrong" but "from-scratch at toy scale lacks the capacity/inductive
bias." Spec 114 proved a HuggingFace `mit_b0` (SegFormer-B0, ~3.7M params) detailer clears its
numeric gate. The path forward is the v7 two-stage shape, at 3–30M params, with pretrained
backbones allowed, on the current v50 signal store.

## Governing Principle

The deployment input is an authored minimap image. Every other inference signal must be predicted
from that image by an independently trained, independently checkpointed small model. Ground-truth
WDL, height, and object masks are training/evaluation evidence only — never inference inputs.
Precise object masks are a **loss signal only** in this lane: no object segmentation,
classification, detection, or retrieval task is created, resumed, or implied.

## Relationship To Existing Specs

- **Supersedes / closes**: Spec 119 (object-library classifier) and Spec 120 (minimap placement
  retrieval). Their negative result is recorded fact, not a failure to retry.
- **Re-uses**: Spec 117's WDL lattice target contract (`wdl_outer_17`/`wdl_inner_16` + present
  flags, already streamed by the C# harvester) and its masked encode/decode loss contract.
- **Re-uses**: Spec 114's detailer precedent (HuggingFace `mit_b0` detailer, coarse-only baseline
  gate, V7 spectral loss terms ported loss-only behind flags).
- **Re-uses**: Spec 118's object-mask loss machinery (`--object-mask-weight`, object-touched vs
  untouched region MAE reporting) and its Full-profile ground-truth arrays.
- **Re-uses**: Spec 116's held-out split machinery (8-neighbour isolation, leakage refusal).
- **Re-uses**: the v50 run-record contract (`v50-model-stage-run-v1`, `promotion_verdict=pending`).
- **Does not amend**: Spec 112/114 geometry trainers remain valid lanes; this spec's Stage B is a
  new checkpoint in the same contract family, not a rewrite.

## Out Of Scope (Explicit)

- Object segmentation, classification, detection, or retrieval **on minimaps**, in any form. This
  includes DINOv2/embedding retrieval, OBB detection, per-instance identification, and object
  library matching.
- Training any model that predicts an object mask as its output. Masks are inputs to the loss,
  never targets.
- New C# harvest work. The v50 store already streams every required signal (Full profile). A store
  rebuild to materialize the Spec 118 arrays is a user-run data prerequisite, not code.
- Models above 30M parameters, DepthAnything/DPT-scale backbones, RunPod envelopes, multi-task
  heads (RULE 7).
- V22 substrate, minimap super-resolution, texture/alpha reconstruction, liquid reconstruction.
- Changing the frozen v50 signal catalog or the WDL grid shape. The C# reader's output shape
  (17×17 outer + 16×16 inner = 545 points) is the contract.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stage A: Minimap → WDL Lattice Prior (Priority: P1)

A model operator trains one small model that predicts the 545-point WDL height lattice (outer
17×17 + inner 16×16, with present-masks) from minimap RGB alone, on the v50 store. A pretrained
HuggingFace backbone (SegFormer family, `mit_b0` precedent) is permitted; a from-scratch fallback
remains selectable. The lattice target reuses the Spec 117 masked contract: absent samples never
affect normalization or loss.

**Why this priority**: The WDL prior is the load-bearing input of the v7 shape. Spec 117 showed
the from-scratch tiny model cannot beat tile-mean; this story tests whether a pretrained-backbone
model in the 3–30M band clears that bar. If it does not, the lane stops here with a reportable
negative result and Stage B is never trained.

**Independent Test**: On the frozen Spec 116 held-out split, Stage A's lattice MAE beats the
tile-mean baseline by a recorded margin, with per-epoch previews showing plausible coarse terrain.

**Acceptance Scenarios**:

1. **Given** the v50 store with `wdl_outer_17`/`wdl_inner_16` arrays and a frozen held-out split,
   **When** Stage A training runs, **Then** the run record contains held-out lattice MAE, the
   tile-mean baseline MAE, and the relative margin, with `--held-out-split` mandatory (no
   fallback).
2. **Given** a tile whose WDL lattice is absent (present flags false), **When** training and
   evaluation run, **Then** that tile contributes nothing to normalization ranges or loss.
3. **Given** a selected backbone, **When** the checkpoint is written, **Then** total parameter
   count is recorded, lies within 3–30M, and the checkpoint is reconstructable from its saved
   config alone (backbone id + revision or from-scratch base width).
4. **Given** a dry-run invocation, **When** the operator runs the CLI without `--confirm-run`,
   **Then** it prints the full plan (params, split counts, baseline, LR schedule) and exits
   without training.

---

### User Story 2 - Stage B: V7-Style Residual Detailer Over the Prior (Priority: P1)

A model operator trains one small detailer that consumes the cleaned minimap plus the upsampled
WDL prior and predicts a residual over the prior at 257×257, producing the final relative-height
field. At inference the prior comes from the frozen Stage A checkpoint; ground-truth WDL is used
only in a labelled ablation. The V7 spectral loss terms (already ported in Spec 114) remain
available as loss-only flags, default off for parity.

**Why this priority**: This is the v7 architecture the user wants back — prior + small residual
refiner — and it is the height product of the lane.

**Independent Test**: On the same frozen split, Stage B (fed the predicted prior) beats the
prior-only baseline (upsampled prior, no detailer) by the recorded gate margin, and the
ground-truth-prior ablation is reported as an upper bound.

**Acceptance Scenarios**:

1. **Given** a frozen Stage A checkpoint and the v50 store, **When** Stage B trains, **Then** its
   inputs are minimap RGB + predicted prior only (plus optional already-existing feature stores),
   and the run record names the Stage A checkpoint sha256.
2. **Given** the frozen split, **When** evaluation runs, **Then** the record contains Stage B MAE,
   prior-only baseline MAE, relative margin, and the ground-truth-prior ablation MAE.
3. **Given** an object-heavy tile, **When** the object-mask weight flag is active, **Then**
   loss contribution at masked pixels is scaled per the documented rule and object-touched vs
   untouched region MAE appears in the record.
4. **Given** the checkpoint, **When** params are counted, **Then** the total is within 3–30M and
   reconstructable from config alone.

---

### User Story 3 - Object Masks as a First-Class Loss Signal (Priority: P2)

A model operator runs the paired comparison this lane exists for: Stage A and Stage B each trained
with object-mask loss weighting off (parity default) vs on, on the identical split, with
object-touched vs untouched relief-stratified MAE reported. The precise masks flow from the v50
Full-profile arrays into the existing `--object-mask-weight` machinery; no new prediction task is
created.

**Why this priority**: This is the user's explicit repurpose of the Spec 118 data. It is P2 only
because the paired comparison requires trained Stage A/B checkpoints to compare.

**Independent Test**: Two paired run records exist per stage, differing only in the mask-weight
flag, and the comparison verdict (helps / hurts / null) is recorded. A null result is a valid,
reportable outcome that closes the question.

**Acceptance Scenarios**:

1. **Given** a store lacking the object-mask arrays, **When** a weighted run is attempted,
   **Then** the trainer warns and disables the weighting (existing behavior) and the run record
   marks the signal absent.
2. **Given** a tile whose mask covers nearly all pixels, **When** weighting is active, **Then**
   the remaining unmasked pixels still produce a finite loss and the tile is not silently dropped.
3. **Given** the paired runs, **When** the comparison is recorded, **Then** promotion of any
   weighted checkpoint remains `pending` until the user's visual gate.

---

### User Story 4 - End-to-End Deployment Chain and Visual Gate (Priority: P3)

A model operator materializes the full chain — authored minimap → Stage A prior → Stage B
residual → final 257×257 height field — over a held-out map and one hand-painted out-of-distribution
tile, producing fixed-row and worst-case validation sheets for the user's visual verdict.

**Why this priority**: Numbers alone have burned this project before (Spec 114's numeric gate
passed before the visual gate). The chain proof is the user's acceptance surface.

**Independent Test**: The materializer runs image-only (no ground-truth signal read at inference),
and the user issues a visual verdict from the sheets.

**Acceptance Scenarios**:

1. **Given** only authored minimaps on disk, **When** the chain materializer runs, **Then** it
   reads no ground-truth WDL, height, or mask array and names both checkpoints in its output
   provenance.
2. **Given** a hand-painted OOD tile, **When** inference runs, **Then** the output sheet is
   produced without crash and flagged OOD in the audit record.

---

### Edge Cases

- Tile with absent WDL lattice (present flags false): excluded from Stage A supervision; at
  inference Stage A still emits a full lattice (that is its purpose).
- Tile with all-object mask: weighted loss degenerates gracefully; tile contributes via Stage A's
  prior rather than dense pixel loss.
- Blank/near-blank minimap: Stage A must not NaN; preview sheets surface the failure mode.
- Backbone unavailable offline (no HuggingFace download): from-scratch fallback config must run
  unchanged, with the understanding that Spec 117's plateau is its known risk.
- Stage A regression after promotion: Stage B's record binds Stage A by sha256, so a swapped
  prior checkpoint is detectable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Stage A MUST predict the WDL lattice (545 values + present-mask semantics) from
  minimap RGB alone, reusing the Spec 117 masked target contract.
- **FR-002**: Stage B MUST predict a residual over an upsampled prior at 257×257 and MUST emit the
  final height field as prior + residual.
- **FR-003**: Each model MUST be independently trained, independently checkpointed, 3–30M
  parameters, and reconstructable from its saved config alone.
- **FR-004**: A frozen held-out split MUST be required by every trainer (no fallback); leakage
  checks must refuse a violating split.
- **FR-005**: Every run record MUST contain the relevant trivial baseline (tile-mean for Stage A;
  prior-only for Stage B) alongside model metrics.
- **FR-006**: Precise object masks MUST be usable only as a loss-side weighting signal
  (`--object-mask-weight`, default 0.0 = parity); no model in this lane may predict an object mask
  or class.
- **FR-007**: With weighting active, run records MUST report object-touched vs untouched region
  MAE.
- **FR-008**: All CLIs MUST be dry-run-first; all training launches are user-run (RULE 0).
- **FR-009**: Pretrained HuggingFace backbones (SegFormer family) MUST be selectable by config;
  backbone id and revision MUST be recorded in the run record; a from-scratch fallback MUST remain
  selectable.
- **FR-010**: At inference, Stage B MUST consume the predicted prior from a frozen Stage A
  checkpoint; ground-truth WDL input is permitted only in a labelled ablation mode.
- **FR-011**: Trainers MUST emit per-epoch validation previews plus final fixed-row and worst-case
  sheets (Spec 117 pattern).
- **FR-012**: Run records MUST reuse `v50-model-stage-run-v1` with `promotion_verdict=pending`;
  promotion is a user gate.
- **FR-013**: This lane MUST NOT require new C# harvest code; it consumes the v50 Full-profile
  store as-is.
- **FR-014**: The deployment chain materializer MUST run image-only and record both checkpoint
  sha256 values as provenance.

### Key Entities

- **WDL lattice prior**: per-tile 545-point low-resolution height field (17×17 outer + 16×16
  inner) with present-masks; ground truth from the store, predicted by Stage A at inference.
- **Object visibility mask**: per-tile 257×257 ground-truth mask of visible object pixels
  (Spec 118 arrays); loss-side weighting evidence only.
- **Residual height field**: Stage B's correction over the upsampled prior at 257×257.
- **Held-out split**: frozen map-grouped partition (Spec 116 machinery) shared by both stages.
- **Model stage run record**: `v50-model-stage-run-v1` JSON binding config, split, baselines,
  metrics, backbone provenance, and promotion verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Stage A held-out lattice MAE is at least 15% below the tile-mean baseline MAE on the
  frozen split. (This is the exact bar Spec 117's from-scratch model failed.)
- **SC-002**: Stage B held-out height MAE is at least 9% below the prior-only baseline on the
  frozen split (Spec 114's detailer precedent cleared 9.1–9.2%).
- **SC-003**: The paired mask-weight on/off comparison is recorded for both stages with
  object-touched vs untouched MAE; any of helps/hurts/null is an acceptable, closable verdict.
- **SC-004**: Each checkpoint is within 3–30M parameters and trains on the user's local GPU at the
  documented batch size without out-of-memory.
- **SC-005**: The end-to-end chain produces validation sheets for one held-out map and one OOD
  hand-painted tile, and the user issues an explicit visual verdict.

## Assumptions

- The v50 Full-profile store has been (or will be, user-run) rebuilt so the Spec 118 object-mask
  arrays and WDL lattice arrays are present; weighted runs degrade gracefully with a warning when
  absent (existing behavior).
- The primary client build remains `0_5_3_3368` under the approved `H:\CLIENTS` root.
- HuggingFace model download is available at training time on the user's machine; otherwise the
  from-scratch fallback is the documented path.
- `transformers` (or the Spec 114 detailer's existing backbone dependency) is already satisfied in
  the `wow-viewer/data-harvester` `uv` environment, since Spec 114's `mit_b0` detailer trained
  successfully.
- The WDL grid shape contract is the C# reader's output (17×17 + 16×16 = 545); this spec invents
  no new shape.
- Specs 119 and 120 will be moved to `specs/archived/` as part of this feature's bookkeeping.
