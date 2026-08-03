# Implementation Plan: Minimap-to-Terrain Reconstruction Stack

**Branch**: `126-minimap-terrain-reconstruction` | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/126-minimap-terrain-reconstruction/spec.md`

## Summary

Invert the minimap: image in, terrain height out, plus a graded best-effort texture decode.

**The plan is deliberately front-loaded with measurement, not construction.** Seven risks (R1-R7)
are currently unresolved, and four of them can invalidate or reshape the architecture. Every one of
those four is answerable by an experiment costing hours, not by a training run costing days. Phase 0
runs those experiments and **no model is trained until it completes**. Each experiment states its
hypothesis, its detector, proof that the detector could find the thing if it were there, a decision
threshold fixed *before* the run, and what specifically changes if the answer comes back the other
way.

This ordering is not caution for its own sake. This project has previously recorded null results from
tests structurally incapable of detecting the effect they were looking for, and has previously
declared a hypothesis refuted using a detector with the wrong world scale and the wrong sun. The cost
of that pattern is measured in weeks. Phase 0 is the cheapest possible place to be wrong.

## Technical Context

**Language/Version**: C# / .NET 10 (synthesizer, codec, harvest); Python 3.11+ / uv (dataset, training, evaluation)

**Primary Dependencies**: PyTorch; Zarr v3; PyArrow/Parquet; Pillow; NumPy. Model backbones from
`timm` / `segmentation_models.pytorch` / HuggingFace `transformers` (SegFormer MiT, ConvNeXt-V2).
Depth-Anything-family models are excluded by standing project policy.

**Storage**: Zarr v3 stores with row-aligned `index.parquet`; PNG tiles for render passes

**Testing**: pytest (Python), xunit (C#). CPU-testable contract gates for every trainer, so
validation logic is verifiable without CUDA.

**Target Platform**: Windows-local training on user CUDA hardware; harvest against a configured
client library

**Project Type**: Multi-stage ML pipeline over an existing C# harvest/render toolchain

**Performance Goals**: Not latency-bound. The binding constraint is total parameter count (<= 200M)
and that a full evaluation pass over the held-out set completes fast enough to iterate daily.

**Constraints**:

- <= 200M parameters across the trained stack
- Evaluation on Kalimdor and Azeroth only; PVPZone02 and Kalidar are never used for validation
- Every dataset bucket stays queryable; filtering that drops rows is prohibited
- Per-signal metrics against per-signal baselines for any multi-output model
- All training and harvest runs are user-executed; commands are handed over PowerShell-ready

**Scale/Scope**: Two maps, roughly 1-2k usable tiles; 8 user stories; 31 functional requirements

**Unresolved before build** (see [research.md](./research.md)): R1 albedo pass and variance split;
R2 shading-law correlation; R3 layer-mask-from-shape and vocabulary; R4 codec fidelity vs authored
bytes. R5-R7 are resolved inside the build phases with their own gates.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All work under `wow-viewer/`. No outward path references. |
| II. Library-First | PASS | Codec, render passes, and dataset logic live in core libraries; scripts are thin wrappers. The DXT1 codec already has one canonical owner per language, parity-locked. |
| III. Real-Data Validation | PASS | Every gate in Phase 0 and every phase exit scores against real client data. Synthetic fixtures are used only to prove detector power, never for signoff. |
| IV. Model Architecture / Evidence | PASS | Under constitution v2.0.0 a multi-output model is permitted. The obligations it replaced the prohibition with are carried as FR-023 (per-signal metrics vs per-signal baselines), FR-024 (independently ablatable heads), and FR-025 (justified per-head loss weighting). Phase 4's exit gate enforces them. |
| V. Streaming-First Dataset Pipeline | PASS | New render passes stream through the existing harvest protocol into Zarr. No intermediate NPZ. |
| VI. No Client Path Assumptions | PASS | Client root stays runtime configuration. |
| Real-Data Validation (training) | PASS | Every training-script change carries a documented reason and validation path per the workflow rule. |
| One Phase at a Time | PASS | Phases below are strictly gated; "done" means validated, not coded. |

**No violations. Complexity Tracking section omitted.**

Note on Principle IV: this feature is the first to be planned under v2.0.0. The amendment retired an
architectural prohibition and replaced it with an evidence obligation. This plan treats that
obligation as a hard gate rather than a reporting nicety — Phase 4 cannot exit while any head sits at
its baseline, regardless of how good the aggregate looks.

## Phase Structure and Gates

Every phase states what result **stops** it. A phase that cannot exit does not roll forward silently.

### Phase 0 — De-risk by measurement (NO MODEL TRAINING)

Clears R1-R4. Full experiment designs in [research.md](./research.md).

| # | Experiment | Clears | Build cost |
|---|-----------|--------|-----------|
| E1 | Shading law: residual vs Lambert over real MCNR normals, sun swept not assumed | R2 | **Zero** — script exists, self-test passes, never run |
| E2 | Unlit-albedo render pass; albedo-vs-shading variance split | R1 | One C# render pass + one measurement script |
| E3 | Codec fidelity vs authored DXT1 bytes; unique-colour band on real output | R4 | Wire an existing unrun function to the CLI |
| E4 | Do layer masks derive from terrain shape? Vocabulary consistency | R3 | Two measurement scripts, reusing archived spec's design |
| E5 | Is the object capture library still lit correctly after the lighting corrections? | R9 | Sample re-render + diff. **Cheap to measure, long to fix** |

**Exit gate**: all five reported with a recorded decision. E1 is the single most important number in
this plan — it is the premise the whole feature rests on, it costs nothing to obtain, and it has
never been run.

E5 is here despite gating Phase 3 rather than Phase 0, because its *remedy* is a long user-run
harvest. Discovering at Phase 3 that every object capture needs regenerating turns a scheduled job
into a blocker.

**STOP conditions**: E1 correlation below the floor means the residual is not the shading field we
believe it is, which removes this spec's central advantage over prior failed attempts. E2 shading
variance below its floor promotes iterative refinement from optional to mandatory. Neither is a
reason to abandon the feature, but both change the architecture, and discovering either *after*
Phase 4 wastes the expensive phase.

### Phase 1 — Core claim: single-tile relief beats the baseline

Clears R5, the risk that no amount of model solves. Runs on the **easiest possible configuration**
first: synthetic, object-free, DXT1-degraded input, one tile, relief only.

**Exit gate**: beats the tile-mean baseline on held-out tiles, reported per tile with the failing
fraction stated. Best-epoch-1 is a structural failure. **Relief correlation >= 0.75 means the project
target is met and this phase is done** — it is a cutoff, not a floor to tune upward from.

**STOP condition**: if the easiest configuration cannot beat a tile-mean baseline, scaling the model
will not fix it. Return to Phase 0 findings and redesign the input representation.

### Phase 2 — Export and mesh

Small, and it makes Phase 1's output a real artifact rather than a tensor. Composes with the WDL
lattice prior for absolute elevation; emits relative-only with an explicit statement when no prior
exists.

**Exit gate**: an exported tile opens in external tooling and meshes without inverted normals or
disconnected spikes.

### Phase 3 — Objects: supervision honesty, library sidecar, corrected lighting

Clears R7, R8, and the remedy for R9. This is where objects enter, and three separate things have to
be true before they can.

**R7 — occlusion masking.** *Correction to the original risk framing*: the occlusion-correct signals
already exist in the v50 manifest — `object_geometry_visible_mask_257`,
`object_geometry_visible_instance_257`, `object_geometry_visible_source_257`. The risk is therefore
*verification*, not construction: are they populated, and do they mark actually-hidden terrain rather
than full ground footprints. `object_precise_mask` is the full-footprint signal, it over-masks
heavily, and it must not be substituted.

**R8 — the per-object library sidecar.** The capture library (`capture_rgb`, `capture_mask`,
`capture_alpha`, `assets.parquet`) exists but sits outside the v50 dataset contract, so no v50 tooling
can join against it. Bind it as a **sidecar**, not a merge: it is one row per capture variant against
the base store's one row per tile, and it regenerates on a different cadence. The sidecar's job is to
map instance IDs in `object_geometry_visible_instance_257` to library asset identity, which is what
turns a masked pixel from "something was here" into a named object.

**R9 — corrected object lighting.** Objects are lit by the model that was corrected after the captures
were made. If E5 says the difference exceeds the codec floor, the re-render must have completed before
this phase trains anything, because object appearance is model *input*.

**Why masking is viable now and was not before.** The earlier attempt at object masking removed too
much terrain to train on — but that was a property of `object_precise_mask`, which is the full ground
footprint and discards most of the terrain under an object even though only the genuinely hidden part
lacks evidence. The choice looked like "hallucinate under objects" or "have no data". The
`object_geometry_visible_*` signals describe what the render actually hid, so the excluded region is
the region that truly carries no ground evidence and the rest stays supervised. The same tiles become
usable at far higher coverage, from data that already exists.

**Exit gate**: a masked run and an identical unmasked run are compared, and the reduction in height
error under object footprints is reported as a number. The instance-to-asset join resolves for a
stated fraction of masked pixels. **Retained-terrain fraction is reported for both the visible mask
and the full-footprint mask on the same tiles**, so the coverage improvement that makes this approach
viable is a measurement rather than an assumption.

**STOP conditions**: if the visible-mask signals are unpopulated or encode full footprints, they must
be regenerated before Phase 4 — an unmasked loss teaches confident hallucination under every object,
and aggregate metrics will not reveal it. If E5 required a re-render and it has not completed, objects
must not be introduced into training input.

### Phase 4 — The multi-signal model

The "big model": minimap in, multiple signals out, <= 200M parameters. This is the expensive phase
and it is deliberately fourth.

**Exit gate**: FR-023 satisfied — every head reports its own metric against its own baseline, and
**no head sits at baseline**. An aggregate win with a dead head is a partial failure and exits
nothing.

### Phase 5 — Texture decode, tiers 1-3

Shaped by E4's answer. If layer masks derive from terrain shape, the decode inherits a strong prior
and joint modelling is justified; if not, decode relies on colour alone and tiers 3-4 get harder.

**Exit gate**: per-tier scores on held-out tiles with tier-0 tiles counted in the denominator; the
measured effect of feeding decoded structure to height reported whatever its sign.

### Phase 6 — Multi-tile and seams

Clears R6. Per-tile min-max normalization makes tiles independent *by construction*, so seams are the
expected failure, not a surprise. Leading candidate (see research.md): predict a gradient field and
integrate globally over the submitted region, which makes continuity structural rather than
post-hoc.

**Exit gate**: edge disagreement no worse than the stated multiple of ground-truth edge disagreement.

### Phase 7 — Iterative refinement

Analysis-by-synthesis with the forward model as referee. Promoted to mandatory if E2 says shading is
a small modulation on a dominant albedo signal.

**Exit gate**: error change reported at 1, 2, and N passes, including when negative.

### Phase 8 — Tier 4 per-layer decode

Gated on Phase 5 meeting tier 3. Reported as blocked, not attempted, if it has not.

## Project Structure

### Documentation (this feature)

```text
specs/126-minimap-terrain-reconstruction/
├── plan.md              # This file
├── research.md          # Phase 0 experiment designs, thresholds, and branches
├── data-model.md        # Entities, stores, and contracts
├── quickstart.md        # PowerShell-ready commands, in phase order
├── contracts/           # Target and store contracts
└── tasks.md             # Created by speckit-tasks, not by this command
```

### Source Code

```text
wow-viewer/
├── src/core/
│   ├── WowViewer.Core.IO/Blp/
│   │   └── Dxt1TileCodec.cs                     # exists; add authored-bytes fidelity path (E3)
│   └── WowViewer.Core.Runtime/World/
│       └── (terrain compositor)                 # add the unlit-albedo pass (E2 / FR-001)
├── tools/harvest/WowViewer.Tool.Harvest/
│   └── Program.cs                               # add --albedo-only; surface round-trip agreement
└── data-harvester/
    ├── src/harvester/v50/
    │   ├── dxt1_approx.py                       # exists, parity-locked
    │   ├── decomposition.py                     # NEW: albedo/shading split + variance measurement
    │   ├── terrain_reconstruction_model.py      # NEW: the multi-signal model
    │   ├── terrain_reconstruction_train.py      # NEW: trainer + per-signal baselines
    │   └── heightmap_export.py                  # NEW: export + mesh
    ├── scripts/
    │   ├── v50_measure_residual_shading_law.py  # exists, UNRUN on real data (E1)
    │   ├── v50_measure_albedo_shading_split.py  # NEW (E2)
    │   ├── v50_measure_codec_fidelity.py        # NEW (E3)
    │   ├── v50_measure_layer_shape_coupling.py  # NEW (E4)
    │   └── v50_build_reconstruction_curriculum.py  # NEW
    └── tests/v50/                               # contract gates, CPU-testable
```

**Structure Decision**: Follows the established split — C# owns rendering, codecs, and harvest;
Python owns dataset assembly, training, and evaluation; both sides of any shared transform are
parity-locked by golden-value tests, as the DXT1 codec already is. No new top-level project.

## What Would Make This Plan Wrong

Recorded deliberately, because the failure mode this project has actually experienced is not bad
code — it is confident measurement of the wrong thing:

- **E1 comes back strong but on contaminated data.** The residual PNGs must come from the same tiles
  and the same era as the normals they are correlated against. A mismatched pairing can manufacture
  or destroy correlation. The row lookup is by `(map, tile_x, tile_y)`; provenance must be checked,
  not assumed.
- **E2's recombination check passes trivially.** If the albedo pass accidentally includes lighting,
  recombination still reproduces the minimap and the variance split is meaningless. The pass must be
  verified to be genuinely unlit before its split is believed.
- **A null result from an underpowered detector.** E4 in particular repeats a test the archived spec
  already flagged as underpowered. Detector power must be demonstrated on planted signal before any
  negative finding is recorded, exactly as E1's script already does.
- **Phase 4 reports an aggregate win.** The per-signal gate exists precisely because a shared trunk
  is where a dead signal hides. It is a stop, not a note.
