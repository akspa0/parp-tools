# Implementation Plan: Revive the v7 terrain regressor on current clean signals

**Branch**: `v0.5.1` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

> **Correction (2026-07-13):** V24 / Spec 094 is NOT functional and is ignored. This plan revives the original **v7** model — a single, basic U-Net (no stages) from ~April 2026 — and runs it on the current clean dataset. The user's claim: v7 was simple, dirty, and worked; with clean signals it should "fly and be perfect."
>
> **Correction (2026-07-14):** Phase 2's procedural-synthetic PoC is demoted to a pipeline smoke test. Procedural patterns (flat/ramp/ridge/crater/plateau) don't replicate real terrain, and the WDL prior trivially solves them (measured: l1_global ≈ 0.0006 at init). The synthetic lane's intended form is **signals synthesized from real terrain** (deterministic shadow/hillshade renders of real height — the T018 lane), never invented terrain. Soundness verification happens on the real curated dataset (Phase 3) with the lean **v8** architecture (see Implementation state).

## Summary

Port the real v7 model (`MultiChannelUNetV7`) into `wow-viewer` and train it on the current clean signals. v7 is one U-Net: **[minimap RGB + normal RGB + WDL prior + aux context] → terrain height**, refining the WDL prior in a single shot (no Stage A/B). It is read-only reference in `gillijimproject_refactor`; we port it, we do not modify it there.

**Source of truth (read-only reference):** `gillijimproject_refactor/src/WoWMapConverter/scripts/` — `v7_model.py` (168 lines, `MultiChannelUNetV7`), `train_v7.py`, `v7_losses.py`, `infer_v7.py`. v7 added 2026-04-14; V7.7 detail head 2026-04-19.

## v7 Contract (from the model + V7.5 guide)

- **Input: 13 channels** — `0–2` minimap RGB (terrain-only cleaned preferred), `3–5` normal-map RGB, `6` WDL height prior (the residual/"trestle" base: `global = wdl_base + tanh(delta)*scale`), `7–12` six auxiliary context channels (alpha/liquid/hole/chunk metadata) marking known minimap losses.
- **Output: 2 channels** — global + local height (V7.7 adds an optional 3rd detail channel), plus a small `bounds` head (4 values). Model interpolates output to `OUTPUT_SIZE` (512 in the original).
- **Core**: 5-level residual U-Net (64→128→256→512→1024→2048 bottleneck), GroupNorm, reflect padding. Architecturally simple; not tiny, but "basic."

## How our current clean signals map to v7's 13 channels

**Pinned from `train_v7.py` (2026-07-13) — full detail in [research-v7-contract.md](research-v7-contract.md):**

| v7 channel | pinned meaning | our signal |
| --- | --- | --- |
| 0–2 minimap RGB | recovery-attenuated, ImageNet-normalized | `minimap_rgb` |
| 3–5 normal RGB | recovery-attenuated, ImageNet-normalized | `normal_xyz` → (n+1)/2 |
| 6 WDL prior | outer 17×17 normalized, align_corners=True upsample; **0.5 fill when missing/dropped** | derived: `height_257[::16, ::16]` |
| 7–8 height hints | constant planes = normalized tile height min/max | `--height-hints gt\|wdl\|none` |
| 9 liquid mask | binary | `liquid_mask` |
| 10 liquid height prior | normalized × mask | `liquid_height` |
| 11 object footprint | context/precise mask | `object_precise_mask` |
| 12 brush imprint | binary | zeros (V18 has none) |

The original guess for aux 7–12 ("alpha, holes_16, chunk metadata") was **wrong** — pinned above.
`v7_losses.derive_recovery_mask_from_inputs` hard-codes ch 9/11/12; the layout is load-bearing.

**Resolution decision (Phase 0 item 3)**: work at 256 (native minimap; v7's 512 was an upscale).
The ported model parameterizes `output_size` — the only deviation from the reference. Vertex
grids (257) resample with align_corners=True.

**Target**: `height_257` → [global absolute-normalized (−1000..3000), local within-tile] + 4-value bounds. This is a supervised reconstruction model; the WDL prior it refines is derived from the same height family, so at deployment the prior must eventually be image-generated — see "Relationship to the image-only law."

## Loss & object handling — quick-and-dirty (decided)

The loss stays simple: plain height regression over all pixels, the way v7 worked. **Object-mask gating in the loss is OFF by default** — masking out object pixels wipes out large swaths of the tile (everything that exists or doesn't), costs hours of extra training, and buys little; v7 proved training straight through object noise works. Object-mask gating is available as an optional flag for later experiments, off unless explicitly requested. Objects are accepted noise; cleanup stays deferred to the output-space segmentation+inpaint lane (spec US3).

**Implemented loss**: the ported `v7_losses.combined_loss` **without** the PatchGAN (v7's own recipe minus the GAN complexity); `--loss l1` gives pure regression for ablation.

## Dataset curation is mandatory (spec Principle #5), not a flag

Clean data in, clean model out. Height under an object is occluded in the minimap, so an
object tile is an **impossible height target** — Principle #5 requires dropping it, not learning
it. `scripts/spec103_curate_dataset.py` is a first-class, auditable pass that buckets every tile
and drops three failure classes, writing a `curation_manifest.parquet` the trainer consumes:

- **object_contaminated** — `object_precise_mask` coverage above `--max-object-coverage`
  (default **0.0** = drop ANY object; `1.0` = v7-faithful keep-all ablation only).
- **blank_minimap** — per-tile RGB std below `--min-rgb-std` (dead-space art; spec edge case).
- **height_normal_mismatch** — flat height but normals show relief (a harvest failure / mismatched signal).
- **missing_signal** — a required array (height / minimap / normals) absent.

Kept tiles are tagged with stratification buckets (map + height-regime tertile) for representative
complete-map holdouts (FR-008). **Recorded V18 zero-object result:** 5134 → 2253 kept (43.9%);
410 blank + 2471 object-contaminated dropped; 0 height/normal mismatch (verified: relief tracks
height-std at r=0.57, and only 2 flat tiles have varied normals — this store is clean on that
axis). The summary's coverage alternatives are pre-other-gate counts: 2650 tiles at zero objects,
3078 at ≤0.5%, and 3540 at ≤2%.
The trainer drops object tiles by default even without a manifest (`--max-object-coverage 0.0`);
`1.0` restores the v7-faithful keep-all behavior for ablation only.

### Pattern-aware reduction: select evidence coverage, not more tiles

The clean-tile manifest is only the first filter. The next bounded slice reduces the remaining
corpus by selecting unique, reusable terrain-art evidence. It does **not** restart alpha analysis:
Spec 076's full-map fractal brush library is the authority for map-canvas segmentation, global
region/family IDs, cross-ADT continuity, and coupled alpha/height/normal/MCLY/object evidence.
Raw 074/tile-local connected components remain supporting evidence only.

The Spec 103 consumer will read that library and emit three auditable curation artifacts:

1. `pattern_evidence_ledger.parquet` — one or more rows per eligible tile and intersecting region,
   preserving build, map, tile ID/coordinates, ADT 16x16 chunk/cell coverage, alpha layer,
   region/family IDs and state, terrain relief, MCLY texture/layer context, and object/liquid
   relationship. It records non-brush, blocky-paste, rectangle-page, composite, and atomic patterns;
   no class is erased merely because it is not a brush stroke.
2. `tile_pattern_coverage.parquet` — one row per tile, aggregating the ledger into deterministic
   family/context coverage and an eligibility decision. This is the directly inspectable answer to
   “what exists in this place on this map?”
3. `curation_manifest.parquet` + `curation_summary.json` — the trainer-facing selection, extended
   with `selection_reason`, representative family/context keys, upstream artifact hashes, and the
   selected/excluded duplicate lineage.

Selection is deliberately conservative: preserve a representative set across map/build, terrain
relief, MCLY context, alpha family/state, and object-placement context; discard only examples whose
coverage is already represented. Family grouping is performed before split assignment, and a family
never spans train and validation. Alpha, mesh, MCLY, and object values are curation observations
only—they never change the eventual image-only deployment tensor or permit ground-truth inference.

The first implementation does not change model architecture, loss, or training. It only produces
the evidence and a smaller manifest. A training claim remains blocked until the user runs a new
training job against that manifest.

## Constitution Check

- **Residual Model Chain**: one model, one signal (terrain height as a residual over the WDL prior). No multi-task, no shared weights. PASS.
- **Repo Independence / Read-Only Reference**: v7 is read from `gillijimproject_refactor` (RULE 1 valid reason: reference for reimplementation) and ported into `wow-viewer/data-harvester/`. The reference repo is not modified. PASS.
- **Real-Data Validation**: code/tests are prepared; the staged-client curation/capture and model
  quality proofs remain USER-run. PREPARED, NOT YET SIGNED OFF.
- **Training-Script Discipline**: the ported trainer documents input-channel layout and losses. PASS.
- **Bite-Sized**: phases ≤10 one-concern steps. PASS.

No violations.

## WDL-prior robustness & the path to image-only (decided + exploratory)

- **Robustness to missing WDL prior (in scope, cheap):** train with WDL-prior **channel dropout** — randomly zero/flatten channel 6 for a fraction of tiles. Because v7's trestle is `height = wdl_base + delta`, a zeroed base makes the model predict the full height as the residual (no prior), a present base makes it refine. One model serves both prior-present and prior-absent tiles. Training-time input augmentation only; no architecture change.
- **Invert to predict the prior (exploratory front-end lane):** once v7 has the WDL↔terrain correlation, train a small `minimap → WDL prior` from the same paired data, closing the image-only loop: image → predicted prior → v7 → terrain.
- **Synthetic universality (exploratory lane):** author synthetic ADT tiles with known height patterns (existing ADT-creation tooling), capture their minimaps in WoWViewer, derive the prior deterministically → perfectly-clean (image, prior, terrain) triples with known ground truth for training and for probing what the model learned; a route to a universal model.

## Terrain shadow & the teacher/student path (exploratory, unlocked by synthetic control data)

- **Shadow ↔ height, now measurable.** Synthetic tiles with known height let us render a deterministic fixed-light terrain shadow (Spec 102 N011-N013 already defines the capture + determinism contract) and directly measure how shadow encodes slope/relief — against perfect ground truth, which the uncooperative renderer blocked before. If the correlation is strong, terrain shadow becomes a legitimate signal (and it is partly baked into the minimap's own lighting).
- **Teacher → student.** Train a TEACHER on the rich clean synthetic signals (minimap + normals + WDL prior + terrain shadow → height, known GT) to prove the mapping is learnable. Then distill it into a STUDENT that consumes only deployment-available inputs (ultimately image-only), inheriting the teacher's mapping while honoring the image-only law. This is the bridge from rich synthetic training to image-only deployment.

### Phase 2C — Correct synthetic illumination before another synthetic capture

1. Bridge existing MCNR and MCSH decode results into both active terrain render paths; preserve the
   format-specific normal decode and transform explicitly rather than hiding Alpha-era inversions in
   a two-sided shader.
2. Replace `abs(N dot L)` with one-sided Lambert, preserve the complete MCSH grid, and make shadow
   strength/version an explicit contract while exact client attenuation remains unrecovered.
3. Replace the capture path's perspective/16-tile framing with a canonical one-tile top-down
   orthographic camera and write a lighting/provenance sidecar for each image.
4. Add grouped time-of-day variants to the synthetic-store builder. Prefer a build-scoped DBCD
   `Light*` export; retain the authored fallback with an unambiguous non-client-exact label.
5. Keep generated/owned and private client-derived sources in separate rights classes. Fail closed
   when the operator requests a clean-synthetic artifact without explicit license, assertion, and
   matching hashes. This is provenance enforcement, not a legal determination.

Detailed evidence and proof gates: [research-synthetic-lighting.md](research-synthetic-lighting.md).

## Relationship to the image-only law (spec Governing Principle)

v7 consumes the WDL prior + normals + aux, which are height-derived — so v7 by itself is the **reconstruction** model, not an image-only one. That is fine and intended for this revival: get v7 flying on the signals we have. Generating those inputs (chiefly the WDL prior) from the image alone is the *separate, later* image-only front-end. The spec's law governs the eventual deployment chain (image → generated prior → v7 → terrain); this plan delivers the proven back half first. Recorded honestly so we don't pretend v7 is image-only.

## Phases (bite-sized)

### Phase 0 — Pin the v7 contract (read-only)

1. Read `train_v7.py` to record the exact 13-channel assembly order and the aux channels 7–12.
2. Read `v7_losses.py` + `infer_v7.py` for the loss terms, output-head modes, and inference resolution.
3. Confirm the working spatial resolution (v7 used 512; our height is 257) and decide the resize convention.

### Phase 1 — Port v7 into wow-viewer

1. Port `MultiChannelUNetV7` (+ losses) to `data-harvester/src/harvester/spec103/v7_model.py`, unchanged in architecture.
2. Write the input assembler that builds the 13-channel stack from the current clean signals (table above), including the derived WDL prior.
3. CPU sanity harness (no GPU run): forward/loss/backward, 13-ch input, output shape, WDL-trestle residual path.

### Phase 2 — Synthetic proof-of-concept (lead experiment, de-risk first)

1. Author a small set of synthetic ADT tiles with known height patterns (ramps, ridges, craters, plateaus) using the existing ADT-creation tooling, used as-is (AlphaWdtWriter stays frozen; no C# rewrites).
2. Capture their minimaps via the WoWViewer capture path; derive the WDL prior deterministically from the synthetic height (`::16`/`8::16`); assemble the 13-channel input.
3. Train v7 on the clean synthetic set (user runs). Verify it reconstructs the known patterns and that prior-dropout tiles still resolve. Catalog every caveat (resolution, channel order, trestle behavior, losses) — this is the whole point of going synthetic-first.

### Phase 3 — Apply the proven recipe to the real clean dataset (user runs)

1. With synthetic caveats resolved, build/point at a real clean store (`minimap_rgb` + `normal_xyz` + `height_257` + alpha/liquid/holes; derive the prior).
2. Train the now-proven v7 on real data with the bounded-trainer conveniences (complete-map holdout, AMP, EMA, warmup+cosine, early-stop, resumable, WDL-prior dropout). User runs.

### Phase 4 — Review + validation

1. Reuse the OBJ/mesh export path to render terrain for eyeball review (the side-by-side you showed).
2. Dev diagnostic: height L1/gradient vs. the WDL-prior baseline; label-free self-consistency (border agreement, plausibility, artifacts) toward the spec's acceptance test.

### Phase 3B — Pattern-aware corpus curation (before any next real-data run)

1. Define the Spec 103 ledger contract as a consumer of the validated Spec 076 full-map primitive
   library; include the complete map/tile/chunk/layer and upstream-artifact identity chain.
2. Build the ledger for every available map and alpha layer, preserving atomic, composite, patterned,
   and non-brush region states plus terrain/MCLY/object context.
3. Aggregate tile coverage and deterministically select representative pattern/context coverage,
   assigning group-safe train/validation partitions and explicit duplicate/exclusion lineage.
4. Validate on a bounded store with schema/provenance/unit tests and review a per-map report; then
   prepare (but do not launch) the user-owned training command against the reduced manifest.

### Phase 5 — Deferred lanes (scoped notes only — T016/T019; no implementation)

- **Image-only WDL-prior front-end** (T016): a small `minimap → outer 17×17 prior` model trained on the same paired data v7 uses, closing the loop image → generated prior → v7 → terrain. Interface already fixed: `assemble_v7_input(wdl_outer_17=...)` accepts a generated prior directly; the trainer's `val_no_prior` column measures how much a front-end would need to add over the flat 0.5 fill. Downstream training obeys FR-003: v7 fine-tunes on the front-end's *generated* priors, not GT.
- **Synthetic-universality scale-up** (T016): grow `spec103_make_synthetic_adts.py` patterns (composed patterns, noise octaves, real-height hybrids) into a large perfectly-labeled corpus; same store schema, no new contracts.
- **Output-space object cleanup** (T016, spec US3): segment object artifacts in the *generated* height field, mask, inpaint from the surrounding lattice; input contract stays image-only. The predicted lattice stays editable (coarse WDL representation).
- **Teacher → student distillation** (T019): TEACHER trained on rich clean synthetic signals (minimap + normals + WDL prior + fixed-light terrain shadow → height, known GT — shadow capture per Spec 102 N011-N013, measured by T018) distilled into a STUDENT consuming deployment-available inputs only (ultimately image-only). The bridge from rich synthetic training to the image-only law; contingent on the T018 shadow↔height correlation result.

### Implementation state (2026-07-15)

Phases 0–1 complete and tested (13/13 CPU sanity tests): pinned contract in
[research-v7-contract.md](research-v7-contract.md), ported lane in
`data-harvester/src/harvester/spec103/` (`v7_model.py`, `v7_losses.py`, `v7_inputs.py`,
`v8_model.py`). **Architecture update (USER decision 2026-07-13):** the primary training
architecture is `V8LeanUNet` (`--arch v8`, trainer default) — a ConvNeXt-V2-style lean U-Net
(6.2M params / 16.4 GFLOPs @256 vs v7's 117M / 119.9) honoring the identical 13-ch/trestle/
bounds contract, built for fast local iteration. v7 stays available (`--arch v7`) as the
faithful ablation. Survey + rationale: [research-v8-optimization.md](research-v8-optimization.md).
Phase 2–4 scripts prepared (`spec103_make_synthetic_adts.py`, `spec103_build_synthetic_store.py`,
`train_spec103_v7.py`, `infer_spec103_v7.py`, `spec103_build_real_store.py`,
`spec103_export_mesh.py`, `validate_spec103_labelfree.py`); commands in
[quickstart.md](quickstart.md). Blocked on USER runs: capture (1d), training (1f/3b),
inference (2a), shadow capture (T018).

Phase 3B is now implemented by `spec103_curate_prefabs.py` plus
`src/harvester/spec103/prefab_curation.py`: typed Parquet evidence ledgers, map-canvas multiscale
and cellular composition features, canonical transform-equivalent prefab families, MCLY/tileset
anomaly evidence, object/liquid context, deterministic coverage selection, representative lineage,
and prefab-family-safe partitions. The bounded real-corpus proof is T030 and remains USER-run.

Phase 2C is implemented across shared I/O/runtime/renderer/capture code: strict shared LIT layouts,
global-clear sampling and `lit profile` evidence export, exact-build DBCD+WoWDBDefs Light* chain,
Z-up sky geometry, MCNR coordinate conversion, complete MCSH propagation, one-sided Lambert,
one-tile orthographic capture sidecars, grouped authored time variants, and fail-closed rights lanes.
LIT supplies recovered colors only; sun direction and MCSH attenuation remain explicitly authored.
Real-client image comparison and the final attenuation calibration remain USER-run proof, not claims
made by this implementation.

## Project Structure

```text
specs/103-image-only-reconstruction/{spec.md, plan.md, checklists/, tasks.md}

wow-viewer/data-harvester/
├── src/harvester/spec103/v7_model.py      # ported from gillijimproject_refactor (read-only ref)
├── src/harvester/spec103/v7_inputs.py     # 13-channel assembler from current signals
└── scripts/train_spec103_v7.py            # lean trainer (reuses bounded-trainer conveniences)
```

**Structure Decision**: new `spec103` lane under `data-harvester/`; port v7 in, do not touch the reference repo, do not touch V24.

## Complexity Tracking

No constitution violations. Deliberate decision: revive the proven single-model v7 rather than the non-functional two-stage V24. The honest caveat (v7 is not image-only; it needs the WDL prior) is recorded above; the image-only front-end is a separate later lane.
