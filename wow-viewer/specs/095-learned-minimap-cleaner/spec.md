# Feature Specification: Learned Minimap Cleaner (V24.1)

**Feature Branch**: `095-learned-minimap-cleaner`
**Created**: 2026-07-07
**Status**: Draft
**Owner**: wow-viewer
**Supersedes placeholder**: Spec 094 FR-010 / Open Question 2 (the NumPy median-fill cleaner)

**Input (user direction, 2026-07-07)**: "you're missing loss signals to clean up the minimap inputs, everything important is fucking ignored in my original spec." Spec 094 made minimap cleaning a *first-class input concern* (US2, FR-009/010) but the implementation placeholdered it as a pure-NumPy median-fill with **no loss signal**. Spec 094's own Open Question 2 deferred a model-based cleaner to "a different spec." This is that spec.

---

## Problem Statement

V24's minimap input is noisy: object roofs, shadows, and compositing artifacts contaminate the terrain signal. Spec 094 correctly identified this and made cleaning first-class, but shipped a NumPy median-fill placeholder ([`clean_minimap.py`](../../../data-harvester/src/harvester/v24/clean_minimap.py:1)). The placeholder:

- Has **no learned component** — it cannot improve with data.
- Has **no loss signal** — nothing in training drives the minimap toward a cleaner input.
- Median-fills object pixels from 8-connected neighbours, which blurs structure and leaves compositing artifacts untouched on builds without `no_object_minimap` (i.e. `3_3_5_12340`).

The result: Stage A and Stage B train on a noisy, unimproved input. The minimap→WDL correlation Stage A must learn is corrupted by object roofs that the cleaner only smears, not removes.

V24.1 replaces the placeholder with a **small learned cleaner that has explicit loss signals** for removing objects and reconstructing object-free terrain appearance.

---

## Relationship To Existing Specs

- **Amends**: Spec 094 (V24). The NumPy cleaner ([`clean_minimap.py`](../../../data-harvester/src/harvester/v24/clean_minimap.py:1)) remains as the fallback; the learned cleaner is an opt-in upgrade via a `--cleaner-checkpoint` flag on `TileSource`.
- **Re-uses**: V18 (Spec 001) substrate — `minimap_rgb`, `object_precise_mask`, `alpha_256`, `normal_xyz`, and `no_object_minimap` (present on `0_5_3_3368` only).
- **Re-uses**: the V18 curation manifest (`kept_tiles.parquet`) for training-tile selection, identical to V24's curated run.
- **Does NOT touch**: V22 per-object mask data (still suspect per the user). V24.1 uses V18's tile-level `object_precise_mask` only.
- **Does NOT change**: Stage A / Stage B architectures. V24.1 only changes the *input* they receive (learned-cleaned vs NumPy-cleaned minimap).
- **Does NOT fix**: the `holes_16` polarity defect (separate spec in `WowViewer.Core.IO`).

---

## Out Of Scope (Explicit)

- V22 per-object precise mask data. Still suspect; out of scope.
- Changing Stage A / Stage B model architectures or param caps.
- DepthAnything / DA-V2 / DPT / LoRA / GPCT / CAI / RunPod. V24.1 is small and local, consistent with V24.
- Editing `gillijimproject_refactor` (RULE 1) or any C#.
- Multi-task terrain prediction. V24.1 predicts one thing: a cleaned minimap (+ an auxiliary object mask).
- Producing game-client files.

---

## The Model

### Architecture

A small U-Net at 256×256, ≤ 1M trainable params (consistent with V24's small-model philosophy). Four-layer encoder / four-layer decoder with skip connections. Two output heads:

| Head | Shape | Activation | Purpose |
|---|---|---|---|
| `object_mask` | (256, 256, 1) | sigmoid | predicts per-pixel object probability (auxiliary supervision + inpaint gate) |
| `cleaned_minimap` | (256, 256, 3) | sigmoid | the object-free minimap reconstruction (the Stage A/B input) |

### Inputs (≈10 channels at 256×256)

1. `minimap_rgb` (256, 256, 3) — raw V18 minimap, float32 [0,1].
2. `object_precise_mask` (257→256 max-pool, 1 channel) — the tile-level object mask (the *supervision* for the mask head, also fed as input so the model knows where to inpaint).
3. `alpha_256` (256, 256, 4) — terrain alpha layers (help the model distinguish terrain from object compositing).
4. `normal_xyz` (257→256 crop, 3 channels) — terrain normals (terrain structure cue independent of object roofs).

The object mask is fed as an input **and** supervised as a target. Feeding it lets the model condition its reconstruction on known object locations at inference; supervising it forces the encoder to learn object semantics.

---

## Loss Signals

This is the core of the spec — the "loss signals to clean up the minimap inputs." The total loss is a weighted sum of four terms. Each term has a documented weight and purpose (RULE 6).

### L1 — Mask loss (all builds)

BCE + Dice between the predicted `object_mask` head and V18's `object_precise_mask` (downsampled to 256 via the same max-pool rule as [`clean_minimap.object_mask_256`](../../../data-harvester/src/harvester/v24/clean_minimap.py:24)).

- **Weight**: `w_mask = 1.0`
- **Purpose**: the model learns to find objects from minimap appearance. This is the only supervision available on `3_3_5_12340` (no `no_object_minimap`).
- **Why both BCE and Dice**: BCE for per-pixel gradient; Dice for class-imbalance robustness (object pixels are a minority).

### L2 — Reconstruction loss (builds with `no_object_minimap` only)

L1 between the predicted `cleaned_minimap` and V18's `no_object_minimap` (the viewer's object-free composite, present on `0_5_3_3368`).

- **Weight**: `w_recon = 1.0` when `no_object_minimap` is present, else 0.
- **Purpose**: direct "produce an object-free minimap" supervision. This is the gold-standard cleaning target — the viewer already renders terrain without objects.
- **Gating**: only on tiles where `no_object_minimap` exists and is non-zero.

### L3 — Identity-preserving loss (all builds)

L1 between the predicted `cleaned_minimap` and the raw `minimap_rgb` **on non-object pixels** (where `object_precise_mask < 0.5`).

- **Weight**: `w_identity = 0.5`
- **Purpose**: the model must not hallucinate terrain where the minimap already shows clean terrain. On non-object pixels, cleaned == raw. This prevents the reconstruction head from drifting on builds without `no_object_minimap`.
- **Gating**: masked to non-object pixels only.

### L4 — Inpainting smoothness loss (all builds)

A Sobel-gradient consistency penalty on the predicted `cleaned_minimap` **on object pixels** (where `object_precise_mask ≥ 0.5`): the inpainted region must be locally smooth and blend with its non-object neighbours.

- **Weight**: `w_smooth = 0.1`
- **Purpose**: on `3_3_5_12340` (no `no_object_minimap`), this is the only signal that drives the reconstruction head to *do something reasonable* on object pixels rather than copy the raw object roof. It penalises high-frequency artifacts in inpainted regions.
- **Form**: `mean(|Sobel(cleaned)| * object_mask)` — minimises gradient energy inside object regions.

### Total

```
loss = w_mask * L1 + w_recon * L2 + w_identity * L3 + w_smooth * L4
```

All weights are documented constants in the training script config and recorded in the checkpoint. No weight is tuned without a validation run (RULE 6).

---

## Training Data

- The curated V18 corpus (`kept_tiles.parquet` from `build_v18_curation_manifest`), the same manifest V24 now uses.
- Both builds where available:
  - `0_5_3_3368` — provides `no_object_minimap` (strong L2 reconstruction supervision).
  - `3_3_5_12340` — mask + identity + smoothness supervision only (L2 gated to 0).
- Tile selection joins on `(build, tile_id)` with `keep == True`, identical to V24's `--curation-manifest` path.
- Train/val split: 80/20, stratified by build and map.

---

## Integration With V24

The learned cleaner is **opt-in**, replacing the NumPy median-fill in [`TileSource.load`](../../../data-harvester/src/harvester/v24/tiles.py:74) when a cleaner checkpoint is provided:

- `TileSource(v24_path, v18_path, cleaner_checkpoint=None)` — when `cleaner_checkpoint` is set, `load()` runs the learned cleaner to produce `cleaned_minimap`; otherwise the NumPy cleaner runs (current behaviour, unchanged).
- Stage A / Stage B are **unchanged** — they consume `record.cleaned_minimap` regardless of source.
- `train_v24_stage_a.py` / `train_v24_stage_b.py` / `infer_v24_stage_b.py` gain an optional `--cleaner-checkpoint` flag that threads through to `TileSource`.

The NumPy cleaner remains the default and the test-suite fallback (so the 30/30 v24 pytest stays green without a trained cleaner).

---

## User Stories & Acceptance Scenarios

### US1 — Trained cleaner produces a better cleaned minimap (P1)

1. **Given** a trained cleaner checkpoint, **When** it runs on a held-out `0_5_3_3368` tile with `no_object_minimap`, **Then** reconstruction L1 (cleaned vs `no_object_minimap`) is lower than the NumPy median-fill's L1 on the same tile.
2. **Given** a trained cleaner checkpoint, **When** it runs on a held-out tile, **Then** the mask head's IoU vs `object_precise_mask` is ≥ 0.7.
3. **Given** a tile with no objects (`object_precise_mask` all-False), **When** the cleaner runs, **Then** the cleaned minimap is identical (within 1e-3) to the raw minimap (identity preservation).

### US2 — Downstream V24 models improve with the learned cleaner (P1)

1. **Given** Stage A trained on the curated corpus with the **NumPy** cleaner, and Stage A trained with the **learned** cleaner (same epochs, same seed), **When** both are validated, **Then** the learned-cleaner Stage A has lower real-cell L1 on the WDL prior.
2. **Given** the full V24 pipeline (Stage A + Stage B) with the learned cleaner, **When** `validate_v24.py` runs, **Then** Stage B final L1 is lower than (or within noise of) the NumPy-cleaner baseline. If it does not improve, the cleaner is not worth promoting and the finding is reported honestly.

### US3 — Determinism + envelope (P2)

1. **Given** two cleaner inference runs with different seeds, same checkpoint, same input, **Then** bit-identical output.
2. **Given** the cleaner runs on a 6 GB consumer GPU at fp16, **Then** peak VRAM < 1 GB and wall-time < 0.5 s/tile.

---

## Requirements

- **FR-001**: A new module `harvester/v24/cleaner_model.py` with the U-Net (≤ 1M params), the two heads, and `build_input` / `build_target` / `loss` functions.
- **FR-002**: A new script `scripts/train_v24_cleaner.py` that trains the cleaner on the curated V18 corpus with the four loss terms. Records all loss weights in the checkpoint config.
- **FR-003**: A new script `scripts/infer_v24_cleaner.py` that runs the cleaner on a tile (or a store) and writes the cleaned minimap NPZ + predicted object mask.
- **FR-004**: `TileSource` gains an optional `cleaner_checkpoint` parameter; when set, `load()` uses the learned cleaner. The NumPy cleaner remains the default.
- **FR-005**: `train_v24_stage_a.py`, `train_v24_stage_b.py`, `infer_v24_stage_b.py` gain an optional `--cleaner-checkpoint` flag threaded to `TileSource`.
- **FR-006**: The cleaner is deterministic under `torch.manual_seed + eval + use_deterministic_algorithms(True)`.
- **FR-007**: The cleaner fits on a 6 GB consumer GPU at fp16: peak VRAM < 1 GB, inference wall-time < 0.5 s/tile.
- **FR-008**: pytest `tests/v24/test_cleaner_model.py` — shape, param cap (≤ 1M), determinism, loss-decreases-on-2-epoch-smoke, identity-preservation on no-object tile.

---

## Success Criteria

- **SC-001**: The trained cleaner's reconstruction L1 vs `no_object_minimap` on held-out `0_5_3_3368` tiles is lower than the NumPy median-fill baseline.
- **SC-002**: The mask head IoU vs `object_precise_mask` is ≥ 0.7 on held-out tiles.
- **SC-003**: Stage A real-cell L1 on the WDL prior is lower with the learned cleaner than with the NumPy cleaner (same epochs/seed/corpus).
- **SC-004**: The cleaner is deterministic (bit-identical across seeds) and fits in < 1 GB VRAM / < 0.5 s/tile.
- **SC-005**: If SC-003 does not improve, the finding is reported honestly and the cleaner is not promoted to default.

---

## Risks

- **Risk 1 (medium)**: On `3_3_5_12340` (no `no_object_minimap`), the reconstruction head has only mask + identity + smoothness supervision. The cleaned minimap on object pixels may not match the "true" object-free appearance. Mitigation: the mask head is the primary deliverable on this build; the reconstruction head's quality is bounded by L4 smoothness, and SC-003 (downstream Stage A improvement) is the real gate.
- **Risk 2 (medium)**: The learned cleaner may not improve downstream Stage A/B L1 if the NumPy cleaner was already "good enough" for the WDL-lattice-resolution target. Mitigation: SC-005 — report honestly, do not promote if no improvement.
- **Risk 3 (low)**: `no_object_minimap` on `0_5_3_3368` is viewer-rendered and may carry its own compositing artifacts. Mitigation: L2 is weighted 1.0 but L3 (identity) anchors non-object pixels to the raw minimap, so the model cannot drift entirely toward the viewer render.
- **Risk 4 (low)**: Object masks are coarse (tile-level, 257→256). Some object pixels will be missed. Mitigation: the mask head learns from appearance, not just the coarse mask, so it can refine; per-object masking remains a future lane blocked on V22 data.

---

## Assumptions

- V18's `object_precise_mask` is a usable (if coarse) supervision signal for object location. The V22 audit (Spec 094 amendment A8) confirmed V24's input signals are sound.
- `no_object_minimap` on `0_5_3_3368` is a valid object-free target. The viewer is the working renderer (per the existing NumPy cleaner's preference for it).
- The curated corpus provides enough object-bearing tiles for the mask head to learn. The curation manifest's `object_cov` column can be used to ensure object coverage in the training split.

---

## Implementation Order (for the tasks skill)

1. `cleaner_model.py` — model + build_input/build_target/loss (FR-001).
2. `tests/v24/test_cleaner_model.py` — shape, params, determinism, identity (FR-008).
3. `train_v24_cleaner.py` — training script with the four loss terms (FR-002).
4. `infer_v24_cleaner.py` — inference script (FR-003).
5. `TileSource.cleaner_checkpoint` integration (FR-004) + `--cleaner-checkpoint` flags on Stage A/B train/infer (FR-005).
6. Train the cleaner on the curated corpus (both builds).
7. Validate SC-001..SC-005; retrain Stage A/B with the learned cleaner; compare downstream L1 vs the NumPy-cleaner baseline.

---

## End of Spec

This spec is bounded, small (≤ 1M params), local, and re-uses the V18 substrate + curation manifest V24 already consumes. The hard precondition — minimap cleaning must be driven by **loss signals**, not a fixed NumPy placeholder — is the entire point. The four loss terms (mask, reconstruction, identity, smoothness) are the "loss signals to clean up the minimap inputs" the user identified as missing from Spec 094.