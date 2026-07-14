# Active Context — wow-viewer

Last updated: 2026-07-14 (curation default tightened to drop ANY object tile)

## Current target — Spec 103: revive the v7 terrain regressor on clean signals

- **Governing law (image-only):** the only deployment input is one image tile. Every other signal is generated from it; no model reads a ground-truth signal at inference; downstream trains on generated (not ground-truth) upstream; a target the image cannot support is invalid. Validation is **label-free** (self-consistency), never label-comparison. See `specs/103-image-only-reconstruction/spec.md`.
- **Implemented (agent side, 2026-07-13):** v7 contract pinned in `specs/103-.../research-v7-contract.md`;
  lane ported to `data-harvester/src/harvester/spec103/` (`v7_model.py` — only deviation: `output_size`
  parameterized, 256 default; `v7_losses.py` verbatim; `v7_inputs.py` 13-ch assembler). 7/7 CPU sanity
  tests green (`tests/spec103/`). All Phase 2–4 scripts written: `spec103_make_synthetic_adts.py`,
  `spec103_build_synthetic_store.py`, `train_spec103_v7.py`, `infer_spec103_v7.py`,
  `spec103_build_real_store.py`, `spec103_export_mesh.py`, `validate_spec103_labelfree.py`.
  Commands: `specs/103-.../quickstart.md`. **Blocked on USER runs**: capture, training, T011 caveat catalog, T018 shadow capture.
- **Pinned 13-ch truth (plan's old aux guess was wrong):** 0-2 minimap, 3-5 normals (both recovery-attenuated
  ×0.85/×0.70 then ImageNet-normalized), 6 WDL prior (outer 17×17 only, align_corners=True, **0.5 fill when
  missing — dropout reuses this**), 7-8 tile height min/max hint planes (`--height-hints gt|wdl|none`),
  9 liquid mask, 10 liquid height, 11 object mask, 12 brush (zeros). Loss reads 9/11/12 — order is load-bearing.
  **The model architecture is unchanged (13 channels).**
- **WDL prior = verified transform:** `outer = height257[::16,::16]`, `inner = height257[8::16,8::16]`.
  Derived at batch time from `height_257` — no reharvest, nothing stored. **Never** `wdl_height_33`.
- **Procedural-synthetic PoC DROPPED as a gate (USER decision 2026-07-14):** flat/ramp/ridge/crater
  patterns don't replicate real terrain, and the WDL prior trivially solves them (v8 run: init
  l1_g ≈ 0.0006 — nothing to learn on the global channel). The intended "synthetic" lane was always
  **signals synthesized FROM real terrain** (deterministic shadow/hillshade renders of real height —
  T018's reinterpretation), never invented terrain. The 10-tile procedural store survives only as a
  pipeline smoke test. **Soundness test = the real-data v8 run** (quickstart §3; everything ready:
  V18 store + curation manifest, 2253 kept, Azeroth holdout 332/1921).
- **Synthetic chain (kept for smoke tests; all existing C# used as-is):** `map generate-blank` (Inspect tool) → known-height .npy →
  `terrain-patch-adt` (Converter) → `Capture render` (perspective-camera caveat recorded) or
  `--synthesize-minimaps` hillshade fallback. Synthetic tiles are placed non-adjacent so the patcher's seam
  stitching never mutates a known pattern.
- **Curation is mandatory (FR-013 / Principle #5), clean-by-default:** object tiles are impossible height
  targets (terrain under an object is occluded in the minimap), so they are DROPPED, not learned — the user
  was right and I initially defaulted keep-all in violation of the spec; fixed. `spec103_curate_dataset.py`
  buckets every tile and drops object_contaminated / blank_minimap / height_normal_mismatch, writes an
  auditable `curation_manifest.parquet` (+ map/height-regime buckets) the trainer consumes via
  `--curation-manifest`. **Default `--max-object-coverage 0.0`** (drop ANY object; was 0.02). V18 at 0.0:
  5134 → 2650 kept. `1.0` is v7-faithful keep-all ablation only. Trainer reports `val_no_prior` every epoch (prior-dropout robustness).
- **v8 is the PRIMARY architecture (USER decision 2026-07-13; implemented + tested):**
  `V8LeanUNet` (`src/harvester/spec103/v8_model.py`, ConvNeXt-V2 blocks, pixel-shuffle decoder,
  global-context mixer) — measured **6.2M params / 16.4 GFLOPs @256** vs v7's 117.06M / 119.9
  (73% of v7's params sat at 8×8–16×16). Identical 13-ch/trestle/bounds contract → loss, trainer,
  inference, previews, harness all unchanged. Trainer default `--arch v8` (`--arch v7` = 117M
  ablation only, NOT a gate); checkpoints record arch, inference auto-resolves. 13/13 CPU tests.
  Driver: v7's ~26 h time-to-signal was unacceptable; v8 targets minutes on synthetic. Survey +
  rationale: `specs/103-image-only-reconstruction/research-v8-optimization.md` (T021).
  Excluded: DA-family (blacklist), diffusion predictors, 100M+ depth foundations.
- **The USER runs all training/capture/heavy jobs.** The agent prepares scripts + commands only (AGENTS RULE 0).

## Dropped / paused

- **V24 / Spec 094 is NOT functional — dropped.** Do not revive it.
- **Spec 102 M0 object-mask lane is paused/superseded** by Spec 103. Preserved: simple M0 trainer
  (`train_spec102_m0_simple.py`) + inference; strict fragment-trace target + 42/42-green tests remain inactive.

## Boundaries

- New work in `wow-viewer/`; `gillijimproject_refactor` is read-only reference (port from, never edit).
- Staged clients only: `output/tmp/wowarchive-clients/`. Never `H:\CLIENTS`.
- Spec 080 owns the UI lane.
