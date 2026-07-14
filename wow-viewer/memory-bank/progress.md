# Progress — wow-viewer

Last updated: 2026-07-13 (evening)

## 2026-07-13 (evening) — Spec 103 Phases 0–4 agent work implemented

- **Contract pinned** (`specs/103-image-only-reconstruction/research-v7-contract.md`): real v7 aux
  channels 7-12 are height-min/max hints, liquid mask, liquid height, object mask, brush — the plan's
  alpha/holes guess was wrong and is corrected in plan.md. Missing/dropped WDL prior = 0.5 fill (v7's own
  fallback). Resolution decision: 256, `output_size` parameterized (the port's only deviation).
- **Lane ported + tested:** `src/harvester/spec103/{v7_model,v7_losses,v7_inputs}.py`; 7/7 CPU sanity
  tests (`tests/spec103/test_v7_sanity.py`): channel order, trestle residual, prior dropout, targets/bounds,
  forward/loss/backward, world-unit round trip.
- **Scripts prepared (USER runs the GPU/dotnet steps — quickstart.md):** synthetic known-height author
  (flat/ramp/ridge/crater/plateau, non-adjacent tiles; prints exact `map generate-blank` +
  `terrain-patch-adt` + `Capture render` commands) → 13-ch store builder (captured PNGs or labeled
  hillshade fallback) → lean trainer (holdout by any index column, AMP/EMA/warmup+cosine/early-stop/resume,
  `--wdl-prior-dropout` with per-epoch `val_no_prior`, `--height-hints gt|wdl|none`, `--loss v7|l1`,
  `--max-object-coverage` clean-tile selection, FR-011 run identity + peak VRAM) → batch inference
  (predicted height_257 npy + paired WDL lattice npz, `terrain-patch-adt`-compatible) → OBJ export →
  label-free harness (border agreement, plausibility, checkerboard/blockiness; `--gt-store` dev-only baselines).
- **Speckit synced same pass:** plan.md (pinned channel table, loss/object decisions, Phase 5 scoped
  deferred lanes T016/T019, implementation state), tasks.md (T001-T010, T012-T017, T019 checked;
  T011/T018 + training runs USER-blocked), quickstart.md new.

## 2026-07-13 — Pivot to Spec 103 (revive v7); image-only law established

- **New governing law** in Spec 103: input is one image; every signal is generated from it; validation is
  label-free. **V24 / Spec 094 dropped** as non-functional. `wdl_height_33` prohibited; the WDL prior is the
  verified `height257[::16]` / `[8::16]` transform. **Spec 102 M0 paused/superseded** but preserved
  (simple trainer + 42/42-green strict tests).

## Key facts for the next session

- Next step is entirely USER runs: quickstart §1 (synthetic authoring → dotnet generate/patch/capture →
  store → training), then T011 caveat catalog in research-v7-contract.md §8, then real-data run (§3).
- v7 reference (read-only): `gillijimproject_refactor/src/WoWMapConverter/scripts/{v7_model,train_v7,v7_losses,infer_v7}.py`.
- Real store = existing V18 `output/datasets/v18/3_3_5_12340.zarr` (5134 tiles; has minimap_rgb, height_257,
  normal_xyz, liquid_mask/height, object_precise_mask — FR-012 satisfied, no copy needed;
  `spec103_build_real_store.py` verifies and pins it).

## Durable boundaries

- `gillijimproject_refactor` read-only (port from, never edit). C# WDL reader + AlphaWdtWriter frozen.
- The USER runs all training/capture/heavy jobs (AGENTS RULE 0). Staged clients only; never `H:\CLIENTS`.
- Older M0 strict-target detail: `memory-bank/archive/2026-07-13-spec102-strict-target-detail.md`.
