# Progress — wow-viewer

Last updated: 2026-07-13 (v8 lean architecture is the primary Spec 103 lane)

## 2026-07-14 — Procedural synthetic dropped as a gate; real data is the proving ground

- **USER decision:** procedural patterns (flat/ramp/ridge/crater/plateau) don't replicate real
  terrain and the WDL prior trivially solves them (v8 smoke run: l1_global ≈ 0.0006 at init and
  at best — the metric is prior-dominated, not learning). The intended synthetic lane =
  **synthesize signals from real terrain** (deterministic shadow/hillshade of real height, T018),
  not invented terrain. Real-data v8 run (quickstart §3) is now the soundness test; ready to run
  (curation manifest 2253 kept, Azeroth 332-tile holdout).
- **Trainer hardening from the smoke run:** batch clamped to train-set size; `drop_last` only
  when ≥2 full batches (tiny sets no longer silently produce 0 train batches); hard exit on an
  empty train loader; loud warning when planned steps are too few for `--ema-decay` (the
  validated EMA model would otherwise stay ~= its initial weights). 13/13 tests green.

## 2026-07-13 (late) — v8 lean architecture implemented; primary lane by USER decision

- **Why:** v7's 117.06M-param U-Net (73% of params at 8×8–16×16; 119.9 GFLOPs @256) meant ~26 h
  before a training run proved sound or not. USER decision: modern lean arch is primary, no
  baseline-first gatekeeping; v7 kept for ablation only.
- **What:** [`v8_model.py`](wow-viewer/data-harvester/src/harvester/spec103/v8_model.py)
  `V8LeanUNet` (`v8-lean-convnextv2-v1`): ConvNeXt-V2 blocks (7×7 reflect DW + GRN), widths
  32-64-128-256-384, pixel-shuffle decoder, pooled global-context mixer + bounds head.
  **Measured 6,204,198 params (25 MB) / 16.4 GFLOPs @256** — 18.9× / 7.3× less than v7. Head,
  trestle residual, clamp modes copied verbatim; the 13-ch contract, `combined_loss`, trainer,
  inference, previews, mesh export, and label-free harness run unchanged.
- **Wiring:** trainer `--arch v8|v7` (v8 default), arch recorded in checkpoints + run identity;
  `infer_spec103_v7.py` auto-resolves arch (pre-v8 checkpoints default to v7). Tests: 6 new v8
  CPU sanity tests incl. a <10M-param budget guard; 13/13 spec103 suite green. Docs synced
  (plan, tasks T021, quickstart, research-v8-optimization.md = survey + decision record).

## 2026-07-14 — Curation default tightened (drop ANY object tile)

- **Curation default tightened:** `--max-object-coverage` default is now `0.0` (drop ANY object) in both
  [`spec103_curate_dataset.py`](wow-viewer/data-harvester/scripts/spec103_curate_dataset.py:59) and
  [`train_spec103_v7.py`](wow-viewer/data-harvester/scripts/train_spec103_v7.py:198). Was 0.02.
  The model architecture is **unchanged** (13 channels) — this is a tile *selection* change only, not an
  architecture change. Object tiles are impossible height targets (spec Principle #5: height under an
  object is occluded in the minimap), so they are dropped, not learned.
- **Tests:** 7/7 CPU sanity green. Docs synced (research-v7-contract, plan, quickstart, spec FR-013, tasks).

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
