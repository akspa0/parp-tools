# Tasks: Revive the v7 terrain regressor on current clean signals (Spec 103)

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Standing constraints (apply to every task)

- **The USER runs all training, capture, and heavy/GPU/harvest jobs.** The agent prepares scripts and exact commands only (AGENTS RULE 0).
- **Read-only reference**: `gillijimproject_refactor` is never modified. C# WDL reader and `AlphaWdtWriter` are frozen — used as-is, never edited.
- **Quick-and-dirty loss**: plain height regression, **no object-mask gating by default** (optional flag, off).
- **Never** resurrect `wdl_height_33`. The WDL prior is the verified `outer=height257[::16,::16]`, `inner=height257[8::16,8::16]`.
- V24 / Spec 094 is dropped — do not reference or revive it.

---

## Phase 1: Setup

- [x] T001 Create the lane: `wow-viewer/data-harvester/src/harvester/spec103/__init__.py` and `wow-viewer/data-harvester/tests/spec103/` (mirror the spec102 layout). *(2026-07-13)*

## Phase 2: Foundational — pin + port v7 (blocks all user stories)

- [x] T002 [P] Read `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` (read-only) and record the exact 13-channel assembly order and aux channels 7-12 in `wow-viewer/specs/103-image-only-reconstruction/research-v7-contract.md`. *(done — real aux set: height-min/max hints, liquid mask, liquid height, object, brush; the plan's alpha/holes guess was wrong)*
- [x] T003 [P] Read `gillijimproject_refactor/src/WoWMapConverter/scripts/v7_losses.py` and `infer_v7.py` (read-only) and record loss terms, output-head modes, and working resolution in `research-v7-contract.md`. *(done — resolution decision: 256, output_size parameterized)*
- [x] T004 Port `MultiChannelUNetV7` (architecture unchanged) into `wow-viewer/data-harvester/src/harvester/spec103/v7_model.py`. *(+ `v7_losses.py` ported verbatim)*
- [x] T005 Implement the 13-channel input assembler (minimap RGB, normal RGB, WDL prior at `::16`/`8::16`, aux 7-12) with WDL-prior channel-dropout support in `wow-viewer/data-harvester/src/harvester/spec103/v7_inputs.py`. *(dropout = v7's 0.5 missing-prior fill; `--height-hints gt|wdl|none`)*
- [x] T006 CPU sanity harness (no GPU run): forward/loss/backward, 13-ch input, output shape, trestle residual + prior-dropout path, in `wow-viewer/data-harvester/tests/spec103/test_v7_sanity.py`. *(7/7 green)*

## Phase 3 (US1): Terrain reconstruction — synthetic-first, then real

**Goal**: v7 reconstructs high-res terrain from `[minimap + normals + WDL prior + aux]`. **Independent test**: on synthetic tiles with known patterns, the output reconstructs the known height within tolerance; prior-dropout tiles still resolve.

- [x] T007 [US1] Write a synthetic-ADT authoring helper (known patterns: flat, ramp, ridge, crater, plateau) using existing ADT tooling as-is (AlphaWdtWriter frozen) in `wow-viewer/data-harvester/scripts/spec103_make_synthetic_adts.py`. *(script written; prints the exact `map generate-blank` / `terrain-patch-adt` commands; tiles non-adjacent so seam stitching never mutates a pattern)* — **USER runs the printed dotnet commands.**
- [x] T008 [US1] Prepare the WoWViewer minimap-capture command for the synthetic ADTs (existing capture path); documented in [quickstart.md](quickstart.md) §1d (perspective-camera caveat recorded; `--synthesize-minimaps` fallback exists). — **USER runs the capture.**
- [x] T009 [US1] Derive the WDL prior (`::16`/`8::16`) and assemble the 13-channel synthetic store in `wow-viewer/data-harvester/scripts/spec103_build_synthetic_store.py`. *(prior derived at batch time from height_257; analytic normals; store schema matches V18 array names)*
- [x] T010 [US1] Port a lean trainer `wow-viewer/data-harvester/scripts/train_spec103_v7.py` (complete-map/complete-pattern holdout, AMP, EMA, warmup+cosine, early-stop, resumable; quick-and-dirty loss, no object-mask gating by default; `--wdl-prior-dropout`; `--max-object-coverage` clean-tile selection; FR-011 run identity + peak VRAM in history.json). Command in quickstart §1f — **USER runs training.**
- [ ] T011 [US1] Verify known-pattern reconstruction on the synthetic run and catalog every caveat (resolution, channel order, trestle behavior, losses) in `research-v7-contract.md` §8. *(blocked on the USER training run)*
- [ ] T018 [US1] Add a deterministic fixed-light terrain-shadow capture for the synthetic tiles (Spec 102 N011 capture/determinism contract; objects/textures/liquids off) and measure the shadow↔height correlation against the known synthetic height; record findings in `research-v7-contract.md`. **USER runs** the capture. *(quickstart §4)*
- [x] T012 [US1] With synthetic caveats resolved, build/point at a real clean store in `wow-viewer/data-harvester/scripts/spec103_build_real_store.py`. *(validator written: V18 store already pairs minimap + height_257 + normals + liquid + object mask — FR-012 satisfied without a copy; prohibits `wdl_height_33`)*
- [x] T020 [US1] **Curate + bucket the corpus (FR-013, Principle #5)** in `wow-viewer/data-harvester/scripts/spec103_curate_dataset.py`: drop object-contaminated / blank / height-normal-mismatch tiles; write an auditable manifest with per-tile reasons + map/height-regime buckets; wire `--curation-manifest` into the trainer (object-drop is the default). *(V18: 5134 → 3131 kept; relief calc validated r=0.57 vs height-std)*
- [x] T013 [US1] Prepare the real-data training command (holdout, prior-dropout, resumable) using `train_spec103_v7.py`. *(quickstart §3b)* — **USER runs training.**

## Phase 4 (US2): Review + label-free validation

**Goal**: judge the generated terrain the way it will be used. **Independent test**: image/prior-fed inference on held-out inputs is self-consistent (border agreement, plausibility, no artifacts) with no labels read.

- [x] T014 [US2] Mesh/OBJ export of predicted terrain for eyeball review (existing export_terrain_obj mesh convention) in `wow-viewer/data-harvester/scripts/spec103_export_mesh.py`. *(plus `infer_spec103_v7.py`, whose per-tile output is `terrain-patch-adt`-compatible for in-viewer review)* — **USER runs** inference/render.
- [x] T015 [US2] Label-free self-consistency harness (adjacent-tile border agreement, plausible height range/gradients, checkerboard + chunk-blockiness detection); ground-truth L1 + prior/flat baselines kept as a dev-only diagnostic (`--gt-store`), in `wow-viewer/data-harvester/scripts/validate_spec103_labelfree.py`.

## Phase 5: Polish & deferred lanes

- [x] T016 [P] Record the deferred follow-on lanes (image-only `minimap → WDL-prior` front-end; synthetic-universality scale-up; output-space object segmentation + inpaint) as scoped notes in the plan; no implementation now. *(plan.md Phase 5)*
- [x] T019 [P] Record the teacher→student distillation lane: a teacher trained on rich clean synthetic signals (minimap + normals + WDL prior + terrain shadow → height, known GT) distilled into an image-only student. Scoped note only; no implementation now. *(plan.md Phase 5)*
- [x] T017 Update `wow-viewer/memory-bank/activeContext.md` and `progress.md` with the Spec 103 v7-revival state (compress hard). *(2026-07-13)*

## Remaining (USER-blocked)

- T011: synthetic training run + caveat catalog (research-v7-contract.md §8).
- T018: shadow capture + shadow↔height correlation.
- Real-data training run (T013's command) and label-free acceptance on its holdout.

---

## Dependencies

- Phase 1 → Phase 2 → Phase 3 → Phase 4. Phase 5 is independent polish.
- T002/T003 are parallel (different reads). T004 depends on T002/T003; T005 on T002; T006 on T004+T005.
- Synthetic PoC (T007-T011) must pass before the real-data run (T012-T013) — the whole point of synthetic-first.
- US2 (validation) depends on at least one trained checkpoint from US1.

## MVP scope

**US1 via the synthetic PoC (T001-T011)** is the MVP: it proves v7 reconstructs known terrain from clean synthetic signals, cataloguing every caveat, before any real-data or validation work.
