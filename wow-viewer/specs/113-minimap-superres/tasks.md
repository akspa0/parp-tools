# Tasks: Minimap Super-Resolution (Real-ESRGAN)

**Input**: Design documents from `specs/113-minimap-superres/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Fixture-first tests are required for every trust/contract boundary (repo convention from
Specs 109-112). Heavy real-data renders and all training runs are **user-executed only** — tooling
prepares and prints the exact command; the assistant never launches them (FR-007/SC-006).

**Hard dependency**: Spec 112's final rebuild must be done — Kalimdor/Azeroth stores must carry
`minimap_rgb_authored` with real coverage before US1's alignment gate and US2's pairing have data.
Code for the detail render (US1) and the CPU-safe test suites can be written before that; the
real-render and alignment *runs* need it.

**Organization**: Tasks follow the three user stories. US1 is a hard gate that can FAIL (spec Edge
Case / SC-002) — US2/US3 are blocked behind an explicit pass. One phase validated before the next
(constitution: One Phase at a Time; each phase ≤10 tasks).

## Phase 1: Setup

- [ ] T001 Verify baseline: `dotnet build wow-viewer/WowViewer.slnx -c Debug` and
  `uv run python -m pytest tests/v50/ -q` (from `wow-viewer/data-harvester/`) both pass before any change

**Checkpoint**: Clean baseline; failures here are pre-existing and out of scope.

---

## Phase 2: Foundational — SR dependencies

- [ ] T002 Add the eval-only metric dependency `lpips` (and confirm `torch` CUDA wheels resolve) to
  `wow-viewer/data-harvester/pyproject.toml`; `uv sync` succeeds and `python -c "import lpips"` works.
  Do NOT add `basicsr`/`realesrgan` (research.md Decision 3 — RRDBNet is vendored, not depended on)

**Checkpoint**: `uv sync` clean; SR metric import works.

---

## Phase 3: User Story 1 — Detail-Rendered HR, Aligned to Authored LR (Priority: P1) — MVP / HARD GATE

**Goal**: A detail-preserving 1024 render that samples real texels, proven more detailed than a
bicubic upscale of the material-average render and spatially registered to the authored minimap.

**Independent Test**: On sampled tiles with both images, the detail render's high-frequency content
exceeds bicubic(material-average) with no moire, and authored↔detail registers under identity within
tolerance OR one fixed transform aligns all sampled tiles.

- [ ] T003 [US1] Add a detail texel-sampling path to `TerrainTextureSampler`
  (`TrySampleTexel(textureId, u, v, out color)`, bilinear read of the decoded BLP at the terrain UV)
  and a `detail` render mode to `TerrainMinimapCompositor` that uses it in `BlendLayers` for the
  1024 pass while the 256 pass stays material-average, in
  `wow-viewer/src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs` (FR-001, contract §1-3)
- [ ] T004 [P] [US1] `TerrainMinimapDetailRenderTests.cs` proving a detail-mode pixel varies with
  position across a synthetic high-frequency texture (not the texture average) and that a 1024
  detail render of a tiling texture shows no moire beyond a stated bound, in
  `wow-viewer/tests/WowViewer.Core.Tests/`
- [ ] T005 [US1] Expose the detail mode on the harvester: `synthetic-minimap --detail` selects the
  detail render for the 1024 pass and records `render_mode=detail` + texel repeat frequency in the
  synthesis manifest, in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-001, contract §6)
- [ ] T006 [US1] Implement the alignment analyzer: for a sample of tiles with both
  `minimap_rgb_authored` and the detail `minimap_rgb_1024`, downsample HR to LR resolution and score
  registration (NCC/phase-correlation) across all 8 dihedral transforms + a small translation;
  report per-tile + aggregate best transform, residual percentiles, and the `gate` verdict
  (`pass_identity`/`pass_with_transform`/`fail_inconsistent`), in
  `wow-viewer/data-harvester/src/harvester/v50/minimap_alignment.py` + thin
  `wow-viewer/data-harvester/scripts/v50_analyze_minimap_alignment.py` (FR-003, contract §gate)
- [ ] T007 [P] [US1] Alignment tests: a synthetically flipped/rotated HR is detected with the exact
  inverse transform and low residual; a per-tile-inconsistent set yields `fail_inconsistent`; an
  identity-aligned pair yields `pass_identity`, in
  `wow-viewer/data-harvester/tests/v50/test_minimap_alignment.py`
- [ ] T008 [US1] Add the SC-001 detail metric (high-frequency energy of detail render vs
  bicubic-upscaled material-average) to the alignment analyzer's report or a sibling helper, so the
  render's detail gain is measured, not assumed
- [ ] T009 [US1] **USER RUNS**: detail-render Kalimdor and Azeroth at 1024 (`--detail`) and run the
  alignment analyzer on a sample — commands + estimates in `quickstart.md` §1.2-1.3. Record the
  alignment report(s) under `wow-viewer/output/reports/v50/v50.1/`
- [ ] T010 [US1] Evaluate the US1 gate from the real reports: confirm SC-001 (detail gain, no moire
  on an eyeballed sample) and SC-002 (registration passes as identity or a single consistent
  transform). Record the verdict and any corrective transform in `quickstart.md`. **If
  `fail_inconsistent`: STOP — surface the finding, do not start US2** (spec Edge Case)

**Checkpoint**: US1 gate PASSED (SC-001 + SC-002) with a recorded corrective transform (or identity).
No US2 work otherwise.

---

## Phase 4: User Story 2 — Aligned SR Pair Set (Priority: P2)

**Goal**: A leak-safe (authored LR, detail HR) pair set from Kalimdor and Azeroth, honest coverage,
deterministic split.

**Independent Test**: The pair set contains only tiles with both sources; excluded tiles counted;
split disjoint by tile and deterministic; PVPZone02/Kalidar absent.

- [ ] T011 [US2] Implement the pair-set builder: from the two stores, include only tiles with both a
  populated `minimap_rgb_authored` and a successful detail `minimap_rgb_1024`; apply the US1
  corrective transform; assign a deterministic per-`source_group_id` within-map split; write a
  `v50-sr-pairset-v1` Zarr store + a schema-conformant summary, in
  `wow-viewer/data-harvester/src/harvester/v50/sr_pairset.py` + thin
  `wow-viewer/data-harvester/scripts/v50_build_sr_pairset.py` (FR-004/FR-005, data-model.md)
- [ ] T012 [P] [US2] Pair-set tests: coverage honesty (a tile missing either source is excluded and
  counted, never zero-filled), leak-free deterministic split, summary validates against
  `contracts/sr-pairset-and-run.schema.json`'s `pairset_summary`, in
  `wow-viewer/data-harvester/tests/v50/test_sr_pairset.py`
- [ ] T013 [US2] Build the real pair set (CPU-side, assistant-runnable once T009's stores exist) as
  `wow-viewer/output/datasets/v50/v50.1/sr-pairset-0_5_3_3368-v1.zarr` (`--val-fraction 0.15`,
  Kalimdor+Azeroth only); verify SC-003 and record pair/coverage counts in `quickstart.md`

**Checkpoint**: SC-003 proven. No US3 work before this.

---

## Phase 5: User Story 3 — Trained SR Model and Evaluation (Priority: P3)

**Goal**: An RRDBNet ×4 model that upscales real authored minimaps, beating bicubic and passing the
user visual gate on held-out tiles.

**Independent Test**: Held-out reference metrics + `beats_bicubic` recorded; SC-005 visual gate.

- [ ] T014 [US3] Vendor a compact RRDBNet ×4 generator (and the optional U-Net-SN discriminator +
  VGG-perceptual + adversarial losses behind a stage flag) in
  `wow-viewer/data-harvester/src/harvester/v50/sr_esrgan_model.py` (research.md Decision 3; single
  output, no multi-task, no shared weights — FR-006)
- [ ] T015 [P] [US3] Model/loss tests (CPU-safe): RRDBNet forward-pass shape (256→1024 ×4), grad
  sanity on a tiny fixture, and that the pair dataset feeds real authored LR (no synthetic
  degradation applied — research.md Decision 4), in
  `wow-viewer/data-harvester/tests/v50/test_sr_esrgan_train.py`
- [ ] T016 [US3] Implement the trainer + evaluator: pair-set-schema gate, real-pair loader (no
  synthetic degradation), patch-based training, stage `psnr` then optional `gan`, held-out
  PSNR/SSIM/LPIPS vs detail HR + `beats_bicubic` on the SC-004 detail metric, Kalimdor/Azeroth-only
  eval guard (FR-009), schema-conformant `v50-sr-run-v1` summary (FR-008), in
  `wow-viewer/data-harvester/src/harvester/v50/sr_esrgan_train.py` + thin
  `wow-viewer/data-harvester/scripts/v50_train_minimap_superres.py`
- [ ] T017 [P] [US3] Trainer contract tests (CPU-safe, no CUDA): out-of-scope-map eval refusal,
  summary fields present after a mocked short loop, bicubic-baseline comparison computed correctly on
  a fixture, GAN stage refuses to run without an `--init` PSNR checkpoint (contract §6), in
  `wow-viewer/data-harvester/tests/v50/test_sr_esrgan_train.py`
- [ ] T018 [US3] **USER RUNS** (stage 1): train the PSNR/L1 generator on the pair set — command +
  estimate in `quickstart.md` §3.1. Assistant reviews the summary against SC-004 and prepares the
  SC-005 side-by-side (model output vs authored LR vs bicubic on held-out Kalimdor/Azeroth tiles)
- [ ] T019 [US3] **USER RUNS** (stage 2, only if stage 1 is too smooth): GAN fine-tune from the
  stage-1 checkpoint — command in `quickstart.md` §3.2. Re-run SC-004 metrics and the SC-005 visual
  gate; watch for hallucinated structure (fails SC-005 even if perceptual metrics improve)

**Checkpoint**: SC-004 numerically proven, SC-005 user-judged. Feature complete (through stage 1 at
minimum; stage 2 conditional).

---

## Phase 6: Polish & Cross-Cutting

- [ ] T020 [P] Document the SR lane in `wow-viewer/docs/dataset-preparation-userguide.md` (detail
  render, alignment gate, pair set, training) and note it is a distinct image-SR model, not a
  terrain-signal model
- [ ] T021 Run the full focused suite (`tests/v50/` + the new C# `TerrainMinimapDetailRenderTests`
  filter), record exact results in `quickstart.md`, and update
  `wow-viewer/memory-bank/activeContext.md` + `progress.md` per Memory Bank Discipline

## Dependencies & Execution Order

- **Phase 1 → 2 → 3 → 4 → 5 → 6** strictly. US1 is a hard gate: T010's PASS is required before any
  US2 task. Real-data steps (T009, T013, T018/T019) depend on Spec 112's authored-minimap rebuild.
- Within US1: T003 before T004/T005; T006 before T007; T009 (user) needs T003+T005+T006; T010 needs
  T009. Within US3: T014 before T015/T016/T017; T018 (user) before T019 (user).
- Code-only tasks (T003-T008, T011-T012, T014-T017) can be written before the Spec 112 rebuild
  lands; only the real *runs* block on it.

## Parallel Opportunities

- T004 ∥ T006's implementation; T007 ∥ T008; T012 ∥ T014 skeleton; T015 ∥ T016 skeleton;
  T017 ∥ T018 prep; T020 ∥ T021.

## Implementation Strategy

US1 is the MVP and the risk: a detail render that's genuinely detailed AND registered to the
authored LR is the whole premise. If the alignment gate fails, that is a real, valuable finding —
it stops us before wasting a GAN training run on invalid pairs. Each user-run gate (T009, T013,
T018, T019) is a hard stop: prepared, printed, waited on.

## Task Summary

- **Total tasks**: 21 (Setup 1, Foundational 1, US1 8, US2 3, US3 6, Polish 2)
- **User-executed**: T009 (detail render + alignment), T018/T019 (training)
- **Blocked on Spec 112 rebuild**: T009, T013, T018, T019 (the real-data runs)
- **Suggested MVP**: T001-T010 (the detail render + the alignment gate verdict)
