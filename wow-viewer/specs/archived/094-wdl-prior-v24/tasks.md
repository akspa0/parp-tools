# Tasks: 094-wdl-prior-v24

**Feature Branch**: `094-wdl-prior-v24`
**Created**: 2026-07-06
**Owner**: wow-viewer
**Source**: [`./spec.md`](./spec.md), [`./plan.md`](./plan.md), [`./data-model.md`](./data-model.md), [`./research.md`](./research.md), [`./checklists/requirements.md`](./checklists/requirements.md)

Tasks are organized by phase (matching the plan.md phases) and within each phase, by user story where applicable. Every task is bite-sized (one concern), independently testable, and follows the strict checklist format.

## Format

- `- [ ] [TaskID] [P?] [Story?] Description with file path`
- `[P]` = parallelizable (different files, no dependencies on incomplete tasks)
- `[US1]`, `[US2]`, etc. = user story label (from spec.md)
- Setup / Foundational / Polish phases have NO story label

> **Execution log 2026-07-06**: implemented with the spec's Implementation
> Amendments (A1–A8) in force. Notable deltas from the original task wording:
> the shim resolves WDLs from MPQs via `NativeMpqService` (no loose `.wdl`
> files exist) and is batch-first; the V24 store uses paired outer/inner
> arrays; Stage A is residual-over-synth (RULE 7) with two grid heads; the
> added V22 audit lane landed as `scripts/audit_v22_dataset.py` +
> `docs/architecture/v22-dataset-audit-2026-07-06.md`.

## Phase 1: Setup (Project Initialization)

- [x] T001 [P] Create `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/` directory tree
- [x] T002 [P] Create empty `wow-viewer/data-harvester/src/harvester/v24/__init__.py`
- [x] T003 [P] Create empty `wow-viewer/data-harvester/tests/v24/__init__.py`

## Phase 2: Foundational (C# WDL Reader Audit + C# CLI Shim)

The C# shim is the foundation for every Python wrapper (synth WDL, real WDL, merged prior). It must be built and validated before any Python wrapper.

- [x] T004 [US1] Locate the existing C# WDL reader in `wow-viewer/src/core/WowViewer.Core.IO/` and document the public API surface (class name, method signature, return type) at `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-06.md`
- [x] T005 [US1] Locate the existing C# terrain→WDL path in `wow-viewer/src/` (used by `WoWViewer`'s "click on map to spawn" visualization) and document the public API surface at `wow-viewer/docs/architecture/wdl-reader-shape-audit-2026-07-06.md` (it is `WdlWriter.ExtractTileHeightsFromAlpha`)
- [x] T006 [US1] Audit real MARE reads for both target builds (687 tiles 3.3.5 / 685 tiles 0.5.3, both 17×17+16×16 int16, MVER 18) and record shape + dtype in the audit doc
- [x] T007 [P] Create `wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead/WowViewer.Tool.WdlRead.csproj` (a new .NET project that references `WowViewer.Core.IO`)
- [x] T008 [P] Create `Program.cs` `read` mode (amended per A2/A3: `--client-root/--map` MPQ resolution + `--wdl` loose-file escape hatch; batch whole-map output)
- [x] T009 Extend `Program.cs` with the `synth --height <npz> [--liquid <npz>]` mode (batch-first; nearest-non-liquid resampling in the shim)
- [x] T010 [P] Add `--help` and `--version` flags to the shim
- [x] T011 Add the shim project to `wow-viewer/WowViewer.slnx`
- [x] T012 Verify `dotnet build` succeeds with 0 errors
- [x] T013 Verify `--help` prints usage
- [x] T014 Verify real reads on both staged builds produce non-zero NPZs with the audited shape
- [x] T015 Verify synth mode emits the audited shape (validated against the real WDL: 100% of cells within 1.0)

## Phase 3: User Story 1 — Synthetic WDL Builder Python Wrapper (P1)

- [x] T016 [US1] Create `harvester/v24/synth_wdl.py` (`build_synth_wdl` / `build_synth_wdl_batch`, thin wrapper over `harvester/v24/shim.py`)
- [x] T017 [P] [US1] Add `tests/v24/test_synth_wdl.py` (single + batch + liquid-resample cases, skipped if shim unbuilt)
- [x] T018 Verify: passes (part of the 30/30 v24 suite)

## Phase 4: User Story 1 (cont.) — Real WDL Reader Python Wrapper (P1)

- [x] T019 [US1] Create `harvester/v24/wdl_reader.py` (`read_wdl_mare` single-tile + `read_wdl_map_tiles` batch entry point; MAHO slot always `None` per A1)
- [x] T020 [P] [US1] Add `tests/v24/test_wdl_reader.py` against the real `3_3_5_12340` staged client (687 tiles), skipped if client/shim absent
- [x] T021 Verify: passes on the real staged client

## Phase 5: User Story 1 (cont.) — Merged WDL Prior Builder + V24 Store (P1)

- [x] T022 [US1] Create `harvester/v24/merged_wdl_prior.py::build_merged_wdl_prior` (paired outer/inner grids per A5, inclusive `<=` threshold)
- [x] T023 [P] [US1] Add `tests/v24/test_merged_wdl_prior.py` (5 sub-tests incl. inclusive-threshold boundary)
- [x] T024 [P] [US1] Create `scripts/build_wdl_prior.py` with `build`/`infer` subcommands (+ `--min-height-std` filter, `--maps`)
- [x] T025 [P] [US1] Add `tests/v24/test_build_wdl_prior.py` (5-tile synthetic V18 store incl. one audit-empty tile)
- [x] T026 Verify: `uv run python -m pytest tests/v24 -m v24 -q` → 30 passed
- [x] T027 Bounded real-data builds run: Northrend 50 tiles (`3_3_5_12340`, 100% real coverage) and Azeroth 50 tiles (`0_5_3_3368`, 85%/15% real/synthetic) — both SC-001 PASS
- [x] T028 Create `scripts/inspect_v24_dataset.py` (`summary`/`tile` subcommands)
- [x] T029 Verify: `inspect_v24_dataset.py summary` reports coverage stats correctly on both bounded builds

## Phase 6: User Story 2 — Minimap Cleaner (P1)

- [x] T030 [P] [US2] Create `harvester/v24/clean_minimap.py::clean_minimap` (8-connected median fill at native 256², amended to prefer `no_object_minimap` where present per A4)
- [x] T031 [P] [US2] Create `scripts/clean_minimap.py` CLI
- [x] T032 [P] [US2] Add `tests/v24/test_clean_minimap.py` (5 sub-tests incl. `no_object_minimap` preference + corner-downsample)
- [x] T033 Verify: passes (part of the 30/30 v24 suite)
- [x] T034 Exercised indirectly via `TileSource.load` in the real bounded builds (T027) — every loaded tile is cleaned

## Phase 7: User Story 3 — Stage A (Minimap → WDL Prior Correlation) (P1)

- [x] T035 [US3] Create `harvester/v24/stage_a.py` (small U-Net, 337,485 params, residual-over-synth-quincunx per RULE 7 — amended from "predict the prior directly")
- [x] T036 [P] [US3] Add `tests/v24/test_stage_a.py` (shape, param cap, zero-init residual identity, determinism, weighted-L1, input-channel assembly — 6 sub-tests)
- [x] T037 [P] [US3] Create `scripts/train_v24_stage_a.py` (AdamW lr=1e-3 cosine, fp16 autocast, synth-dropout 0.5 for the minimap-only regime)
- [x] T038 [P] [US3] Loss-decrease covered by the 50-epoch real-data run (T041) rather than a separate synthetic 2-epoch test
- [x] T039 [P] [US3] Create `scripts/infer_v24_stage_a.py` (audit-empty → `prior_unavailable=True` short-circuit)
- [x] T040 Verify: passes (part of the 30/30 v24 suite)
- [x] T041 Bounded real-data training on Northrend rough-50 (mixed real/synthetic coverage): 50 epochs, val L1 (cheat) 0.501 world units vs minimap-only 172.6 — see Phase 8 validation report for the SC-002 baseline comparison
- [x] T042 Verified via `validate_v24.py`'s wall-time/VRAM checks (SC-005) rather than a standalone timing script

## Phase 8: User Story 4 — Stage B (Lattice Detailer) (P1)

- [x] T043 [US4] Create `harvester/v24/stage_b.py` (conv-deconv, 827,681 params; input includes `object_precise_mask` per spec, not a separate `object_mask`)
- [x] T044 [P] [US4] Add `tests/v24/test_stage_b.py` (shape, param cap, determinism, gated-L1, exact quincunx upsample — 4 sub-tests)
- [x] T045 [P] [US4] Determinism test lives in `validate_v24.py`'s `_determinism_check` (two seeds, real checkpoints, `np.array_equal`) rather than a standalone pytest
- [x] T046 [P] [US4] Create `scripts/train_v24_stage_b.py` (loss gated non-liquid/non-object/non-hole; `holes_16` polarity auto-corrected per the V22 audit finding — see `harvester/v24/tiles.py::_normalize_holes`)
- [x] T047 [P] [US4] Loss-decrease covered by the 50-epoch real-data run (T050)
- [x] T048 [P] [US4] Create `scripts/infer_v24_stage_b.py` (single entry point, Stage A + Stage B chained)
- [x] T049 Verify: passes (part of the 30/30 v24 suite)
- [x] T050 Bounded real-data training on Northrend rough-50: 50 epochs, val final L1 0.031 world units (upsampled-prior baseline 0.868; block_reduce+bilinear baseline ~0.000 because the val split is mostly real-WDL-agreeing tiles — see validation report for the full SC-003 reading)
- [x] T051 Verified via `validate_v24.py` (SC-005 wall-time + VRAM checks)

## Phase 9: User Story 5 — Validation Report (P2)

- [x] T052 [US5] Create `scripts/validate_v24.py` (coverage + confidence bound, Stage A real/synth/baseline L1, Stage B final/prior/block_reduce L1, determinism, VRAM/wall-time, preview PNG)
- [x] T053 [P] [US5] Covered by running `validate_v24.py` directly against the real Northrend rough-50 run rather than a synthetic-store pytest (real data was available and preferred per coding_standards.md)
- [x] T054 Full pipeline run on Northrend rough-50 (build → Stage A → Stage B → validate) — see `output/v24_validation/v24_northrend_rough50_20260706/report.json`
- [x] T055 SC-001..SC-005 results recorded in the validation doc (Phase 10)

## Phase 10: Polish & Cross-Cutting Concerns

- [x] T056 [P] Update `wow-viewer/memory-bank/activeContext.md` to add Spec 094
- [x] T057 [P] Update `wow-viewer/memory-bank/progress.md` with a 2026-07-06 entry
- [x] T058 [P] Write `wow-viewer/docs/architecture/v24-validation-2026-07-06.md` with the validation report summary
- [x] T059 Verify: `uv run python -m pytest tests/v24 -m v24 -q` → 30 passed

## Added: V22 Dataset Audit (user-directed scope addition, amendment A8)

- [x] Create `scripts/audit_v22_dataset.py` (C#-grounded: re-extracts reference signals via `WowViewer.Tool.Harvest extract-unified`, Python only compares)
- [x] Run against the canonical `output/datasets/v22/3_3_5_12340.zarr` (6 sampled tiles across 4 maps)
- [x] Write `docs/architecture/v22-dataset-audit-2026-07-06.md` — found the `holes_16` polarity defect (root-caused to `AdtTensorPackBuilder.ReadMcrfAndHoles`'s flags-based hole derivation) and two per-tile coverage gaps; confirmed V24's actual input signals are sound

## Phase Dependencies

```
Phase 1 (T001-T003)        ── independent setup, all parallel
       ↓
Phase 2 (T004-T015)        ── C# audit + C# shim; T007-T010 parallel after T004-T006
       ↓
Phase 3 (T016-T018)        ── synth WDL wrapper; T017-T018 parallel
       ↓
Phase 4 (T019-T021)        ── real WDL wrapper; T020-T021 parallel
       ↓
Phase 5 (T022-T029)        ── merged prior + V24 store; T023, T024, T025, T028 parallel
       ↓
Phase 6 (T030-T034)        ── minimap cleaner; T030, T031, T032 parallel
       ↓
Phase 7 (T035-T042)        ── Stage A; T036, T037, T038, T039 parallel
       ↓
Phase 8 (T043-T051)        ── Stage B; T044, T045, T046, T047, T048 parallel
       ↓
Phase 9 (T052-T055)        ── validation report; T053 parallel
       ↓
Phase 10 (T056-T059)       ── memory bank + docs; T056, T057, T058 parallel
```

## Parallel Execution Examples

### Phase 1 (all parallel)
```bash
# T001, T002, T003 can all run in parallel
mkdir -p wow-viewer/tools/wdl-read/WowViewer.Tool.WdlRead
touch wow-viewer/data-harvester/src/harvester/v24/__init__.py
touch wow-viewer/data-harvester/tests/v24/__init__.py
```

### Phase 7 (Stage A — multiple parallel tasks)
```bash
# T036, T037, T039 can all run in parallel after T035
# T036: synthetic test
# T037: training script
# T039: inference script
# T038 (loss-decreases test) depends on T037
# T040, T041, T042 depend on the script being correct
```

## Independent Test Criteria Per User Story

| User Story | Independent Test |
|---|---|
| US1 (Merged WDL Prior Coverage) | `uv run python -m pytest tests/v24/ -q` passes; bounded 5-tile real-data build produces a V24 store with `wdl_prior.shape` matching the C# reader's shape, source in {0,1,2}, confidence in [0,1], combined real+synthetic coverage ≥ 95%. |
| US2 (Minimap Cleaning) | `uv run python -m pytest tests/v24/test_clean_minimap.py -q` passes; bounded clean on 5 V18 tiles produces 5 cleaned NPZs with object pixels replaced. |
| US3 (Stage A) | `uv run python -m pytest tests/v24/test_stage_a.py -q` passes; Stage A model has ≤ 1M params; 2-epoch training on 5-tile V24 subset shows loss decreasing; 10-epoch training on 50-tile V24 real-data subset shows loss decreasing; inference on 1 tile is < 1 s. |
| US4 (Stage B) | `uv run python -m pytest tests/v24/test_stage_b.py -q` passes; Stage B model has ≤ 2M params; 2-epoch training on 5-tile V24 subset shows loss decreasing; 10-epoch training on 50-tile V24 real-data subset shows loss decreasing; full pipeline inference on 1 tile is < 3 s. |
| US5 (Validation) | `validate_v24.py` runs on a 50-tile V24 real-data validation set; SC-001 through SC-005 pass. |

## MVP Scope (Recommended First Slice)

**Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5** are the MVP. That's:
- The C# WDL reader audit (T004-T006).
- The C# CLI shim (T007-T015).
- The Python wrappers (T016-T021).
- The merged prior + V24 store + inspection (T022-T029).

This is the data side of V24. Once this lands, V24 has a complete V24 Zarr store for any V18 + staged-client combination, and the model side (Stage A, Stage B) is a clean read-train-eval loop on real data.

**Not in the MVP**: the model training itself. That comes after the data side is validated.

## Format Validation

Every task follows the strict checklist format:
- `- [ ]` checkbox ✓
- Sequential Task ID ✓
- `[P]` marker where parallelizable ✓
- `[US#]` story label for user story phases ✓
- No story label for setup/foundational/polish phases ✓
- Description with exact file path ✓

Total tasks: **59** across 10 phases.

## Notes

- Every training script change is a separate, testable commit (per RULE 6).
- Every phase ends with a `pytest` run or a CLI validation (per RULE 8).
- The C# shim is the only C# code added. C# is not modified anywhere else.
- The user's "no over-engineering" directive is enforced by the hard caps (≤ 1M + ≤ 2M params, no RunPod, no DA-V2, L1 loss only) and by the small, sequential phase structure.
- The `block_reduce(height_257)` baseline is the trivial "no learning" answer. Stage A and Stage B both need to beat it to pass the success criteria.
