# Tasks: 096-v24-minimap-deploy

**Status**: Draft
**Created**: 2026-07-09
**Parent spec**: [`spec.md`](spec.md)
**Parent plan**: [`plan.md`](plan.md)

Each task is one focused, testable commit. Do not bundle. Do not skip validation.

---

## Slice 1 — Train the minimap-only Stage A checkpoint

- [ ] **T-096-001**: Verify the curated V24 store path. If `output/datasets/v24/3_3_5_12340_openworld_curated.zarr` does not exist, run `build_wdl_prior.py build` first (with the same args as the 2026-07-09 curated run). Document the actual path used.
- [ ] **T-096-002**: Add two tests to `tests/v24/test_stage_a.py`:
  - `test_stage_a_minimap_only_forward_shape_and_params` — assert Stage A minimap-only forward returns (B,17,17)+(B,16,16) and param count ≤ 1M.
  - `test_stage_a_minimap_only_pre_train_is_constant` — assert the zero-init head output is constant across spatial positions on a random input.
  - Run `uv run python -m pytest tests/v24/test_stage_a.py -m v24 -q` and confirm green.
- [ ] **T-096-003**: Run the training command (one shell exec):
  ```
  cd wow-viewer/data-harvester
  uv run python scripts/train_v24_stage_a.py \
    --v24-store output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
    --v18-store output/datasets/v18/3_3_5_12340.zarr \
    --output output/v24_validation/v24_minimap_only_3_3_5_12340_20260709 \
    --minimap-only --epochs 50 --seed 94 \
    --autotune-batch-size --log-interval 1
  ```
  Capture the entire stdout (epoch-by-epoch log lines). Commit it as a build artifact under `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/train.log` if the directory is gitignored; otherwise do not commit.
- [ ] **T-096-004**: Validation gate for Slice 1:
  - `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a.pt` exists, ≤ 5 MB, loads as `StageAMinimapOnly(base=28, in_channels=3)`, config contains `"minimap_only": true`.
  - `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/stage_a_metrics.json` reports `train_tiles + val_tiles = 2011`, `epochs_run == 50` (no early stop), `peak_vram_gb < 2.0`, `params ≤ 1_000_000`.
  - `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/loss_history.jsonl` has 50 lines.
  - `uv run python -m pytest tests/v24 -m v24 -q` is green.
  - **Record** `best_val_l1` in a one-line note for Slice 3.
- [ ] **T-096-005**: Commit Slice 1 (one commit): "V24: train minimap-only Stage A on curated 3_3_5_12340 corpus".

## Slice 2 — `infer_v24_stage_a_png.py` standalone inference script

- [ ] **T-096-006**: Write `wow-viewer/data-harvester/scripts/infer_v24_stage_a_png.py` per the design in `plan.md` Slice 2. Target: ≤ 120 lines, no matplotlib, only PIL + numpy + torch + the existing `harvester.v24` module. Document every CLI flag in the module docstring.
- [ ] **T-096-007**: Write `wow-viewer/data-harvester/tests/v24/test_infer_stage_a_png.py` with three tests per `plan.md`. Use `tmp_path` fixtures for the PNG and the output NPZ. Use the Slice 1 checkpoint as the test fixture (or, if it is too large for a unit test, train a 2-epoch minimap-only model in a fixture — that is a known cheap path).
- [ ] **T-096-008**: Validation gate for Slice 2:
  - `uv run python -m pytest tests/v24/test_infer_stage_a_png.py -m v24 -q` passes.
  - Full suite: `uv run python -m pytest tests/v24 -m v24 -q` is green (≥ 36 tests).
  - Manual: `python scripts/infer_v24_stage_a_png.py --checkpoint <slice 1 ckpt> --image <some V18-derived PNG> --output /tmp/prior.npz --preview /tmp/prior.png` runs end-to-end, NPZ has the expected keys and shapes, preview PNG exists, exit code 0, wall-time < 200 ms on CPU.
- [ ] **T-096-009**: Commit Slice 2 (one commit): "V24: standalone minimap-only Stage A inference (PNG -> WDL prior NPZ)".

## Slice 3 — Validation comparison (minimap-only vs cheat)

- [ ] **T-096-010**: Modify `validate_v24.py` to add `--minimap-only-checkpoint <ckpt>`. When given:
  - Load the minimap-only Stage A.
  - Evaluate on the same held-out rows used for the cheat regime.
  - Add a `stage_a_minimap_only` block to the report: `params`, `val_l1_minimap_only`, `block_reduce_baseline_l1` (the same baseline as the cheat regime).
  - Add the SC-002-MINIMAP check: `stage_a_minimap_only_l1 < block_reduce_baseline_l1`.
  - Add the determinism re-check on the minimap-only pipeline.
  - Keep all existing checks intact.
- [ ] **T-096-011**: Run validation:
  ```
  uv run python scripts/validate_v24.py \
    --v24-store output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
    --v18-store output/datasets/v18/3_3_5_12340.zarr \
    --stage-a-checkpoint <cheat ckpt> \
    --stage-b-checkpoint <cheat ckpt> \
    --minimap-only-checkpoint <slice 1 ckpt> \
    --run-id v24_minimap_only_3_3_5_12340_20260709_validation
  ```
  Confirm the report is written and the SC-002-MINIMAP check is recorded (pass or fail, honestly).
- [ ] **T-096-012**: Validation gate for Slice 3:
  - `report.json` contains the `stage_a_minimap_only` block and the new SC-002-MINIMAP check.
  - SC-004 (determinism) is still PASS for the minimap-only pipeline.
  - The gap `cheat_l1 - minimap_only_l1` is a recorded number, not hidden.
- [ ] **T-096-013**: Commit Slice 3 (one commit): "V24: validation report includes minimap-only comparison + SC-002-MINIMAP gate".

## Slice 4 — Memory bank + progress sync + summary doc

- [ ] **T-096-014**: Write `wow-viewer/docs/architecture/v24-minimap-deploy-2026-07-09.md` with the training result, the inference script, the determinism check, the hardware envelope, and an honest section on "minimap-only vs cheat" with the actual numbers.
- [ ] **T-096-015**: Update `wow-viewer/memory-bank/activeContext.md` "WDL prior + lattice detailer lane (V24)" section with: minimap-only training result, the new inference script, the SC-002-MINIMAP outcome, the link to the new doc. Also fix the `>>>>> REPLACE` marker in `progress.md` that was left dangling.
- [ ] **T-096-016**: Update `wow-viewer/memory-bank/progress.md` with a 2026-07-09 entry summarising the slice.
- [ ] **T-096-017**: Commit Slice 4 (one commit): "V24: memory bank + progress + architecture doc for minimap-only deployment slice".

## Final gate (post all slices)

- [ ] **T-096-018**: Run `uv run python -m pytest tests/v24 -m v24 -q` and confirm ≥ 36 tests pass.
- [ ] **T-096-019**: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeds (no C# changes, but be sure).
- [ ] **T-096-020**: Confirm `git status` shows the expected four commits, no untracked artifacts in the working tree outside `output/v24_validation/v24_minimap_only_3_3_5_12340_20260709/`.

---

## End of Tasks

Each task is bounded, single-commit, single-validation-gate. RULE 8 (one phase at a time) and RULE 11 (bite-sized) are honored.
