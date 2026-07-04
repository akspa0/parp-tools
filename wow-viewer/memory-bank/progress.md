# Progress — wow-viewer

Keep this file to last-week truth. Older history moved to `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## 2026-07-04

### Documentation audit and rewrite

- Rewrote `wow-viewer/AGENTS.md` to current repo truth.
- Rewrote root `README.md`, `docs/PLANS-OVERVIEW.md`, `docs/WoWViewer/README.md`, `docs/WoWViewer/USERGUIDE.md`, and `data-harvester/README.md`.
- Added `docs/DOCUMENTATION-STATUS.md` as canonical doc map.
- Removed dead links and stale path guidance from high-traffic docs.

### Spec 089 local 12 GB pivot

- `train_v23_height.py` now applies real memory profiles, honors `grad_accum_steps`, records `peak_vram.json`, and retries OOM by shrinking batch size, then GPCT-K, then AMP mode.
- Default target VRAM is now 12 GB, not 22 GB.
- Focused validation passed: `uv run python -m pytest tests/v23/test_train_profiles.py tests/v23/test_train_smoke.py -m v23 -q` -> `3 passed`.
- Next proof owner = real local T035 CUDA run, not remote Pod setup.

### Spec 080 compatibility slice in `MdxViewer`

- Bottom display bar now owns terrain/world toggles.
- Top toolbar now acts as launcher strip for minimap, terrain workbench, PM4 workbench, and capture automation.
- `DrawPm4ObjectMatchWindow()` and `DrawPm4WmoCorrelationWindow()` are wired back into `DrawUI()` and exposed from `Tools`.
- Legacy build still fails on broad pre-existing missing refs outside touched slice. Status = source-complete only.

### V23 remote selector preference cleanup

- `setup_v23_runpod.py` now prefers `3090 -> 4090 -> 5090` when no explicit GPU list is given.
- Focused test passed: `tests/test_setup_v23_runpod.py` -> `1 passed`.

## 2026-07-03

### Spec 089 local stack reached bundle boundary

- V23 encoder, head, model, losses, trainer, inference, checkpoint, and RunPod bundle surfaces all landed.
- Local proof suite passed: `uv run python -m pytest tests/v23 -m v23 -q` -> `28 passed, 14 warnings`.
- Real Pod creation happened, but upload and remote smoke remain open. Not proof owner.

### Spec 088 real-data V22 path repaired

- `WowViewer.Tool.V22Enrich` builds again and the Python writer now emits coherent `index.parquet`, `placements.parquet`, `asset_inventory.parquet`, and `finalization.json`.
- Canonical V22 stores now exist for `0_5_3_3368` and `3_3_5_12340`.
- Contract is `paths_only`, with provenance sidecars and no embedded asset payload blobs.
- Remaining bounded gate: run same proof for `4_0_0_11927`.

### Environment repair

- Stale `.venv` moved aside.
- Fresh env rebuilt on `C:\Python314\python.exe`.
- `pyproject.toml` now exposes `src/` package metadata and missing `setuptools`.

## 2026-06-30

### Spec 088 replaced broken V22 payload plans

- Specs 086 and 087 were superseded by Spec 088.
- New route = `V22Enrich` + paths-only V22 store built from V18 substrate.
- This remains live background because Spec 089 depends on it.

## 2026-06-29

### Spec 077 masking correction

- `HeightOnlyPriorDataset` weight gating now prefers `object_precise_mask`, then filtered, then coarse fallback.
- RunPod slim bundles must carry `object_precise_mask` before more trust in that lane.

### Viewer animation/UI source fixes

- Model animation controls resurfaced in default info surfaces.
- Save-dialog-backed animation state export landed.
- Shell-wrapper blocker was later replaced by real legacy build failures outside the UI slice.
