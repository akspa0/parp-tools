# Quickstart: V50-Native Height-First Terrain Model

**Status**: dataset correction and dual-source curriculum are complete on real 0.5.3.3368 data;
the relative-height model/trainer and CPU contract tests are implemented. The remaining gate is the
user-run CUDA training command in Phase 2.

All commands run from `wow-viewer/data-harvester/` unless noted otherwise. The C# fixes (Decisions
1-2) are built via the existing `dotnet build wow-viewer/WowViewer.slnx -c Debug` before any Phase 1
dataset step.

## Phase 1 — Dataset corrections (US1)

### 1.1 Fix and prove the C# gaps

```powershell
dotnet build ../WowViewer.slnx -c Debug
dotnet test ../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~AlphaTensorPackBuilder|FullyQualifiedName~NativeMpqService"
```

Expect a focused regression test proving `AlphaTensorPackBuilder` now assigns real `McnkFlags16`
data (Decision 1), and — if Decision 2's race is confirmed — a test proving concurrent
`NativeMpqService.ReadFile` calls no longer corrupt/lose reads under `Parallel.ForEach`.

### 1.2 Regenerate the manifest template from the frozen catalog

```powershell
uv run python scripts/v50_generate_manifest_template.py `
  --catalog-doc ../docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md `
  --release v50.1 `
  --build-id 0_5_3_3368 `
  --output ./v50_configs/v50-manifest-template-0_5_3_3368.json
```

The four catalog-dropped signals (`mddf_mask`, `modf_mask`, `object_filtered_mask`,
`model_focus_mask`) must not appear in the regenerated template; `mccv_rgb` must be absent (era
restriction, Decision 1 of research.md's Decision 3/data-model's `era_available` field) rather than
declared-and-zero-filled.

### 1.3 Rebuild Kalimdor and Azeroth

```powershell
uv run python scripts/v50_build_dataset.py build `
  --harvest-project ../tools/harvest/WowViewer.Tool.Harvest `
  --clients-root H:\CLIENTS --map Kalimdor --stream-profile v22 `
  --signals-config ./v50_configs/v50-signals-0_5_3_3368.json `
  --manifest-template ./v50_configs/v50-manifest-template-0_5_3_3368.json `
  --report ../output/reports/v50/v50.1/build-0_5_3_3368-Kalimdor.json `
  --write-store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --write-manifest ../output/reports/v50/v50.1/build-manifest-0_5_3_3368-Kalimdor.json `
  --confirm-run
```

(Repeat for Azeroth. This reuses the existing Spec 109 `build`/`finalize` commands unchanged —
see `specs/109-v50-clean-room-audit/quickstart.md` §5 for the finalize step and the reason
`--write-manifest`'s output, never `--manifest-template`, feeds `finalize`.)

### 1.4 Prove the corrections landed

```powershell
uv run python scripts/v50_audit_signal_coverage.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr `
  --output ../output/reports/v50/v50.1/coverage-audit-0_5_3_3368-Kalimdor.json
```

Report conforms to `contracts/coverage-audit-report.schema.json`. Expect zero
`zero_coverage_unexplained` signals and `minimap_resolution_parity.parity == true`. Repeat for
Azeroth. This is the SC-001/SC-002 proof.

**Real proof (2026-07-18)**:

- Kalimdor: 951 rows; `mcnk_flags_16` 948/951 (99.68%); synthesized minimaps 731/951 at both
  resolutions; authored minimap 951/951; parity true; `mccv_rgb` explicitly era-unavailable.
- Azeroth: 685 rows; `mcnk_flags_16` 671/685 (97.96%); synthesized minimaps 642/685 at both
  resolutions; authored minimap 678/685; parity true; `mccv_rgb` explicitly era-unavailable.
- Reports: `output/reports/v50/v50.1/coverage-audit-0_5_3_3368-{Kalimdor,Azeroth}.json`.

## Phase 1 — Curriculum (US2)

```powershell
uv run python scripts/v50_build_training_curriculum.py `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Kalimdor.zarr --curation-manifest ../output/datasets/v50/v50.1/curation-0_5_3_3368-Kalimdor-terrain `
  --store ../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr  --curation-manifest ../output/datasets/v50/v50.1/curation-0_5_3_3368-Azeroth-terrain `
  --output ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --val-fraction 0.15
```

Note the absence of `--store`/`--curation-manifest` pairs for PVPZone02 or Kalidar — the builder's
map allow-list (data-model.md) refuses them for this lane even if supplied. Verify the curriculum's
signal list against the frozen catalog's populated set (SC-003).

**Real proof (2026-07-18)**: `curriculum-0_5_3_3368-dual_v1.zarr` contains 2,990 rows
(2,545 train / 445 val), 1,629 authored + 1,361 synthetic inputs, and only Kalimdor/Azeroth.
Validation contributes 248 Kalimdor and 197 Azeroth rows. The summary is stored beside the Zarr
store as `summary.json`.

## Phase 2 — Model (US3, user-executed)

```powershell
uv run python scripts/v50_train_height_relative.py `
  --store ../output/datasets/v50/v50.1/curriculum-0_5_3_3368-dual_v1.zarr `
  --val-key split --val-value val `
  --output ../output/v50/v50.1/height_relative_v1 `
  --epochs 100 --batch 32 --workers 4 --patience 15
```

**This command is printed for the user to run, never executed by the assistant** (contracts/
relative-height-target-contract.md, execution contract). Expect the training summary
(`training_summary.json`, per data-model.md) to show `best_epoch > 1` and a `tile_mean_baseline`
comparison the trained model beats — this is SC-004. SC-005 (visual relief-structure judgment) is a
separate user review step over reconstructed held-out tiles from both maps, following the same
side-by-side discipline as Spec 110/111's minimap fidelity gates.

Estimated runtime on the user's RTX 4070 Ti SUPER: roughly 15–30 minutes for 100 epochs with early
stopping, writing checkpoints, `run_identity.json`, and `training_summary.json` under
`output/v50/v50.1/height_relative_v1/`. This is an estimate; the command remains user-launched.

## Implementation proof (2026-07-18)

- `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/test_v50_build_command.py -q`
  → **175 passed, 4 skipped** (2026-07-19; includes the later Spec 113 visual/pair contracts).
- `dotnet test ... --filter "FullyQualifiedName~AlphaMcnkFlagsTests|FullyQualifiedName~TerrainMinimapDetailRenderTests"`
  → **7 passed** (3 Spec 112 MCNK-flag tests + 4 Spec 113 detail-render tests).
- `dotnet build WowViewer.slnx -c Debug` → **0 errors**; existing analyzer/package warnings remain.
- Model/trainer scripts and all new v50 modules pass `py_compile`.

## Explicitly out of scope for verification here

- PVPZone02 and Kalidar: never appear in any curriculum, training, or evaluation command in this
  spec. Their stores remain on disk, untouched, for whatever future lane wants them.
- The legacy `v50_train_wdl_prior.py`/spec103 lane: not modified, not re-run, not compared against
  as a promotion baseline — it is rejected, not superseded-and-benchmarked.
