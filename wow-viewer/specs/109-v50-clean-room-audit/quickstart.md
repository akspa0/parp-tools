# V50 Clean-Room Reset Quickstart

These are the intended operator flows. Commands whose implementation is still pending are marked
`PLANNED`; do not run them yet. Codex does not launch the full verification, dataset build, training,
or cleanup apply operations.

Run Python commands from `wow-viewer/data-harvester`.

## 0. Empty legacy output roots — COMPLETE 2026-07-16

The guarded cleanup completed successfully and recovered more than 200 GB. The commands below are
retained only as the reproducible cleanup contract; do not rerun them unless new disposable legacy
outputs have accumulated.

First inspect the exact top-level targets and measure expected reclaimed bytes:

```powershell
pwsh -File ..\scripts\clean-legacy-outputs.ps1 `
  -WorkspaceRoot I:\parp\parp-tools `
  -MeasureBytes `
  -ReportPath C:\tmp\v50-legacy-output-cleanup.json
```

After reviewing the report, empty both legacy output roots:

```powershell
pwsh -File ..\scripts\clean-legacy-outputs.ps1 `
  -WorkspaceRoot I:\parp\parp-tools `
  -Apply `
  -Confirmation DELETE-LEGACY-OUTPUTS
```

The script refuses tracked files, reparse points, paths outside the two exact output roots, and a
missing confirmation. It retains the empty root directories. `H:\CLIENTS` is outside its allowed
scope and cannot be touched.

Some pytest outputs were created under the sandbox-only account and may be unreadable to the normal
user. Measurement records these as incomplete instead of aborting. If apply reports ACL failures,
open **Command Prompt as Administrator** and repair only the two approved output roots:

```bat
takeown /F "I:\parp\parp-tools\output" /R /D Y
icacls "I:\parp\parp-tools\output" /grant "%USERDOMAIN%\%USERNAME%:(OI)(CI)F" /T /C
takeown /F "I:\parp\parp-tools\wow-viewer\output" /R /D Y
icacls "I:\parp\parp-tools\wow-viewer\output" /grant "%USERDOMAIN%\%USERNAME%:(OI)(CI)F" /T /C
```

Then rerun the exact apply command. Do not run these ACL commands against the workspace root or
`H:\CLIENTS`.

## 0.5. Phase 2 foundational contracts — COMPLETE 2026-07-17

`harvester.v50.contracts`/`identity`/`client_evidence`/`path_policy` are implemented with
fixture-only tests (no client data, no GPU, nothing destructive). `harvester.v50_contract` is now a
thin re-export shim over `harvester.v50.contracts` so existing Spec 103/108 callers are unaffected.

```powershell
uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ -q
```

Result: 90 passed, 2 skipped, 0 failed. The 2 skips are the symlink-escape cases
(`test_rejects_a_symlink_that_escapes_every_approved_root`,
`test_rejects_a_symlink_that_redirects_into_a_protected_root`) -- this host's account cannot
create symlinks without elevated privilege/Developer Mode, so they self-skip via a runtime probe
rather than failing on an environment limitation. They have not yet been observed passing on a
host with symlink privilege; re-run on Developer Mode or an elevated shell to close that gap. The
non-symlink containment tests (protected-root-wins-when-nested, out-of-root rejection,
nonexistent-path rejection) exercise the same resolution logic and all pass.

## 1. Configure the faster client library

The root is runtime configuration and is not committed:

```powershell
$FastClientsRoot = 'H:\CLIENTS'
```

Repository policy explicitly approves this known-good SSD library. Each child build remains
fingerprinted independently; approval of the root does not promote any old dataset.

## 2. Read-only inventory — PLANNED

```powershell
uv run python scripts/v50_audit_artifacts.py inventory `
  --dataset-root ../output/datasets `
  --output-root ../output `
  --model-root ../models `
  --report ../output/reports/v50/v50.1/inventory.json
```

This reads metadata and sizes only. It writes no trust promotions and deletes nothing.

## 3. Sample one V18 build against its client — PLANNED, USER RUNS

```powershell
uv run python scripts/v50_audit_artifacts.py verify-v18 `
  --store ../output/datasets/v18/3_3_5_12340.zarr `
  --clients-root $FastClientsRoot `
  --build 3_3_5_12340 `
  --sample 16 `
  --report ../output/reports/v50/v50.1/v18-3_3_5_12340-sample.json
```

This uses the existing C# harvester as the independent reader. It reports every signal separately;
it never promotes the complete store because a subset of signals passed.

## 4. Full verification and selective migration — PLANNED, USER RUNS

After reviewing the sample report, the intended command will be:

```powershell
uv run python scripts/v50_build_dataset.py migrate-v18 `
  --source ../output/datasets/v18/3_3_5_12340.zarr `
  --clients-root $FastClientsRoot `
  --build 3_3_5_12340 `
  --release v50.1 `
  --output ../output/datasets/v50/v50.1/3_3_5_12340.zarr `
  --full-verify `
  --allow-zarr-write
```

Passing payloads are copied bit-for-bit. Known-defective or failed signals are freshly extracted or
recorded unavailable. `holes_16` is never copied from the old V18 store. `liquid_mask` and
`liquid_height` are fresh-only: historic liquid payloads are rejected, and a WL source must declare
`wl_liquid_surface_quads_v1`.

**Current status**: this command is intentionally unavailable. `v50_build_dataset.py` returns a
non-zero result instead of delegating to the legacy mixed-copy builder until Spec 109 implements
the fixture-proven migration and client-backed fresh-extraction owner.

## 5. Fresh build for an additional SSD client — PLANNED, USER RUNS

```powershell
uv run python scripts/v50_build_dataset.py build `
  --clients-root $FastClientsRoot `
  --build <build-id> `
  --release v50.1 `
  --output ../output/datasets/v50/v50.1/<build-id>.zarr `
  --allow-zarr-write
```

The command writes one complete canonical per-build store and finalization manifest. It does not
create a training-specific mixed copy.

## 6. Create curriculum manifests — PLANNED

```powershell
uv run python scripts/v50_build_dataset.py curriculum `
  --release-root ../output/datasets/v50/v50.1 `
  --policy <reviewed-policy.json> `
  --output ../output/datasets/v50/v50.1/curricula/<name>.parquet
```

Curricula reference canonical rows and contain no array payloads.

## 7. Cleanup dry run — PLANNED

```powershell
uv run python scripts/v50_cleanup_artifacts.py plan `
  --inventory ../output/reports/v50/v50.1/inventory.json `
  --release-manifest ../output/datasets/v50/v50.1/release-manifest.json `
  --output ../output/reports/v50/v50.1/cleanup-plan.json
```

Review exact targets, dependencies, and expected recovered bytes. The plan must include old datasets,
temporary project-local client copies, obsolete models/checkpoints, cloud downloads, and caches only
when their replacement/dependency checks pass.

## 8. Cleanup apply — PLANNED, USER RUNS ONLY

The implemented tool will require the reviewed cleanup-plan hash and will refuse paths outside
approved generated roots. The exact apply command will be added only after fixture tests and dry-run
review; do not use a manual recursive-delete command.
