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

## 0.6. Phase 3 fail-closed trust boundary — COMPLETE 2026-07-17

`harvester.v50.inventory`/`verify_v18`/`verify_store` implemented with fixture tests, plus the
real, working `scripts/v50_audit_artifacts.py` CLI (`inventory`, `verify-v18` subcommands).

```powershell
uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ -q
```

Result: 114 passed, 2 skipped (symlink privilege, same as Phase 2), 0 failed.

The `inventory` command was also run for real (read-only) against everything currently on disk:

```powershell
uv run python scripts/v50_audit_artifacts.py inventory `
  --output-root ../output --model-root ../models `
  --extra-root checkpoints=./checkpoints --extra-root tmp=./tmp --extra-root harvester_models=./models `
  --report ../output/reports/v50/v50.1/inventory.json
```

12 artifacts, ~15.6 GB, every one `trust_state=unverified`/`disposition=quarantine` regardless of
path or filename. Full results in
`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`.

`verify-v18` needs a real V18 Zarr store + `index.parquet` and a `--signals-config` JSON file
(`{"signals": [{"name": ..., "blacklisted": ..., "blacklist_reason": ..., "has_flag_name": ...}]}`).
It does not yet cross-validate against a fresh client extraction -- that is deferred until Spec 109
T002 freezes the v50 signal catalog (see the architecture doc for the exact gap statement).

## 0.7. Phase 4 reviewable cleanup planning — COMPLETE 2026-07-17

`harvester.v50.dependencies`/`cleanup` implemented with fixture tests, plus a real, working
`scripts/v50_cleanup_artifacts.py plan` command. No `apply` command exists yet -- that is Phase 7,
separate and user-run-only.

```powershell
uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ -q
```

Result: 121 passed, 2 skipped (symlink privilege), 0 failed.

A real local dry run against the Phase 3 inventory, with zero dispositions reviewed, correctly
produced zero targets (nothing has been human-reviewed yet, so nothing is proposed):

```powershell
uv run python scripts/v50_cleanup_artifacts.py plan `
  --inventory ../output/reports/v50/v50.1/inventory.json `
  --approved-root ../output --approved-root ../models --approved-root ./checkpoints --approved-root ./tmp --approved-root ./models `
  --protected-root ../specs `
  --output ../output/reports/v50/v50.1/cleanup-plan-noop.json
```

With the two genuinely-disposable `data-harvester/tmp/v18_smoke` and `.../v22_smoke` scratch
artifacts explicitly dispositioned `remove-candidate` and given a replacement-proof string (both
supplied via `--dispositions`/`--replacement-proofs` JSON, never inferred automatically), the same
command produced a real plan: 2 targets, 12,901,439 bytes expected recovered, `dry_run_complete:
true`. **Nothing was deleted** -- there is no apply command yet.

## 0.8. Phase 5 complete v50 dataset builder — COMPLETE 2026-07-17

`harvester.v50.store`/`migrate`/`build`/`curriculum` implemented with fixture tests, plus the real,
working `scripts/v50_build_dataset.py` (`migrate-v18`, `build`, `verify`, `finalize`, `curriculum`
subcommands -- this replaces the old thin fail-closed placeholder entirely).

```powershell
uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ tests/spec111/ -q
```

Result: 164 passed, 2 skipped (symlink privilege), 0 failed.

All 5 subcommands were also smoke-tested end-to-end against a synthetic 3-row V18 Zarr fixture
(real Zarr/Parquet I/O, nothing mocked):

1. `migrate-v18 --write-store` -- audited all 3 rows as copy-eligible for `height_257`, wrote a
   partial v50 store with `finalization_state=incomplete`.
2. `finalize` against the stale manifest template -- correctly refused with
   `finalization_state=incomplete`, exit 1 (the template's placeholder `content_identity` didn't
   match what was actually written -- this is the check working, not a bug).
3. `finalize` against the manifest read back from the real written store --
   `finalization_state=complete`, exit 0.
4. `verify` against the finalized manifest with the matching observed hash -- `passed: true`,
   `proof_level: full`, all 5 checks (`schema_dtype_shape`, `row_count_agreement`,
   `required_signal_truthfulness`, `content_integrity`, `partition_leakage`).
5. `verify` again with a deliberately wrong observed hash -- correctly failed closed:
   `passed: false`, `proof_level: contract`, `failure_reasons` names the exact signal and both
   hash values.
6. `curriculum` over 3 row references across a train/val split -- wrote a manifest containing only
   `{store_id, row_id, source_group, split}` references, no array payloads.

The real client-backed paths (`build` launching the C# harvester against `H:\CLIENTS`, and running
`migrate-v18`/`verify` against a real V18 build) are implemented and gated behind `--confirm-run`
or explicit user execution, but have not been run against real client data yet -- see sections 3-5
below.

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

## 4. Selective migration from a verified V18 build — IMPLEMENTED, USER RUNS

After reviewing the sample report, migrate only the signals that passed the V18 audit
(`--signals-config` is the same per-signal blacklist/policy JSON used in step 3; `--manifest-template`
is a hand-reviewed manifest shell with the real `store_id`/`build_id`/`producer_identity`/
`client_build_evidence_id` for this build -- see `contracts/v50-provenance.schema.json`):

```powershell
uv run python scripts/v50_build_dataset.py migrate-v18 `
  --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
  --signals-config ./v50-signals-3_3_5_12340.json `
  --manifest-template ./v50-manifest-template-3_3_5_12340.json `
  --report ../output/reports/v50/v50.1/migrate-3_3_5_12340.json `
  --write-store ../output/datasets/v50/v50.1/3_3_5_12340.zarr
```

Passing payloads are copied bit-for-bit (`copy_signal_row`, hash-checked). Known-defective, failed,
or missing signals are recorded `unavailable` (never partially copied) and require a fresh
extraction via step 5. `holes_16` is never copy-eligible. `liquid_mask`/`liquid_height` are
fresh-only and are rejected by the audit regardless of V18 pass/fail.

The `--write-store` output is partial (`finalization_state=incomplete`) until every required signal
either copied or was topped up by a fresh build. Run `finalize` afterward:

```powershell
uv run python scripts/v50_build_dataset.py finalize `
  --store ../output/datasets/v50/v50.1/3_3_5_12340.zarr `
  --manifest ./v50-manifest-template-3_3_5_12340.json `
  --row-lineages ../output/reports/v50/v50.1/migrate-3_3_5_12340.json `
  --output ../output/datasets/v50/v50.1/3_3_5_12340.manifest.json
```

`finalize` recomputes hashes from the store actually on disk and only reports `complete` (exit 0)
if every required signal's `content_identity` matches; otherwise `incomplete`, exit 1. Then run
`verify` against the finalized manifest as the promotion gate (FR-005) before anything downstream
treats this store as trusted:

```powershell
uv run python scripts/v50_build_dataset.py verify `
  --manifest ../output/datasets/v50/v50.1/3_3_5_12340.manifest.json `
  --row-lineages ../output/reports/v50/v50.1/migrate-3_3_5_12340.json `
  --observed-hashes ./v50-observed-hashes-3_3_5_12340.json
```

`verify` fails closed on any hash mismatch, row-count disagreement, or partition leakage -- see the
Phase 5 smoke test above for both the pass and deliberate-failure cases.

## 5. Fresh build for an additional SSD client — IMPLEMENTED, USER RUNS

```powershell
uv run python scripts/v50_build_dataset.py build `
  --harvest-project ..\..\tools\harvest\WowViewer.Tool.Harvest `
  --clients-root $FastClientsRoot `
  --map Azeroth `
  --stream-profile v22 `
  --confirm-run
```

Without `--confirm-run` the command only prints the exact C# harvester invocation and launches
nothing (`run_fresh_extraction(confirm_run=False)` returns `None`). Consuming the resulting
`harvest-stream` into a v50 store writer is not yet wired end-to-end in this pass -- today the
stream can be read with `harvester.v50.build.read_harvest_stream()`, but turning that into store
rows still requires the same manual `migrate-v18 --write-store` / `finalize` steps above.

## 6. Create curriculum manifests — IMPLEMENTED

```powershell
uv run python scripts/v50_build_dataset.py curriculum `
  --release v50.1 `
  --rows ./v50-curriculum-rows.json `
  --selection-reason "<why these rows/splits were chosen>" `
  --policy-identity sha256:<hash of the reviewed selection policy> `
  --output ../output/datasets/v50/v50.1/curricula/<name>.json
```

`--rows` is `{"rows": [{"store_id", "row_id", "source_group", "split"}, ...]}`. Curricula reference
canonical rows only -- no array payloads -- and partition-leakage across splits is rejected using
the same `validate_source_group_split` check as `verify`.

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
