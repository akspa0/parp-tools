# Implementation Plan: V50 Clean-Room Dataset and Repository Reset

**Branch**: `v0.5.1` (Spec Kit logical feature `109-v50-clean-room-audit`) | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

**Input**: Recover the missing v50 rename and dataset plan, verify V18 before porting any data,
build a complete v50 canonical dataset, use a configured faster-SSD client library, and reclaim disk
space by deleting only reviewed obsolete artifacts.

## Summary

Create one fail-closed v50 data authority rather than relabeling historical stores. A read-only audit
first inventories old artifacts and verifies V18 signal-by-signal against the existing C# harvester
and configured client builds. Verified V18 payloads may then be copied bit-for-bit into new per-build
v50 stores with immutable lineage; defective or missing signals are freshly extracted or explicitly
recorded as unavailable. Training curricula reference canonical rows through manifests instead of
copying complete stores. After v50 proof, a hash-bound cleanup manifest supports user-reviewed,
user-run deletion of old datasets, temporary client copies, model outputs, and other generated files.

## Technical Context

**Language/Version**: Python 3.11+ under `wow-viewer/data-harvester`; existing .NET 10 harvester/IO tools

**Primary Dependencies**: uv, NumPy, Zarr v3, PyArrow/Parquet, pytest; existing `WowViewer.Tool.Harvest`

**Storage**: Complete per-build Zarr stores under `wow-viewer/output/datasets/v50/v50.1/`; JSON and
Parquet provenance/audit/cleanup manifests; user-configured client library on a faster SSD

**Testing**: Fixture-only pytest contract/unit tests first; read-only sampled real-data audit and full
migration/build commands are prepared for the user

**Target Platform**: Windows local workstation and the existing portable Python/.NET command surface

**Project Type**: Library plus thin CLI workflows

**Performance Goals**: Inventory should avoid reading chunk payloads; migration streams one signal or
bounded row batch at a time; curriculum manifests add negligible storage relative to canonical stores

**Constraints**: No Codex-launched harvest/training/GPU/full-dataset/delete operations; no parser
rewrite; no hardcoded client root; no writes outside `wow-viewer`; no deletion before replacement and
dependency proof; preserve the dirty worktree

**Scale/Scope**: Six known V18 per-build stores plus additional builds in the configured client
library; dozens of dataset/model/tmp roots; all v50 dataset signals and row lineage

## Constitution Check

*GATE: Re-evaluated after Phase 1 design.*

| Gate | Status | Evidence / required handling |
|---|---|---|
| Repo independence | Pass | All new source and manifests stay under `wow-viewer`; client roots are runtime configuration only. |
| Library-first / one owner | Pass by design | `harvester.v50` owns contracts and workflows; scripts remain thin. Historical Spec 103/108 code is compatibility-only after migration. |
| Real-data validation | Pass | The user approved `H:\CLIENTS` as the known-good fast SSD library; commands still receive it as runtime configuration and fingerprint each build. |
| Existing readers remain authoritative | Pass | Verification/fresh extraction calls the existing C# harvester; no format reader is rewritten. |
| One phase at a time | Pass | Each phase below ends with an explicit gate; later phases cannot start early. |
| User owns heavy/destructive work | Pass | Full verification, migration, fresh builds, training, and cleanup application are commands for the user. |
| Small residual model policy | Not changed | This feature changes data/repo ownership only; no model head, loss, or training architecture change is authorized. |
| Documentation and memory sync | Pass | Spec 109, architecture audit, quickstart, and continuity files are explicit deliverables. |

Repository policy now recognizes `H:\CLIENTS` as the user-approved preferred fast SSD client library.
The path remains runtime configuration rather than a source-code default, and each build must be
fingerprinted before it can support a v50 provenance claim.

## Project Structure

### Documentation (this feature)

```text
specs/109-v50-clean-room-audit/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── v50-provenance.schema.json
│   └── v50-cleanup-plan.schema.json
└── tasks.md

docs/architecture/
└── v50-clean-room-dataset-repo-audit-2026-07-15.md
```

### Source Code (repository root)

```text
data-harvester/
├── scripts/
│   ├── v50_audit_artifacts.py
│   ├── v50_build_dataset.py
│   └── v50_cleanup_artifacts.py
├── src/harvester/v50/
│   ├── __init__.py
│   ├── contracts.py
│   ├── inventory.py
│   ├── verify_v18.py
│   ├── migrate.py
│   ├── store.py
│   └── cleanup.py
└── tests/v50/
    ├── test_contracts.py
    ├── test_inventory.py
    ├── test_verify_v18.py
    ├── test_migrate.py
    └── test_cleanup.py
```

**Structure Decision**: Dataset truth, verification, migration, and cleanup live in a dedicated
`harvester.v50` package. The three commands expose read-only audit, dataset build/migration, and
cleanup plan/apply respectively. Existing C# readers remain the game-data authority.

## Delivery Phases

### Phase 0 — Authority and contract lock

1. Reconcile workspace policy with the user-selected configurable faster-SSD client library.
2. Freeze the v50 complete-store signal table and explicitly blacklist known-defective V18 signals.
3. Freeze provenance, verification, migration-ledger, release-manifest, and cleanup-plan schemas.
4. Record protected roots and generated-data roots eligible for cleanup consideration.
5. Prove schemas and path policies with fixture-only tests.

**Gate**: Docs agree, no client path is hardcoded in source, and all unresolved policy conflicts are closed.

The user has also approved a pre-build clean-slate bootstrap: after tracked files and retained M2
evidence are removed from the two output roots, `scripts/clean-legacy-outputs.ps1` may empty
`<workspace>/output` and `<workspace>/wow-viewer/output`. This user-run cleanup precedes dataset
implementation because it establishes the requested empty v50 workspace; it does not promote or
migrate any old dataset.

### Phase 1 — Read-only inventory and fail-closed foundations

1. Rehome the existing release helpers into the canonical `harvester.v50` package.
2. Implement metadata-only artifact inventory with no trust-by-name behavior.
3. Implement complete store/index/lineage validation against the frozen schema.
4. Implement dependency discovery from manifests, checkpoints, reports, and known output layouts.
5. Emit deterministic inventory and protected-root reports.
6. Validate missing/mismatched identities and unsafe paths with fixtures.

**Gate**: Inventory is deterministic and cannot promote, mutate, or delete an artifact.

### Phase 2 — V18 signal audit and selective migration

1. Add a V18 audit adapter that reports each signal independently rather than certifying a whole row.
2. Bind sampled reference extraction to the configured client build and existing C# harvester.
3. Reject known-bad `holes_16`, coverage gaps, false `has_*` flags, non-finite values, and lineage gaps.
4. Add bit-preserving copy for verified signal payloads plus fresh-extraction slots for rejected data.
5. Write per-row/per-signal migration ledgers and recompute v50 store/index hashes.
6. Prove mixed pass/fail rows, interrupted resume, and exact copies with fixtures.
7. Prepare sampled and full user-run verification/migration commands for each candidate build.

**Gate**: One bounded build passes sampled proof and produces a complete ledger; user approves the
full migration command before it is run.

### Phase 3 — Fresh v50 builds from the faster client library

1. Add runtime client-root configuration and logical client-build fingerprinting.
2. Route new builds through the existing C# harvester into the v50 store writer.
3. Cover all frozen required signals, sidecars, placement tables, and explicit unavailable states.
4. Build immutable curriculum manifests over canonical stores; do not copy mixed training stores.
5. Add finalization that accounts for every expected row/signal and writes the release manifest.
6. Prepare per-build user-run commands and duration/output expectations.

**Gate**: The user-run complete v50 build passes contract and full-content verification before any
legacy dataset is marked removable.

### Phase 4 — V50 rename and command ownership

1. Move v50 implementation ownership out of Spec 103/108 modules into `harvester.v50`.
2. Keep only bounded compatibility shims where current tests or callers require them.
3. Update command help, checkpoint metadata, generated-prior metadata, docs, and tests to v50 names.
4. Prove old checkpoints/stores cannot masquerade as v50 and v50 releases cannot cross-load.
5. Remove obsolete wrappers only after caller search and compatibility proof.

**Gate**: One canonical v50 owner exists for each retained workflow and all focused tests pass.

### Phase 5 — Reviewed disk reclamation and documentation compression

1. Inventory byte sizes without following links or leaving approved generated roots.
2. Classify old datasets, tmp client copies, model outputs, cloud downloads, and caches.
3. Require replacement/dependency proof for every deletion candidate.
4. Emit a deterministic cleanup manifest and dry-run report with expected recovered bytes.
5. Have the user review and run the hash-bound cleanup apply command.
6. Verify post-cleanup survivors, v50 integrity, and actual recovered space.
7. Compress continuity docs and archive stale chronology after the active truth is preserved.

**Gate**: The cleanup report accounts for every target, protected roots remain untouched, and the
complete v50 dataset revalidates after deletion.

## Post-Design Constitution Check

- The design introduces no parser duplication and no path hardcoding.
- Complete stores have one v50 owner; curricula are manifests rather than duplicated payloads.
- The V18 port is signal-scoped and evidence-gated, so old metadata cannot launder trust.
- Destructive work is separated into a reviewed manifest and a user-run apply step.
- The faster-SSD client-library policy is aligned; per-build provenance remains mandatory.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Temporary compatibility shims may remain in historical command modules | Preserve dirty-worktree callers during the ownership move | Renaming or deleting them immediately could destroy active user work and hide compatibility regressions |
