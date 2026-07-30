# Implementation Plan: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Branch**: `122-dataset-curation` | **Date**: 2026-07-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/122-dataset-curation/spec.md`

## Summary

Every terrain-model generation on this project (V16 → V18 → V22 → V50, and the specs built on top
of V50) has needed a "is this tile clean, and clean how" answer, and every generation has
re-answered it with new, throwaway Python: `v16_curation.py` (difficulty buckets, blank-tile
detection), `mismatch_detector.py` (height-vs-normal mismatch), `spec111/lighting_buckets.py`
(lighting-bucket reconciliation), plus several one-off audit scripts. None of it is shared across
generations, and the underlying failure mode is worse than duplication: today's tooling treats
"bad" as something to filter and forget, so the excluded population — including the project's own
synthetic minimap renders, whose shading does not match authored minimaps — is not durably
queryable on its own terms.

This feature consolidates that logic into one canonical, C#-produced classification layer
(`WowViewer.Core.Curation`), invoked as a `curate` subcommand on the existing
`WowViewer.Tool.Harvest`, that reads already-decoded tensor-pack signals and writes a durable,
row-per-tile Parquet manifest plus a row-per-finding Parquet table alongside a v50 store. Every
tile gets a bucket assignment and zero or more mismatch findings (including a synthetic-vs-authored
minimap fidelity finding built on the existing `MinimapShadingMatch` correlation machinery); nothing
is ever dropped from the output. The six legacy Python scripts either become thin readers of this
new manifest or are documented as retired, matching this repo's established shim convention (Spec
109 Phase 6). No model architecture, loss function, or training behavior is decided here — this
feature only makes "which tiles are clean, and why" answerable from one place, for every future
spec.

## Technical Context

**Language/Version**: C# / .NET 10 for the new canonical curation library and its `curate` CLI
subcommand. Python 3.11+ (managed by `uv`, under `wow-viewer/data-harvester/`) only for the thin
manifest-reader shims that replace the legacy scripts' own logic — no new curation *logic* is
written in Python by this feature.

**Primary Dependencies**: No new external dependency. C# side reuses the existing tensor-pack
model (`TerrainTileTensorPack`), the existing `MinimapShadingMatch`/`TerrainMinimapCompositor`
correlation machinery, and the existing Parquet-writing capability already used elsewhere in
`WowViewer.Core.IO` (`decoded_metadata.parquet` sidecar precedent). Python side reuses `pyarrow`
(already a project dependency) to read the new manifest.

**Storage**: Two new Parquet files written alongside an existing v50 store's own `index.parquet`:
`curation_manifest.parquet` (one row per tile: identity, bucket assignments, and synthetic-fidelity
summary) and `curation_findings.parquet` (one row per finding: tile identity, category, severity,
reason, and evaluability state). Plus one small JSON run record (`v50-curation-run-v1`, mirroring
the existing `v50-model-stage-run-v1` provenance convention) per curation invocation. The v50 Zarr
store's own writer contract is untouched — curation is strictly read-only with respect to the
signals it classifies (FR-014).

**Testing**: C# unit tests under a new `wow-viewer/tests/WowViewer.Core.Curation.Tests/` project
(mirrors the existing `WowViewer.Core.PM4.Tests` convention) against synthetic in-memory tensor-pack
fixtures for each bucket rule and mismatch detector. A real (non-fixture) smoke run of `curate`
against an existing on-disk v50 store, verifying full tile coverage (SC-006). A comparison pass
(SC-003) running the legacy `mismatch_detector.py` and the new C# check against the same real store
and diffing the flagged sets, before any legacy script is marked retired. `pytest` for the new
Python-side manifest-reader shims under `data-harvester/tests/`.

**Target Platform**: Windows-native .NET 10 CLI (matches the existing `WowViewer.Tool.Harvest`
target); no GPU/CUDA involvement anywhere in this feature — curation is CPU-only classification
logic over already-decoded arrays.

**Project Type**: One new shared library (`WowViewer.Core.Curation`, library-first per constitution
II) + one new subcommand on the existing `WowViewer.Tool.Harvest` (no new tool project — curation
is "another pass over harvested tensor packs," and the existing tool already hosts a wide range of
harvest-adjacent analysis subcommands). Thin Python reader shims only, no new Python package beyond
a single `harvester/curation_store.py` module.

**Performance Goals**: A `curate` pass over one map's worth of tiles (hundreds to ~1,000 tiles per
the existing v50 corpus sizes recorded in memory-bank) should complete in low-single-digit minutes
on the existing harvest-tool performance envelope — it reads data already produced by harvest and
does bounded per-tile arithmetic (no rendering beyond the existing shading-match sweep, which
already runs today as part of Spec 111's streaming pathway).

**Constraints**: Curation MUST NOT modify, delete, or move any harvested signal (FR-014). Every
tile MUST receive a record — full coverage is a hard gate (FR-008, SC-006), not a best-effort
target. The CLI MUST be dry-run-first, matching every other CLI in this repository (prints planned
tile count, checks, and output paths; requires an explicit write flag to persist anything).

**Scale/Scope**: Scoped to the existing v50 signal catalog and existing on-disk v50 stores/builds
(0.5.3.3368 and any future build with its own `v50-signals-*.json`/`v50-manifest-template-*.json`
pair). No new signal extraction, no new harvest pass, no new client-build support.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Repo Independence | PASS | New C# library and Python shim both live entirely under `wow-viewer/`; no path outside it. |
| II. Library-First | PASS | Canonical logic lives in a new `WowViewer.Core.Curation` library; `curate` is a thin subcommand on the existing harvest tool, not a second implementation. One canonical owner for curation logic, matching the "one canonical owner per format surface" spirit extended to "one canonical owner per curation concern." |
| III. Real-Data Validation | PASS | SC-003/SC-006 both require validation against a real, already-built on-disk v50 store, not only fixtures. |
| IV. Residual Model Chain | N/A | This feature defines no model; explicitly out of scope per spec FR-016. |
| V. Streaming-First Dataset Pipeline | PASS | No new intermediate NPZ files; the two new Parquet outputs follow the same "Parquet for index/metadata" pattern already established (`index.parquet`, `decoded_metadata.parquet`). The Zarr store remains the sole store-of-record for signal arrays; curation adds a companion classification, not a new signal store. |
| VI. No Game Client Path Assumptions | PASS | `curate` takes a configured store path and (where a live re-derivation is needed) a configured client root, exactly like every existing harvest command; nothing hardcoded. |
| Read-Only Reference Codebase | PASS | No writes to `gillijimproject_refactor`. |
| Format Reader/Writer Ownership | PASS | Curation reads already-decoded tensor-pack output; it does not add or duplicate an ADT/WDT/WDL/MCNK/MCAL reader. It reuses `MinimapShadingMatch`/`TerrainMinimapCompositor` rather than reimplementing minimap comparison. |
| Terrain Alpha Risk Area | N/A | No MCAL decode, edge-fix, texture-sourcing, or shader-blending change. |
| AlphaWdtWriter Frozen | N/A | Not touched. |
| One Phase at a Time | PASS | Phases below are ordered US1→US2→US3→US4 (spec priority order); each ends with a real-data validation gate before the next starts. |
| Spec Docs Source of Truth | PASS | This plan + spec.md + the Phase 1 artifacts below are the source of truth; `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`'s frozen signal catalog gets a short pointer addendum once curation ships (not a rewrite). |
| Training Script Changes Require Validation | N/A | No training script is touched; explicitly out of scope (FR-016). |
| Bite-Sized Plans | PASS | Max 10 steps per phase, generated by `speckit-tasks`, one concern per step. |

**Gate verdict**: PASS, no violations, no Complexity Tracking entries required.

### Post-design re-check (after Phase 1 data-model + contracts)

Re-evaluated after `data-model.md` and `contracts/` were written. All principles still PASS. The
Parquet-companion storage decision (D-02 in research.md) confirms zero changes are required to the
v50 Zarr store writer (`harvester/v50/store.py`) or its schema — "Streaming-First Dataset Pipeline"
and "Format Reader/Writer Ownership" have nothing new to validate on the write side.
**Post-design verdict: PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/122-dataset-curation/
├── plan.md              # This file
├── research.md          # Phase 0 output (decisions D-01 through D-05+, external precedent)
├── data-model.md         # Phase 1 output (entities, Parquet schemas, run-record schema)
├── quickstart.md         # Phase 1 output (user-run commands)
├── contracts/            # Phase 1 output (CLI contract + Parquet schema contracts)
└── tasks.md              # Phase 2 output (speckit-tasks — NOT created by this plan)
```

### Source Code (repository root)

```text
wow-viewer/
├── src/core/
│   └── WowViewer.Core.Curation/                    # NEW library — canonical curation logic
│       ├── WowViewer.Core.Curation.csproj
│       ├── Buckets/
│       │   ├── DifficultyBucketClassifier.cs        # ports v16_curation.py DIFFICULTY_BUCKETS
│       │   ├── CoverageBucketClassifier.cs           # alpha/mcly/liquid/object coverage buckets
│       │   └── LightingBucketClassifier.cs           # ports spec111/lighting_buckets.py
│       ├── Mismatch/
│       │   ├── HeightNormalMismatchDetector.cs       # ports mismatch_detector.py
│       │   ├── NonFiniteSignalDetector.cs            # ports verify_v18-style checks
│       │   ├── HasFlagTruthfulnessDetector.cs
│       │   └── SyntheticFidelityDetector.cs          # builds on MinimapShadingMatch
│       ├── BlankTileDetector.cs                      # ports is_blank_what_plate
│       ├── CurationRecord.cs                         # TileCurationRecord / MismatchFinding types
│       ├── CurationManifestWriter.cs                 # Parquet writer for both output tables
│       └── CurationRunRecord.cs                      # v50-curation-run-v1 JSON record
├── tools/harvest/WowViewer.Tool.Harvest/
│   └── Program.cs                                    # + `curate` subcommand (dry-run-first)
├── tests/
│   └── WowViewer.Core.Curation.Tests/                # NEW test project
│       ├── DifficultyBucketClassifierTests.cs
│       ├── HeightNormalMismatchDetectorTests.cs
│       ├── BlankTileDetectorTests.cs
│       ├── SyntheticFidelityDetectorTests.cs
│       └── CurationManifestWriterTests.cs
└── data-harvester/
    ├── src/harvester/
    │   └── curation_store.py                         # NEW thin Parquet reader for the C# manifest
    ├── scripts/
    │   ├── v16_curation.py                            # -> becomes a thin reader/shim (FR-015)
    │   ├── mismatch_detector.py                        # -> becomes a thin reader/shim (FR-015)
    │   ├── spec111/lighting_buckets.py                 # -> becomes a thin reader/shim (FR-015)
    │   └── build_v16_curation_manifest.py               # -> documented retired header (FR-015)
    └── tests/
        └── test_curation_store.py                      # NEW — C#/Python manifest contract test
```

**Structure Decision**: One new C# library (`WowViewer.Core.Curation`) alongside the existing
`WowViewer.Core`/`WowViewer.Core.IO` libraries — a distinct canonical owner because curation is a
derived-analysis concern over already-decoded signals, not a format reader/writer (which
`WowViewer.Core.IO` owns) and not general shared primitives (which `WowViewer.Core` owns). It is
exposed through the existing `WowViewer.Tool.Harvest` as one more subcommand rather than a new
tool project, keeping with constitution II's "CLI tools are thin wrappers" and avoiding a fourth
tool project for what is fundamentally one more pass over harvest output. Python involvement is
deliberately minimized to thin, mechanical readers — the point of this feature is that curation
*logic* has exactly one home from now on, in C#.

## Complexity Tracking

*No violations. Constitution Check passed cleanly; this table is intentionally empty.*

## Phases

> Phases follow the spec's user-story priority (US1 canonical classification → US2 full bucket
> access → US3 synthetic-fidelity finding → US4 legacy consolidation). Each phase ends with
> validation against a real, already-built v50 store — never fixtures alone for the final gate.

### Phase 0 — Research (this plan, Phase 0 output: research.md)

Resolve the concrete decisions the spec deliberately left open: C# project/namespace placement and
CLI surface (D-01), on-disk manifest shape (D-02), invocation point relative to the harvest
pipeline (D-03), legacy-script migration path per script (D-04), and test/validation strategy
(D-05). Also folds in the external research pass comparing the v50 signal catalog against
comparable remote-sensing/game-terrain ML projects, specifically their documented data-curation and
spatial-leakage-handling practices, as corroborating (not gating) evidence for the bucket/finding
design.

### Phase 1 — Design & Contracts (this plan, Phase 1 output: data-model.md, contracts/, quickstart.md)

Entity model (`TileCurationRecord`, `QualityBucket`, `MismatchFinding`, `CurationManifest`,
`SelectionRecord`, `CurationRunRecord`), the two Parquet table schemas, the `curate` CLI contract,
and the user-run quickstart. No code.

### Phase 2 — Tasks (speckit-tasks output: tasks.md)

Dependency-ordered, bite-sized implementation tasks, generated by the `speckit-tasks` command.

### Implementation phases (after tasks.md, executed one at a time)

- **Phase A — US1 canonical classification** (library + CLI, no legacy retirement yet). Build
  `WowViewer.Core.Curation` with the difficulty/coverage/lighting bucket classifiers and the
  height-normal-mismatch/non-finite/has-flag-truthfulness detectors; wire the `curate` subcommand;
  write the two Parquet tables + run record. Validate: real dry-run against an existing v50 store
  prints correct planned counts; real write produces full-coverage output (every tile classified).
- **Phase B — US2 full bucket access proof**. Add/validate the query surface (whether that is
  "just read the Parquet with pandas/pyarrow, filtered by column" or a thin helper) and prove, on
  the real store, that a non-clean bucket returns its full tile set with the same effort as the
  clean bucket. No new storage format — this phase is a proof/documentation phase over Phase A's
  output, per FR-009/SC-002.
- **Phase C — US3 synthetic-fidelity finding**. Extend `MinimapShadingMatch`'s existing
  correlation output into a durable `SyntheticFidelityDetector` finding on tiles with both a
  synthesized and authored minimap; validate against real tiles a human can visually judge as
  good/bad synthetic matches (SC-004).
- **Phase D — US4 legacy consolidation**. Run the SC-003 comparison (legacy `mismatch_detector.py`
  vs. the new C# detector on identical real tiles); on a passing/justified comparison, convert each
  of the six named scripts to a thin reader/shim or a documented-retired header per the per-script
  disposition in research.md D-04; update `docs/architecture/v50-clean-room-dataset-repo-audit-
  2026-07-15.md` with a short pointer to the new canonical curation manifest.
