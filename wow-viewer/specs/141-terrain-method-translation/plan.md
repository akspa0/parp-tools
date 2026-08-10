# Implementation Plan: Terrain Method Translation and Evidence Gates

**Branch**: `141-terrain-method-translation` | **Date**: 2026-08-10 | **Spec**: [spec.md](spec.md)

## Summary

Create a small, project-owned evidence lane that records external LiDAR/DSM and aerial-image methods, enforces their input modality, and compares only the methods that can be translated honestly into WoW minimap reconstruction. The first executable branch is RGB-only object-aware terrain completion. DSM and point-cloud methods remain offline diagnostic references until a legitimate source and contract exist.

**Implementation status (2026-08-10)**: Phase 0 and Phase 1 are complete. The v60 library now exposes
six versioned external method records, four input-contract branches, canonical signal aliases, and
fail-closed input-read audits. The dry-run CLI and 17 focused tests are proven; no external weights,
client data, corpus build, or training run was used.

Phase 2 is now implemented as a manifest-only planner. The real `object-library-sieve-v3` artifact
produced a valid 540-row control plan (304 train / 236 validation), but it is explicitly not runtime
RGB-compatible because its input is `objectified_terrain_shadow_256`. The authored raw-RGB condition
remains pending until its user-built corpus exists. The planner adds 5 focused tests; full v60 proof is
111 passing tests.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: Existing `data-harvester` v60 contracts, NumPy, Zarr/Parquet readers, PyTorch, Pillow, and existing test/lint tooling. No new external model dependency for the first slice.

**Storage**: Versioned JSON evidence manifests and reports; existing v60 corpora and project-owned Zarr sources remain read-only inputs unless a later task explicitly builds a derived corpus.

**Testing**: Focused pytest contract tests, `ruff`, `py_compile`, deterministic hash checks, dry-run CLI audits, and user-run CUDA evaluation when authorized by a completed plan gate.

**Target Platform**: Windows PowerShell development; CPU-compatible inspection and dry-run; user-owned CUDA for heavy training.

**Project Type**: Python research/data library plus CLI evidence tools.

**Performance Goals**: Method and contract audits must complete on small fixtures without GPU. Corpus generation and training performance are user-run measurements, not claims of this plan.

**Constraints**: No target-derived deployment input; no external weights/datasets in the first benchmark; no broad harvest or training launched by Codex; all client roots configured and provenance-bound.

**Scale/Scope**: Initial ledger of six method families, one RGB-only benchmark plan, one optional DSM/point-cloud diagnostic contract, and bounded research-lead records.

## Constitution Check

| Gate | Status | Evidence |
|---|---|---|
| Repo independence | PASS | All future code and documents remain under `wow-viewer/`. |
| Real-data validation | PASS | Real-data claims require configured client/source provenance; synthetic controls are explicitly labeled controls. |
| Evidence per signal | PASS | Clean, contaminated, mask, cross-tile, family, and baseline metrics remain separate. |
| No hidden deployment inputs | PASS | Modality contracts and forbidden-read audit fail closed. |
| User owns heavy work | PASS | Training, broad harvest, and external-weight use are deferred and confirmation-gated. |
| One phase at a time | PASS | Every phase has an independent proof gate before the next phase. |
| Documentation hygiene | PASS | Spec, research, contract, quickstart, tasks, and memory-bank updates are part of the delivery slice. |

## Project Structure

### Documentation

```text
wow-viewer/specs/141-terrain-method-translation/
├── spec.md
├── research.md
├── plan.md
├── data-model.md
├── quickstart.md
├── contracts/method-translation.schema.md
├── contracts/rgb-method-benchmark.schema.md
├── checklists/requirements.md
└── tasks.md
```

### Current Source Code

```text
wow-viewer/data-harvester/
├── src/harvester/v60/
│   ├── terrain_method_translation.py   # method records, modality audit, decisions
│   └── research_leads.py               # bounded hypothesis/provenance records
├── scripts/
│   ├── v60_audit_terrain_methods.py   # ledger and contract dry-run
│   └── v60_build_rgb_method_benchmark.py # no-mask/predicted-mask/withheld-mask plan
└── tests/v60/
    ├── test_terrain_method_translation.py
    └── test_research_leads.py
```

**Structure Decision**: Keep the method ledger, contract audit, research leads, and benchmark preparation in the existing v60 Python library/CLI surface. Do not create a second training pipeline, modify format readers, or add external model runtime dependencies in this feature’s first phase.

## Phase Plan and Proof Gates

### Phase 0 — Method evidence ledger

1. Freeze the six initial method records and source URLs in a versioned evidence artifact.
2. Record input modality, outputs, domain gap, license/weights status, and translation status.
3. Add deterministic validation for missing provenance and contradictory modality claims.

**Gate 0**: Every initial method has complete ledger fields and a reference/diagnostic/candidate status.

### Phase 1 — Modality and forbidden-read contract

1. Implement the four branch values and signal allowlists.
2. Validate accepted RGB-only, height-prior, point-cloud, combined, and forbidden fixtures.
3. Emit an exact forbidden-read audit and fail closed for deployment claims.

**Gate 1**: Representative contracts classify correctly with zero forbidden deployment reads.

### Phase 2 — RGB-only object-aware benchmark preparation

1. Define no-mask, predicted-mask, and withheld-mask benchmark conditions.
2. Reuse project-owned object-library controls and authored raw-RGB provenance without serializing target-side masks as inference inputs.
3. Emit deterministic split, baseline, artifact, and user-run command plans.

**Gate 2**: The CPU dry plan is reproducible and reports all required independent metrics before any GPU run is recommended.

### Phase 3 — Optional height-prior diagnostic

1. Add a diagnostic-only contract for an explicitly supplied DSM or point-cloud source.
2. Record SMRF/CSF/DSM2DTM/ResDepth comparisons without calling them RGB-compatible.
3. Hold the phase if no valid project-owned source exists; absence is a valid result.

**Gate 3**: Every height-prior result is provenance-bound and cannot enter the RGB deployment report.

### Phase 4 — Research leads and translation decision

1. Add hypothesis, observation, provenance, falsification, result, confidence, and next-action records.
2. Join method evidence runs to translation decisions.
3. Update Spec 139/140 handoff notes only when evidence passes the declared gates.

**Gate 4**: Each lead and method has an explicit next action or a justified hold/rejection; no unsupported discovery is promoted.

## Risks and Mitigations

- **Input analogy error**: make modality classification mandatory before comparing metrics.
- **Object-mask leakage**: distinguish predicted, supervision-only, and validation-only masks in the manifest.
- **Domain mismatch**: external aerial models remain reference/candidate until WoW-specific evidence exists.
- **False motif or correlation discovery**: require independent source groups, split identity, and falsification tests.
- **Aggregate metric concealment**: report per-signal and per-family metrics and retain partial failures.
- **Scope creep into a general UniqueID or LiDAR product**: keep those as future research surfaces outside this feature.

## Complexity Tracking

No constitution violations are required. The optional height-prior branch is intentionally separate rather than adding complexity to the RGB-only contract.
