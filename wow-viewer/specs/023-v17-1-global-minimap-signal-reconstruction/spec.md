# Feature Specification: V17.1 Global Minimap-Signal Reconstruction Contract

**Feature Branch**: `023-v17-1-global-minimap-signal-reconstruction`

**Created**: 2026-05-24

**Status**: Draft

## Problem Statement

V16 base-style multi-signal training is not the target architecture for current work. It mixes too many weakly-convergent objectives and causes repeated confusion about what each model is supposed to learn.

V16.1 direction is the intended architecture:

- one model per signal
- single-purpose supervision
- explicit composition of outputs

For the normal model specifically, the contract is:

- input: minimap RGB only
- output: normals only
- height is training-time supervision only (teacher/prior), not a model input/output contract

This spec ratifies a global V17.1 contract so future work does not drift back to base V16 behavior.

## Scope

Define one canonical, repo-wide contract for:

1. Per-signal model boundaries (especially normals)
2. Training/inference contracts for each signal model
3. Zarr reconstruction pipeline (generated signals)
4. Export/validation flow from reconstructed signals

## User Scenarios & Testing

### User Story 1 — Normals Model Contract Is Unambiguous (Priority: P1)

A researcher launches normals training and can verify from config/logs that the model learns only `minimap -> normals` with no extra model outputs.

**Independent Test**: 1-epoch sanity run prints resolved contract and writes it in config.

**Acceptance Scenarios**:

1. **Given** a V17.1 normals run, **When** startup logs print resolved contract, **Then** they show `input=minimap_rgb`, `output=normals_xyz`, and `height_supervision_only=true`.
2. **Given** a run config, **When** reading it after epoch 1, **Then** it contains explicit contract fields and no normal-model multi-output fields.

---

### User Story 2 — Per-Signal V16.1 Decomposition Is Enforced (Priority: P1)

A developer cannot accidentally route V17.1 training through base V16 multi-signal paths.

**Independent Test**: attempt to launch V17.1 with base-V16 trainer path fails fast.

**Acceptance Scenarios**:

1. **Given** a V17.1 command, **When** it resolves training entrypoint, **Then** only V16.1 per-signal trainers are allowed.
2. **Given** conflicting flags or legacy paths, **When** run starts, **Then** it exits with a clear error (no silent fallback).

---

### User Story 3 — Reconstructed Zarr Is Canonical Integration Surface (Priority: P1)

Researcher runs per-signal models, writes reconstructed signals to a generated Zarr datastore, and validates/exports from that datastore.

**Independent Test**: reconstruction run produces generated-signal Zarr and export tools consume it.

**Acceptance Scenarios**:

1. **Given** trained per-signal checkpoints, **When** reconstruction runs, **Then** outputs are written to a generated-signal Zarr root with stable field names.
2. **Given** generated-signal Zarr, **When** terrain/object export runs, **Then** validation artifacts can be produced without re-reading training internals.

---

### User Story 4 — Data Curation Rejects Cross-Signal Mismatch (Priority: P1)

Training pools exclude tiles with obvious minimap/height/normal mismatch.

**Independent Test**: curation summary shows mismatch rejection counts and selected tiles pass thresholds.

**Acceptance Scenarios**:

1. **Given** curated pools, **When** evidence is written, **Then** mismatch metrics and rejection counts are recorded.
2. **Given** selected train/val sets, **When** previewing validation rows, **Then** severe minimap-height-normal mismatch cases are absent.

---

### User Story 5 — MdxViewer Precise Object Mask Capture Is Manifest-Driven (Priority: P1)

Researcher captures precise object masks only for a pre-curated manifest tile set, instead of full-map/full-client brute force capture.

**Independent Test**: run capture using a curated manifest; only listed tiles are processed; output completeness is measured against manifest.

**Acceptance Scenarios**:

1. **Given** a curated manifest with tile bucketing, **When** precise mask capture starts, **Then** capture jobs are spawned only for manifest tiles.
2. **Given** capture output with missing tiles, **When** training dataset selection is built from the same manifest, **Then** missing tiles are excluded automatically and training proceeds without requiring global capture completeness.
3. **Given** per-tile capture evidence, **When** run summary is generated, **Then** it reports captured/failed/skipped tile counts keyed by manifest IDs.

## Requirements

### Functional Requirements

- **FR-001**: V17.1 normals model MUST be single-output (`normals_xyz`) and single-input (`minimap_rgb`) at model contract level.
- **FR-002**: Height usage in V17.1 normals training MUST be supervisor-only (teacher/prior/loss shaping), not a persisted model I/O contract.
- **FR-003**: V17.1 workflow MUST reject base V16 multi-signal training paths for normals experiments.
- **FR-004**: Trainer MUST write resolved contract fields in logs and config for every run.
- **FR-005**: Curation MUST include explicit mismatch filters/metrics for minimap-height-normal coherence.
- **FR-006**: Validation pools MUST support controlled rotation while preserving a fixed holdout for comparability.
- **FR-007**: Reconstructed/generated signals MUST be written to a dedicated Zarr surface for downstream export/validation.
- **FR-008**: Export/validation tools MUST consume generated-signal Zarr without requiring training-time code paths.
- **FR-009**: MdxViewer capture improvements for object-mask generation MUST support batched tile capture to reduce capture overhead in dataset synthesis workflows.
- **FR-010**: Precise object-mask capture MUST support manifest-driven tile targeting and MUST NOT require full-map/full-client capture as a prerequisite.
- **FR-011**: Dataset selection MUST be able to consume only tiles with completed capture artifacts from the same manifest, tolerating holes outside the selected manifest scope.
- **FR-012**: Capture pipeline MUST emit per-manifest completion evidence (captured/failed/skipped), so training reproducibility is tied to manifest state.

### Key Entities

- **Per-Signal Model**: a model that predicts one signal only.
- **Supervisor-Only Signal**: a signal used for training guidance but not as model input/output contract.
- **Generated-Signal Zarr**: canonical reconstructed datastore produced from checkpoint inference.
- **Mismatch Curation Gate**: curation filter that rejects low-coherence cross-signal tiles.

## Success Criteria

- **SC-001**: Normals runs consistently report and persist `minimap->normals` + `height_supervision_only` contract fields.
- **SC-002**: No V17.1 normals run can be launched through base V16 trainer paths.
- **SC-003**: Curated training/validation pools include explicit mismatch rejection metrics and improved validation coherence.
- **SC-004**: Generated-signal Zarr is produced and used for at least one end-to-end export/validation flow.
- **SC-005**: Manifest-driven precise-mask capture completes for a bounded curated tile set and trains successfully without requiring global tile coverage.

## Assumptions

- V16.1 family remains the active foundation for per-signal decomposition.
- Existing V16 dataset roots and curation outputs remain available under `wow-viewer/output/datasets/v16`.
- Height supervision improves normal convergence when used as training guidance, even when normals model I/O stays minimap->normals only.
