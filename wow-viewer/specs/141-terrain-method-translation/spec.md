# Feature Specification: Terrain Method Translation and Evidence Gates

**Feature Branch**: `141-terrain-method-translation`

**Created**: 2026-08-10

**Status**: Draft

**Input**: User description: "Use Speckit to plan the right things for translating LiDAR/DSM terrain methods and external aerial-image models into the WoW minimap reconstruction research, while preserving the project's role as a signal-discovery and evidence-led research lane."

## Context

External terrain research separates two different problems:

1. **Surface-observed ground extraction**: LiDAR point clouds or DSMs contain object-top elevation and can be filtered into bare-earth terrain.
2. **RGB-only terrain completion**: an aerial image can suggest object masks and surrounding structure, but it does not directly reveal terrain hidden beneath an object.

WoW minimap reconstruction currently belongs to the second category. This feature records and tests useful methods without allowing a DSM, LiDAR point cloud, MCSH, height target, or target-derived object mask to silently become a deployment input. It also formalizes a small research-lead record so an unusual observation can be preserved as a hypothesis, tested against client-backed evidence, and handed to later investigators without being promoted to fact prematurely.

Spec 139 remains the owner of clean-signal geometry reconstruction. Spec 140 remains the owner of terrain paste, fractal, alpha, and motif archaeology. This feature owns the translation boundary and the evidence needed to decide whether an external method belongs in either lane.

## User Scenarios & Testing

### User Story 1 - Maintain a method evidence ledger (Priority: P1)

A researcher can record an external method, its input modality, output, domain assumptions, source link, license status, reproducibility state, and WoW translation status in one reviewable ledger.

**Why this priority**: The project needs to benefit from adjacent research without repeatedly rediscovering the same methods or importing a method whose input assumptions do not match minimap inference.

**Independent Test**: Build the ledger from the initial candidate set and verify that every method has a modality classification, source provenance, domain-gap note, and explicit status of `reference`, `diagnostic`, `candidate`, `rejected`, or `promoted`.

**Acceptance Scenarios**:

1. **Given** DSM2DTM, ResDepth, SMRF, CSF, aerial object-mask models, and Prithvi are recorded, **when** the ledger is validated, **then** each entry states whether it requires RGB, DSM, point cloud, or a combined input.
2. **Given** a source has unknown license or unavailable weights, **when** it is recorded, **then** it remains reference-only and cannot be treated as a project dependency.

### User Story 2 - Enforce the modality and provenance boundary (Priority: P1)

A researcher can classify a proposed experiment as RGB-only, height-prior, or point-cloud diagnostic and receive a fail-closed decision when its inputs exceed the declared runtime contract.

**Why this priority**: The most dangerous failure is a strong result that only works because it reads a signal that the real minimap never carries.

**Independent Test**: Validate representative manifests for an accepted RGB-only experiment, a DSM diagnostic, a LiDAR diagnostic, and forbidden target-derived input; only the first three receive their appropriate diagnostic classification, and the forbidden case fails.

**Acceptance Scenarios**:

1. **Given** `minimap_rgb` and an optional predicted object mask, **when** the experiment is classified, **then** it is eligible for the RGB-only branch and records the mask as predicted rather than ground truth.
2. **Given** `height_257`, `terrain_shadow_256`, `shadow_mask`, or target-side object masks as model inputs, **when** the experiment is classified for deployment, **then** it is rejected with the exact forbidden signal.
3. **Given** a DSM or point-cloud method, **when** it is evaluated without a declared DSM/point-cloud source, **then** it remains an offline method reference and no runtime claim is emitted.

### User Story 3 - Compare object-aware RGB terrain completion (Priority: P2)

A researcher can compare RGB-only terrain completion with no object mask, a predicted object mask, and a deliberately withheld mask baseline using the existing synthetic object-library controls and authored minimap RGB corpus.

**Why this priority**: This is the closest WoW analogue to DSM-to-DTM research while respecting that the runtime only sees image evidence.

**Independent Test**: Run a CPU dry plan and a user-owned training/evaluation plan that report clean-height, contaminated-input, object-mask, cross-tile, and family metrics independently against identity and tile-mean baselines.

**Acceptance Scenarios**:

1. **Given** a synthetic object-library control with a known clean target, **when** the RGB-only benchmark runs, **then** ground-truth object masks are used only for evaluation or loss supervision and never serialized as inference inputs.
2. **Given** an authored minimap RGB row without a target-side object mask, **when** the benchmark runs, **then** it can evaluate RGB-only behavior without claiming that the hidden terrain is observed.
3. **Given** the predicted-mask branch improves aggregate height error but fails the cross-tile or clean-identity gate, **when** the report is written, **then** promotion is held and the failed signal is named independently.

### User Story 4 - Preserve novel research leads for follow-up (Priority: P3)

A researcher can record an unusual correlation or recovered signal as a bounded lead with source rows, hypothesis, test, result, confidence, and next action.

**Why this priority**: The project’s value includes noticing signals that are not yet documented elsewhere. A lead should survive beyond the first glance without becoming an unsupported historical claim.

**Independent Test**: Create a lead from a synthetic control and a client-backed observation, then validate that the record preserves provenance, separates observation from interpretation, and cannot mark itself promoted without a linked evidence report.

**Acceptance Scenarios**:

1. **Given** a candidate relationship between a minimap feature and terrain structure, **when** it is logged, **then** the record includes the exact source group, build, tile/window, signal availability, and proposed falsification test.
2. **Given** a lead with only one visual example, **when** it is reviewed, **then** its state remains `unconfirmed` and the missing evidence is explicit.

## Edge Cases

- A method may accept RGB and DSM together; it must not be classified as RGB-only.
- A predicted object mask may be useful but wrong; mask quality and terrain quality must be reported separately.
- A DSM/DTM method may remove walls or bridges as non-ground; this is a diagnostic limitation, not a reason to relabel the output as WoW terrain truth.
- A source repository may change, disappear, or lack reproducible weights; the ledger must preserve the access date and keep the method reference-only.
- A real minimap may have missing alpha, auxiliary, or object signals; unavailable data remains unavailable.
- Similar fractal or albedo patterns may produce a false motif match; recurrence requires independent source groups and spatial evidence.
- A result can beat aggregate MAE while failing clean identity, cross-tile continuity, or one output head; it must remain a partial failure.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST record each investigated external method with source URL, source kind, access date, input modality, expected output, domain assumptions, license/weight status, and WoW translation status.
- **FR-002**: The system MUST classify experiments into `rgb_only`, `height_prior`, `point_cloud`, or `combined` input contracts before evaluation.
- **FR-003**: The system MUST fail closed when a deployment-bound experiment reads WDL, height targets, MCSH/shadow targets, target-side object masks, or any other declared forbidden signal.
- **FR-004**: The system MUST distinguish an offline DSM/point-cloud diagnostic from a runtime-compatible RGB-only experiment in manifests and reports.
- **FR-005**: The system MUST preserve the difference between an observed RGB feature, a predicted object mask, a source-side supervision mask, and a target-side validation mask.
- **FR-006**: The system MUST report clean-height, contaminated-input, object-mask, cross-tile, family, and baseline-relative metrics independently whenever those signals are present.
- **FR-007**: The system MUST provide a deterministic RGB-only comparison plan covering no-mask, predicted-mask, and withheld-mask conditions before any user-owned heavy run.
- **FR-008**: The system MUST not require external model weights or copyrighted datasets for the first project-owned benchmark; external methods may supply architecture and algorithm references.
- **FR-009**: The system MUST record a promotion decision of `reference`, `diagnostic`, `candidate`, `hold`, `rejected`, or `promoted`, including the evidence report that justifies it.
- **FR-010**: The system MUST support research-lead records containing hypothesis, observation, provenance, falsification test, result, confidence, and next action.
- **FR-011**: The system MUST keep heavy corpus generation, training, and broad client harvests user-launched and confirmation-gated.
- **FR-012**: The system MUST update the relevant Speckit plan and memory-bank continuity files at the end of each completed implementation slice.

### Key Entities

- **External Method Record**: A researched method and its source, assumptions, modality, reproducibility, and translation status.
- **Input Contract**: The declared observable and forbidden signals for one experiment branch.
- **Method Evidence Run**: A deterministic evaluation plan/result tied to a method, corpus, split, baseline, and report.
- **Research Lead**: A falsifiable observation or hypothesis with client/data provenance and follow-up state.
- **Translation Decision**: The evidence-backed classification that determines whether a method remains reference-only, becomes a diagnostic, or enters a candidate model branch.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The initial external-method set has 100% modality, provenance, domain-gap, and translation-status coverage before implementation proceeds.
- **SC-002**: The modality audit emits zero false-positive RGB-only classifications for representative DSM, point-cloud, combined-input, and target-derived fixtures.
- **SC-003**: The first RGB-only benchmark reports all three mask conditions plus identity and tile-mean baselines with deterministic split and provenance hashes.
- **SC-004**: Every candidate output reports independent clean-height, contaminated-input, cross-tile, and family metrics; no aggregate-only promotion is permitted.
- **SC-005**: A promoted RGB-only branch beats its declared baseline on both final-height and boundary/cross-tile acceptance metrics without violating the forbidden-read audit; otherwise the decision is `hold` or `rejected`.
- **SC-006**: Every external method in the initial ledger has a reproducible local note or an explicit reason it remains reference-only.
- **SC-007**: Every research lead contains at least one source-group/build/tile provenance reference and one falsification or follow-up test before it can leave `unconfirmed`.
- **SC-008**: Repeating a dry-run with identical inputs produces identical method IDs, contract decisions, row hashes, split assignments, and report schema.

## Assumptions

- The current deployed observation is RGB minimap data; no LiDAR point cloud or DSM is assumed to exist at runtime.
- `height_257`, `terrain_shadow_256`, raw MCSH/shadow masks, and target-side object masks remain training/evaluation references and are not deployment inputs.
- Spec 139 supplies the clean-signal geometry model and Spec 140 supplies motif/alpha archaeology; this feature adds evidence and translation gates rather than replacing either owner.
- Existing project-owned synthetic controls, the object-library sieve, and the authored raw-RGB corpus are the first benchmark sources.
- External Hugging Face/GitHub models may be inspected and cited, but external weights and datasets are not required for the first benchmark.
- The user runs any large corpus build, CUDA training, or broad real-data harvest after the dry plan and proof gates pass.
- UniqueID timeline archaeology is a related future research surface; this feature preserves the same hypothesis-to-evidence discipline but does not implement a general UniqueID timeline analyzer.

## Out of Scope

- Shipping a generic LiDAR or aerial-imagery product.
- Treating DSM2DTM, ResDepth, SMRF, or CSF as drop-in RGB-minimap inference.
- Importing external weights or datasets into the project-owned training corpus before licensing and domain evidence are resolved.
- Claiming terrain hidden below an object is directly observed by an RGB minimap.
- Broad multi-client harvesting, a new renderer, exact object identity, or a general UniqueID timeline product.
- Launching user-owned GPU training or long-running harvests from Codex.
