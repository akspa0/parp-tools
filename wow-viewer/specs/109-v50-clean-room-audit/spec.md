# Feature Specification: V50 Clean-Room Dataset and Repository Audit

**Feature Branch**: `109-v50-clean-room-audit`

**Created**: 2026-07-15

**Status**: Active audit

**Input**: User description: "Begin the v50 clean-room dataset and repo cleanup audit. Do not
reuse or trust old datasets until verified. Recover the missing v50 rename plan, build one complete
new v50 dataset, port sane V18 data only after verification, then delete approved old datasets and
temporary files to recover disk space. Use a user-configured faster-SSD client library instead of
keeping project-local temporary client copies."

## User Scenarios & Testing

### User Story 1 - Establish a Fail-Closed Dataset Trust Boundary (Priority: P1)

The operator can distinguish verified v50 inputs from historical, incomplete, or unknown-origin
artifacts before any dataset build, training, inference, or evaluation consumes them.

**Why this priority**: A release label alone cannot establish data correctness. Reusing an
unverified store would contaminate every downstream result.

**Independent Test**: Present the workflow with an old store, an unversioned store, a store with a
missing provenance field, and a fully verified store. Only the fully verified store is eligible for
promotion.

**Acceptance Scenarios**:

1. **Given** a pre-v50 dataset or a dataset with unknown provenance, **When** it is inventoried,
   **Then** it is marked unverified and cannot be consumed by a v50 run.
2. **Given** an artifact carrying a v50 label but lacking independent source proof, **When** it is
   audited, **Then** the label is rejected as proof of trust.
3. **Given** a dataset whose provenance, schema, content checks, and row lineage all pass,
   **When** the audit is reviewed, **Then** it may be promoted as an explicit v50 source.

---

### User Story 2 - Classify Repository Cleanup Without Losing Work (Priority: P2)

The maintainer receives an evidence-backed disposition for legacy commands, duplicate workflow
owners, generated artifacts, local environments, stale documentation, and user-authored changes.

**Why this priority**: Cleanup must reduce ambiguity without deleting active or unreviewed work.

**Independent Test**: Run the inventory on a dirty worktree and confirm every identified item has a
reasoned `keep`, `quarantine`, `verify`, `migrate`, or `remove-candidate` state while all user changes
remain intact.

**Acceptance Scenarios**:

1. **Given** an untracked or modified user file, **When** the audit runs, **Then** it is never deleted
   or overwritten.
2. **Given** two command surfaces claiming the same workflow, **When** ownership is audited,
   **Then** one canonical owner and a bounded migration or retirement action are identified.
3. **Given** a file that violates a repository boundary, **When** it is classified,
   **Then** removal is deferred until its dependencies and owner are verified.

---

### User Story 3 - Build and Promote the Complete V50 Baseline (Priority: P3)

The operator can build and review a complete v50 dataset whose per-build stores contain every
approved signal, whose training subsets are manifests rather than duplicate data copies, and whose
release manifest binds source evidence, dataset identity, splits, and downstream compatibility.

**Why this priority**: A clean baseline makes later small-model experiments reproducible without
reopening historical datasets by accident.

**Independent Test**: Starting from the approved evidence set, independently reproduce the v50
store identities, confirm every expected signal and row is accounted for, and verify that a
mismatched dataset, prior archive, or checkpoint fails closed.

**Acceptance Scenarios**:

1. **Given** an approved v50 dataset, **When** its evidence is rechecked, **Then** its identity and
   row lineage reproduce exactly.
2. **Given** artifacts from different v50 releases, **When** they are combined, **Then** the workflow
   refuses the combination before expensive work starts.
3. **Given** a V18 signal or row that passes the independent audit, **When** the v50 dataset is built,
   **Then** it is copied bit-for-bit with source lineage rather than reinterpreted or merely relabeled.
4. **Given** a V18 signal or row that fails or lacks proof, **When** the v50 dataset is built,
   **Then** it is re-extracted from an approved client build or omitted with an explicit gap record.
5. **Given** a historical liquid signal whose WL fallback lacks continuous-surface provenance,
   **When** the v50 dataset is built, **Then** `liquid_mask` and `liquid_height` are freshly
   extracted and the historical payload is rejected rather than copied or relabeled.

### Edge Cases

- A legacy dataset has valid-looking arrays but no reproducible source manifest.
- A v50-labeled dataset was created by copying or relabeling old rows without verifying them.
- Source paths still exist but their contents changed after the dataset was created.
- A dataset contains zero-filled optional arrays while its index claims those signals are present.
- Multiple rows share a source group across training and validation partitions.
- A cleanup candidate is needed by local tooling outside the v50 workflow.
- Ignored output roots contain valuable user artifacts that are invisible to normal Git status.
- The faster-SSD client library moves or changes after a dataset manifest is written.
- A verified V18 row contains one sound signal and one known-defective signal.
- Cleanup is interrupted after some approved artifacts have been deleted.

## Requirements

### Functional Requirements

- **FR-001**: Every dataset, model, checkpoint, generated prior, manifest, and derived report that
  predates the verified v50 baseline MUST begin in the `unverified` state.
- **FR-002**: A v50 name, directory, attribute, or release string MUST NOT by itself establish trust.
- **FR-003**: The audit MUST record each artifact's location, kind, owner, observed identity,
  provenance evidence, verification status, and proposed disposition.
- **FR-004**: Dispositions MUST be one of `keep`, `quarantine`, `verify`, `migrate`, or
  `remove-candidate`, with a stated reason and proof needed for the next transition.
- **FR-005**: Promotion of a dataset MUST require reproducible source identity, extraction/build
  identity, schema and dtype/shape validation, row-count agreement, row-level lineage, partition
  leakage checks, required-signal truthfulness, and content-integrity hashes.
- **FR-006**: A builder MUST NOT convert an unverified source into a trusted v50 artifact merely by
  copying rows or writing v50 metadata.
- **FR-007**: Downstream v50 workflows MUST fail before training or inference when dataset,
  generated-prior, checkpoint, manifest, or release identities are absent or incompatible.
- **FR-008**: The repository audit MUST identify historical command wrappers, duplicate owners,
  stale spec labels, misplaced environment files, ignored generated roots, and oversized continuity
  documents that obscure the current route.
- **FR-009**: The audit MUST preserve all pre-existing modified and untracked user files. Deletion
  MUST occur only through a reviewed cleanup manifest after its v50 replacements and retained
  evidence are verified.
- **FR-010**: No training, harvest, broad capture, GPU job, or destructive cleanup may be launched as
  part of this audit.
- **FR-011**: The canonical v50 command and module surface MUST not depend on a historical spec name
  once it is promoted as the owner. Proven reusable logic may be migrated only after focused proof.
- **FR-012**: Audit reports MUST distinguish metadata/contract proof, sampled content proof,
  full-content proof, and model-quality proof.
- **FR-013**: Trusted game inputs MUST come from an approved configured client library, and every
  validation report MUST name the client-library identity, exact build, and fingerprint used.
- **FR-014**: Client roots MUST be runtime configuration. The faster-SSD client library MUST be
  approved in repository policy and MUST NOT be hardcoded into source files. `H:\CLIENTS` is the
  current user-approved preferred root for v50 work.
- **FR-015**: The v50 dataset MUST be a new canonical per-build store family, not a renamed V18,
  Spec 103, or Spec 108 store and not a mixed copy created solely for one training run.
- **FR-016**: Every sound V18 signal considered for migration MUST be audited independently. Passing
  one signal MUST NOT promote other signals in the same row or store.
- **FR-017**: Known-defective V18 data, including uncorrected hole masks and every historical
  `liquid_mask`/`liquid_height` payload without `wl_liquid_surface_quads_v1` provenance, MUST NOT
  be ported. Liquid signals are `fresh-only` for v50 until a client-backed extraction records its
  authoritative source; known-defective inputs MUST be re-extracted or recorded as unavailable.
- **FR-018**: V50 training curricula MUST select rows by immutable manifests or views over canonical
  stores so repeated experiments do not duplicate full dataset payloads.
- **FR-019**: The v50 rename MUST establish canonical v50 module and command ownership. Historical
  spec-named modules may remain only as tested compatibility shims during migration.
- **FR-020**: Cleanup planning MUST measure candidate size, verify that no retained manifest depends
  on the candidate, bind the candidate path and identity into a cleanup manifest, and support dry-run
  review before deletion.
- **FR-021**: Cleanup application MUST refuse paths outside approved generated-data roots and MUST
  never delete configured client libraries, source code, specs, audit evidence, or the active v50
  release.
- **FR-022**: Old project-local client copies, obsolete datasets, duplicate model outputs, cloud
  downloads, and temporary artifacts MAY be deleted only after the reviewed cleanup manifest marks
  them approved and the user runs the cleanup command.

### Key Entities

- **Artifact Record**: One discovered dataset, model, checkpoint, prior archive, manifest, report,
  local environment, command surface, or documentation owner and its audit state.
- **Provenance Record**: Reproducible source identity, producing workflow identity, parameters,
  timestamps, content hashes, and parent-child lineage.
- **Verification Record**: Checks performed, scope of proof, result, evidence location, and reviewer.
- **Disposition Record**: Current state, reason, dependencies, next proof, and approved action.
- **V50 Release Manifest**: The reviewed set of compatible source, dataset, split, model, and derived
  artifact identities for one v50 release.
- **Cleanup Manifest**: Immutable list of approved deletion targets, measured sizes, dependency
  checks, replacement evidence, and expected recovered bytes.

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% of discovered historical dataset and model roots are explicitly classified; none
  are silently treated as trusted.
- **SC-002**: 100% of promoted v50 datasets have complete provenance, schema, lineage, partition,
  signal-truthfulness, and content-integrity evidence.
- **SC-003**: Every missing or mismatched identity tested is rejected before any expensive operation
  begins.
- **SC-004**: No pre-existing modified or untracked user file is lost during the audit.
- **SC-005**: The final repository report names one canonical owner for every retained v50 workflow
  and gives a bounded disposition for every duplicate owner.
- **SC-006**: A reviewer can reproduce the identity of every promoted v50 artifact from its manifest
  without relying on chat history or an old dataset name.
- **SC-007**: The complete v50 dataset accounts for 100% of expected rows and required signals as
  verified, freshly extracted, or explicitly unavailable; no unknown state remains.
- **SC-008**: Every copied V18 payload matches its verified source hash, while every known-defective
  or unverified payload is absent from the migrated set.
- **SC-009**: The cleanup dry run reports exact candidate paths and expected recovered bytes, and the
  post-cleanup report accounts for every approved target without touching protected roots.

## Assumptions

- Existing stores may be inspected as evidence, but inspection does not authorize their reuse.
- Full verification, migration, fresh dataset building, and cleanup application remain user-run
  operations because they are resource intensive or destructive; Codex prepares and tests the tools.
- Existing v50 worktree changes are user work and remain untouched unless a later bounded fix is
  clearly within the approved audit plan.
- The faster-SSD client-library path is `H:\CLIENTS`. It is supplied as runtime configuration and
  recorded in operator evidence, not baked into source code.
- Repository cleanup inventory and dry-run review precede deletion or archival actions.
- The user explicitly designated both legacy output trees as disposable clean-slate state on
  2026-07-15. Tracked code/evidence is relocated or removed from tracking before the guarded cleanup
  command is allowed to empty those roots.
