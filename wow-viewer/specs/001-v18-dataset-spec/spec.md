# Feature Specification: V18 Dataset Canonical Contract

**Feature Branch**: `001-v18-dataset-spec`

**Created**: 2026-05-27

**Status**: Draft

**Input**: User description: "let's turn the v18 dataset into a real spec - use speckit"

## Problem Statement

The current V18 terrain workflow has a strong training and refinement identity, but
the dataset-build contract beneath that workflow is still split across the older
V16 creation path plus a growing set of follow-up patch steps.

That split creates four practical risks:

1. operators can finish a nominal dataset build and still need extra patch steps
   before the dataset actually reflects the current expected signal surface,
2. decoded metadata and newly promoted signals can drift from tile index
   coverage if they are not first-class build outputs,
3. merge flows can weaken provenance when source datasets were created before the
   newer metadata and signal surfaces were mandatory, and
4. future raw-blob preservation work can accidentally destabilize the main build
   contract if it is not kept explicitly additive.

This spec defines V18 as the canonical successor to the current V16 dataset
creation flow: same basic build intent, versioned forward, but with decoded
metadata and the currently expected patch-on signal families promoted into the
main build contract so the first completed V18 build is already the publishable
dataset.

## Goal

Define a canonical V18 dataset-build contract that:

1. promotes the current V16 dataset creation workflow into a versioned V18
   build path rather than inventing a separate dataset pipeline,
2. folds decoded metadata and the currently expected patch-on signal families
   directly into the canonical V18 build output,
3. produces a publishable dataset in one end-to-end workflow,
4. enforces validation parity between tile index coverage, decoded metadata
   coverage, and required promoted signal coverage, and
5. keeps raw blob preservation as an additive sidecar path rather than a
   breaking change to the decoded contract.

## Scope Boundaries

### In Scope

- the canonical definition of a finalized V18 dataset build
- the rule that V18 is the direct dataset-build successor to the current V16
  creation flow
- mandatory dataset artifacts and validation outputs
- promotion of decoded metadata and current patch-on signal families into the
  main build contract
- decoded metadata completeness and parity rules
- merge expectations for datasets with and without preexisting decoded metadata
- additive raw blob preservation as a future-safe sidecar path

### Out of Scope

- changing model architectures, loss functions, or training-run behavior
- replacing current decoded signal extraction semantics with a wholly new build
  family
- redesigning the end-to-end dataset pipeline to bypass the current raw array /
  shard interchange shape entirely
- requiring immediate archival completeness for every undecoded source byte
- rewriting client file readers or introducing a second permanent dataset
  contract for the same decoded outputs

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Promote the Current V16 Build Flow into Canonical V18 (Priority: P1)

As a dataset operator, I can run one canonical dataset-build workflow that
behaves like the current V16 dataset creation flow but publishes a complete V18
dataset contract, including current required signals and validation outputs,
without requiring a separate patch phase to make the dataset publishable.

**Why this priority**: This is the core reliability requirement. The user wants
V18 to be a straightforward versioned successor to V16, not a new parallel
pipeline with more moving parts.

**Independent Test**: Execute one canonical V18 build for a staged client build
and selected map set, then verify that it produces the full expected artifact
set and does not require any follow-up patch step for signals that are already
part of the current promoted V18 contract.

**Acceptance Scenarios**:

1. **Given** a staged client build and selected map set, **When** the canonical
   build flow is run successfully, **Then** it produces a finalized dataset with
   all mandatory artifacts and validation reports in the same end-to-end run.
2. **Given** a canonical build output, **When** its validation reports are
   reviewed, **Then** they confirm both signal coverage integrity and decoded
   metadata parity for the finalized dataset.
3. **Given** the current V16 operator workflow, **When** an operator switches to
   the V18 build surface, **Then** the workflow remains recognizably the same in
   purpose and sequencing while producing the richer V18 output contract.
4. **Given** a canonical build running in strict mode, **When** required signal
   validation or decoded metadata validation fails, **Then** the build does not
   produce a publishable finalized dataset status.

---

### User Story 2 - Promote Metadata and Patched Signals into First-Class V18 Outputs (Priority: P1)

As a data consumer, I can rely on V18 dataset builds to include decoded metadata
for every accepted tile and all signal families that are currently being patched
onto V16 datasets, so no required tile-level context is stranded in a separate
post-build workflow.

**Why this priority**: The user explicitly wants the extra metadata and newly
required signals folded into V18 itself instead of remaining bolt-on repairs to
V16.

**Independent Test**: Compare tile coverage between index records, decoded
metadata records, and promoted signal coverage records for a finalized dataset,
then validate that each required V18-promoted surface is present without
separate patch completion.

**Acceptance Scenarios**:

1. **Given** a finalized dataset, **When** tile identifiers from the index and
   decoded metadata surfaces are compared, **Then** coverage is one-to-one with
   no missing, duplicate, or extra tile identifiers.
2. **Given** decoded metadata rows, **When** required payload fields are
   validated, **Then** each row provides a valid decoded metadata object, tile
   identity, source provenance, and decoded-structure summary fields.
3. **Given** a finalized V18 dataset, **When** signal families that previously
   required patch-on promotion are reviewed, **Then** they are present as part
   of the canonical dataset contract rather than as optional after-the-fact
   additions.
4. **Given** a tile whose decoded metadata is minimal, placeholder-backed, or
   sourced from a legacy fallback path, **When** the dataset is inspected,
   **Then** the tile remains present in the dataset with explicit metadata state
   indicators rather than being silently dropped.

---

### User Story 3 - Preserve Raw Blob Expansion Path Without Breaking Current Consumers (Priority: P2)

As a format-research and migration owner, I can extend the dataset with raw blob
preservation artifacts while keeping current decoded-contract consumers stable.

**Why this priority**: This enables future undecoded-byte preservation and
research while avoiding regressions in existing V16/V18 decoded-contract
consumers.

**Independent Test**: Confirm decoded-contract artifacts remain unchanged for
existing consumers when raw blob preservation is disabled, and confirm that
sidecar blob artifacts can be produced with traceable provenance when enabled.

**Acceptance Scenarios**:

1. **Given** raw blob preservation is disabled, **When** the canonical build
   runs, **Then** the decoded-contract outputs remain complete, valid, and fully
   consumable on their own.
2. **Given** raw blob preservation is enabled, **When** the canonical build
   runs, **Then** sidecar blob manifest and payload artifacts are produced with
   traceable linkage back to tile and chunk provenance.
3. **Given** a finalized dataset that includes raw blob sidecars, **When** a
   decoded-contract consumer reads only the mandatory V18 dataset artifacts,
   **Then** its required decoded workflow remains stable without mandatory blob
   awareness.

### Edge Cases

- A harvested tile may be valid for required signals but only contain a minimal
  decoded metadata payload; the dataset must preserve the tile with an explicit
  metadata state instead of omitting it.
- A resumed or incremental build may find partial decoded metadata artifacts
  from an earlier interrupted run; the system must distinguish in-progress work
  from a finalized validated dataset.
- A migrated V18 build may start from the same baseline signals as V16 but must
  still include all newly required promoted signals without a second patch pass.
- A merge may consume legacy source datasets created before decoded metadata was
  mandatory; the merged result must still provide complete decoded metadata row
  coverage.
- A tile may contain malformed or non-deserializable metadata content even when
  required signals exist; strict validation must prevent final publication of an
  invalid decoded metadata contract.
- Placeholder map labels or anomalous source-path fields may appear in source
  records; these anomalies must be represented explicitly without dropping the
  affected tile rows.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST define V18 as the direct versioned successor to
  the current V16 dataset creation workflow rather than as a separate parallel
  dataset-build family.
- **FR-002**: The canonical V18 workflow MUST preserve the core purpose and
  operator-facing build shape of the current V16 workflow while producing the
  richer V18 dataset contract.
- **FR-003**: The canonical V18 workflow MUST produce a complete, consumable
  dataset in one end-to-end run without requiring patch or fixup commands for
  any signal family that is part of the approved V18 contract.
- **FR-004**: The canonical workflow MUST distinguish between in-progress build
  outputs and a finalized publishable dataset state.
- **FR-005**: A dataset MUST only be considered finalized when all mandatory
  dataset artifacts and validation reports required by this contract are present.
- **FR-006**: The canonical workflow MUST emit an index surface describing all
  harvested tiles accepted into the dataset.
- **FR-007**: The canonical workflow MUST emit placement coverage data for tile
  object placements when placements are present for accepted tiles.
- **FR-008**: The canonical workflow MUST emit a decoded metadata surface
  containing exactly one row per harvested tile included in the index surface.
- **FR-009**: Each decoded metadata row MUST include tile identity fields,
  provenance path fields, decoded-structure summary fields, a serialized decoded
  metadata payload, and an explicit metadata state indicator.
- **FR-010**: The canonical V18 workflow MUST promote all signal families that
  are currently required through V16 post-build patch workflows into first-class
  outputs of the main V18 build contract.
- **FR-011**: The dataset contract MUST preserve accepted tiles even when their
  decoded metadata payload is minimal, placeholder-backed, or derived from a
  fallback path, as long as required contract fields remain present.
- **FR-012**: The system MUST validate decoded metadata parity against the tile
  index, including row-count parity, one-to-one tile identifier coverage, and
  structural validity of decoded metadata payloads.
- **FR-013**: The system MUST validate coverage for all mandatory V18 signal
  families, including the families promoted from prior V16 patch workflows.
- **FR-014**: The system MUST generate machine-readable validation report
  artifacts for both signal coverage integrity and decoded metadata integrity.
- **FR-015**: Build workflows operating in strict mode MUST fail finalization
  when decoded metadata validation or required signal validation fails.
- **FR-016**: Resumed or incremental build workflows MUST regenerate or refresh
  required validation outputs before an interrupted build can be promoted to
  finalized status.
- **FR-017**: Merge workflows MUST preserve and remap decoded metadata coverage
  consistently for merged tile identifiers.
- **FR-018**: When merging source datasets that lack decoded metadata surfaces,
  the merge workflow MUST still provide complete decoded metadata row coverage
  for all merged tiles using explicit fallback rows.
- **FR-019**: Validation command surfaces MUST validate both signal coverage and
  decoded metadata integrity for finalized datasets.
- **FR-020**: The system MUST represent placeholder map labels, missing source
  path details, or other provenance anomalies explicitly in dataset outputs or
  validation findings rather than silently dropping affected rows.
- **FR-021**: Documentation MUST define V18 dataset contract artifacts,
  validation expectations, finalized-status rules, the no-patch/fixup
  completeness rule, and the rule that V18 is the versioned successor to the
  current V16 build path.
- **FR-022**: The raw blob preservation path MUST remain an additive sidecar
  that does not remove, replace, or weaken decoded-contract completeness
  requirements.
- **FR-023**: When raw blob preservation is enabled, the system MUST provide
  traceable linkage between sidecar blob records and the related tile or chunk
  provenance they preserve.
- **FR-024**: The decoded-contract artifacts required for canonical V18 dataset
  consumption MUST remain complete and consumable whether or not raw blob
  preservation is enabled.

### Key Entities *(include if feature involves data)*

- **Dataset Build Run**: One execution of the canonical V18 workflow for a
  staged input build and selected map set, resulting in either an in-progress or
  finalized dataset state.
- **Promoted Signal Family**: A signal surface that previously required
  post-build V16 patching but is mandatory as part of the canonical V18 build
  contract.
- **Dataset Tile Record**: A harvested tile accepted into the canonical dataset,
  identified by tile identity and coordinate fields, with associated derived
  supervision signals.
- **Decoded Metadata Record**: One record per dataset tile that captures tile
  identity, provenance, decoded metadata summary fields, a full serialized
  decoded metadata payload, and an explicit metadata state.
- **Placement Coverage Record**: A placement-oriented record associated with a
  tile, describing instance-level object placement context when present.
- **Validation Report**: A machine-readable artifact that records pass/fail
  integrity checks for signal coverage, decoded metadata parity, and finalized
  dataset readiness.
- **Merge Mapping Record**: A record that explains how tile identifiers or
  provenance were remapped when multiple source datasets were merged.
- **Raw Blob Manifest Record**: An optional sidecar record linking preserved
  undecoded payload bytes to tile or chunk provenance without changing the
  decoded-contract surface.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of canonical dataset builds that complete successfully
  produce all mandatory core artifacts in the same run, including tile index
  coverage, decoded metadata coverage, and machine-readable validation reports.
- **SC-002**: 100% of signal families that are in scope for V18 promotion are
  produced directly by the canonical V18 build without requiring a separate
  patch phase.
- **SC-003**: For every finalized dataset build, decoded metadata coverage
  matches index coverage exactly, with 100% row-count parity and a 100% one-to-
  one tile identifier match.
- **SC-004**: 100% of decoded metadata rows in finalized dataset builds contain
  structurally valid decoded metadata payload objects plus the required identity,
  provenance, and metadata-state fields.
- **SC-005**: Newly built datasets require zero post-build patch or fixup steps
  to satisfy the canonical publication contract.
- **SC-006**: Strict validation gates prevent 100% of attempted finalized
  publications that have decoded metadata parity failures or signal validation
  failures.
- **SC-007**: Merged dataset outputs preserve complete decoded metadata coverage
  for 100% of merged tile records.
- **SC-008**: When raw blob preservation is enabled, 100% of sidecar blob
  records have traceable linkage to the preserved tile or chunk provenance
  without changing mandatory decoded-contract artifact counts.

## Dependencies & Constraints

- Canonical dataset inputs come from trusted staged client roots in the project-
  local staging area.
- The current V16 dataset creation behavior and currently required patch-on
  signal families define the baseline that V18 is promoting into the main build
  contract.
- The V18 dataset contract must remain compatible with the repo's Bring Your Own
  Data policy and cannot depend on shipping proprietary harvested outputs.
- The dataset contract defined here sits beneath the broader V18 training and
  refinement namespace and must not force unrelated changes to model or trainer
  ownership.

## Assumptions

- Existing decoded-signal extraction behavior remains the baseline; this feature
  formalizes a versioned V16-to-V18 promotion of completeness, provenance, and
  validation guarantees rather than replacing extraction semantics with a new
  unrelated pipeline.
- A future V20 dataset effort may replace the current intermediate raw-array /
  NPZ-shaped interchange with a direct parser → decoded → dataset pipeline, but
  that redesign is explicitly deferred and is not part of V18 scope.
- Legacy patch or fixup commands may still exist for remediation or historical
  stores, but they are not part of the canonical completeness path for newly
  built datasets.
- Raw blob sidecar preservation remains optional and additive in the near term,
  while decoded metadata completeness is mandatory.
- Existing downstream training and inference consumers continue to use the
  current decoded-contract artifacts and should not require migration to consume
  the initial V18 canonical dataset outputs.
- Canonical V18 dataset publication may happen per staged client build and later
  merge into larger corpora, but each published dataset must independently obey
  this contract.
- Image-derived promoted signals depend on separately proven real object-loading
  and capture lanes; until that proof is refreshed, those signals must remain
  explicitly bounded or experimental rather than silently treated as closed.
