# Feature Specification: 046 - PM4 Asset Matching

**Feature Branch**: `046-pm4-asset-matching`
**Created**: 2026-06-03
**Status**: Draft
**Input**: User description - "Export PM4 Obj Set - causes the whole program to freeze up, instead of exporting things via the `ck24ObjectId`, which seems to be the way to segment all the surfaces into individual objects. We need to figure out a way to build a matching algorithm with real WMO's and M2's matching the various surfaces we have access to within the pm4 data, to help us in generating replacement placement data for missing development tiles, soley from the pm4 data. We have rudimentary manual matching tools built in to the ui that don't work, I'd rather we automate that. use zarr for dataset signals, use speckit to implement a plan"

## Context

`WoWViewer` already has PM4 research, cross-tile stitching evidence, and partial manual matching surfaces, but the current workflow is not usable for production reconstruction work:

- `Export PM4 Obj Set` can freeze the viewer instead of producing reliable output.
- the current manual PM4 matching tools in the UI are not trusted as the design owner for reconstruction.
- the user wants automation rooted in PM4 object segmentation rather than hand-driven matching sessions.
- the downstream goal is replacement placement data for missing development tiles, derived from PM4-owned evidence and matched against real staged WMO/M2 assets.

This feature defines a new automation lane that treats PM4-derived object segments as the authoritative inputs, ranks candidate WMO/M2 matches automatically, and emits bounded replacement-placement proposals. The manual UI tools may remain as review surfaces later, but they are not the primary workflow owner.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Export PM4 Object Segments Without Freezing (Priority: P1)

As a PM4 researcher, I need a deterministic export path that segments PM4 surfaces into object candidates without freezing the whole viewer so I can build reconstruction data from the corpus instead of fighting the UI.

**Why this priority**: If object export is unreliable or blocks the shell, the rest of the matching pipeline cannot start.

**Independent Test**: Export object segments from a bounded PM4 tile set and a directory-scale PM4 corpus using the new automation path. Verify that the run completes with a structured output manifest and does not require the manual `Export PM4 Obj Set` UI flow.

**Acceptance Scenarios**:

1. **Given** a PM4 tile or directory, **When** I run the export workflow, **Then** it emits a structured set of PM4 object segments instead of freezing the application.
2. **Given** PM4 surfaces that share one object identity candidate, **When** the export workflow segments them, **Then** the segment output preserves the chosen grouping identity consistently across every emitted record.
3. **Given** a corpus-scale export, **When** progress is requested, **Then** the workflow exposes progress and completion status without depending on a blocking viewer interaction.
4. **Given** a PM4 surface family that cannot be confidently segmented, **When** export completes, **Then** the output marks the ambiguity instead of silently dropping the data.

---

### User Story 2 - Rank Real WMO/M2 Matches Automatically (Priority: P1)

As a reconstruction researcher, I need the system to rank real staged WMO/M2 assets against PM4 object segments automatically so I am no longer relying on broken manual matching tools.

**Why this priority**: Automated candidate ranking is the core value of the feature. Without it, replacement placement still depends on manual archaeology.

**Independent Test**: Run the matcher on a validation set with known placed assets. Verify that every eligible PM4 object segment gets a ranked candidate list and that the report explains why each top candidate scored well or poorly.

**Acceptance Scenarios**:

1. **Given** exported PM4 object segments and a staged client asset reference corpus, **When** I run the matcher, **Then** each eligible WMO/M2-like PM4 segment receives a ranked candidate list.
2. **Given** a PM4 segment that spans multiple tiles or multiple `MSHD.Field04` buckets, **When** the matcher scores it, **Then** it treats the segment as one object candidate rather than splitting it by tile-level bookkeeping.
3. **Given** a candidate asset with poor agreement, **When** the report is generated, **Then** the mismatch is visible in the scoring breakdown rather than hidden behind one opaque score.
4. **Given** a PM4 segment with no credible asset match, **When** scoring completes, **Then** the result is explicitly unresolved rather than forced to the nearest weak candidate.

---

### User Story 3 - Generate Replacement Placement Proposals For Missing Tiles (Priority: P2)

As a development-map rebuilder, I need the system to turn matched PM4 segments into replacement placement proposals for missing tiles so reconstruction can proceed from PM4-owned evidence instead of manual placement guessing.

**Why this priority**: Matching alone is not enough; the real downstream deliverable is placement data that can fill missing development tiles.

**Independent Test**: Run the placement synthesis workflow on a bounded missing-tile set and verify that it emits a machine-readable replacement-placement proposal set with provenance back to PM4 segments and candidate assets.

**Acceptance Scenarios**:

1. **Given** a matched PM4 segment, **When** placement synthesis runs, **Then** the output includes a replacement placement proposal with asset identity, transform, and provenance.
2. **Given** multiple PM4 segments that map to the same candidate asset family, **When** placement synthesis runs, **Then** each output placement remains tied to its original PM4 evidence instead of collapsing unrelated objects together.
3. **Given** a low-confidence or ambiguous match, **When** placement synthesis runs, **Then** the output marks it as review-needed instead of presenting it as final truth.
4. **Given** a missing tile with no usable PM4 segments, **When** synthesis runs, **Then** the workflow reports that no placement proposal could be produced.

---

### User Story 4 - Review Automation Results Without Returning To Broken Manual Tools (Priority: P3)

As a researcher validating automated reconstruction, I need a bounded review surface for export, matching, and placement results so I can inspect the automation output without going back to the old broken manual PM4 matching workflow.

**Why this priority**: The first priority is automation, not UI. But the results still need a credible review path.

**Independent Test**: Open a produced match report and placement proposal set through a bounded CLI or viewer review surface and verify that a user can inspect the chosen PM4 segment, top candidates, and final placement proposal without re-running the old manual matching steps.

**Acceptance Scenarios**:

1. **Given** an automated match report, **When** I inspect it, **Then** I can see the PM4 segment identity, candidate list, and chosen proposal in one review flow.
2. **Given** a rejected or unresolved match, **When** I inspect it, **Then** I can see why it failed without manually re-deriving the PM4 object segmentation.

### Edge Cases

- What happens when a `ck24ObjectId` is reused across multiple type bytes or cross-tile spans?
- What happens when a PM4 object segment has too little geometry or signal evidence to compare confidently?
- How does the workflow handle PM4 segments whose best candidate asset exists only in some staged client builds but not others?
- What happens when the asset reference corpus and PM4 segment corpus were built from incompatible client roots or mismatched signal definitions?
- How are terrain-like or nav-mesh-only PM4 segments excluded from WMO/M2 matching without losing audit visibility?
- What happens when a candidate score ties across multiple assets or every candidate falls below the confidence floor?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a non-blocking PM4 object export workflow that does not rely on the current freeze-prone `Export PM4 Obj Set` interaction as the primary owner.
- **FR-002**: The PM4 export workflow MUST emit deterministic object-segment records suitable for downstream matching and placement synthesis.
- **FR-003**: The export workflow MUST preserve the object identity layer chosen for segmentation consistently across every output record for that segment.
- **FR-004**: The system MUST produce an automated candidate-matching workflow that compares PM4 object segments against real staged WMO/M2 assets.
- **FR-005**: The matching workflow MUST produce a ranked candidate list plus scoring rationale for each eligible PM4 object segment.
- **FR-006**: The matching workflow MUST distinguish at least these states: matched, ambiguous, unresolved, and ineligible.
- **FR-007**: The system MUST support corpus-scale signal storage for both PM4 segments and asset reference entries so the matcher can operate without repeated ad hoc recapture.
- **FR-008**: The replacement-placement workflow MUST emit machine-readable placement proposal records for matched segments targeting missing development tiles.
- **FR-009**: Every placement proposal MUST retain provenance linking it back to the PM4 segment and candidate asset evidence that produced it.
- **FR-010**: The workflow MUST allow bounded validation against known placed assets on reference tiles before treating synthesis output as replacement data for missing tiles.
- **FR-011**: The first signed-off workflow MUST be automation-first and MUST NOT require the current manual PM4 matching UI as a mandatory step.
- **FR-012**: The implementation MUST reuse existing PM4 decoding and staged-client asset access surfaces instead of introducing duplicate readers or parallel parser stacks.
- **FR-013**: The workflow MUST surface unresolved or low-confidence outputs explicitly rather than silently forcing weak matches or final placements.
- **FR-014**: The feature MUST provide a bounded review/report surface for export, match, and placement outputs so researchers can inspect the automation results.
- **FR-015**: The automation lane MUST remain reproducible from staged client data and PM4 inputs without referencing untrusted client roots or out-of-repo dependencies.

### Key Entities *(include if feature involves data)*

- **PM4 Object Segment**: A grouped PM4 surface set treated as one object candidate for export, scoring, and placement synthesis.
- **PM4 Segment Signal Record**: The derived geometry, topology, and diagnostic signals associated with one PM4 object segment.
- **Asset Reference Signal Record**: A comparable signal record derived from one staged WMO or M2 asset for matching.
- **Candidate Match**: A scored relationship between one PM4 object segment and one asset reference entry, including ranking and rationale.
- **Replacement Placement Proposal**: A machine-readable placement record synthesized from a PM4 object segment and a selected candidate asset, with confidence and provenance.
- **Match Run Manifest**: The bounded report describing one export/match/synthesis run, including inputs, outputs, and unresolved cases.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can export PM4 object segments for a bounded development corpus run without the primary workflow freezing the viewer shell.
- **SC-002**: The automated matcher produces a ranked candidate list for 100% of eligible WMO/M2-like PM4 object segments in the chosen validation run.
- **SC-003**: For a validation set with known placed assets, the ground-truth asset appears within the top 10 ranked candidates for at least 70% of eligible PM4 object segments.
- **SC-004**: Replacement placement proposals can be generated for a bounded missing-tile target set without requiring the old manual PM4 matching UI.
- **SC-005**: Every emitted placement proposal can be traced back to the PM4 object segment and candidate-match evidence that produced it.

## Assumptions

- `ck24ObjectId` remains the starting segmentation owner for the first automation slice, even though some reuse and ambiguity may require follow-up heuristics.
- The first signed-off export/matching/synthesis lane is offline and automation-first; the viewer review surface is secondary.
- Existing staged client roots under `output/tmp/wowarchive-clients/` remain the authoritative source for real WMO/M2 reference assets.
- Replacement placement output is a proposal/export workflow first, not immediate direct mutation of development map files.
- Existing PM4 research and cross-tile evidence remain the source of truth for what counts as one PM4 object candidate during this first automation slice.
