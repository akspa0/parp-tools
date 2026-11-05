# Feature Specification: Visual PM4 Shape Library and Human-in-the-Loop Pairing

**Feature Branch**: `295-pm4-visual-shape-library`

**Created**: 2026-11-05

**Status**: Draft

**Input**: User description: "The shape library thing does not work, it matches to all the wrong 'centroids', it's off by a hair though, along the -X axis. We need a 'smarter' shape library analysis tool that identifies and pairs automatically, and uses the human to decide if the object is right or not, or to replace it with another possible match. We need the library to be visual in nature, not text."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Browse the shape library visually (Priority: P1)

An operator opens the PM4 workbench and sees the shape library as a visual catalog: each entry shows a recognizable preview of the model or hull, its footprint proportions, and the assets that share that shape. The operator can filter, search, and compare entries without reading a text table.

**Why this priority**: The current text-only library is the root usability failure. A visual catalog is the minimum useful slice and is independently valuable even before automatic pairing changes.

**Independent Test**: Open the library, filter to a known model family, and confirm the previews and proportions match the assets; select an entry and see its candidate hulls.

**Acceptance Scenarios**:

1. **Given** a populated shape library, **When** the operator opens the library, **Then** every entry shows a visual preview and its shape proportions.
2. **Given** a search term or filter, **When** the operator applies it, **Then** the visible entries update without losing the selected entry.
3. **Given** two entries, **When** the operator compares them, **Then** both previews and their proportions are visible side by side.

---

### User Story 2 - Automatic proposals with corrected centroids (Priority: P1)

The operator runs analysis on a map or tile. The system identifies PM4 hulls, proposes the most likely object for each hull, and shows the proposal in the 3D view with the candidate model aligned to the hull. The known small offset along the -X axis is corrected by a calibrated placement adjustment rather than being baked into every asset.

**Why this priority**: Wrong centroids make every downstream pairing untrustworthy. Correct alignment is required before human review is meaningful.

**Independent Test**: Run analysis on a tile with known authored objects and confirm the proposed model sits on the hull, not a hair off along -X.

**Acceptance Scenarios**:

1. **Given** a PM4 hull with a known authored object, **When** analysis runs, **Then** the top proposal is that object or a visually equivalent asset.
2. **Given** the calibrated -X correction, **When** a proposal is previewed, **Then** the model's footprint center coincides with the hull's footprint center within the stated tolerance.
3. **Given** a hull with no credible candidate, **When** analysis runs, **Then** the hull is reported as unmatched rather than assigned a low-confidence guess.

---

### User Story 3 - Accept, reject, or replace a proposal (Priority: P1)

The operator reviews proposals one at a time or in a queue. For each proposal they can accept it, reject it, or replace it with another candidate from the visual library. The decision is recorded and the scene updates immediately.

**Why this priority**: The operator must remain the authority. Automatic pairing without a review gate is not acceptable.

**Independent Test**: Review a proposal, reject it, choose an alternate candidate, and confirm the scene and the saved project reflect the replacement.

**Acceptance Scenarios**:

1. **Given** a proposal, **When** the operator accepts it, **Then** the pairing is committed and the next proposal is shown.
2. **Given** a proposal, **When** the operator rejects it, **Then** the hull remains unpaired and the rejection is recorded.
3. **Given** a proposal, **When** the operator replaces it, **Then** the chosen candidate becomes the pairing and the previous candidate is retained in the audit history.
4. **Given** a completed review, **When** the operator reopens the project, **Then** every decision is restored exactly.

---

### User Story 4 - Batch analysis and audit (Priority: P2)

The operator runs analysis across a whole map, watches progress, and receives a summary of matched, unmatched, and low-confidence hulls. The summary can be exported for review, and every decision carries enough provenance to explain why it was proposed.

**Why this priority**: Whole-map work is the real workload; a per-tile tool alone does not scale.

**Independent Test**: Run a whole-map analysis, confirm progress and summary counts, export the summary, and verify each row names the hull, the chosen asset, the score, and the decision source.

**Acceptance Scenarios**:

1. **Given** a map with many hulls, **When** batch analysis runs, **Then** progress is visible and the run can be cancelled without corrupting existing decisions.
2. **Given** a completed run, **When** the operator opens the summary, **Then** matched, unmatched, and low-confidence counts are shown.
3. **Given** an exported summary, **When** the operator inspects a row, **Then** it identifies the hull, the proposed asset, the score, and whether the decision was automatic, accepted, rejected, or replaced.

### Edge Cases

- A hull has several visually identical candidates; the operator must be able to see and choose among them.
- A hull is rotated or mirrored relative to the candidate; the preview must show the transformed alignment.
- A candidate asset is missing or fails to load; the proposal must degrade to a visible placeholder and remain reviewable.
- A hull is degenerate or has too few vertices; it must be reported as unmatched, not forced into a shape.
- Two hulls claim the same object; the system must surface the conflict instead of silently duplicating.
- The -X correction is applied twice or with the wrong sign; the preview must make the error obvious and the calibration must be resettable.
- A batch run is cancelled midway; already accepted decisions must remain intact.
- The library contains stale entries for assets that no longer exist; they must be marked and excluded from automatic proposals.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST present the shape library as a visual catalog with a preview for every entry, not as a text-only list.
- **FR-002**: The visual catalog MUST show each entry's footprint proportions and the assets associated with that shape.
- **FR-003**: The operator MUST be able to search, filter, and compare library entries without losing the current selection.
- **FR-004**: The system MUST analyze PM4 hulls and produce ranked candidate proposals with a confidence score.
- **FR-005**: The system MUST apply a calibrated placement correction for the observed -X centroid offset, and the correction MUST be visible and resettable.
- **FR-006**: The system MUST preview a proposal in the 3D view with the candidate aligned to the hull before any decision is committed.
- **FR-007**: The operator MUST be able to accept, reject, or replace each proposal.
- **FR-008**: Replacing a proposal MUST offer alternative candidates from the visual library, ranked by the same evidence used for the original proposal.
- **FR-009**: The system MUST NOT commit an automatic pairing without an explicit operator acceptance, unless the operator has enabled a clearly labeled auto-accept mode for a bounded run.
- **FR-010**: Every decision MUST be persisted with the hull identity, chosen asset, score, decision source, and timestamp.
- **FR-011**: The system MUST retain rejected and replaced candidates in an audit history for the hull.
- **FR-012**: The system MUST report unmatched and low-confidence hulls separately from matched hulls.
- **FR-013**: The system MUST support a batch analysis over a map or a selected tile range with visible progress and cancellation.
- **FR-014**: Cancelling a batch run MUST preserve decisions already accepted.
- **FR-015**: The system MUST detect and report conflicts where multiple hulls claim the same object.
- **FR-016**: The system MUST mark library entries whose assets are missing or unreadable and exclude them from automatic proposals.
- **FR-017**: The system MUST allow the operator to export an analysis summary that identifies each hull, proposal, score, and decision.
- **FR-018**: The system MUST keep the existing manual pairing and curation paths available; this feature adds a review workflow, it does not remove manual control.

### Key Entities

- **Shape Library Entry**: A visual catalog record for one shape family, with preview imagery, footprint proportions, and the assets that share the shape.
- **Shape Candidate**: One asset proposed for a hull, with its shape distance, alignment, and confidence.
- **Match Proposal**: The ranked candidate set for one PM4 hull, including the top candidate and the evidence used to rank it.
- **Match Decision**: The operator's accept, reject, or replace action for a proposal, with provenance and timestamp.
- **Centroid Calibration**: The measured placement correction applied to proposals, including its sign, magnitude, and reset state.
- **Analysis Run**: A batch or single-tile analysis with scope, progress, cancellation state, and summary counts.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a validation tile with known authored objects, at least 90% of accepted proposals place the model's footprint center within 1 yard of the hull's footprint center after calibration.
- **SC-002**: The operator can review and decide on 50 proposals in under 5 minutes using the visual queue.
- **SC-003**: 100% of accept, reject, and replace decisions survive a save and reopen without loss.
- **SC-004**: A whole-map analysis reports matched, unmatched, and low-confidence counts, and the counts reconcile with the exported summary.
- **SC-005**: The visual library opens with at least 200 entries and remains responsive enough for the operator to filter and select without a noticeable stall.
- **SC-006**: No automatic pairing is committed without an explicit operator decision unless auto-accept was deliberately enabled for that run.

## Assumptions

- Existing PM4 hull extraction, model catalog, and 3D preview infrastructure remain the foundation; this feature adds analysis, review, and visual catalog behavior on top.
- The -X offset is a calibration value, not a per-asset constant; the operator can inspect and reset it.
- The operator owns the final decision for every pairing; automatic analysis is advisory.
- The existing manual pairing and curation project remain authoritative for saved decisions.
- Model previews can be rendered from the same assets the viewer already loads; missing assets are reported rather than hidden.
- The first validation target is a development map with known authored objects, not a full production map.
