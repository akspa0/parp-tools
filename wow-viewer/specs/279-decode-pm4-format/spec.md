# Feature Specification: PM4 Format Comprehension and Evidence-Based Reference

**Feature Branch**: `279-decode-pm4-format`

**Created**: 2026-10-20

**Status**: Specified; first documentation/research phase authorized by the operator's request

**Input**: “continue. We want to fully comprehend the pm4 format, including all the things we do
not understand right now, to make it all work better in the viewer, and matching tooling. Can we
get a nice writeup of the format now that you understand it better? use speckit”

**Owner**: Epic 253, remaining decode/field-map/documentation work. Adopts Spec 270's measured
whole-map receipts; does not duplicate or reopen its placement-generation implementation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read an honest, usable format reference (Priority: P1)

As an operator or tool developer, I can read one current PM4 reference that explains the file's
structure, the relationships we have measured, and what remains unknown, without mistaking old
field names or plausible guesses for decoded facts.

**Why this priority**: Viewer and matching decisions are unsafe when parsing a field is confused
with understanding its meaning.

**Independent Test**: Compare the reference with the supported chunk inventory and current real
map evidence. Every supported chunk and raw field is covered; every semantic claim has an
evidence category and a source, and unresolved meanings are explicitly listed.

**Acceptance Scenarios**:

1. **Given** the measured development corpus, **when** I read about a surface connection,
   **then** the reference explains its target-tile context, evidence, missing targets and limits.
2. **Given** a legacy field called group, type or attribute, **when** I look it up,
   **then** I can distinguish its raw layout, observed relationships and unresolved meaning.
3. **Given** old contradictory notes, **when** I follow their current-documentation link,
   **then** I reach the reference and can identify superseded assumptions.

---

### User Story 2 - Resolve unknowns through discriminating experiments (Priority: P1)

As a researcher, I can select one unresolved relationship, test alternatives against the entire
map context with meaningful controls, and record whether the result supports, refutes or fails
to distinguish them.

**Why this priority**: Full comprehension is a research goal; apparent coincidences must not
become dependencies merely because they improve one matching example.

**Independent Test**: Inspect a completed experiment's question, alternative explanations,
positive/wrong-target controls, source fingerprints, denominators and actual output. A null or
inconclusive result is retained rather than promoted into a new meaning.

**Acceptance Scenarios**:

1. **Given** a candidate index or ownership field, **when** its interpretation is tested,
   **then** a mere in-range value is not called a decoded relation without independent evidence.
2. **Given** references crossing files, **when** I evaluate a hypothesis,
   **then** neighboring files and absent targets remain accounted for.
3. **Given** operator-reported editor path nodes, **when** I examine point streams,
   **then** authored-node identity is not conflated with generated surface connectivity.

---

### User Story 3 - Apply findings without breaking proven behavior (Priority: P2)

As a viewer/matching user, I can see which verified findings are safe to consume, which proposed
changes still require proof, and why navigation connectivity alone does not identify a placed M2.

**Why this priority**: The existing WMO baseline is useful and must not be sacrificed to an
unvalidated format reinterpretation.

**Independent Test**: Trace any proposed consumer change to a verified finding and a separately
validated phase. The reference distinguishes navigation relationships, source-object ownership,
operator candidate-pool context and authored route nodes.

**Acceptance Scenarios**:

1. **Given** a coherent navigation component containing several scaled assets,
   **when** matching is evaluated, **then** component membership is not asserted as object identity.
2. **Given** a new format inference, **when** integration is proposed,
   **then** real behavior proof is required before viewer/matcher behavior is changed.

### Edge Cases

- Empty/header-only files, missing neighboring tiles and unresolved addressed records.
- A surface index that fits both the source file and a different addressed tile.
- Repeated or overlapping chunk records, unknown signatures and partial typed projections.
- Group-like values reused across tiles, equal-height instances, merged or disconnected copies,
  varying scales and sparse/absent M2 collision surfaces.
- Different source builds or corpus subsets with incompatible denominators.
- Legacy names, old unreceipted claims and screenshots whose exact chunk mapping is unknown.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Provide one current, readable reference covering every PM4 chunk and raw field
  supported by the project's existing reader, plus container structure and reference domains.
- **FR-002**: Distinguish raw-layout facts, corpus-measured relationships, working hypotheses,
  operator evidence, refuted interpretations and unresolved semantics.
- **FR-003**: Link semantic claims to actual evidence, with corpus identity, denominators,
  controls and limitations; compilation and synthetic controls alone do not prove real semantics.
- **FR-004**: Treat the PM4 map as one dataset with file-qualified references and retained
  absent-target context; isolated-tile results are identified as local diagnostics only.
- **FR-005**: Maintain an unknown register spanning headers, surfaces, connections, point/reference
  streams and auxiliary state/filename records, with an explicit next test for each open question.
- **FR-006**: Continue one independently verifiable research concern at a time and update the
  reference/evidence ledger when it supports, refutes or cannot distinguish an interpretation.
- **FR-007**: Separate authored route-node hypotheses, generated surface connectivity, geometric
  fragments and source-object ownership; do not equate any of them without evidence.
- **FR-008**: Account for merged/scaled M2 copies, independently authored Museum placements and
  operator design-kit/pool hints without presenting expectations as confirmed source identities.
- **FR-009**: Preserve existing readers, working WMO matching, source data and unrelated changes.
  Consumer changes require a separately validated implementation phase and explicit evidence.
- **FR-010**: Keep the earlier work accessible as history, clearly route stale documentation to
  the current reference, and synchronize the research plan, receipt-gated tasks and live handoff.

### Key Entities *(include if feature involves data)*

- **Format element**: Raw chunk/record/field, its layout and currently supported relationships.
- **Claim**: Interpretation, confidence category, evidence source, scope and known limitations.
- **Unknown**: Unresolved question, competing explanations and a discriminating next experiment.
- **Corpus**: Complete map input inventory, fingerprint and separately identified source revision.
- **Experiment**: One question, controls, actual observations, result and reproducible receipt.
- **Consumer implication**: Safe read-only use or proposed viewer/matching behavior change with proof gate.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The reference covers 100% of currently supported PM4 chunks and their raw fields;
  no supported entry is silently omitted or assigned an invented meaning.
- **SC-002**: Every asserted semantic relationship has an evidence category and traceable source;
  all currently known unresolved meanings appear in the unknown register.
- **SC-003**: A reader can trace the cross-tile surface/link example from raw fields to the
  addressed record and inspect the actual positive/negative evidence without opening old session logs.
- **SC-004**: Each completed research phase has a reproducible real-file receipt with controls,
  explicit outcome and limitations, including inconclusive results.
- **SC-005**: No claim of full PM4 comprehension, correct M2 ownership or improved viewer/matching
  behavior is made while the corresponding unresolved questions and proof gates remain open.

## Assumptions

- “Fully comprehend” is the research program's goal, not a promise that all unknowns will be solved
  in this documentation pass. Unknown outcomes remain valid and visible research results.
- The immediate deliverable is the current writeup plus a phased investigation of remaining
  unknowns. Existing Spec 270 measurements are adopted as evidence, not repeated unnecessarily.
- Earlier ChatGPT-era interpretations are not authority without receipts; familiar field names
  do not settle semantics. Observations are scoped to the fingerprinted corpus, not every build.
- Operator-reported WMO accuracy (~70%) and screenshot path-node evidence remain attributed to
  the operator; neither is independently re-certified by documentation or unit tests.
- Large exports, training/GPU jobs and real-client visual proof remain operator-owned. Lightweight
  read-only summaries may run with measured bounds and repository-contained outputs.
- PM4 is the current scope. PD4 similarities can be noted, but a full PD4 decode or generation
  feature is not implicitly added. Viewer annotation UI remains in its existing owning tasks.
