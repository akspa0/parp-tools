# Implementation Plan: PM4 Format Comprehension and Evidence-Based Reference

**Branch**: `279-decode-pm4-format` | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)

**Input**: Operator request for full PM4 comprehension, a clear current writeup, and continued
investigation of unresolved semantics for safer viewer/matching work.

## Summary

Deliver a readable, exhaustive **current** reference first, without claiming a complete decode.
Adopt existing measured map evidence, audit raw layouts against the unchanged reader, and mark
all uncertain meanings. Then test one raw-field relationship at a time over the correctly
tile-addressed graph. Keep point-node identity, source-object ownership and navigation distinct.

This is Epic 253's selected remaining-decode/documentation work, not a second placement generator.
Spec 270 remains the owner of its ledger/restoration/pool tooling and existing implementation gaps.

## Technical Context

**Language/Version**: Markdown reference/spec artifacts; C#/.NET 10 only for bounded research probes.
**Primary Dependencies**: Existing Core.PM4 research reader/models, tile-address/geometry probes,
Spec 270 fingerprinted reports; unchanged Core.IO placement reader where independently needed.
**Storage**: Small repository-contained JSON/text evidence summaries; no archive/container writes,
no dataset/storage migration and no large decoded-data export in the immediate phase.
**Testing**: Source/layout/claim coverage checks; focused xUnit controls before real-file inference;
complete-map summaries with hashes, denominators, wrong-domain controls and explicit null outcomes.
**Target Platform**: Existing Windows development environment; portable input paths and PS7 commands.
**Project Type**: Documentation + shared research library + thin evidence runner.
**Performance Goals**: Bounded summary passes comparable to existing 4–7 second corpus probes;
stop/re-scope if a pass becomes long/heavy. These are analysis budgets, not viewer FPS goals.
**Constraints**: No working reader, WMO matcher, renderer or god-class changes in reference/probe
phases; preserve dirty work and operator-owned training/export/runtime proof boundaries.
**Scale/Scope**: Fingerprinted 616-file, 169.08 MiB development PM4 corpus; 16 signatures parsed by
the existing PM4 reader, with all supported raw fields covered. Other builds remain unverified.

## Constitution Check

- **Repo independence/library-first**: New reusable analysis, if needed, lives in Core.PM4;
  runners are thin and all paths stay under wow-viewer or explicit operator inputs.
- **Real-data validation**: Adopt receipts with their actual subset/build limitations; new semantics
  require real inputs plus controls. Synthetic results prove detector behavior only.
- **Reader ownership/base preservation**: Read layouts from proven code; no parser rewrites or
  renames are bundled into documentation. Raw payload preservation and typed losses are documented.
- **One phase at a time**: Phase 0 must have its validation receipt before Phase 1 code. Each later
  phase has its own gate; checked tasks require evidence, including inconclusive experiments.
- **Source budget/UI boundaries**: New research classes remain below 2,000 lines; no ViewerApp or
  WorldScene members. Viewer UI is not implicit scope.
- **Canonical artifacts**: Current AGENTS.md governs placement: spec/plan/tasks live here, not
  memory-bank or architecture plans. The reference is format documentation, not a hidden plan.

No constitutional exception is requested.

## Project Structure

### Documentation (this feature)

```text
specs/279-decode-pm4-format/
  spec.md
  plan.md
  tasks.md
  research.md
  data-model.md
  quickstart.md
  contracts/evidence-contract.md
  checklists/requirements.md
  evidence/
docs/research/pm4-format-reference.md
```

The new reference is the current PM4 reading route. The older combined PM4/PD4 wiki draft is
preserved with a prominent supersession notice; it is not duplicated as a competing authority.
Existing sources/receipts are adopted by links, not copied as supposedly new measurements.

### Source Code (only when its phase is active)

```text
src/core/WowViewer.Core.PM4/Research/Pm4FieldDomainProbe.cs
tests/WowViewer.Core.PM4.Tests/Pm4FieldDomainProbeTests.cs
specs/279-decode-pm4-format/evidence/field-domain-probe/
```

Existing PM4 readers/models/matchers are inputs to the analysis, not a write set. Point/ownership
probe filenames are selected in their bounded phase after the preceding result, not speculative
classes added up front.

## Adopted Evidence and Design

- [Spec 270 structural receipt](../270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md)
  and its hashed whole-map window, path-contact and paired cross-tile reports are the measured baseline.
- [Epic 253](../253-epic-pm4-navmesh-research/spec.md) owns remaining decode/field-map/documentation.
- Archived 130/185/188/189 designs remain historical background adopted only for their evidence
  discipline/remaining questions, **not** their unverified semantic assertions. Do not reload them
  merely to override current raw measurements.
- [Research decisions](research.md), [claim/data model](data-model.md), and
  [evidence contract](contracts/evidence-contract.md) define the new investigation discipline.

## Phases and Gates

### Phase 0 — US1: Current reference and unknown ledger

1. Inventory the unchanged reader's 16 parsed signatures and every record field; distinguish
   raw/layout support, current typed projections and losses.
2. Write the reader-friendly reference: overview, confidence legend, relationships, layouts,
   corpus evidence, implications and unknown register, including all operator corrections.
3. Route old wiki/memory references to it; register Spec 279 under Epic 253 without reopening
   unrelated backlog items or claiming the whole format is decoded.
4. Validate chunk/field coverage, linked evidence, local links and absence of placeholder claims;
   write a receipt and update continuity. **Gate R0:** readable reference and honest scope validated.

### Phase 1 — US2: Raw link +0x04 domain/reverse-link hypothesis

Prerequisite: Gate R0 and the correctly addressed graph from Spec 270.

1. Define the narrow question: is raw link +0x04 an index into a record domain, a reverse-link
   pointer, or merely a recurring value? Do not presume the legacy GroupObjectId alias is identity.
2. Add a shared read-only probe and synthetic controls for genuine reverse-link indexing,
   coincidental in-range values, wrong-domain/offset targets and missing addressed tiles.
3. Run a bounded complete-map summary with same-byte hashes. Report domain fits separately
   from actual back-reference evidence and break results down by raw link kind/zero context.
   Include a compact header/chunk/version inventory for the reference, not a raw corpus export.
4. Publish supported/refuted/inconclusive outcomes and revise the reference/unknown register.
   **Gate R1:** a reproducible discriminating result, not a required “discovery”.

### Phase 2 — US2: Authored-point and reference streams

Prerequisite: Gate R1; choose the next one concern from its result and unknown register.

**Selected sub-step 2A (operator continuation 2026-10-20):** determine whether raw link +0x04
selects a geometric point/boundary element. Exact write scope: new
`Research/Pm4PointReferenceProbe.cs`, `Pm4PointReferenceProbeTests.cs`, and this spec's thin
`evidence/point-reference-probe/` runner/receipts; update the current reference only after evidence.
Compare source MSCN, MSVI→MSVT, MSVT, MSPV/MSPI and raw/permuted MPRL point candidates; retain
addressed-MSCN as a scope alternative, and +1/+17/hashed indices as wrong-target controls.
Measure point-to-vertex/3D segment/centroid contacts, not bounds alone. A separate row family tests
raw subtype as zero-/one-based source/target edge ordinal, against a shifted-ordinal control;
out-of-range ordinals are counted, never coerced into validity. Retain link-kind/zero-bucket strata,
missing addressed tiles and invalid coordinates. No result identifies authored editor nodes or
source assets automatically. Use synthetic true-reference/wrong-domain/missing-target controls
before a same-byte-fingerprinted complete-map summary. Stop/re-scope if the read/analysis exceeds
a short bounded CPU pass. MPRR and screenshot correspondence are not certified by sub-step 2A.

Conditional follow-up within 2A: if an edge-ordinal hypothesis strongly beats controls, test that
selected boundary against auxiliary geometry in a short edges-only pass. Keep exact 3D edge/
endpoint contact separate from membership in the selected edge's vertical plane; XY coplanarity
does not prove equal height or navigability. Retain shifted/wrong-side controls and raw-kind strata.

**Selected sub-step 2B (after 2A receipt, 2026-10-20):** test whether raw +0x04 behaves as an
edge/connection identifier rather than a point reference or asset ID. Exact write scope: new
`Research/Pm4EdgeIdentityProbe.cs`, `Pm4EdgeIdentityProbeTests.cs`, and the existing thin point
runner's explicit `--edge-identity` summary mode. Compare the raw value on independently resolved
unique reverse records; do not select reverse records by equality of the field under test. Retain
multiple/no reverse cases and wrong-next-record control. Also measure per-file value reuse across
directed source-edge keys and unordered addressed surface pairs; local equality is not automatically
global identity. Synthetic true-edge-ID and deliberately reused-value controls precede a bounded
all-map summary with hashes. Publish either support or rejection and leave source-asset identity
and authored-node semantics open.

1. Audit MSCN and MPRL point counts, raw coordinate conventions and candidate reference domains
   against the addressed graph; test wrong axes/domains rather than inheriting old labels.
2. Test MPRR record/run hypotheses independently of fitting screenshots; retain all fields and
   distinguish sentinels from established graph structure.
3. Compare an explicitly identified point-layer hypothesis with operator screenshot evidence and
   real map context; if screenshots are unavailable, leave that visual correspondence open.
4. **Gate R2:** per-question receipts with limitations; no point stream is called editor-authored
   merely because it looks like a waypoint list.

### Phase 3 — US2: Auxiliary state and source ownership

Prerequisite: Gate R2 or an explicit operator-selected independent question.

1. Test remaining header/normal/plane/raw-word/state record relationships with wrong-domain controls.
2. Assess proposed ownership separators against independently authored Museum M2 paths/transforms/
   scales, retaining merged copies, absent collision geometry and equal-value counterexamples.
3. Distinguish navigation grouping from source ownership and operator tile-pool priors. Publish
   failures as well as successes. **Gate R3:** no global object ID without independent identity proof.

### Phase 4 — US3: Consumer proposals and proof-gated integration

1. Produce a bounded viewer/matcher impact list tied to verified claims and baseline receipts.
2. Obtain operator approval for the particular behavior change before editing frozen/shared paths.
3. Implement one approved owned-service/adapter slice and run focused regressions.
4. Operator supplies real viewer/matching proof. **Gate R4:** runtime/accuracy criteria remain open
   until real output supports them; compilation does not certify them.

## Validation Strategy

Phase 0 is source/evidence documentation work. Use a project-contained validation command to compare
the reference's chunk inventory and field offsets with current raw declarations, resolve local
Markdown links, and verify adopted report fingerprints/counts. Store exact commands and statuses.
Phase 1 tests fail on constructed wrong interpretations before running real data. Corpus reports
must include all PM4 filenames, diagnostics, subset denominators, source hashes and explicit absent
targets. Do not run a large Museum/raw export to validate this initial reference.

## Complexity Tracking

No exceptions. No new infrastructure, dependency, format reader, renderer path, automatic semantic
renaming or duplicated placement-generation pipeline is introduced by this plan.
