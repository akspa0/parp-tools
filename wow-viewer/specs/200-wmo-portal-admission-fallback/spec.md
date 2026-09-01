# Feature Specification: WMO Portal Admission — Eliminating the Conservative Fallback

**Feature Branch**: `200-wmo-portal-admission-fallback`

**Created**: 2026-09-01

**Status**: Draft

**Input**: Operator flight telemetry over a 5.0.1 map, 2026-09-01, plus the admission
instrumentation spec 151 landed.

## Context

Spec 151 established that WMO **admission**, not batching, owns the remaining renderer cost,
and built the instrumentation to say why. That instrumentation has now produced its first
real reading, from an operator flight over the Wandering Isle (MoP Beta 5.0.1.15464):

```
Placements: considered 145   admitted 145   rejected 0
Groups:     considered 806   admitted 806   rejected 0
Placement evaluations: 290   mean admitted/placement 2.8
worst placement: world\wmo\pandaria\generic\temple

Admitted by rule:
  portal conservative fallback   313  (38.8%)
  gpu-instanced shell            309  (38.3%)
  portal + frustum               146  (18.1%)
  frustum union only              38   (4.7%)
  portal traversal only            0
Portal fallback fired on 119 of 290 evaluations
```

**Nothing is ever rejected.** Not one placement of 145, not one group of 806. Portal culling
is running and reducing nothing, and the panel's own text names the cause: a conservative
fallback admits every group in the placement, and while it fires, no other rule gets a chance
to reduce anything.

The fallback is not a mystery — it is deliberate, and it already records *why* it fired.
`WmoPortalVisibilityDecision` has five named reasons: `visibility_input_invalid`,
`groups_absent`, `portal_data_absent`, `portal_edges_absent`, and the group/portal map build
errors. The viewer surfaces the first reason. What is not known is the **distribution** —
which reason dominates, on which WMOs, and whether each is a data limitation or a bug in how
we build the portal graph.

This is a correctness question before it is a performance one. A fallback that fires on 41% of
evaluations either means the portal data genuinely cannot be used, or means we are failing to
build a graph from data that is present. Those have opposite fixes, and today we cannot tell
them apart.

Scope note: spec 151 also carries game-mode and viewer-surface work. This spec takes **only**
the admission slice, so it can be finished without waiting on the rest of 151.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The fallback reasons are counted, not just sampled (Priority: P1)

An operator flying a route sees the distribution of fallback reasons across all evaluations —
how many fired for absent portal data, how many for absent edges, how many for invalid input —
and which WMOs produced each.

**Why this priority**: The panel today reports the *first* reason seen. A first reason cannot
distinguish "this map's WMOs have no portals" from "we fail to build edges from portals that
exist". Every fix in this spec depends on knowing which.

**Independent Test**: Fly a route and read the per-reason counts with their offending WMO
paths. Delivers a decision-ready diagnosis with no culling behaviour changed.

**Acceptance Scenarios**:

1. **Given** a flight over a map, **When** the admission panel is read, **Then** each fallback
   reason has its own count and the WMOs that triggered it are named.
2. **Given** a WMO that triggers a fallback, **When** its reason is reported, **Then** the
   report distinguishes "the file has no portal data" from "the file has portal data we did
   not turn into a usable graph".

---

### User Story 2 - Portal culling rejects something (Priority: P1)

On a map whose WMOs carry usable portal data, flying outside and inside buildings rejects
groups that cannot be seen, and the rejection counts are non-zero.

**Why this priority**: The measurable goal. 806 of 806 groups admitted is the defect; any
honest reduction is the fix. Equal priority to US1 because the diagnosis and the fix are the
same work — US1 without US2 is a nicer readout of the same problem.

**Independent Test**: Fly the same route before and after, and compare admitted-group counts
and frame time on an unchanged scene.

**Acceptance Scenarios**:

1. **Given** a WMO with valid portal data and a camera outside it, **When** the scene renders,
   **Then** groups not reachable through any visible portal are rejected.
2. **Given** the same route flown before and after, **When** admitted-group counts are
   compared, **Then** the count falls and no visible geometry disappears.
3. **Given** a WMO whose portal data genuinely cannot support culling, **When** it is
   evaluated, **Then** it still admits conservatively and is counted as such — correctness is
   never traded for a reduction.

---

### User Story 3 - The GPU-instanced shell path is accounted for (Priority: P2)

The 38.3% of admissions attributed to the GPU-instanced shell are explained: whether that path
is expected to bypass portal culling, or whether it is a second way of admitting everything.

**Why this priority**: It is the same size as the fallback (309 vs 313) and may be a second
independent cause. Deferred behind US1/US2 only because the fallback is the one the
instrumentation already names as the thing to fix.

**Independent Test**: Read the admission breakdown with the shell path disabled and compare.

**Acceptance Scenarios**:

1. **Given** admissions attributed to the GPU-instanced shell, **When** the breakdown is read,
   **Then** it states whether portal culling was applicable to those and skipped, or not
   applicable.

---

### Edge Cases

- A WMO with exactly one group and no portals — conservative admission is correct; must not be
  counted as a failure.
- Camera exactly on a portal plane, or inside no group.
- A placement whose model bounds are missing — already noted in
  `WorldObjectVisibilityCollector` as having only the conservative fallback available.
- Interior vs exterior cameras: the decision already distinguishes them, and the fix must not
  regress interiors to fix exteriors.
- A map where every WMO legitimately lacks portals — the honest outcome is no reduction, and
  the report must say so rather than the spec appearing to fail.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST report fallback counts **per reason**, not just the first reason
  observed, aggregated across a flight.
- **FR-002**: For each fallback reason, the system MUST name the WMOs that triggered it, so a
  data limitation can be told apart from a graph-construction failure.
- **FR-003**: The system MUST distinguish "portal data absent from the file" from "portal data
  present but no usable graph was built from it". These have opposite fixes.
- **FR-004**: Where portal data is present and usable, the evaluator MUST produce a decision
  that can reject groups, rather than falling back.
- **FR-005**: The system MUST NOT reject any group that is actually visible. A reduction that
  removes visible geometry is a regression, not a success.
- **FR-006**: Admissions attributed to the GPU-instanced shell MUST record whether portal
  culling was applicable to them.
- **FR-007**: Conservative admission MUST remain the behaviour whenever the data cannot support
  culling, and MUST remain counted as such.
- **FR-008**: The change MUST be comparable within one session — the previous admission
  behaviour reachable at runtime, so a before/after does not require reflying a route from a
  cold start.

### Key Entities

- **Fallback reason**: One of the evaluator's named causes, with a count and the WMOs that
  produced it.
- **Admission attribution**: Which rule admitted a group — portal traversal, portal+frustum,
  frustum union, conservative fallback, or GPU-instanced shell.
- **Portal graph availability**: Per WMO — whether portal data exists in the file, and whether
  a traversable graph was built from it.
- **Route comparison**: Admitted placements, admitted groups, and frame-time distribution for
  the same route before and after.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a flown route, every fallback is attributed to a named reason, and the
  reason distribution is reported with counts summing to the total fallback count.
- **SC-002**: The proportion of WMOs whose portal data is present but unusable is measured. If
  it is zero, the fallback is a data limitation and this is recorded as the finding.
- **SC-003**: On a map with usable portal data, admitted groups for a fixed route fall from
  the measured baseline of 806 of 806.
- **SC-004**: No geometry visible before the change is missing after it, confirmed by capture
  comparison on the same route.
- **SC-005**: Frame-time distribution for the same route is captured before and after and the
  change recorded — including no change, which would mean admission was not the cost after all.
- **SC-006**: The GPU-instanced shell's 38.3% share is explained as either applicable-and-
  skipped or not-applicable.

## Assumptions

- Spec 151's admission instrumentation is the measurement surface; this spec extends its
  reporting rather than building a second one.
- The Wandering Isle flight (145 placements, 806 groups, 290 evaluations, 119 fallbacks) is the
  baseline. Re-flights for comparison are operator-run per the execution boundary.
- Opaque MDX batching remains **off** during comparison flights, so the admission change is not
  confounded by spec 153 US3's toggle. If it is turned on, the baseline is re-established
  first.
- Whether the fallback is a data limitation or a bug is genuinely unknown; this spec is
  written so that "the data does not support culling" is a legitimate, reportable outcome
  rather than a failure to deliver.
