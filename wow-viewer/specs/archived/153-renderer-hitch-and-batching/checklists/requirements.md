# Specification Quality Checklist: Renderer hitch elimination and MDX batching restoration

**Purpose**: Validate specification completeness before implementation
**Created**: 2026-08-15
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Focused on user value (smooth movement in dense scenes)
- [x] All mandatory sections completed
- [x] Findings are measurements, not inferences

Note: this spec names source symbols and prints real numbers. That is deliberate — the measured
evidence is the asset being preserved, and losing it would mean re-running the whole investigation.

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers
- [x] Requirements testable against the frame history
- [x] Success criteria measurable, each with a recorded baseline
- [x] Acceptance scenarios defined
- [x] Edge cases identified
- [x] Scope bounded (four defects, named)
- [x] Assumptions identified

## Evidence quality

- [x] Every finding has a number and a zone it was observed in
- [x] Ruled-out theories recorded **with the evidence that killed them**, so they are not re-proposed
- [x] Defect A is honestly marked as localised-to-a-pass but **not yet named to a call**
- [x] The refuted allocation hypothesis is recorded as refuted rather than quietly dropped

## Open at time of writing

- Defect A's specific operation is unnamed. Phase 0 exists solely to name it, with an explicit
  instruction to subdivide further rather than guess if the sub-probes do not match.
- No fix has been attempted, so no success criterion is met yet. Every SC has a baseline recorded to
  compare against.

## Notes

- Two defects are independent and must not be credited with fixing each other: the periodic ~212 ms
  stall (SC-001) and the sustained unbatched-MDX cost (SC-003).
- Stranglethorn Vale is the standard benchmark for every before/after in this spec.
