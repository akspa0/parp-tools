# Specification Quality Checklist: M2 and WMO Shader Permutation System

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-01
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

Validation run 2026-09-01, one iteration.

Two items required a spec edit before passing:

- **"No implementation details"** — the Context section names concrete files
  (`M2Renderer.cs`, `M2MaterialPassProfile`) and binary addresses. Retained deliberately:
  they are *measured evidence of the current gap*, not prescriptions for how to build the
  feature. No requirement or success criterion names a file, type, or API. The distinction
  this project cares about — established repeatedly in specs 184/188 — is that a spec must
  not assert unmeasured structure; citing where a measurement came from is the opposite of
  that failure.
- **"Scope is clearly bounded"** — initial draft did not separate this work from spec 136
  (batching) or the lighting work. FR-008 and the closing assumption now state both
  boundaries explicitly, and Context names 136's actual state (9/11, remaining tasks
  operator-owned).

Deliberate design choices recorded so planning does not relitigate them:

- FR-002 and FR-004 exist because this project has twice shipped decoders that assigned
  meaning to unmeasured fields (`MCXH`, `Ck24Type`). A permutation the data does not
  actually determine must surface as unresolved, never as a plausible guess.
- FR-007 and SC-005 exist because AGENTS.md forbids changing base renderer behaviour to make
  new tooling pass. The old path must remain reachable and pixel-identical.
- SC-006 turns "how much is left" into an enumerated output of the system rather than an
  estimate, which is what makes the remaining permutations plannable.
