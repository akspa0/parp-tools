# Specification Quality Checklist: MCP Tooling Harness

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-02
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [~] Written for non-technical stakeholders
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

## Validation Notes

**Iteration 1 — issues found and fixed:**

1. *The contract-integrity requirement was originally phrased as "schemas should match the CLI".*
   "Should match" is not testable and is exactly how the documented drift happened in the first place.
   FR-007 now requires both surfaces to be **derived from one shared definition**, FR-008 requires the
   build to fail on divergence, and SC-002 makes it a measured 100%. The Assumptions section states
   explicitly that a hand-maintained parallel schema does not satisfy FR-007.
2. *The boundary story was initially a single line about "not exposing dangerous tools".* Promoted to
   US5 at P1 with its own acceptance scenarios, six functional requirements, and a negative test suite
   in SC-006. A server fronting ten tools that read client archives and write files needs its boundary
   designed in from the first exposed tool, not retrofitted.
3. *Deferred client work had no home.* US6 is recorded at P4 with its acceptance criteria explicitly
   left unwritten, so the server contract is designed knowing a client phase is coming without this
   spec pretending to specify it.
4. *A protocol version was almost pinned in the spec.* Removed. The specification revises on its own
   schedule; pinning belongs in `plan.md`, and the spec states the intent (current published version)
   rather than a number that will rot.

**Deliberate deviations from the generic checklist:**

- *"No implementation details"* passes for Requirements and Success Criteria. The **Context** section
  names the concrete tool projects (`inspect`, `harvest`, `capture`, …) because the inventory of what
  exists *is* the scope of what gets exposed, and an agent reading this needs it. No FR or SC names a
  transport, library, language, or serialisation format — those are planning decisions and the spec
  says so.
- *"Written for non-technical stakeholders"* is marked partial. The reader is the operator and the
  authors of the external orchestration project. Protocol vocabulary is unavoidable in a spec about
  conforming to a protocol.

**Open risks carried into planning (not spec defects):**

- **Per-tool result shaping.** Several tools today report by printing to standard output. FR-004
  requires structured results, but *what* the structured result is for each tool has to be decided per
  tool. This is the largest unestimated part of the work and `plan.md` must enumerate it tool by tool
  rather than treating "expose the tools" as uniform.
- **Transport and specification version** are deliberately unpinned here; `plan.md` pins both against
  the specification current at planning time.
- **Process/file locking.** The standing constraint that the viewer must be closed before tests run
  has a direct analogue in a long-lived server process holding build output open. Flagged in Edge
  Cases; the operational answer belongs in `plan.md`.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
