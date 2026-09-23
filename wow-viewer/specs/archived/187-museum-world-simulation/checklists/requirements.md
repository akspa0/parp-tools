# Specification Quality Checklist: Single-player museum world simulation

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-23
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

- **Split out of the earlier combined draft.** Ingest and content browsing moved to spec 186, which
  delivers standalone value without any simulation. This spec depends on 186 and adds nothing to its
  requirements — the same one-way rule spec 178 established for MCP, applied here between specs.
- **Determinism (FR-001, FR-007, FR-012) is the architectural spine and a design *resolution*, not a
  preference.** An automated renderer-proof loop and model-driven actors pull against each other: put
  model output inside the simulation and no run reproduces, so a passing scenario proves nothing and a
  failing one cannot be investigated. Modelling every intelligent actor as an *input source* whose
  decisions are recorded resolves it — the model becomes replayable without the model.
- **The integrity boundary (FR-008/009/010) is the other half.** The model's real job is replacing the
  per-entity scripting layer, not flavour text; that is what "a very simple set of core server needs"
  buys. But a model that paraphrases a stored fact corrupts the museum silently and unfalsifiably.
  Hence the split: stored facts and invariants never touch a model; open decisions may, validated and
  recorded; retrieval ranks but never authors. SC-009 is a path inspection, not prompt discipline.
- **FR-002 and FR-015 are stated separately on purpose.** One says the simulation implements no game
  protocol; the other says it never touches a live client's traffic. They are different claims and the
  project's footing depends on both, so neither is left implied.
- **FR-014 (no wall-clock dependence) is load-bearing for US5.** Scenarios must run faster than real
  time to be useful in a feedback loop; behaviour keyed to wall clock would make results depend on
  machine speed, reintroducing exactly the non-determinism FR-001 exists to remove.
- **FR-017 mirrors 186's no-normalisation rule at runtime.** 186 keeps contradictory sources side by
  side; a simulation must nonetheless pick one behaviour to execute. Requiring it to *report which
  source it used* preserves the distinction rather than blending it into an average that belongs to
  nobody.
- **FR-006's deletion test is what keeps MCP honest.** The user asked for MCP in and out, and that is
  honoured — as an adapter over a first-class library surface the viewer already uses in-process, not
  as a peer with a vote in core design. Spec 178 established that rule after an earlier draft let MCP
  shape the editor.
- **Model choice is deliberately absent from the requirements.** The assumptions name a Gemma-class
  model served through `llama.cpp` over MCP as the user's setup; every requirement says "a decision
  source with a fallback". The museum should outlive any particular model, and FR-011 means it still
  runs when the model does not.
