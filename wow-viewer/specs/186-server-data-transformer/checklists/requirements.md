# Specification Quality Checklist: Server data transformer and world content browsing

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

- **This spec was split out of an earlier combined draft** that mixed the transformer with a world
  simulation. The user's framing — "just building the tooling around letting someone go find all the
  sql db's" and "might as well make the transformer for the data" — makes the transformer a
  standalone deliverable with its own payoff: seeing NPCs, spells and dense text in the viewer needs
  no simulation at all. The simulation moved to spec 187, which depends on this one and adds nothing
  to its requirements.
- **Toolmaker, not archivist (FR-017).** The project ships the transformer; the operator supplies the
  sources and owns their licensing. This deliberately removes redistribution from the critical path —
  an earlier draft had it as a spec-level concern, which was the wrong owner.
- **Losslessness (FR-002) is justified as a tool property here, not as curatorial duty.** The server
  half is the half that erodes; client tables are preserved many times over. A transformer that drops
  an uninterpreted column destroys the specific fix the operator was trying to keep. SC-001's
  reconstruct-and-compare is the gate, and it is pass/fail rather than a percentage target.
- **FR-005 forbids the normalisation instinct.** Two projects disagreeing about one creature is a
  record of two people solving the same problem differently. Any merge that silently picks a winner
  destroys the most interesting and least recoverable content in the store.
- **FR-009 (label which half a fact came from) exists because the join is the point.** A spell's
  mechanics come from client tables; which creature casts it comes from a server database. Presenting
  them as one undifferentiated fact sheet would hide which parts are Blizzard's data and which are a
  fan project's reconstruction — exactly the distinction a user of this tool needs.
- **FR-010 and FR-016 are the integrity pair.** Unresolvable references are reported, never filled in;
  and no model output may stand in for a stored value. A plausible substitute is worse than a gap
  because it is unfalsifiable. SC-012 makes this mechanical — inspect the serving path — rather than a
  matter of prompt discipline.
- **FR-012 (dialect extensibility) reflects who chooses the sources.** The operator does. A transformer
  that only understands the dialects that happened to be on hand during development is not the tooling
  that was asked for, so extensibility is a requirement rather than a design preference.
- **FR-018 accepts partial sources on purpose.** Real dumps are truncated, mid-migration, or corrupted
  in places. Refusing them wholesale would lose recoverable work; the spec requires progress plus a
  gap report instead.
- **Open, deferred to planning**: which second dialect is targeted. SC-007 requires two plus
  demonstrated extensibility to a third, and the MaNGOS family is the obvious candidate given the
  projects the user named, but committing to a schema belongs where real dumps can be inspected.
