# Specification Quality Checklist: Minimap DXT1 Artifact Inversion

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-02
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

Validation performed 2026-08-02. Three iterations of self-review; issues found and corrected:

**Update 2026-08-02 (second pass)**: Spec extended with two user additions. (1) The **global lighting
normalisation** hypothesis — authored tiles of a map may share a common lighting baseline — is now a
first-class deliverable (FR-016, SC-011, Lighting Baseline entity) rather than an inherited
assumption, because it is a second, independent confound on top of the codec. (2) The synthesizer now
MUST emit a **DXT1-compressed parity companion** alongside the pristine render (FR-015, SC-010, Parity
Companion entity), so authored and synthetic tiles compare on equal terms without a comparison-time
encode step. Both additions keep the spec technology-agnostic: they describe the outcome (a shared
baseline measured and accounted for; a compressed variant available) without naming an implementation.

1. **First pass** named DXT1/RGB565 throughout the requirements. Those are the measured properties of
   the *input data*, not a technology choice being specified, so they are retained in Background and
   Assumptions but removed from FR/SC wording in favour of behavioural phrasing ("the same lossy
   encoding-and-decoding cycle the authored tile carries"). This keeps the requirements valid if a
   build turns out to use a different codec.

2. **Second pass** had success criteria written as model-quality metrics (PSNR-style). Replaced with
   outcome statements measurable without knowing the implementation — improvement over the encoded
   input, agreement after re-encoding, and change-on-undamaged-input.

3. **Third pass** added FR-013 and SC-007. The original draft assumed DXT1 everywhere on the strength
   of one map from one build; the survey requirement makes verifying that assumption a deliverable
   rather than a footnote. This matches the project's era-gating discipline, where an unrecognised
   build must be flagged rather than silently defaulted.

No [NEEDS CLARIFICATION] markers were needed. Two decisions that could have been clarifications were
resolved from existing project constraints instead:

- **Encoder fidelity** — initially written off as "unknown and unknowable" and moved out of scope.
  **Corrected 2026-08-02 after user challenge.** The BLP format and DXT1 decoding are fully public and
  already in this codebase (`SereniaBLPLib`, wrapped by `BlpRgbReader`) — there is nothing to
  reverse-engineer on the read side, and calling it unknowable was wrong. What is genuinely
  undetermined is which DXT1 *encoder implementation* Blizzard used, since encoding is a lossy fitting
  problem with many valid answers. But that is measurable, not unknowable, so it became FR-014/FR-015
  and SC-009 rather than an assumption. The measurement is also expected to show encoder choice
  barely matters: DXT1 re-encoding is near-idempotent on decoded data, because a decoded block's four
  colours already sit on a line between two exactly-RGB565-representable endpoints.
- **Super-resolution scope** — resolved as out of scope and separated by FR-012, because no
  client-side ground truth exists above native resolution and mixing it with artifact removal would
  make both unmeasurable.

**Resolved for planning — encoder availability (checked 2026-08-02)**: `SereniaBLPLib` is
**decode-only**. Its whole DXT surface is one method, `DXTDecompression.DecompressImage(...)`; there
is no compress/encode path. A copy lives in-tree at
`wow-viewer/libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/` (the `gillijimproject_refactor` copy is
read-only per project rules and must not be edited). A `BLPSharp` 0.1.0 package reference also exists
and has not been checked for encode support.

So the decode half is solved and in-tree, and **the encode half must be supplied**. That is the one
new component this feature needs. It is well-trodden — DXT1 block encoding is a small, fully
specified fitting problem — and FR-015 exists so the choice among candidates is made on measured
re-encode agreement rather than convenience.

**Risk to carry into planning**: User Story 3 is the only story that can fail on quality rather than
correctness. Its gate (FR-010, SC-006) is deliberately strict because a restoration model that
invents plausible terrain detail would be worse than no restoration — it would contaminate a
restoration corpus with convincing fabrications. Stories 1 and 2 deliver value independently and
should not be blocked on Story 3 succeeding.
